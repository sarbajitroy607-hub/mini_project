import csv
import io
import os
import re
import secrets
import sqlite3
import zipfile
from xml.sax.saxutils import escape as xml_escape
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, Response, abort, flash, jsonify, redirect,
                   g, render_template, request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import db
import face_engine

app = Flask(__name__)
# Set SECRET_KEY in the environment for real deployments.
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
db.init_db()


# ------------------------------------------------------------------ helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals["csrf_token"] = csrf_token


@app.before_request
def check_csrf():
    # Preserve the actor for logout logging, where the route clears session.
    g.audit_actor = session.get("user")
    g.audit_role = session.get("role")
    # Sign-up is deliberately public; every other state-changing API requires
    # an authenticated session and the normal CSRF token below.
    public_api = {"api_login", "api_signup"}
    if (request.method == "POST" and request.path.startswith("/api/")
            and request.endpoint not in public_api and "user" not in session):
        return jsonify(error="Session expired - log in again."), 401
    if request.endpoint in public_api:
        return None
    sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token")
    if request.method == "POST" and (not sent or sent != session.get("_csrf")):
        abort(400, "Invalid CSRF token - go back, refresh the page and try again.")


@app.errorhandler(400)
@app.errorhandler(403)
@app.errorhandler(404)
def handle_api_http_errors(e):
    if request.path.startswith("/api/"):
        code = getattr(e, "code", 400)
        desc = getattr(e, "description", str(e))
        return jsonify(error=desc), code
    return e


@app.after_request
def camera_security_headers(response):
    """Keep webcam access scoped to this Flask origin."""
    response.headers.setdefault("Permissions-Policy", "camera=(self)")
    return response


@app.after_request
def audit_state_changes(response):
    """Record meaningful state changes without logging passwords or camera frames."""
    if (request.method in {"POST", "PUT", "PATCH", "DELETE"}
            and request.endpoint not in {"api_scan", None}
            and response.status_code < 500):
        actor = session.get("user") or getattr(g, "audit_actor", None)
        role = session.get("role") or getattr(g, "audit_role", None)
        try:
            db.log_audit(actor, role, request.endpoint or request.path, request.path,
                         request.method, response.status_code, request.remote_addr)
        except sqlite3.Error:
            # Audit failure must never block the user's primary action.
            pass
    return response


@app.context_processor
def inject_globals():
    return {"settings": db.get_settings(), "current_user": session.get("user"),
            "role": session.get("role")}


STAFF_ROLES = frozenset({"admin", "teacher"})
VALID_ROLES = STAFF_ROLES | {"student"}


def login_required(admin_only=False, roles=None):
    """Require an authenticated staff member and enforce the route's role policy.

    Teachers may use normal staff routes.  Routes marked ``admin_only`` are
    reserved for administrators; API clients receive a JSON 403 while browser
    users are sent back to the dashboard with an explanatory message.
    """
    allowed_roles = set(roles) if roles is not None else ({"admin"} if admin_only else STAFF_ROLES)

    def deco(fn):
        @wraps(fn)
        def wrapper(*a, **kw):
            if "user" not in session:
                if request.path.startswith("/api/"):
                    return jsonify(error="Session expired - log in again."), 401
                return redirect(url_for("login", next=request.path))
            if session.get("role") not in allowed_roles:
                if request.path.startswith("/api/"):
                    return jsonify(error="Admin access required." if admin_only else "Access denied."), 403
                flash("Admin access required." if admin_only else "Access denied.", "danger")
                return redirect(url_for("student_portal" if session.get("role") == "student" else "dashboard"))
            return fn(*a, **kw)
        return wrapper
    return deco


def valid_date(s):
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------- auth
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,30}$")


@app.route("/api/signup", methods=["POST"])
def api_signup():
    """Create a self-service teacher account and sign the new user in.

    The API never accepts a role field: administrator accounts can only be
    created by an existing administrator through the staff-management route.
    """
    payload = request.get_json(silent=True) or request.form
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    password_confirm = str(payload.get("password_confirm", password))

    if not USERNAME_RE.fullmatch(username):
        return jsonify(error="Username must be 3-30 characters: letters, numbers, ., _, or -."), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400
    if password != password_confirm:
        return jsonify(error="Passwords do not match."), 400

    conn = db.get_conn()
    try:
        cur = conn.execute("INSERT INTO staff(username, password_hash, role) VALUES(?,?,?)",
                           (username, generate_password_hash(password), "teacher"))
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify(error="Username is already taken."), 409
    finally:
        conn.close()

    session.clear()
    session["user"], session["role"] = username, "teacher"
    return jsonify(message="Account created.", csrf_token=csrf_token(),
                   user={"id": cur.lastrowid, "username": username, "role": "teacher"}), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    """Authenticate a staff account and start a session for API clients."""
    payload = request.get_json(silent=True) or request.form
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))

    conn = db.get_conn()
    row = conn.execute("SELECT * FROM staff WHERE username=?", (username,)).fetchone()
    conn.close()
    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify(error="Wrong username or password."), 401

    session.clear()
    session["user"], session["role"] = row["username"], row["role"]
    return jsonify(message="Signed in.", csrf_token=csrf_token(),
                   user={"id": row["id"], "username": row["username"], "role": row["role"]})


@app.route("/api/logout", methods=["POST"])
@login_required()
def api_logout():
    """End the current authenticated API/browser session."""
    session.clear()
    return jsonify(message="Signed out.")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        conn = db.get_conn()
        row = conn.execute("SELECT * FROM staff WHERE username=?",
                           (request.form.get("username", "").strip(),)).fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], request.form.get("password", "")):
            session.clear()
            session["user"], session["role"] = row["username"], row["role"]
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//")
                            else url_for("dashboard"))
        flash("Wrong username or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/student/register", methods=["GET", "POST"])
def student_register():
    """Public student enrolment; face capture remains an admin-supervised step."""
    if request.method == "POST":
        fields = {key: request.form.get(key, "").strip()
                  for key in ("name", "roll_no", "email", "department")}
        password = request.form.get("password", "")
        password_confirm = request.form.get("password_confirm", "")
        if not NAME_RE.fullmatch(fields["name"]):
            flash("Enter a valid name using 2-40 letters, numbers, spaces, ., _, or -.", "danger")
        elif not fields["roll_no"]:
            flash("Roll number is required.", "danger")
        elif len(password) < 8:
            flash("Password must be at least 8 characters.", "danger")
        elif password != password_confirm:
            flash("Passwords do not match.", "danger")
        else:
            conn = db.get_conn()
            try:
                department_id = db.ensure_department(conn, fields["department"])
                conn.execute("""INSERT INTO users(name, roll_no, email, department, department_id, password_hash, created_at)
                                VALUES(?,?,?,?,?,?,?)""",
                             (fields["name"], fields["roll_no"], fields["email"], fields["department"],
                              department_id, generate_password_hash(password), datetime.now().isoformat(timespec="seconds")))
                conn.commit()
                flash(f"Registration complete for {fields['name']}. You can now sign in; ask an administrator to register your face before attendance can be marked.", "success")
                return redirect(url_for("student_login"))
            except sqlite3.IntegrityError:
                flash("A student with that name is already registered.", "danger")
            finally:
                conn.close()
    return render_template("student_register.html")


@app.route("/student/login", methods=["GET", "POST"])
def student_login():
    """Sign a student in using their roll number and registration password."""
    if request.method == "POST":
        roll_no = request.form.get("roll_no", "").strip()
        password = request.form.get("password", "")
        conn = db.get_conn()
        user = conn.execute("SELECT * FROM users WHERE roll_no=? ORDER BY id LIMIT 1", (roll_no,)).fetchone()
        conn.close()
        if user and user["active"] and user["password_hash"] and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user"], session["role"], session["student_id"] = user["name"], "student", user["id"]
            return redirect(url_for("student_portal"))
        flash("Invalid roll number or password, or the student account is inactive.", "danger")
    return render_template("student_login.html")


@app.route("/student/portal")
@login_required(roles={"student"})
def student_portal():
    """A student can view only their own profile and attendance records."""
    uid = session.get("student_id")
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    records = conn.execute("""SELECT date, time, status, method FROM attendance
                              WHERE user_id=? ORDER BY date DESC, time DESC LIMIT 60""", (uid,)).fetchall()
    conn.close()
    if not user:
        session.clear()
        return redirect(url_for("student_login"))
    start = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    working, rows = db.student_report(start, db.today())
    summary = next((row for row in rows if row["id"] == uid), None)
    return render_template("student_portal.html", user=user, records=records, summary=summary,
                           working=working, has_face=db.has_face_data(user["name"]))


@app.route("/api/student/attendance", methods=["POST"])
@login_required(roles={"student"})
def api_student_attendance():
    """Mark attendance only after the live frame matches the signed-in student."""
    uid = session.get("student_id")
    conn = db.get_conn()
    user = conn.execute("SELECT id, name, active FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not user or not user["active"]:
        return jsonify(error="Student account is inactive or unavailable."), 403
    if not db.has_face_data(user["name"]):
        return jsonify(error="Your face profile has not been registered yet. Contact an administrator."), 409
    img = _image_from_request()
    try:
        faces, _ = face_engine.recognize(img)
    except RuntimeError as error:
        return jsonify(error=str(error)), 503
    match = next((face for face in faces if face["name"] == user["name"]), None)
    if not match:
        return jsonify(error="Face verification failed. Ensure only your face is visible and try again."), 401
    confidence = round(max(0.0, 1 - match["distance"] / face_engine.THRESHOLD), 2)
    inserted, status = db.log_attendance(user["name"], "student_portal", confidence, allow_create=False)
    return jsonify(ok=True, marked=inserted, status=status, confidence=confidence,
                   message=("Attendance marked successfully." if inserted else
                            "Attendance was already marked for today."))


@app.route("/api/student/face/photo", methods=["POST"])
@login_required(roles={"student"})
def api_student_face_photo():
    """Allow a signed-in student to submit a photo for their face enrollment."""
    uid = session.get("student_id")
    conn = db.get_conn()
    user = conn.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not user:
        return jsonify(error="Student not found"), 404
    name = user["name"]
    img = _image_from_request()
    try:
        ok, msg, count = face_engine.save_photo(name, img)
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    return jsonify(ok=ok, message=msg, count=count)


@app.route("/api/student/face/train", methods=["POST"])
@login_required(roles={"student"})
def api_student_face_train():
    """Compute and save embeddings for the signed-in student's face profile."""
    uid = session.get("student_id")
    conn = db.get_conn()
    user = conn.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not user:
        return jsonify(error="Student not found"), 404
    name = user["name"]
    if face_engine.photo_count(name) < 3:
        return jsonify(ok=False, message="Capture at least 3 photos first (8-10 is best)."), 400
    try:
        ok = face_engine.train(name)
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    dimension = face_engine.embedding_dimension(name) if ok else 0
    photos = face_engine.photo_count(name)
    conn = db.get_conn()
    try:
        if ok:
            db.upsert_face_profile(conn, uid, name, dimension, photos)
        else:
            db.delete_face_profile(conn, uid)
        conn.commit()
    finally:
        conn.close()
    return jsonify(ok=ok, message="Face data saved - your profile is now registered for attendance!" if ok
                   else "No usable face found in the photos.",
                   embedding_dimension=dimension, photos_used=photos, user_id=uid)


@app.route("/account", methods=["GET", "POST"])
@login_required()
def account():
    if request.method == "POST":
        conn = db.get_conn()
        row = conn.execute("SELECT * FROM staff WHERE username=?", (session["user"],)).fetchone()
        new = request.form.get("new_password", "")
        if not check_password_hash(row["password_hash"], request.form.get("old_password", "")):
            flash("Current password is wrong.", "danger")
        elif len(new) < 6:
            flash("New password must be at least 6 characters.", "danger")
        else:
            conn.execute("UPDATE staff SET password_hash=? WHERE id=?",
                         (generate_password_hash(new), row["id"]))
            conn.commit()
            flash("Password changed.", "success")
        conn.close()
    return render_template("account.html")


@app.route("/audit-log")
@login_required(admin_only=True)
def audit_log():
    q = request.args.get("q", "").strip()
    conn = db.get_conn()
    if q:
        rows = conn.execute("""SELECT * FROM audit_logs
                               WHERE actor LIKE ? OR action LIKE ? OR target LIKE ?
                               ORDER BY id DESC LIMIT 200""", (f"%{q}%",) * 3).fetchall()
    else:
        rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 200").fetchall()
    conn.close()
    return render_template("audit_log.html", rows=rows, q=q)


@app.route("/api/audit-log")
@login_required(admin_only=True)
def api_audit_log():
    limit = min(max(request.args.get("limit", 100, type=int), 1), 200)
    conn = db.get_conn()
    rows = conn.execute("SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return jsonify(entries=[dict(row) for row in rows])


# ---------------------------------------------------------------- landing + dashboard
@app.route("/")
def landing():
    """Public entry point with working sign-in and teacher sign-up flows."""
    if "user" in session:
        return redirect(url_for("student_portal" if session.get("role") == "student" else "dashboard"))
    return render_template("landing.html")


@app.route("/dashboard")
@login_required()
def dashboard():
    date = request.args.get("date") or db.today()
    if not valid_date(date):
        abort(400, "Invalid date")
    return render_template("dashboard.html", today=date)


@app.route("/api/live")
@login_required()
def api_live():
    """Polled every few seconds by the dashboard for real-time updates."""
    d = request.args.get("date") or db.today()
    if not valid_date(d):
        return jsonify(error="Invalid date"), 400
    return jsonify(summary=db.day_summary(d), recent=db.recent(10, d),
                   hourly=db.hourly(d), trend=db.trend(14, d),
                   departments=db.department_summary(d),
                   server_time=datetime.now().strftime("%H:%M:%S"))


# --------------------------------------------------------- attendance log
@app.route("/attendance")
@login_required()
def attendance():
    date = request.args.get("date") or db.today()
    if not valid_date(date):
        abort(400, "Invalid date")
    q = request.args.get("q", "").strip()
    conn = db.get_conn()
    rows = conn.execute("""SELECT a.rowid AS id, a.name, a.time, a.status, a.method, u.roll_no
                           FROM attendance a LEFT JOIN users u ON u.id=a.user_id
                           WHERE a.date=? AND a.name LIKE ? ORDER BY a.time""",
                        (date, f"%{q}%")).fetchall()
    absent = conn.execute("""SELECT id, name, roll_no FROM users WHERE active=1 AND id NOT IN
                             (SELECT user_id FROM attendance WHERE date=? AND user_id IS NOT NULL)
                             ORDER BY name""", (date,)).fetchall()
    conn.close()
    return render_template("attendance.html", date=date, q=q, rows=rows, absent=absent,
                           summary=db.day_summary(date))


@app.route("/attendance/mark", methods=["POST"])
@login_required()
def mark_manual():
    date = request.form.get("date", "")
    time = request.form.get("time") or datetime.now().strftime("%H:%M")
    if not valid_date(date):
        abort(400, "Invalid date")
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (request.form.get("user_id"),)).fetchone()
    if not user:
        abort(404)
    status = request.form.get("status", "Present")
    if status not in ("Present", "Late"):
        abort(400)
    conn.execute("""INSERT OR REPLACE INTO attendance(name,time,date,user_id,status,method)
                    VALUES(?,?,?,?,?,'manual')""",
                 (user["name"], time + ":00" if len(time) == 5 else time, date, user["id"], status))
    conn.commit()
    conn.close()
    flash(f"Marked {user['name']} as {status}.", "success")
    return redirect(url_for("attendance", date=date))


@app.route("/attendance/<int:rowid>/delete", methods=["POST"])
@login_required(admin_only=True)
def delete_record(rowid):
    conn = db.get_conn()
    row = conn.execute("SELECT date FROM attendance WHERE rowid=?", (rowid,)).fetchone()
    if not row:
        abort(404)
    conn.execute("DELETE FROM attendance WHERE rowid=?", (rowid,))
    conn.commit()
    conn.close()
    flash("Record deleted.", "info")
    return redirect(url_for("attendance", date=row["date"]))


# ----------------------------------------------------------------- reports
def _range():
    end = request.args.get("end") or db.today()
    start = request.args.get("start") or (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    if not (valid_date(start) and valid_date(end)):
        abort(400, "Invalid date")
    return start, end


def _report_data(start, end):
    """Return the filtered summary and attendance records used by every export."""
    working, summary = db.student_report(start, end)
    conn = db.get_conn()
    records = [dict(row) for row in conn.execute(
        """SELECT a.date, a.time, a.name, u.roll_no, u.department, a.status, a.method, a.confidence
           FROM attendance a LEFT JOIN users u ON u.id=a.user_id
           WHERE a.date BETWEEN ? AND ? ORDER BY a.date, a.time""", (start, end))]
    conn.close()
    return working, summary, records


def _xlsx_cell(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"<c><v>{value}</v></c>"
    return f"<c t=\"inlineStr\"><is><t>{xml_escape(str(value or ''))}</t></is></c>"


def _xlsx_sheet(rows):
    body = "".join("<row>" + "".join(_xlsx_cell(value) for value in row) + "</row>" for row in rows)
    return ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">"
            f"<sheetData>{body}</sheetData></worksheet>")


def _xlsx_report(start, end, working, summary, records):
    """Create a dependency-free, Excel-compatible two-sheet attendance workbook."""
    summary_rows = [["Attendance Report", f"{start} to {end}"], ["Working days", working], []]
    summary_rows += [["Name", "Roll No", "Department", "Present", "Late", "Absent", "Working days", "Attendance %"]]
    summary_rows += [[r["name"], r["roll_no"] or "", r["department"] or "", r["present"], r["late"],
                      r["absent"], working, r["percent"]] for r in summary]
    record_rows = [["Date", "Time", "Name", "Roll No", "Department", "Status", "Method", "Confidence"]]
    record_rows += [[r["date"], r["time"], r["name"], r["roll_no"] or "", r["department"] or "",
                     r["status"], r["method"], r["confidence"] if r["confidence"] is not None else ""] for r in records]
    workbook = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" "
                "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
                "<sheets><sheet name=\"Summary\" sheetId=\"1\" r:id=\"rId1\"/>"
                "<sheet name=\"Records\" sheetId=\"2\" r:id=\"rId2\"/></sheets></workbook>")
    rels = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet1.xml\"/>"
            "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet2.xml\"/>"
            "</Relationships>")
    content_types = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                     "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
                     "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
                     "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
                     "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>"
                     "<Override PartName=\"/xl/worksheets/sheet1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
                     "<Override PartName=\"/xl/worksheets/sheet2.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>"
                     "</Types>")
    root_rels = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                 "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                 "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>"
                 "</Relationships>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", _xlsx_sheet(summary_rows))
        archive.writestr("xl/worksheets/sheet2.xml", _xlsx_sheet(record_rows))
    return buf.getvalue()


@app.route("/reports")
@login_required()
def reports():
    start, end = _range()
    working, rows = db.student_report(start, end)
    return render_template("reports.html", start=start, end=end, working=working, rows=rows)


@app.route("/reports/export.xlsx")
@login_required()
def export_excel():
    start, end = _range()
    working, summary, records = _report_data(start, end)
    data = _xlsx_report(start, end, working, summary, records)
    return Response(data,
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename=attendance_{start}_to_{end}.xlsx"})


@app.route("/reports/export.pdf")
@login_required()
def export_pdf():
    """Export the selected attendance summary as a print-ready PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph
    except ImportError:
        return jsonify(error="PDF export requires reportlab. Run pip install -r requirements.txt."), 503

    start, end = _range()
    working, summary, _ = _report_data(start, end)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter), rightMargin=28, leftMargin=28,
                            topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    institution = db.get_settings()["institution"]
    title = Paragraph(f"{institution} - Attendance Report", styles["Title"])
    subtitle = Paragraph(f"{start} to {end} | {working} working day(s)", styles["Normal"])
    data = [["Name", "Roll No", "Department", "Present", "Late", "Absent", "Attendance"]]
    data += [[r["name"], r["roll_no"] or "", r["department"] or "", r["present"], r["late"],
              r["absent"], f"{r['percent']}%"] for r in summary]
    table = Table(data, repeatRows=1, colWidths=[1.7*inch, 1.1*inch, 1.6*inch, .75*inch, .65*inch, .75*inch, .9*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#212529")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#ced4da")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("ALIGN", (3, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    doc.build([title, Spacer(1, 6), subtitle, Spacer(1, 14), table])
    return Response(buf.getvalue(), mimetype="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=attendance_{start}_to_{end}.pdf"})


@app.route("/reports/export.csv")
@login_required()
def export_csv():
    start, end = _range()
    kind = request.args.get("kind", "records")
    buf = io.StringIO()
    w = csv.writer(buf)
    if kind == "summary":
        working, rows = db.student_report(start, end)
        w.writerow(["Name", "Roll No", "Department", "Present", "Late", "Absent",
                    "Working days", "Attendance %"])
        for r in rows:
            w.writerow([r["name"], r["roll_no"] or "", r["department"] or "", r["present"],
                        r["late"], r["absent"], working, r["percent"]])
    else:
        conn = db.get_conn()
        w.writerow(["Date", "Time", "Name", "Roll No", "Status", "Method"])
        for r in conn.execute("""SELECT a.date, a.time, a.name, u.roll_no, a.status, a.method
                                 FROM attendance a LEFT JOIN users u ON u.id=a.user_id
                                 WHERE a.date BETWEEN ? AND ? ORDER BY a.date, a.time""",
                              (start, end)):
            w.writerow([r["date"], r["time"], r["name"], r["roll_no"] or "", r["status"], r["method"]])
        conn.close()
    return Response(buf.getvalue(), mimetype="text/csv", headers={
        "Content-Disposition": f"attachment; filename=attendance_{kind}_{start}_to_{end}.csv"})


# ---------------------------------------------------------- user management
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,38}[A-Za-z0-9]$")   # becomes a folder name -> keep it safe


@app.route("/api/users", methods=["POST"])
@login_required(admin_only=True)
def api_create_user():
    """Create a student/user record.  Only administrators may add users."""
    payload = request.get_json(silent=True) or request.form
    fields = {
        key: str(payload.get(key) or "").strip()
        for key in ("name", "roll_no", "email", "department")
    }
    if not NAME_RE.fullmatch(fields["name"]):
        return jsonify(error="Name must be 2-40 characters using letters, numbers, spaces, ., _, or -."), 400

    conn = db.get_conn()
    try:
        department_id = db.ensure_department(conn, fields["department"])
        cur = conn.execute(
            """INSERT INTO users(name, roll_no, email, department, department_id, created_at)
               VALUES(?,?,?,?,?,?)""",
            (fields["name"], fields["roll_no"], fields["email"], fields["department"], department_id,
             datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return jsonify(error="A user with that name already exists."), 409
    finally:
        conn.close()

    return jsonify(message="User created.", user={"id": cur.lastrowid, **fields,
                   "department_id": department_id, "active": True}), 201


@app.route("/api/users/<int:uid>", methods=["PATCH", "PUT"])
@login_required(admin_only=True)
def api_update_user(uid):
    """Update a student's profile.  This route is restricted to admins."""
    payload = request.get_json(silent=True) or request.form
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        conn.close()
        return jsonify(error="User not found."), 404

    fields = {
        key: str(payload.get(key, user[key]) or "").strip()
        for key in ("name", "roll_no", "email", "department")
    }
    if not NAME_RE.fullmatch(fields["name"]):
        conn.close()
        return jsonify(error="Name must be 2-40 characters using letters, numbers, spaces, ., _, or -."), 400
    name_owner = conn.execute("SELECT id FROM users WHERE name=? AND id!=?", (fields["name"], uid)).fetchone()
    if name_owner:
        conn.close()
        return jsonify(error="A user with that name already exists."), 409

    active_value = payload.get("active", user["active"])
    if isinstance(active_value, str):
        active_value = active_value.strip().lower() not in {"0", "false", "no", "off"}
    active = 1 if bool(active_value) else 0
    try:
        department_id = db.ensure_department(conn, fields["department"])
        if fields["name"] != user["name"]:
            face_engine.rename_person(user["name"], fields["name"])
        conn.execute("""UPDATE users SET name=?, roll_no=?, email=?, department=?, department_id=?, active=? WHERE id=?""",
                     (fields["name"], fields["roll_no"], fields["email"], fields["department"], department_id,
                      active, uid))
        if fields["name"] != user["name"]:
            db.rename_face_profile(conn, uid, fields["name"])
        conn.execute("UPDATE attendance SET name=? WHERE user_id=?", (fields["name"], uid))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.rollback()
        return jsonify(error="A user with that name already exists."), 409
    finally:
        conn.close()

    return jsonify(message="User updated.", user={"id": uid, **fields,
                   "department_id": department_id, "active": bool(active)})


@app.route("/api/users/<int:uid>/deactivate", methods=["POST"])
@login_required(admin_only=True)
def api_deactivate_user(uid):
    """Deactivate a student so their face scans no longer mark attendance."""
    conn = db.get_conn()
    user = conn.execute("SELECT id, name, active FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        conn.close()
        return jsonify(error="User not found."), 404
    conn.execute("UPDATE users SET active=0 WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    return jsonify(message="User deactivated.", user={"id": uid, "name": user["name"], "active": False})


@app.route("/api/users/<int:uid>/activate", methods=["POST"])
@login_required(admin_only=True)
def api_activate_user(uid):
    """Reactivate a previously deactivated student."""
    conn = db.get_conn()
    user = conn.execute("SELECT id, name FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        conn.close()
        return jsonify(error="User not found."), 404
    conn.execute("UPDATE users SET active=1 WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    return jsonify(message="User activated.", user={"id": uid, "name": user["name"], "active": True})


def _filtered_users(conn, q="", status="all", department=""):
    """Return users matching the shared web-page/API search filters."""
    clauses, params = [], []
    q = q.strip()
    if q:
        clauses.append("(name LIKE ? OR roll_no LIKE ? OR email LIKE ? OR department LIKE ?)")
        params.extend([f"%{q}%"] * 4)
    if status == "active":
        clauses.append("active=1")
    elif status == "inactive":
        clauses.append("active=0")
    if department:
        clauses.append("department=?")
        params.append(department)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return conn.execute(f"SELECT * FROM users{where} ORDER BY active DESC, name", params).fetchall()


@app.route("/api/users", methods=["GET"])
@login_required()
def api_users():
    """Search and filter users by text, activity status, and department."""
    q = request.args.get("q", "")
    status = request.args.get("status", "all").lower()
    department = request.args.get("department", "").strip()
    if status not in {"all", "active", "inactive"}:
        return jsonify(error="status must be all, active, or inactive."), 400
    conn = db.get_conn()
    rows = _filtered_users(conn, q, status, department)
    conn.close()
    return jsonify(users=[dict(row) for row in rows], count=len(rows),
                   filters={"q": q, "status": status, "department": department})


@app.route("/api/departments", methods=["GET"])
@login_required()
def api_departments():
    """List departments with the number of linked users."""
    conn = db.get_conn()
    rows = conn.execute("""SELECT d.id, d.name, COUNT(u.id) AS user_count
                           FROM departments d LEFT JOIN users u ON u.department_id=d.id
                           GROUP BY d.id, d.name ORDER BY d.name""").fetchall()
    conn.close()
    return jsonify(departments=[dict(row) for row in rows])


@app.route("/users")
@login_required()
def users():
    conn = db.get_conn()
    q = request.args.get("q", "")
    status = request.args.get("status", "all").lower()
    department = request.args.get("department", "").strip()
    if status not in {"all", "active", "inactive"}:
        status = "all"
    rows = _filtered_users(conn, q, status, department)
    departments = [r["department"] for r in conn.execute(
        "SELECT DISTINCT department FROM users WHERE department IS NOT NULL AND department != '' ORDER BY department")]
    staff = conn.execute("SELECT id, username, role FROM staff ORDER BY username").fetchall()
    conn.close()
    face = {r["name"]: db.has_face_data(r["name"]) for r in rows}
    return render_template("users.html", users=rows, staff=staff, face=face, departments=departments,
                           q=q, status=status, department=department)


@app.route("/users/new", methods=["GET", "POST"])
@app.route("/users/<int:uid>/edit", methods=["GET", "POST"])
@login_required(admin_only=True)
def user_form(uid=None):
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone() if uid else None
    if uid and not user:
        abort(404)
    if request.method == "POST":
        f = {k: request.form.get(k, "").strip() for k in ("name", "roll_no", "email", "department")}
        if not NAME_RE.match(f["name"]):
            flash("Name: 2-40 chars, letters/digits/space/._- only, starting and ending with a letter or digit (it must match the face folder name).", "danger")
        else:
            try:
                if user:
                    if user["name"] != f["name"]:
                        face_engine.rename_person(user["name"], f["name"])
                    department_id = db.ensure_department(conn, f["department"])
                    conn.execute("""UPDATE users SET name=?, roll_no=?, email=?, department=?, department_id=?, active=?
                                    WHERE id=?""", (f["name"], f["roll_no"], f["email"], f["department"], department_id,
                                                    1 if request.form.get("active") else 0, uid))
                    if user["name"] != f["name"]:
                        db.rename_face_profile(conn, uid, f["name"])
                    conn.execute("UPDATE attendance SET name=? WHERE user_id=?", (f["name"], uid))
                else:
                    department_id = db.ensure_department(conn, f["department"])
                    conn.execute("INSERT INTO users(name,roll_no,email,department,department_id,created_at) VALUES(?,?,?,?,?,?)",
                                 (f["name"], f["roll_no"], f["email"], f["department"], department_id,
                                  datetime.now().isoformat(timespec="seconds")))
                conn.commit()
                flash("User saved.", "success")
                conn.close()
                return redirect(url_for("users"))
            except sqlite3.IntegrityError:
                flash("A user with that name already exists.", "danger")
    conn.close()
    return render_template("user_form.html", user=user)


@app.route("/users/<int:uid>/toggle", methods=["POST"])
@login_required(admin_only=True)
def user_toggle(uid):
    conn = db.get_conn()
    conn.execute("UPDATE users SET active = 1 - active WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    return redirect(url_for("users"))


@app.route("/users/<int:uid>/delete", methods=["POST"])
@login_required(admin_only=True)
def user_delete(uid):
    conn = db.get_conn()
    row = conn.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        abort(404)
    conn.execute("DELETE FROM attendance WHERE user_id=?", (uid,))
    db.delete_face_profile(conn, uid)
    conn.execute("DELETE FROM users WHERE id=?", (uid,))
    conn.commit()
    conn.close()
    face_engine.delete_person(row["name"])
    flash("User, their attendance records and face data deleted.", "info")
    return redirect(url_for("users"))


@app.route("/users/<int:uid>")
@login_required()
def user_detail(uid):
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    if not user:
        abort(404)
    recs = conn.execute("SELECT date, time, status, method FROM attendance WHERE user_id=? "
                        "ORDER BY date DESC LIMIT 60", (uid,)).fetchall()
    conn.close()
    start = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    working, rows = db.student_report(start, db.today())
    me = next((r for r in rows if r["id"] == uid), None)
    return render_template("user_detail.html", user=user, recs=recs, me=me, working=working,
                           has_face=db.has_face_data(user["name"]))


@app.route("/api/users/<int:uid>/profile")
@login_required()
def api_user_profile(uid):
    """Return one user's profile, enrollment state, and recent attendance."""
    conn = db.get_conn()
    user = conn.execute("""SELECT u.*, COALESCE(d.name, u.department) AS department_name,
                                  fp.embedding_dimension AS face_embedding_dimension,
                                  fp.photo_count AS face_photo_count, fp.updated_at AS face_profile_updated_at
                           FROM users u LEFT JOIN departments d ON d.id=u.department_id
                           LEFT JOIN face_profiles fp ON fp.user_id=u.id
                           WHERE u.id=?""", (uid,)).fetchone()
    if not user:
        conn.close()
        return jsonify(error="User not found."), 404
    attendance = conn.execute("""SELECT date, time, status, method, confidence
                                 FROM attendance WHERE user_id=? ORDER BY date DESC, time DESC LIMIT 60""",
                              (uid,)).fetchall()
    conn.close()

    start = (datetime.now() - timedelta(days=29)).strftime("%Y-%m-%d")
    working, reports = db.student_report(start, db.today())
    summary = next((row for row in reports if row["id"] == uid), None)
    profile = dict(user)
    profile["department"] = profile.pop("department_name")
    profile["active"] = bool(profile["active"])
    embedding_dimension = profile.pop("face_embedding_dimension") or 0
    profile["face_profile"] = {
        "generated": embedding_dimension == 128,
        "embedding_dimension": embedding_dimension,
        "photos": profile.pop("face_photo_count") or 0,
        "updated_at": profile.pop("face_profile_updated_at", None),
    }
    return jsonify(user=profile, attendance=[dict(row) for row in attendance],
                   attendance_summary={"working_days": working, **dict(summary)} if summary else None)


@app.route("/users/rebuild-features", methods=["POST"])
@login_required(admin_only=True)
def rebuild_features():
    """Re-train every person from their photos (same result as features_extraction_to_csv.py)."""
    try:
        n = face_engine.train_all()
        flash(f"Face data rebuilt for {n} person(s).", "success")
    except Exception as e:
        flash(f"Could not rebuild face data: {e}", "danger")
    return redirect(url_for("users"))


# ------------------------------------------------- face scan + registration
def _image_from_request():
    f = request.files.get("image")
    if not f:
        abort(400, "No image")
    try:
        return face_engine.decode_image(f.read())
    except ValueError as e:
        abort(400, str(e))


@app.route("/scan")
@login_required()
def scan():
    ok, msg = face_engine.status()
    names, _ = face_engine.load_known()
    return render_template("scan.html", engine_ok=ok, engine_msg=msg, known=len(names))


@app.route("/api/scan", methods=["POST"])
@login_required()
def api_scan():
    """Receives one webcam frame, returns face boxes + names, and marks attendance."""
    img = _image_from_request()
    try:
        faces, (w, h) = face_engine.recognize(img)
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    recognized_names = [f["name"] for f in faces if f["name"]]
    user_ids = {}
    if recognized_names:
        conn = db.get_conn()
        marks = ",".join("?" for _ in recognized_names)
        user_ids = {row["name"]: row["id"] for row in conn.execute(
            f"SELECT id, name FROM users WHERE name IN ({marks})", recognized_names)}
        conn.close()
    out = []
    for f in faces:
        item = {"box": f["box"], "name": f["name"], "status": "unknown", "time": None,
                "user_id": user_ids.get(f["name"]), "confidence": None}
        if f["name"]:
            conf = round(max(0.0, 1 - f["distance"] / face_engine.THRESHOLD), 2)
            item["confidence"] = conf
            inserted, status = db.log_attendance(f["name"], "face", conf, allow_create=False)
            item["status"] = ("inactive" if status == "inactive" else
                              "unregistered" if status == "unregistered" else
                              "marked" if inserted else "already")
            item["label"] = status if inserted or status == "unregistered" else None
        out.append(item)
    return jsonify(faces=out)


@app.route("/face-registration")
@login_required()
def face_registration_hub():
    """Face Registration directory: view biometric enrollment status and enroll faces."""
    conn = db.get_conn()
    users = conn.execute("SELECT id, name, roll_no, department, active FROM users ORDER BY name COLLATE NOCASE").fetchall()
    conn.close()
    profiles = {u["name"]: face_engine.photo_count(u["name"]) for u in users}
    embeddings = {u["name"]: (face_engine.embedding_dimension(u["name"]) > 0) for u in users}
    ok, msg = face_engine.status()
    return render_template("face_registration_hub.html", users=users, profiles=profiles,
                           embeddings=embeddings, engine_ok=ok, engine_msg=msg)


@app.route("/users/<int:uid>/face")
@login_required()
def face_register(uid):
    conn = db.get_conn()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not user:
        abort(404)
    ok, msg = face_engine.status()
    return render_template("face_register.html", user=user, engine_ok=ok, engine_msg=msg,
                           count=face_engine.photo_count(user["name"]))


def _user_name(uid):
    conn = db.get_conn()
    row = conn.execute("SELECT name FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not row:
        abort(404)
    return row["name"]


@app.route("/api/users/<int:uid>/face/photo", methods=["POST"])
@login_required()
def api_face_photo(uid):
    name = _user_name(uid)
    img = _image_from_request()
    try:
        ok, msg, count = face_engine.save_photo(name, img)
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    return jsonify(ok=ok, message=msg, count=count)


@app.route("/api/users/<int:uid>/face/train", methods=["POST"])
@login_required()
def api_face_train(uid):
    name = _user_name(uid)
    if face_engine.photo_count(name) < 3:
        return jsonify(ok=False, message="Capture at least 3 photos first (8-10 is best)."), 400
    try:
        ok = face_engine.train(name)
    except RuntimeError as e:
        return jsonify(error=str(e)), 503
    dimension = face_engine.embedding_dimension(name) if ok else 0
    photos = face_engine.photo_count(name)
    conn = db.get_conn()
    try:
        if ok:
            db.upsert_face_profile(conn, uid, name, dimension, photos)
        else:
            db.delete_face_profile(conn, uid)
        conn.commit()
    finally:
        conn.close()
    return jsonify(ok=ok, message="Face data saved - this person can now be recognised." if ok
                   else "No usable face found in the photos.",
                   embedding_dimension=dimension, photos_used=photos, user_id=uid)


@app.route("/api/users/<int:uid>/face/embedding")
@login_required()
def api_face_embedding_status(uid):
    """Report whether a user's 128D face embedding has been generated.

    Biometric vector values intentionally remain server-side; callers only
    receive the enrollment status and dimensionality.
    """
    name = _user_name(uid)
    conn = db.get_conn()
    profile = conn.execute("SELECT * FROM face_profiles WHERE user_id=?", (uid,)).fetchone()
    conn.close()
    dimension = profile["embedding_dimension"] if profile else 0
    return jsonify(user_id=uid, generated=dimension == 128, embedding_dimension=dimension,
                   photos=profile["photo_count"] if profile else face_engine.photo_count(name),
                   updated_at=profile["updated_at"] if profile else None)


@app.route("/users/<int:uid>/face/delete", methods=["POST"])
@login_required()
def face_delete(uid):
    face_engine.delete_person(_user_name(uid))
    conn = db.get_conn()
    db.delete_face_profile(conn, uid)
    conn.commit()
    conn.close()
    flash("Face data deleted.", "info")
    return redirect(url_for("users"))


# ----- staff accounts + settings (admin)
@app.route("/staff/new", methods=["POST"])
@login_required(admin_only=True)
def staff_new():
    u, p, r = (request.form.get("username", "").strip(), request.form.get("password", ""),
               request.form.get("role", "teacher"))
    if not re.match(r"^[A-Za-z0-9_.-]{3,30}$", u) or len(p) < 6 or r not in ("admin", "teacher"):
        flash("Username 3-30 chars, password 6+ chars, valid role required.", "danger")
    else:
        try:
            conn = db.get_conn()
            conn.execute("INSERT INTO staff(username,password_hash,role) VALUES(?,?,?)",
                         (u, generate_password_hash(p), r))
            conn.commit()
            conn.close()
            flash(f"Staff account '{u}' created.", "success")
        except sqlite3.IntegrityError:
            flash("Username already taken.", "danger")
    return redirect(url_for("users") + "#staff")


@app.route("/staff/<int:sid>/delete", methods=["POST"])
@login_required(admin_only=True)
def staff_delete(sid):
    conn = db.get_conn()
    row = conn.execute("SELECT username FROM staff WHERE id=?", (sid,)).fetchone()
    admins = conn.execute("SELECT COUNT(*) FROM staff WHERE role='admin'").fetchone()[0]
    target = conn.execute("SELECT role FROM staff WHERE id=?", (sid,)).fetchone()
    if not row:
        abort(404)
    if row["username"] == session["user"] or (target["role"] == "admin" and admins <= 1):
        flash("You can't delete yourself or the last admin.", "danger")
    else:
        conn.execute("DELETE FROM staff WHERE id=?", (sid,))
        conn.commit()
        flash("Staff account deleted.", "info")
    conn.close()
    return redirect(url_for("users") + "#staff")


@app.route("/settings", methods=["POST"])
@login_required(admin_only=True)
def save_settings():
    late = request.form.get("late_after", "")
    if re.match(r"^([01]\d|2[0-3]):[0-5]\d$", late):
        db.set_setting("late_after", late)
    inst = request.form.get("institution", "").strip()[:60]
    if inst:
        db.set_setting("institution", inst)
    flash("Settings saved.", "success")
    return redirect(url_for("users") + "#staff")


if __name__ == "__main__":
    app.run(debug=True)
