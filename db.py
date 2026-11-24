"""Shared database layer for the web app (app.py) and the camera script
(attendance_taker.py). Uses the same attendance.db file as the original
project, and upgrades an old database in place (no data is lost)."""
import csv
import os
import sqlite3
from datetime import datetime, timedelta

from werkzeug.security import generate_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "attendance.db")
FACES_DIR = os.path.join(BASE, "data", "data_faces_from_camera")

DEFAULT_SETTINGS = {
    "late_after": "09:30",          # check-ins after this time are marked Late
    "institution": "My College",    # shown in navbar + reports
}


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _columns(conn, table):
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def init_db():
    """Create tables and upgrade the original (name, time, date) table."""
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS departments (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL UNIQUE,      -- must match the face-folder name
        roll_no     TEXT,
        email       TEXT,
        department  TEXT,
        department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
        password_hash TEXT,
        active      INTEGER NOT NULL DEFAULT 1,
        created_at  TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS staff (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'teacher'   -- 'admin' or 'teacher'
    );
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attendance (
        name TEXT, time TEXT, date DATE, UNIQUE(name, date)
    );
    CREATE TABLE IF NOT EXISTS face_profiles (
        user_id             INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
        person_name         TEXT NOT NULL,
        embedding_dimension INTEGER NOT NULL,
        photo_count         INTEGER NOT NULL,
        updated_at          TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS audit_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at  TEXT NOT NULL,
        actor       TEXT,
        actor_role  TEXT,
        action      TEXT NOT NULL,
        target      TEXT,
        method      TEXT,
        status_code INTEGER,
        ip_address  TEXT
    );
    """)
    # Upgrade the original table with the new columns
    cols = _columns(conn, "attendance")
    for col, ddl in [("user_id", "INTEGER"),
                     ("status", "TEXT DEFAULT 'Present'"),
                     ("method", "TEXT DEFAULT 'face'"),
                     ("confidence", "REAL")]:
        if col not in cols:
            conn.execute(f"ALTER TABLE attendance ADD COLUMN {col} {ddl}")
    user_cols = _columns(conn, "users")
    if "department_id" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN department_id INTEGER")
    if "password_hash" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_att_date ON attendance(date)")
    # One attendance record per registered user per calendar day. The original
    # name/date constraint remains for backward compatibility with old data.
    conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_att_user_date
                    ON attendance(user_id, date) WHERE user_id IS NOT NULL""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_department ON users(department_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_logs(created_at DESC)")

    # Preserve existing department text while creating normalized department
    # records and linking every existing user to the matching record.
    now = datetime.now().isoformat(timespec="seconds")
    conn.execute("""INSERT OR IGNORE INTO departments(name, created_at)
                    SELECT DISTINCT TRIM(department), ? FROM users
                    WHERE department IS NOT NULL AND TRIM(department) != ''""", (now,))
    conn.execute("""UPDATE users SET department_id =
                    (SELECT id FROM departments WHERE departments.name = TRIM(users.department))
                    WHERE department IS NOT NULL AND TRIM(department) != ''""")

    for k, v in DEFAULT_SETTINGS.items():
        conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))

    # Old records: create a user for every name already in attendance
    conn.execute("""INSERT OR IGNORE INTO users(name, created_at)
                    SELECT DISTINCT name, ? FROM attendance WHERE name IS NOT NULL""", (now,))
    conn.execute("""UPDATE attendance SET user_id =
                    (SELECT id FROM users WHERE users.name = attendance.name)
                    WHERE user_id IS NULL""")

    # First run: create a default admin (CHANGE THE PASSWORD after logging in)
    if conn.execute("SELECT COUNT(*) FROM staff").fetchone()[0] == 0:
        conn.execute("INSERT INTO staff(username, password_hash, role) VALUES (?,?,?)",
                     ("admin", generate_password_hash("admin123"), "admin"))
    conn.commit()
    conn.close()


def ensure_department(conn, name):
    """Return the ID for a department, creating it when a non-empty name is supplied."""
    name = (name or "").strip()
    if not name:
        return None
    conn.execute("INSERT OR IGNORE INTO departments(name, created_at) VALUES (?,?)",
                 (name, datetime.now().isoformat(timespec="seconds")))
    return conn.execute("SELECT id FROM departments WHERE name=?", (name,)).fetchone()[0]


def upsert_face_profile(conn, user_id, person_name, embedding_dimension, photo_count):
    """Persist the face-profile metadata against the stable database user ID."""
    conn.execute("""INSERT INTO face_profiles
                    (user_id, person_name, embedding_dimension, photo_count, updated_at)
                    VALUES(?,?,?,?,?)
                    ON CONFLICT(user_id) DO UPDATE SET
                      person_name=excluded.person_name,
                      embedding_dimension=excluded.embedding_dimension,
                      photo_count=excluded.photo_count,
                      updated_at=excluded.updated_at""",
                 (user_id, person_name, embedding_dimension, photo_count,
                  datetime.now().isoformat(timespec="seconds")))


def rename_face_profile(conn, user_id, person_name):
    """Keep profile metadata aligned when a user name/folder is renamed."""
    conn.execute("UPDATE face_profiles SET person_name=?, updated_at=? WHERE user_id=?",
                 (person_name, datetime.now().isoformat(timespec="seconds"), user_id))


def delete_face_profile(conn, user_id):
    conn.execute("DELETE FROM face_profiles WHERE user_id=?", (user_id,))


def log_audit(actor, actor_role, action, target=None, method=None, status_code=None, ip_address=None):
    """Append a security-relevant action to the application audit trail."""
    conn = get_conn()
    try:
        conn.execute("""INSERT INTO audit_logs
                        (created_at, actor, actor_role, action, target, method, status_code, ip_address)
                        VALUES(?,?,?,?,?,?,?,?)""",
                     (datetime.now().isoformat(timespec="seconds"), actor, actor_role, action, target,
                      method, status_code, ip_address))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------- settings
def get_settings():
    conn = get_conn()
    s = {r["key"]: r["value"] for r in conn.execute("SELECT * FROM settings")}
    conn.close()
    return {**DEFAULT_SETTINGS, **s}


def set_setting(key, value):
    conn = get_conn()
    conn.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()
    conn.close()


# ----------------------------------------------------------------- logging
def log_attendance(name, method="face", confidence=None, when=None, conn=None, allow_create=True):
    """Mark `name` present for today. Returns (inserted: bool, status: str).

    Legacy camera scripts may auto-register unknown names. Web recognition sets
    ``allow_create=False`` so only users registered in the web database can
    receive attendance from a matched face embedding.
    """
    own = conn is None
    conn = conn or get_conn()
    when = when or datetime.now()
    date, tm = when.strftime("%Y-%m-%d"), when.strftime("%H:%M:%S")
    try:
        user = conn.execute("SELECT id, active FROM users WHERE name=?", (name,)).fetchone()
        if user is None:
            if not allow_create:
                return False, "unregistered"
            cur = conn.execute("INSERT INTO users(name, created_at) VALUES(?,?)",
                               (name, when.isoformat(timespec="seconds")))
            user_id = cur.lastrowid
        else:
            if not user["active"]:
                return False, "inactive"
            user_id = user["id"]
        late_after = get_settings()["late_after"]
        status = "Late" if tm[:5] > late_after else "Present"
        cur = conn.execute(
            "INSERT OR IGNORE INTO attendance(name,time,date,user_id,status,method,confidence)"
            " VALUES(?,?,?,?,?,?,?)", (name, tm, date, user_id, status, method, confidence))
        conn.commit()
        return cur.rowcount == 1, status
    finally:
        if own:
            conn.close()


# --------------------------------------------------------------- analytics
def today():
    return datetime.now().strftime("%Y-%m-%d")


def day_summary(date):
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM users WHERE active=1").fetchone()[0]
    rows = conn.execute("""SELECT status, COUNT(*) c FROM attendance
                           WHERE date=? GROUP BY status""", (date,)).fetchall()
    conn.close()
    by = {r["status"] or "Present": r["c"] for r in rows}
    present, late = by.get("Present", 0), by.get("Late", 0)
    marked = present + late
    return {"total": total, "present": present, "late": late, "marked": marked,
            "absent": max(total - marked, 0),
            "rate": round(100 * marked / total, 1) if total else 0.0}


def trend(days=14, end_date=None):
    """Attendance totals for the selected day and its preceding days."""
    end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else datetime.now().date()
    start = end - timedelta(days=days - 1)
    conn = get_conn()
    rows = conn.execute("SELECT date, COUNT(*) c FROM attendance WHERE date>=? GROUP BY date",
                        (start.isoformat(),)).fetchall()
    conn.close()
    got = {r["date"]: r["c"] for r in rows}
    labels = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    return {"labels": labels, "values": [got.get(d, 0) for d in labels]}


def hourly(date):
    conn = get_conn()
    rows = conn.execute("""SELECT substr(time,1,2) h, COUNT(*) c FROM attendance
                           WHERE date=? GROUP BY h""", (date,)).fetchall()
    conn.close()
    got = {r["h"]: r["c"] for r in rows}
    hours = [f"{h:02d}" for h in range(6, 20)]
    return {"labels": [f"{h}:00" for h in hours], "values": [got.get(h, 0) for h in hours]}


def department_summary(date):
    """Today's attendance totals and rates grouped by student department."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT COALESCE(NULLIF(u.department, ''), 'Unassigned') AS department,
               COUNT(u.id) AS total,
               COALESCE(SUM(a.user_id IS NOT NULL), 0) AS marked,
               COALESCE(SUM(a.status='Present'), 0) AS present,
               COALESCE(SUM(a.status='Late'), 0) AS late
        FROM users u LEFT JOIN attendance a ON a.user_id=u.id AND a.date=?
        WHERE u.active=1
        GROUP BY COALESCE(NULLIF(u.department, ''), 'Unassigned')
        ORDER BY department
    """, (date,)).fetchall()
    conn.close()
    items = []
    for row in rows:
        item = dict(row)
        item["absent"] = item["total"] - item["marked"]
        item["rate"] = round(100 * item["marked"] / item["total"], 1) if item["total"] else 0.0
        items.append(item)
    return items


def recent(limit=10, date=None):
    conn = get_conn()
    if date:
        rows = conn.execute("""SELECT name, time, date, status, method FROM attendance
                               WHERE date=? ORDER BY time DESC LIMIT ?""", (date, limit)).fetchall()
    else:
        rows = conn.execute("""SELECT name, time, date, status, method FROM attendance
                               ORDER BY date DESC, time DESC LIMIT ?""", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def student_report(start, end):
    """Per-student totals between two dates (inclusive). Working days = days
    on which at least one student was marked."""
    conn = get_conn()
    working = conn.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE date BETWEEN ? AND ?",
                           (start, end)).fetchone()[0]
    rows = conn.execute("""
        SELECT u.id, u.name, u.roll_no, u.department,
               COALESCE(SUM(a.status='Present'),0) AS present,
               COALESCE(SUM(a.status='Late'),0)    AS late
        FROM users u
        LEFT JOIN attendance a ON a.user_id=u.id AND a.date BETWEEN ? AND ?
        WHERE u.active=1 GROUP BY u.id ORDER BY u.name""", (start, end)).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        attended = d["present"] + d["late"]
        d["attended"] = attended
        d["absent"] = max(working - attended, 0)
        d["percent"] = round(100 * attended / working, 1) if working else 0.0
        out.append(d)
    return working, out


def has_face_data(name):
    """True if a face folder exists and the 128D embedding is trained."""
    if not os.path.isdir(FACES_DIR):
        return False
    folder_exists = any(f.split("_", 2)[-1] == name for f in os.listdir(FACES_DIR))
    if not folder_exists:
        return False
    # Check features_all.csv if present
    csv_file = os.path.join(BASE, "data", "features_all.csv")
    if os.path.isfile(csv_file):
        try:
            with open(csv_file, newline="") as f:
                for row in csv.reader(f):
                    if row and row[0] == name and len(row) >= 129:
                        return True
        except Exception:
            pass
    return folder_exists
