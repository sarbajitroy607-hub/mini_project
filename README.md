# Face Recognition Based Attendance System

A Flask-based face recognition attendance system for college use. It provides
browser-based face registration and scanning, real-time attendance monitoring,
user management, reports, audit logs, and administrator controls.

**Developed by:** Sarbajit Roy

## Features

- Secure staff login with administrator and teacher roles
- Real-time face recognition through the browser
- Individual face enrollment using a webcam
- Automatic and manual attendance marking
- Present, late, and absent status tracking
- Daily attendance records and date-based filtering
- Date-range reports with CSV, Excel, PDF, and print output
- Student and staff account management
- Department and institution settings
- Live dashboard analytics and recent check-in activity
- Audit logging for important system actions

## Requirements

- Python 3.10, 3.11, or 3.12
- Webcam access
- Python packages listed in `requirements.txt`
- Google Chrome or another modern browser

Python 3.14 is not recommended because some face-recognition dependencies
may not support it.

## Installation

1. Open a terminal in the project directory.

2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the application:

   ```powershell
   python app.py
   ```

5. Open `http://127.0.0.1:5000` in the browser.

For the initial login, use:

- Username: `admin`
- Password: `admin123`

The default password should be changed from the account menu after logging in.

## How to Use

### 1. Add a user

Open **User Management** and add a student or staff profile with a name,
roll or employee number, department, and email address.

### 2. Register a face

Open the person's profile and select **Register face**. Start the webcam,
capture several photos from slightly different angles, and select
**Save face data**.

Use clear lighting and keep only one person visible while registering.

### 3. Take attendance

Open **Live Attendance** and start the camera. The system detects registered
faces and creates one attendance record for each person per day.

Green results indicate a newly marked person. Blue results indicate that a
person is already marked for the day.

### 4. View records and reports

- Use **Attendance Records** to search by date or name.
- Administrators can delete incorrect records or manually mark attendance.
- Use **Reports & Analytics** to generate date-range attendance summaries.
- Export summaries as CSV, Excel, or PDF.

### 5. Optional demo data

To populate the database with sample users and attendance records:

```powershell
python seed_demo.py
```

## Camera Requirements

The browser can use the webcam on `localhost`, `127.0.0.1`, or an HTTPS
website. Camera access must be allowed in the browser.

When testing locally, use:

```text
http://127.0.0.1:5000
```

## Project Structure

| File or folder | Purpose |
|----------------|---------|
| `app.py` | Flask application and routes |
| `db.py` | SQLite database operations |
| `face_engine.py` | Face detection and recognition engine |
| `attendance_taker.py` | Optional standalone camera attendance script |
| `features_extraction_to_csv.py` | Face feature extraction utility |
| `get_faces_from_camera_tkinter.py` | Optional desktop face-capture utility |
| `seed_demo.py` | Demo data generator |
| `templates/` | HTML pages |
| `static/` | CSS and frontend assets |
| `data/` | Registered face images and feature data |
| `attendance.db` | SQLite attendance database |

## Configuration

Set a secret key before deploying outside a trusted local environment:

```powershell
$env:SECRET_KEY="replace-with-a-random-secret"
python app.py
```

The institution name and late-attendance cut-off time can be changed from
**User Management > System Settings** when signed in as an administrator.

## Limitations

- Recognition accuracy depends on image quality and lighting.
- This project uses SQLite and Flask's development server, so it is suitable
  for a college mini-project and small local deployments rather than a large
  production environment.
- Face recognition without liveness detection can potentially be fooled by a
  photograph or a screen.

