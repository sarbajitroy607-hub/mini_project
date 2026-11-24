"""Face detection / recognition used by the web app (Scan + Register pages).

Uses the SAME dlib models and the SAME data files as the original scripts:
  data/data_faces_from_camera/person_<n>_<name>/*.jpg   (photos)
  data/features_all.csv                                  (name + 128 numbers per row)
so attendance_taker.py and the web app can be used side by side.

Importing this module never fails: if dlib or the model files are missing,
`status()` explains what is wrong and the web pages show that message."""
import csv
import glob
import importlib.util
import os
import shutil
import threading

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(BASE, "data", "data_faces_from_camera")
FEATURES_CSV = os.path.join(BASE, "data", "features_all.csv")
MODEL_DIR = os.path.join(BASE, "data", "data_dlib")
LANDMARK_FILE = "shape_predictor_68_face_landmarks.dat"
RESNET_FILE = "dlib_face_recognition_resnet_model_v1.dat"

# Same value as the original attendance_taker.py (smaller = stricter).
THRESHOLD = float(os.environ.get("FACE_THRESHOLD", "0.4"))
SCAN_WIDTH = 480            # frames are shrunk to this width before detection (speed)

_lock = threading.Lock()    # dlib objects are not thread-safe
_state = {"loaded": False, "error": None}
_known = {"mtime": None, "names": [], "matrix": np.zeros((0, 128))}


# ------------------------------------------------------------------ loading
def _find_model(filename):
    p = os.path.join(MODEL_DIR, filename)
    if os.path.isfile(p):
        return p
    # fallback: `pip install face_recognition_models` ships the same two files
    spec = importlib.util.find_spec("face_recognition_models")
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            p = os.path.join(loc, "models", filename)
            if os.path.isfile(p):
                return p
    return None


def _load():
    if _state["loaded"] or _state["error"]:
        return
    try:
        import cv2  # noqa: F401
        import dlib
    except ImportError as e:
        _state["error"] = f"Missing package: {e.name}. Run: pip install -r requirements.txt"
        return
    lm, rn = _find_model(LANDMARK_FILE), _find_model(RESNET_FILE)
    if not lm or not rn:
        _state["error"] = ("dlib model files not found. Either put the two .dat files in "
                           "data/data_dlib/ or run: pip install face_recognition_models")
        return
    _state["dlib"] = dlib
    _state["detector"] = dlib.get_frontal_face_detector()
    _state["predictor"] = dlib.shape_predictor(lm)
    _state["model"] = dlib.face_recognition_model_v1(rn)
    _state["loaded"] = True


def status():
    """(ok, message)"""
    _load()
    return (True, "ready") if _state["loaded"] else (False, _state["error"])


# ------------------------------------------------------------------ helpers
def decode_image(data: bytes):
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a valid image")
    return img


def _rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _descriptor(rgb, rect):
    shape = _state["predictor"](rgb, rect)
    return np.array(_state["model"].compute_face_descriptor(rgb, shape))


def _largest(faces):
    return max(faces, key=lambda r: r.width() * r.height())


# ----------------------------------------------------------- liveness & anti-spoofing
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


def calculate_ear(eye_pts):
    """Eye Aspect Ratio (EAR) for blink detection."""
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    h = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return float((v1 + v2) / (2.0 * max(h, 1e-6)))


def estimate_head_pose(pts, img_shape):
    """Estimate 3D head pose (pitch, yaw, roll) using solvePnP."""
    import cv2
    image_points = np.array([
        pts[30], pts[8], pts[36], pts[45], pts[48], pts[54]
    ], dtype=np.float64)
    h, w = img_shape[:2]
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, _ = cv2.solvePnP(_MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
    return float(angles[0]), float(angles[1]), float(angles[2])


def evaluate_liveness(bgr_img, rect, shape):
    """Multi-layer passive & active liveness and anti-spoofing evaluation.
    
    Analyzes:
      1. Eye aspect ratio (blinking & open eye geometry).
      2. 3D perspective head pose realism.
      3. YCrCb human skin chrominance density vs digital screen / print spectrum.
      4. High-frequency Laplacian texture & specular screen glare.
      
    Returns (is_real: bool, score: float, reason: str, metrics: dict).
    """
    import cv2
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float64)
    h, w = bgr_img.shape[:2]
    left = max(0, int(rect.left()))
    top = max(0, int(rect.top()))
    right = min(w, int(rect.right()))
    bottom = min(h, int(rect.bottom()))
    crop = bgr_img[top:bottom, left:right]
    
    if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
        return False, 0.0, "Face region too small to assess", {}
        
    # 1. Eye Aspect Ratio (EAR)
    left_ear = calculate_ear(pts[36:42])
    right_ear = calculate_ear(pts[42:48])
    ear = (left_ear + right_ear) / 2.0
    
    # 2. Head Pose (pitch, yaw, roll)
    pitch, yaw, roll = estimate_head_pose(pts, (h, w))
    
    # 3. YCrCb Skin Chrominance Density
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    skin_mask = (y >= 15) & (y <= 245) & (cr >= 125) & (cr <= 180) & (cb >= 75) & (cb <= 135)
    skin_ratio = float(np.mean(skin_mask))
    
    # 4. Specular Reflection / Glare (common on phone screens & glossy photos)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    glare_ratio = float(np.mean(gray >= 250))
    
    # 5. Texture & Laplacian variance
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Scoring system (0.0 to 1.0)
    score = 0.5
    reasons = []
    
    # Skin color realism
    if skin_ratio >= 0.70:
        score += 0.25
    elif skin_ratio >= 0.45:
        score += 0.10
    else:
        score -= 0.35
        reasons.append("Abnormal skin spectrum (Screen/Print detected)")
        
    # Screen Glare check
    if glare_ratio > 0.08:
        score -= 0.30
        reasons.append("Excessive specular glare (Digital screen reflection)")
    elif glare_ratio < 0.04:
        score += 0.10
        
    # Texture / sharpness check
    if lap_var < 15.0:
        score -= 0.25
        reasons.append("Texture lacks detail / blurry image")
    elif 30.0 <= lap_var <= 1200.0:
        score += 0.15
        
    # Natural 3D pose (extreme distorted 2D perspective)
    if abs(yaw) > 60.0 or abs(roll) > 50.0:
        score -= 0.20
        reasons.append("Unrealistic face angle")
        
    score = float(np.clip(score, 0.0, 1.0))
    is_real = score >= 0.52
    reason_str = "Live person verified" if is_real else (reasons[0] if reasons else "Spoof suspected")
    
    metrics = {
        "score": round(score, 2),
        "ear": round(ear, 3),
        "skin_ratio": round(skin_ratio, 2),
        "glare_ratio": round(glare_ratio, 3),
        "lap_var": round(lap_var, 1),
        "pose": {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}
    }
    return is_real, score, reason_str, metrics


# ----------------------------------------------------------- known faces
def load_known(force=False):
    """(names, matrix) from features_all.csv; reloads when the file changes."""
    try:
        m = os.path.getmtime(FEATURES_CSV)
    except OSError:
        _known.update(mtime=None, names=[], matrix=np.zeros((0, 128)))
        return _known["names"], _known["matrix"]
    if force or _known["mtime"] != m:
        names, rows = [], []
        with open(FEATURES_CSV, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 129:
                    continue
                try:
                    vec = [float(x) if x != "" else 0.0 for x in row[1:129]]
                except ValueError:
                    continue
                if not any(vec):          # person with no usable photos
                    continue
                names.append(row[0])
                rows.append(vec)
        _known.update(mtime=m, names=names, matrix=np.array(rows).reshape(-1, 128))
    return _known["names"], _known["matrix"]


def recognize(img):
    """Detect + identify every face in a BGR frame with multi-face batching and anti-spoofing.
    Returns (faces, (w, h)); each face = {
        box: [x, y, w, h] as 0-1 fractions,
        name: str | None,
        distance: float | None,
        is_real: bool,
        liveness_score: float,
        liveness_reason: str,
        ear: float,
        pose: dict
    }."""
    ok, msg = status()
    if not ok:
        raise RuntimeError(msg)
    import cv2
    h0, w0 = img.shape[:2]
    if w0 > SCAN_WIDTH:
        img = cv2.resize(img, (SCAN_WIDTH, int(h0 * SCAN_WIDTH / w0)))
    h, w = img.shape[:2]
    rgb = _rgb(img)
    names, matrix = load_known()
    out = []

    with _lock:
        detected_rects = list(_state["detector"](rgb, 0))
        if not detected_rects:
            return out, (w, h)

        # Batch extract shapes and descriptors for all detected faces
        shapes = [_state["predictor"](rgb, r) for r in detected_rects]
        descriptors = [np.array(_state["model"].compute_face_descriptor(rgb, s)) for s in shapes]

    # Vectorized multi-face distance comparison
    desc_matrix = np.array(descriptors)  # (M, 128)
    if len(names) and matrix.shape[0] > 0:
        # Distance matrix (M, N) via broadcasting across all detected faces and registered vectors
        dists_matrix = np.linalg.norm(desc_matrix[:, None, :] - matrix[None, :, :], axis=2)
    else:
        dists_matrix = None

    for i, (r, shape) in enumerate(zip(detected_rects, shapes)):
        name = None
        dist = None
        if dists_matrix is not None:
            min_idx = int(np.argmin(dists_matrix[i]))
            min_d = float(dists_matrix[i, min_idx])
            dist = min_d
            if min_d < THRESHOLD:
                name = names[min_idx]

        # Multi-layer liveness evaluation (passive & active checks)
        is_real, score, reason, metrics = evaluate_liveness(img, r, shape)

        x, y = max(r.left(), 0), max(r.top(), 0)
        bw = max(r.right() - x, 1)
        bh = max(r.bottom() - y, 1)

        out.append({
            "box": [x / w, y / h, bw / w, bh / h],
            "name": name,
            "distance": dist,
            "is_real": is_real,
            "liveness_score": score,
            "liveness_reason": reason,
            "ear": metrics.get("ear", 0.3),
            "pose": metrics.get("pose", {})
        })

    return out, (w, h)


# ---------------------------------------------------------- face folders
def _folders():
    if not os.path.isdir(FACES_DIR):
        return []
    return sorted(f for f in os.listdir(FACES_DIR) if os.path.isdir(os.path.join(FACES_DIR, f)))


def person_folder(name):
    for f in _folders():
        if f.split("_", 2)[-1] == name:
            return os.path.join(FACES_DIR, f)
    return None


def photo_count(name):
    p = person_folder(name)
    return len(glob.glob(os.path.join(p, "*.jpg"))) if p else 0


def _photo_quality_error(img, face, cv2):
    """Return a user-facing reason when an enrollment frame is unsuitable."""
    h, w = img.shape[:2]
    left, top = max(face.left(), 0), max(face.top(), 0)
    right, bottom = min(face.right(), w), min(face.bottom(), h)
    face_w, face_h = right - left, bottom - top
    if face_w < 80 or face_h < 80 or face_w * face_h < w * h * 0.03:
        return "Move closer to the camera so your face fills more of the frame."

    crop = img[top:bottom, left:right]
    if crop.size == 0:
        return "Face could not be measured. Please reposition and try again."
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    if brightness < 30:
        return "The image is too dark. Move to better lighting and try again."
    if brightness > 235:
        return "The image is overexposed. Avoid strong light behind or on your face."
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < 20:
        return "The image is too blurry. Hold still and let the camera focus."
    return None


def save_photo(name, img):
    """Save one training photo for `name`. Requires exactly one face in the frame.
    Returns (ok, message, photo_count)."""
    ok, msg = status()
    if not ok:
        raise RuntimeError(msg)
    import cv2
    h0, w0 = img.shape[:2]
    if w0 > 640:
        img = cv2.resize(img, (640, int(h0 * 640 / w0)))
    with _lock:
        faces = _state["detector"](_rgb(img), 1)
    if len(faces) == 0:
        return False, "No face found - face the camera in good light.", photo_count(name)
    if len(faces) > 1:
        return False, "More than one face in the frame - only one person please.", photo_count(name)
    quality_error = _photo_quality_error(img, faces[0], cv2)
    if quality_error:
        return False, quality_error, photo_count(name)
    # Reserve the next filename while locked. This keeps rapid or concurrent
    # multi-photo captures from overwriting a previously saved image.
    with _lock:
        folder = person_folder(name)
        if not folder:
            nums = [int(f.split("_")[1]) for f in _folders() if len(f.split("_")) > 1 and f.split("_")[1].isdigit()]
            folder = os.path.join(FACES_DIR, f"person_{max(nums, default=0) + 1}_{name}")
            os.makedirs(folder, exist_ok=True)
        existing = [int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
                    for p in glob.glob(os.path.join(folder, "img_face_*.jpg"))
                    if os.path.splitext(os.path.basename(p))[0].split("_")[-1].isdigit()]
        path = os.path.join(folder, f"img_face_{max(existing, default=0) + 1}.jpg")
        saved = cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not saved:
        return False, "Could not save the photo. Please try again.", photo_count(name)
    return True, "Photo saved.", photo_count(name)


def _person_mean(folder):
    """Mean 128-d descriptor over the photos in a folder (None if no face found)."""
    import cv2
    vecs = []
    for p in sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))):
        img = cv2.imread(p)
        if img is None:
            continue
        rgb = _rgb(img)
        with _lock:
            faces = _state["detector"](rgb, 1)
            if len(faces) == 0:
                # photos from the Tkinter tool are already tight face crops
                faces = [_state["dlib"].rectangle(0, 0, img.shape[1], img.shape[0])]
            vecs.append(_descriptor(rgb, _largest(faces)))
    return np.mean(vecs, axis=0) if vecs else None


def _read_csv():
    rows = {}
    if os.path.isfile(FEATURES_CSV):
        with open(FEATURES_CSV, newline="") as f:
            for row in csv.reader(f):
                if len(row) >= 129:
                    rows[row[0]] = row[1:129]
    return rows


def embedding_dimension(name):
    """Return the stored embedding size for a person (zero when not trained)."""
    return len(_read_csv().get(name, []))


def _write_csv(rows):
    os.makedirs(os.path.dirname(FEATURES_CSV), exist_ok=True)
    with open(FEATURES_CSV, "w", newline="") as f:
        w = csv.writer(f)
        for name, vec in rows.items():
            w.writerow([name, *vec])
    load_known(force=True)


def train(name):
    """(Re)compute one person's features and update features_all.csv."""
    ok, msg = status()
    if not ok:
        raise RuntimeError(msg)
    folder = person_folder(name)
    mean = _person_mean(folder) if folder else None
    rows = _read_csv()
    if mean is None:
        rows.pop(name, None)
        _write_csv(rows)
        return False
    rows[name] = [repr(float(x)) for x in mean]
    _write_csv(rows)
    return True


def train_all():
    """Rebuild features_all.csv from every folder. Returns number of people trained."""
    ok, msg = status()
    if not ok:
        raise RuntimeError(msg)
    rows = {}
    for f in _folders():
        mean = _person_mean(os.path.join(FACES_DIR, f))
        if mean is not None:
            rows[f.split("_", 2)[-1]] = [repr(float(x)) for x in mean]
    _write_csv(rows)
    return len(rows)


def delete_person(name):
    """Remove a person's photos and their row in features_all.csv."""
    folder = person_folder(name)
    if folder:
        shutil.rmtree(folder, ignore_errors=True)
    rows = _read_csv()
    if name in rows:
        rows.pop(name)
        _write_csv(rows)


def rename_person(old, new):
    folder = person_folder(old)
    if folder:
        head = os.path.basename(folder).split("_", 2)
        os.rename(folder, os.path.join(FACES_DIR, f"{head[0]}_{head[1]}_{new}"))
    rows = _read_csv()
    if old in rows:
        rows = {(new if k == old else k): v for k, v in rows.items()}
        _write_csv(rows)
