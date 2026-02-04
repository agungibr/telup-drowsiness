import cv2
import numpy as np
import mediapipe as mp
from utils.config import Config

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection_model = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=Config.FACE_DETECTION_CONFIDENCE
)

face_mesh_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=Config.FACE_MESH_CONFIDENCE
)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
LIPS_IDX = [78, 81, 13, 311, 308, 402, 14, 178]
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float64)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

def crop_face(img, detection):
    h, w = img.shape[:2]
    if not detection or not detection.location_data:
        return img
    box = detection.location_data.relative_bounding_box
    xmin, ymin = int(box.xmin * w), int(box.ymin * h)
    width, height = int(box.width * w), int(box.height * h)
    
    margin_x = int(width * Config.FACE_MARGIN)
    margin_y = int(height * Config.FACE_MARGIN)
    
    x1, y1 = max(0, xmin - margin_x), max(0, ymin - margin_y)
    x2, y2 = min(w, xmin + width + margin_x), min(h, ymin + height + margin_y)
    return img[y1:y2, x1:x2]

def calculate_ear(points, indices):
    pts = points[indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def calculate_mar(points, indices):
    pts = points[indices]
    A = np.linalg.norm(pts[1] - pts[7])
    B = np.linalg.norm(pts[2] - pts[6])
    C = np.linalg.norm(pts[3] - pts[5])
    D = np.linalg.norm(pts[0] - pts[4])
    return (A + B + C) / (3.0 * D + 1e-6)

def estimate_head_pose(landmarks, w, h):
    key_indices = [1, 152, 33, 263, 61, 291]
    img_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in key_indices], dtype=np.float64)
    cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    try:
        success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS_3D, img_points, cam_matrix, np.zeros((4,1)))
        if not success: return 0.0, 0.0, 0.0
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(cv2.hconcat([rot_mat, trans_vec]))
        return np.clip(euler[0,0]/90, -1, 1), np.clip(euler[1,0]/90, -1, 1), np.clip(euler[2,0]/90, -1, 1)
    except:
        return 0.0, 0.0, 0.0

def extract_geometric_features(frame_rgb):
    h, w = frame_rgb.shape[:2]
    try:
        res = face_mesh_model.process(frame_rgb)
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            px = np.array([(p.x * w, p.y * h) for p in lms])
            ear = (calculate_ear(px, LEFT_EYE_IDX) + calculate_ear(px, RIGHT_EYE_IDX)) / 2.0
            mar = calculate_mar(px, LIPS_IDX)
            pitch, yaw, roll = estimate_head_pose(lms, w, h)
            return np.array([ear, mar, pitch, yaw, roll], dtype=np.float32)
    except Exception:
        pass
    return np.zeros(5, dtype=np.float32)