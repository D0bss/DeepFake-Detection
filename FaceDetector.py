import dlib
import numpy as np
from PIL import Image

# Load CNN face detector (do this once)
cnn_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

def detect_faces_images(image , margin=20):
    rgb_img = np.array(image)

    # Detect faces
    detections = cnn_detector(rgb_img)

    if not detections:
        return [],[],[], "❌ No faces detected."

    face_images = []
    face_boxes = []
    confidences = []

    for det in detections:
        rect = det.rect
        conf = det.confidence
        x1, y1 = max(rect.left() - margin, 0), max(rect.top() - margin, 0)
        x2, y2 = min(rect.right() + margin, rgb_img.shape[1]), min(rect.bottom() + margin, rgb_img.shape[0])

        face_crop = rgb_img[y1:y2, x1:x2]
        face_pil = Image.fromarray(face_crop)

        face_images.append(face_pil)
        face_boxes.append((x1, y1, x2, y2))
        confidences.append(conf)

    return face_images, face_boxes, confidences, f"✅ {len(face_images)} face(s) detected."


