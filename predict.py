from FaceDetector import detect_faces_images
from VGG16 import load_model
from preprocess import preprocess_face
import torch
import cv2
from PIL import Image
from scripts import draw_boxes_on_image
import os
from tempfile import NamedTemporaryFile
from fpdf import FPDF
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

last_mode = None
last_summary = ""
last_raw_images = []
last_annotated_images = []
last_face_images = []

# Load the model
model = load_model("models/My_Model.pth").to(device)

def predict_image(image):

    global last_mode, last_summary, last_raw_images, last_annotated_images, last_face_images
    last_mode = "image"
    last_raw_images = [image]

    faces, boxes, confidences, message = detect_faces_images(image)
    if not faces:
        last_annotated_images = [image]
        last_face_images = []
        last_summary = message
        return [], image, message
    
    predictions = []
    labels = []

    for i, face in enumerate(faces):
        face_tensor = preprocess_face(face).to(device)
        with torch.no_grad():
            output = model(face_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(torch.softmax(output, dim=1)).item()
            label = "Fake" if predicted_class == 0 else "Real"
            pred_text = f"{label} with confidence score: {confidence * 100:.0f}%"
            predictions.append(pred_text)
            labels.append(pred_text)

    annotated_image = draw_boxes_on_image(image, boxes, labels)
    last_annotated_images = [annotated_image]
    summary = f"{len(faces)} face(s) detected: {', '.join(predictions)}"
    last_summary = f"{len(faces)} face(s) detected: {', '.join(predictions)}"


    return faces, annotated_image, summary

def predict_video(video_path):

    global last_mode, last_summary, last_raw_images, last_annotated_images, last_face_images
    last_mode = "video"
    last_raw_images = []
    last_annotated_images = []
    last_face_images = []

    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    annotated_frames = []
    predictions = []
    all_faces = []

    frame_index = 0
    processed_count = 0
    max_processed_frames = 15  # Process 15 frames maximum

    while True:
        ret, frame = cap.read()
        if not ret or processed_count >= max_processed_frames:
            break

        if frame_index % 10 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            last_raw_images.append(pil_frame)


         
            raw_frames.append(pil_frame)

            # Detect faces
            faces, boxes, confidences, _ = detect_faces_images(pil_frame)
            annotated = frame_rgb.copy()

            for (x1, y1, x2, y2), face, conf in zip(boxes, faces, confidences):
                face_tensor = preprocess_face(face).to(device)
                with torch.no_grad():
                    output = model(face_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                    prob = torch.softmax(output, dim=1)[0][predicted_class].item()
                    label = "Fake" if predicted_class == 0 else "Real"
                    text = f"{label} (confidence: {prob:.2f})"
                    predictions.append(label)
                    all_faces.append(face)

                    # Draw box and label
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            last_annotated_images.append(Image.fromarray(annotated))
            processed_count += 1

        frame_index += 1

    cap.release()

    last_face_images = all_faces

    if not last_face_images:
        last_summary = "No faces detected in video"
    else:
        most_common = max(set(predictions), key=predictions.count)
        last_summary = f"{len(last_face_images)} faces detected. Most common prediction: {most_common}"

    return last_raw_images, last_face_images, last_annotated_images, last_summary


def generate_pdf():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "DeepFake Detection Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on: {now}", ln=True)

    if last_mode == "image":
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Uploaded Image:", ln=True)
        for img in last_raw_images:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=100)
            os.remove(img_path)

        pdf.cell(0, 10, "Annotated Image:", ln=True)
        for img in last_annotated_images:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=100)
            os.remove(img_path)

        pdf.cell(0, 10, "Detected Faces:", ln=True)
        for img in last_face_images:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=40)
            os.remove(img_path)

    elif last_mode == "video":
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Video Frames:", ln=True)
        for img in last_raw_images[:5]:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=60)
            os.remove(img_path)

        pdf.cell(0, 10, "Annotated Frames:", ln=True)
        for img in last_annotated_images[:5]:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=60)
            os.remove(img_path)

        pdf.cell(0, 10, "Detected Faces:", ln=True)
        for img in last_face_images[:10]:
            img_path = save_temp_image(img)
            pdf.image(img_path, w=40)
            os.remove(img_path)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, last_summary or "No summary available.")

    # Save to temporary file and return path
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        return tmp.name


def save_temp_image(img):
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        return tmp.name







