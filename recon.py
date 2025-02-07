import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from utils import load_model, allowed_file, draw_hud

app = Flask(__name__, template_folder="templates")

# Load trained deep learning model for truthfulness detection
MODEL_PATH_TF = "models/truthfulness_model.h5"
if not os.path.exists(MODEL_PATH_TF):
    raise FileNotFoundError("Missing truthfulness model! Please train and save 'truthfulness_model.h5'")

truthfulness_model = load_model(MODEL_PATH_TF)

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Store total truthfulness scores for averaging
total_truthfulness_scores = []

def preprocess_face(face_roi):
    """ Preprocess face for TensorFlow model (resize & normalize) """
    if face_roi is None or face_roi.size == 0:
        return np.zeros((1, 64, 64, 3))  # Return blank if no face detected
    
    face_resized = cv2.resize(face_roi, (64, 64))
    face_resized = face_resized.astype("float32") / 255.0
    face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
    return face_resized

def detect_truthfulness(frame):
    """Detects faces in the frame and predicts truthfulness score using the model."""
    faces = detector(frame)
    results = []

    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_roi = frame[y:y+h, x:x+w]

        if face_roi is None or face_roi.size == 0:
            continue

        processed_face = preprocess_face(face_roi)
        raw_score = truthfulness_model.predict(processed_face)[0][0]

        # Normalize score dynamically (0.10 - 0.90 range)
        truthfulness_score = 0.10 + (raw_score * 0.80)

        results.append((face, truthfulness_score))
    
    return results

def process_video(video_path, output_path):
    """Processes the video frame by frame, detects faces, applies truthfulness detection, and saves the output."""
    global total_truthfulness_scores
    total_truthfulness_scores = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return  # Exit function if video cannot be opened

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs("processed", exist_ok=True)  # Ensure processed folder exists
    print(f"Saving processed video to: {output_path}")

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Warning: No more frames to read, stopping process.")
            break
        
        results = detect_truthfulness(frame)
        
        # Accumulate scores across frames
        if results:
            frame_scores = [score for _, score in results]
            total_truthfulness_scores.append(np.mean(frame_scores))
        
        avg_truthfulness = np.mean(total_truthfulness_scores) if total_truthfulness_scores else 0.0
        frame = draw_hud(frame, results, avg_truthfulness)
        out.write(frame)

    cap.release()
    out.release()

    # Ensure the file exists before returning
    if not os.path.exists(output_path):
        print(f"Error: Processed video was not saved properly at {output_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handles video file upload, processes it, and returns the processed video."""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file format", 400

    input_path = os.path.join("uploads", file.filename)
    output_path = os.path.join("processed", file.filename)

    os.makedirs("uploads", exist_ok=True)
    os.makedirs("processed", exist_ok=True)
    file.save(input_path)

    process_video(input_path, output_path)
    return send_file(output_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
