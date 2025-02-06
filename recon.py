import os
import cv2
import dlib
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file
import urllib.request
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates")

# Load the trained deep learning model for truthfulness detection
MODEL_PATH_TF = "truthfulness_model.h5"
if not os.path.exists(MODEL_PATH_TF):
    raise FileNotFoundError("Missing truthfulness model! Please train and save 'truthfulness_model.h5'")

truthfulness_model = tf.keras.models.load_model(MODEL_PATH_TF)

# Download shape predictor model if not exists
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

if not os.path.exists(MODEL_PATH):
    print("Downloading shape_predictor_68_face_landmarks.dat...")
    urllib.request.urlretrieve(MODEL_URL, "shape_predictor_68_face_landmarks.dat.bz2")
    import bz2
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as fr, open(MODEL_PATH, "wb") as fw:
        fw.write(fr.read())
    print("Download complete.")

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Store total truthfulness scores for averaging
total_truthfulness_scores = []

def preprocess_face(face_roi):
    if face_roi is None or face_roi.size == 0:
        print("Error: Received an empty face image for preprocessing.")
        return np.zeros((1, 64, 64, 3))
    face_resized = cv2.resize(face_roi, (64, 64)).astype("float32") / 255.0
    return np.expand_dims(face_resized, axis=0)

def detect_truthfulness(frame):
    faces = detector(frame)
    results = []
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_roi = frame[y:y+h, x:x+w]
        if face_roi is None or face_roi.size == 0:
            continue
        processed_face = preprocess_face(face_roi)
        raw_score = truthfulness_model.predict(processed_face)[0][0]
        truthfulness_score = raw_score if raw_score > 0.7 else 1 - raw_score
        total_truthfulness_scores.append(truthfulness_score)
        results.append((face, truthfulness_score))
    return results

def draw_hud(frame, results, avg_truthfulness, conclusion):
    for face, truthfulness_score in results:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Define color based on truthfulness
        color = (0, int(255 * truthfulness_score), int(255 * (1 - truthfulness_score)))

        # Draw bounding box (Green)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Adjust the black background box to match the green bounding box width
        overlay_x, overlay_y = x, y + h + 30
        overlay_w, overlay_h = w, 120

        # Draw Black Background Box
        cv2.rectangle(frame, (overlay_x, overlay_y), 
                      (overlay_x + overlay_w, overlay_y + overlay_h), (0, 0, 0), -1)

        # Truthfulness Progress Bar **above the text**
        bar_x_start = overlay_x + 10
        bar_x_end = overlay_x + overlay_w - 10
        bar_y_start = overlay_y + 10
        bar_y_end = bar_y_start + 10
        bar_width = int((bar_x_end - bar_x_start) * truthfulness_score)
        
        cv2.rectangle(frame, (bar_x_start, bar_y_start), 
                      (bar_x_start + bar_width, bar_y_end), color, -1)

        # Proper spacing for text below the bar
        text_start_y = bar_y_end + 20

        cv2.putText(frame, f"Truthfulness : {truthfulness_score:.2f}", 
                    (overlay_x + 10, text_start_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Total Score : {avg_truthfulness:.2f}", 
                    (overlay_x + 10, text_start_y + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2)

        # Assign correct color and label based on total score
        if avg_truthfulness >= 0.71:
            conclusion = "Truthful"
            conclusion_color = (0, 255, 0)  # Green
        elif avg_truthfulness >= 0.41:
            conclusion = "Uncertain"
            conclusion_color = (0, 255, 255)  # Yellow
        else:
            conclusion = "Deception"
            conclusion_color = (0, 0, 255)  # Red

        cv2.putText(frame, f"Conclusion : {conclusion}", 
                    (overlay_x + 10, text_start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conclusion_color, 2)

        # Add Developer Credit
        cv2.putText(frame, "By Opposite6890", 
                    (overlay_x + 10, text_start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

def process_video(video_path, output_path):
    global total_truthfulness_scores
    total_truthfulness_scores = []
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = detect_truthfulness(frame)
        avg_truthfulness = np.mean(total_truthfulness_scores) if total_truthfulness_scores else 0.0
        conclusion = "Truthful" if avg_truthfulness > 0.5 else "Deception"
        frame = draw_hud(frame, results, avg_truthfulness, conclusion)
        out.write(frame)
    cap.release()
    out.release()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
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
