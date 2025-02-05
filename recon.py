import cv2
import dlib
import numpy as np
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import urllib.request

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

# Initialize face detector and tracker
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def analyze_face(frame):
    faces = detector(frame)
    results = []
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_roi = frame[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(face_roi, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            results.append((face, analysis[0]))
        except Exception as e:
            print("Error in DeepFace analysis:", e)
    return results

def draw_hud(frame, results):
    for face, analysis in results:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{analysis['dominant_emotion']}, {analysis['age']}, {analysis['gender']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = analyze_face(frame)
        frame = draw_hud(frame, results)
        cv2.imshow('Face Analysis', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

async def start(update, context):
    await update.message.reply_text("Send me a video, and I'll analyze faces in it!")

async def handle_video(update, context):
    if not update.message.video:
        await update.message.reply_text("No video file detected. Please send a valid video.")
        return
    
    file = await update.message.video.get_file()
    file_path = await file.download()
    process_video(file_path)
    await update.message.reply_text("Analysis complete!")

def main():
    app = Application.builder().token("7807412883:AAH6HjlVA9kxPVCsFvQFSxCmh4GHumgCvuc").build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.run_polling()

if __name__ == "__main__":
    main()
