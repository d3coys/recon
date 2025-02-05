Deception Face Analysis

A web-based application that detects truthfulness or deception in video footage using deep learning and facial analysis. This system processes uploaded video files, analyzes facial expressions, and assigns a truthfulness score for each detected face. The final result is a processed video with truthfulness visualizations, including a progress bar and a final conclusion.

🚀 Features

✅ Facial Detection: Automatically detects faces in the video.

✅ Truthfulness Prediction: Uses a trained deep learning model to analyze facial expressions.

✅ Dynamic Truthfulness Score: Displays a real-time score for detected faces.

✅ Graphical Overlay: Adds a truthfulness progress bar and a final truthfulness score.

✅ Conclusion Classification: Determines Truthful (🟢 Green) or Deception (🔴 Red).

✅ Download Processed Video: After analysis, users can download the output video.


📦 Installation

1️⃣ Clone the Repository

git clone https://github.com/d3coys/recon.git

cd recon


2️⃣ Create a Virtual Environment

python3 -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows


3️⃣ Install Dependencies

pip install -r requirements.txt


4️⃣ Download Required Models

The application requires two pre-trained models:
	1.	shape_predictor_68_face_landmarks.dat (for facial landmark detection)
	2.	truthfulness_model.h5 (for deep learning-based truthfulness detection)

Ensure these files exist in the project folder before running the program.

🛠 Usage

Run the Web Application

python recon.py

Once the server starts, open http://127.0.0.1:5000/ in your web browser.

Upload a Video for Analysis
	1.	Click “Choose File” and select an MP4, AVI, MOV, or MKV video.
	2.	Click “Upload” to start processing.
	3.	The system will analyze the video and display truthfulness scores frame by frame.
	4.	Once the analysis is complete, you can download the processed video with truthfulness overlays.
 

📡 Deployment on GitHub & Heroku


1️⃣ Push to GitHub

git init  
git add .  
git commit -m "Initial commit - Truthfulness Detector"  
git branch -M main  
git remote add origin https://github.com/YOUR_USERNAME/truthfulness-detector.git  
git push -u origin main  


2️⃣ Deploy to Heroku (Optional)

heroku login  
heroku create truthfulness-detector  
git push heroku main  
heroku open  

🔧 Dependencies (requirements.txt)

Below is the list of required Python packages:

Flask
opencv-python
dlib
tensorflow
numpy
matplotlib
urllib3
bz2file
gunicorn

📌 System Requirements
	•	Python 3.8+
	•	RAM: At least 8GB (Recommended for deep learning processing)
	•	A dedicated GPU (Optional, but speeds up processing)
	•	Operating System: Windows / macOS / Linux


📜 File Structure

📂 truthfulness-detector

│── 📂 templates/                # Contains HTML files for the web interface

│── 📂 uploads/                  # Stores uploaded video files

│── 📂 processed/                # Stores processed output videos

│── 📜 recon.py                  # Main Flask application

│── 📜 truthfulness_model.h5      # Pre-trained TensorFlow model

│── 📜 shape_predictor_68_face_landmarks.dat  # Dlib face landmark model

│── 📜 requirements.txt           # Python dependencies

│── 📜 README.md                  # Project documentation


📸 Example Output

The processed video will display:
	•	A bounding box around the detected face.
	•	A Truthfulness Score (Dynamic) below the face.
	•	A Progress Bar changing color based on truthfulness.
	•	Final Conclusion: Truthful (🟢) or Deception (🔴).


👨‍💻 Authors
	•	[Oppsoite6890] - Developer
	•	GitHub: https://github.com/decoys

📝 License

This project is licensed under the MIT License.
