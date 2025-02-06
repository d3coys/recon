📌 Methodology Used in This Project

The methodology for this Truthfulness Detection System follows a structured approach combining deep learning, computer vision, and statistical analysis. Below is a breakdown of the key methodologies used:

1️⃣ Data Collection & Preprocessing

🔹 Dataset Preparation

	•	The AI model (truthfulness_model.h5) is trained on a dataset containing facial expressions labeled as truthful or deceptive.
	•	The dataset includes various emotional cues like eye movement, lip tension, and microexpressions.

🔹 Face Detection

	•	Uses dlib’s frontal face detector to identify faces in each video frame.
	•	Extracts Region of Interest (ROI) for each detected face.

🔹 Image Preprocessing

	•	Resize each detected face to 64×64 pixels (standard input size for CNN models).
	•	Normalize pixel values between [0,1] to improve model efficiency.
	•	Convert the face image into a format suitable for the AI model.

2️⃣ Machine Learning & Deep Learning Model

🔹 Model Type

	•	The truthfulness_model.h5 is a Convolutional Neural Network (CNN) trained using TensorFlow/Keras.
	•	CNNs are chosen due to their strong capability in extracting spatial patterns from facial images.

🔹 Training Approach

	•	The model is trained on labeled facial expression data using supervised learning.
	•	It uses techniques like:
	•	Feature extraction (to detect truthfulness indicators).
	•	Backpropagation & Gradient Descent (for model optimization).
	•	Cross-validation (to ensure accuracy and prevent overfitting).

🔹 Model Prediction

	•	The trained model predicts a truthfulness score between 0 and 1 for each detected face.
	•	This score is dynamically adjusted to range between 0.10 - 0.80 to prevent extreme bias.

3️⃣ Statistical Analysis & Decision Making

🔹 Score Calculation

	•	Frame-Level Score: The model assigns a truthfulness score per frame.
	•	Total Score: The average truthfulness score is calculated across all frames.

🔹 Conclusion Classification

The system categorizes the Total Score into one of three truthfulness levels:

Score Range	Conclusion	Color Representation

0.00 - 0.40	🔴 Deception	High chance of lying

0.41 - 0.70	🟡 Uncertain	Needs further review

0.71 - 1.00	🟢 Truthful	Likely honest

4️⃣ Video Processing & Visualization

🔹 Real-time Overlay & Graphics

	•	OpenCV (cv2) is used to draw bounding boxes, progress bars, and text overlays.
	•	The progress bar dynamically changes based on the truthfulness score.

🔹 Flask Web Integration

	•	The system is deployed as a web-based application using Flask.
	•	Users upload a video, and the system processes it to return a processed video with truthfulness insights.

5️⃣ Deployment & Optimization

🔹 Web Deployment

	•	Flask handles video upload, processing, and download.
	•	The AI model runs locally but can be deployed to cloud servers.

🔹 Performance Optimization

	•	Batch processing is used for frame-by-frame analysis.
	•	The model uses GPU acceleration (if available) to speed up processing.

📌 Summary of Methodology

This project combines computer vision, deep learning, and statistical analysis to detect deception in videos. 

The process involves:

✔ Face detection & preprocessing

✔ CNN-based truthfulness prediction

✔ Score calculation & classification

✔ Real-time visualization & web deployment
