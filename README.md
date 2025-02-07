# Deception Face Analysis

## Overview
This project is a deep learning-based **Truthfulness Detection System** that analyzes facial expressions to determine the likelihood of truthfulness or deception in video footage. It utilizes **Dlib** for face detection and a **Convolutional Neural Network (CNN)** trained on a dataset of truthful and deceptive facial expressions. The model is integrated into a Flask-based web application that allows users to upload videos for analysis.

---

## Features
- **Face Detection**: Utilizes Dlib's frontal face detector.
- **Deep Learning Model**: CNN-based classifier trained on labeled truthful and deceptive expressions.
- **Dynamic Score Calculation**: Truthfulness is measured frame by frame, and a cumulative score is computed.
- **Flask Web Interface**: Users can upload videos and receive a visual overlay with truthfulness analysis.
- **Progress Bar & Color Coding**: Visual indicators dynamically adjust based on detection results.

---

## Installation
### Prerequisites
- Python 3.10+
- Virtual environment (recommended)
- TensorFlow, OpenCV, Dlib

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/d3coys/recon.git
   cd recon
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv-tf
   source venv-tf/bin/activate  # On macOS/Linux
   venv-tf\Scripts\activate     # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the dataset is structured correctly:
   ```bash
   dataset/
   ├── train/
   │   ├── truthful/
   │   ├── deceptive/
   ├── val/
   │   ├── truthful/
   │   ├── deceptive/
   ```

5. Train the model (optional if pre-trained model exists):
   ```bash
   python train_model.py
   ```

6. Run the web application:
   ```bash
   python recon.py
   ```

---

## Methodology
### 1. **Dataset Preparation**
- The dataset consists of labeled **truthful** and **deceptive** facial expressions.
- Images are **preprocessed** (resized to 64x64, normalized) before training.
- The dataset is split into **80% training** and **20% validation** for evaluation.

### 2. **Model Architecture**
- A **CNN model** is used for feature extraction from facial expressions.
- The model is trained with **binary cross-entropy loss** and **Adam optimizer**.
- Dropout layers are added to prevent overfitting.

### 3. **Training & Optimization**
- Model is trained for **20 epochs** with early stopping to prevent overfitting.
- Data augmentation (random flips, rotations) improves generalization.
- The trained model is saved as `truthfulness_model.h5`.

### 4. **Inference & Real-time Processing**
- The model processes each frame from a video.
- A **truthfulness score** (0 to 1) is assigned per frame.
- The **total score** is an average of all frame scores.
- The result is overlaid on the video with a **progress bar** and **color-coded conclusion**.

---

## Validation & Accuracy
- **Validation Accuracy**: ~63-66% (varies based on dataset quality and augmentation).
- **Loss Convergence**: Trained with validation loss monitoring.
- **Evaluation**: Confusion matrix and F1-score were used for final model assessment.
- **Future Improvements**:
  - Expand dataset with more diverse facial expressions.
  - Use transformer-based architectures (e.g., Vision Transformers).
  - Implement a multi-modal approach (combine facial analysis with audio tone detection).

---

## Contributors
- **Opposite6890** (Lead Developer)
- **https://instagram.com/opposite6890.recon**
- **https://t.me/opposite6890_xfire**
- **Skylark** (Testing & Implementation)

For issues and improvements, please raise a GitHub issue.

---

## License
MIT License

