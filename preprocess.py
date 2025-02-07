import os
import cv2
import dlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

def load_data(data_dir, target_size=(64, 64), batch_size=32):
    """Load and preprocess dataset for training with data augmentation"""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory '{data_dir}' not found!")

    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, shear_range=0.15, zoom_range=0.2,
        horizontal_flip=True, validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=batch_size,
        class_mode='binary', subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=batch_size,
        class_mode='binary', subset='validation'
    )

    print(f"✅ Loaded dataset: {train_generator.samples} training, {val_generator.samples} validation images")
    return train_generator, val_generator

def detect_faces(frame):
    """Detect faces in an image using dlib's face detector"""
    faces = detector(frame)
    
    if len(faces) == 0:
        print("⚠️ Warning: No faces detected.")
    
    return faces
