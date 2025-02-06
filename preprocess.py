import os
import cv2
import dlib
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

detector = dlib.get_frontal_face_detector()

def load_data(data_dir, target_size=(64, 64)):
    """Load and preprocess dataset for training"""
    datagen = ImageDataGenerator(
        rescale=1./255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
        horizontal_flip=True, validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=32, class_mode='binary', subset='training'
    )

    val_generator = datagen.flow_from_directory(
        data_dir, target_size=target_size, batch_size=32, class_mode='binary', subset='validation'
    )

    return train_generator, val_generator

def detect_faces(frame):
    """Detect faces using dlib"""
    return detector(frame)
