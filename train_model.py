import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Load and preprocess dataset
print("Loading dataset...")
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(TRAIN_DIR, target_size=(64, 64), batch_size=32, class_mode='binary')
val_generator = datagen.flow_from_directory(VAL_DIR, target_size=(64, 64), batch_size=32, class_mode='binary')

# Build model
print("Building model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# **FIXED: Compile the model**
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(train_generator, validation_data=val_generator, epochs=20)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/truthfulness_model.h5")

print("✅ Model training complete & saved to models/truthfulness_model.h5")
