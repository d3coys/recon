import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Create a simple CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: Probability of truthfulness
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate random data for training (Replace this with real dataset)
X_train = np.random.rand(1000, 64, 64, 3)  # 1000 random face images
y_train = np.random.randint(0, 2, 1000)  # 0 = Deception, 1 = Truthful

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the trained model
model.save("truthfulness_model.h5")

print("Model saved successfully as 'truthfulness_model.h5'")
