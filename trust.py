import tensorflow as tf
import numpy as np

# Load model
model = tf.keras.models.load_model("truthfulness_model.h5")

# Generate a test face input (random noise)
test_face = np.random.rand(1, 64, 64, 3)  # Simulating a processed face input
prediction = model.predict(test_face)[0][0]

print(f"Test Prediction: {prediction}")
