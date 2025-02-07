import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt

# Dataset paths
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# Data Augmentation for Training Set
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.3, height_shift_range=0.3,
    shear_range=0.3, zoom_range=0.3, horizontal_flip=True
)

# Validation Set (No Augmentation, Only Rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
print("📂 Loading dataset...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(64, 64), batch_size=32, class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(64, 64), batch_size=32, class_mode='binary'
)

# **🔹 Load EfficientNetB0 Pretrained Model (Feature Extractor)**
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(64, 64, 3))

# **🔹 Freeze Base Model (Only train new layers initially)**
for layer in base_model.layers:
    layer.trainable = False

# **🔹 Define Custom Classifier on Top**
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Convert feature maps into single vector
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)  # Binary classification

model = Model(inputs=base_model.input, outputs=x)

# **🔹 Compile Model**
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# **🔹 Callbacks for Better Training**
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
checkpoint = ModelCheckpoint("models/best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max")

# **🚀 Train Model**
print("🚀 Training model...")
history = model.fit(
    train_generator, validation_data=val_generator, epochs=20,  # EfficientNet converges faster
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# **🔹 Unfreeze Some Layers & Fine-Tune**
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers
    layer.trainable = True

# **🔹 Recompile Model for Fine-Tuning**
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # Lower LR for fine-tuning
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

# **🚀 Fine-Tuning Model**
print("🔄 Fine-tuning model...")
history_fine = model.fit(
    train_generator, validation_data=val_generator, epochs=10,
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Save Final Model
os.makedirs("models", exist_ok=True)
model.save("models/truthfulness_model.h5")
print("✅ Model training complete & saved to models/truthfulness_model.h5")

# **📊 Plot Accuracy & Loss Curves**
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.plot(history_fine.history["accuracy"], label="Fine-tune Train Accuracy")
plt.plot(history_fine.history["val_accuracy"], label="Fine-tune Val Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.plot(history_fine.history["loss"], label="Fine-tune Train Loss")
plt.plot(history_fine.history["val_loss"], label="Fine-tune Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("models/training_plot.png")  # Save plot
plt.show()
