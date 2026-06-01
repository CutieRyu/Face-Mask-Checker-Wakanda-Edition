import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─── CONFIG ───────────────────────────────────────────────
IMG_SIZE    = (224, 224)   # Resize all images to 128x128
BATCH_SIZE  = 32
EPOCHS      = 30
DATASET_DIR = "dataset"    # Change this to your dataset path
MODEL_PATH  = "mask_detector.h5"

# ─── 1. DATA PREPROCESSING ────────────────────────────────
print("\n[1] Loading and preprocessing data...")

# Training augmentation — helps the model generalize better
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,           # Normalize pixel values to [0, 1]
    rotation_range=15,           # Random rotations
    zoom_range=0.1,              # Random zoom
    width_shift_range=0.1,       # Horizontal shift
    height_shift_range=0.1,      # Vertical shift
    horizontal_flip=True,        # Mirror images
    fill_mode="nearest"
)

# Validation/Test — only normalize, no augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"          # 0 = WithMask, 1 = WithoutMask
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, "Test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print(f"   Classes: {train_generator.class_indices}")
print(f"   Training samples  : {train_generator.samples}")
print(f"   Test samples      : {test_generator.samples}")

# ─── 2. BUILD THE CUSTOM CNN ──────────────────────────────
print("\n[2] Building custom CNN model...")

model = Sequential([
    # --- Block 1 ---
    Conv2D(32, (3, 3), activation="relu", padding="same",
           input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # --- Block 2 ---
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # --- Block 3 ---
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    # --- Block 4 ---
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ─── 3. TRAINING ──────────────────────────────────────────
print("\n[3] Training the model...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=callbacks
)

# ─── 4. EVALUATION ────────────────────────────────────────
print("\n[4] Evaluating model...")

loss, accuracy = model.evaluate(test_generator)
print(f"   Test Accuracy : {accuracy * 100:.2f}%")
print(f"   Test Loss     : {loss:.4f}")

# Classification report
y_true = test_generator.classes
y_pred = (model.predict(test_generator) > 0.5).astype(int).flatten()
labels = list(train_generator.class_indices.keys())

print("\n   Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# ─── 5. PLOTS ─────────────────────────────────────────────
print("\n[5] Saving training plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history.history["accuracy"],     label="Train Accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

# Loss plot
axes[1].plot(history.history["loss"],     label="Train Loss")
axes[1].plot(history.history["val_loss"], label="Val Loss")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_results.png")
print("   Saved → training_results.png")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("   Saved → confusion_matrix.png")

print(f"\n✅ Done! Model saved to '{MODEL_PATH}'")
print("   Next: run 'python app.py' to start the web server.")
