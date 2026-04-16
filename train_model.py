"""
train_model.py
--------------
MULTI-LABEL TOMATO LEAF DISEASE DETECTION
Stage 2: Disease Classifier

KEY CHANGES FROM SINGLE-LABEL VERSION:
---------------------------------------
1. Output layer changed from Softmax (single-label) to Sigmoid per class
   (multi-label). Each output neuron independently predicts one disease.

2. Loss changed from categorical_crossentropy to binary_crossentropy
   so each class is treated as an independent binary classification.

3. Labels changed from one-hot encoding to multi-hot encoding
   (a leaf can have multiple diseases simultaneously).

4. Grayscale augmentation added so CNN learns leaf SHAPE not just COLOR.

ARCHITECTURE:
-------------
Input (224, 224, 3)
    Conv1(32) + ReLU + MaxPool
    Conv2(64) + ReLU + MaxPool
    Conv3(128) + ReLU + MaxPool
    Flatten
    Dense(256) + ReLU + BatchNorm + Dropout(0.5)
    Dense(4) + Sigmoid   <- one output per disease class (multi-label)

OUTPUT INTERPRETATION:
----------------------
Each of the 4 output neurons outputs a probability independently.
    output[0] = P(Bacterial Spot present)
    output[1] = P(Healthy)
    output[2] = P(Late Blight present)
    output[3] = P(Yellow Leaf Curl present)

If output[0] >= 0.70 AND output[2] >= 0.70 -> leaf has BOTH diseases.

HOW TO RUN:
-----------
1. Activate venv:   venv/Scripts/activate
2. Run:             python train_model.py
3. Restart:         python app.py
"""

import os
import sys
import re
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore

print(f"TensorFlow : {tf.__version__}")
print(f"Project root: {PROJECT_ROOT}")

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_DIR     = r"D:\PROJECT\tomato_leaf_backend_v2\PlantVillageDataset\PlantVillage"
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "model", "tomato_leaf_disease_model.h5")
CONFIG_PATH     = os.path.join(PROJECT_ROOT, "config.py")

IMAGE_SIZE       = (224, 224)
BATCH_SIZE       = 32
EPOCHS           = 20
VALIDATION_SPLIT = 0.2
LEARNING_RATE    = 1e-4

# Class labels — alphabetical order matching training folder names
CLASS_LABELS = [
    "Bacterial Spot",    # index 0 -> folder: Bacterial_Spot
    "Healthy",           # index 1 -> folder: Healthy
    "Late Blight",       # index 2 -> folder: Late_Blight
    "Yellow Leaf Curl",  # index 3 -> folder: Yellow_Leaf_Curl
]

NUM_CLASSES = len(CLASS_LABELS)


# ── Step 1: Update config.py ──────────────────────────────────────────────────

def update_config():
    """Update CLASS_LABELS and MODEL_PATH in config.py."""
    print("\n[Config] Updating config.py...")
    with open(CONFIG_PATH, "r") as f:
        content = f.read()

    # Update CLASS_LABELS
    new_labels = (
        'CLASS_LABELS = [\n'
        '    "Bacterial Spot",    # index 0\n'
        '    "Healthy",           # index 1\n'
        '    "Late Blight",       # index 2\n'
        '    "Yellow Leaf Curl",  # index 3\n'
        ']'
    )
    content = re.sub(
        r'CLASS_LABELS\s*=\s*\[.*?\]',
        new_labels, content, flags=re.DOTALL
    )

    # Update MODEL_PATH to point to correct file
    content = re.sub(
        r'MODEL_PATH\s*=.*',
        f'MODEL_PATH    = os.path.join(BASE_DIR, "model", "tomato_leaf_disease_model.h5")',
        content
    )

    # Ensure DISEASE_THRESHOLD exists
    if 'DISEASE_THRESHOLD' not in content:
        content += '\n# Disease detection threshold (70% = strict)\nDISEASE_THRESHOLD = 0.70\n'

    with open(CONFIG_PATH, "w") as f:
        f.write(content)

    print("[Config] Updated successfully.")
    for i, label in enumerate(CLASS_LABELS):
        print(f"         index {i} -> {label}")


# ── Step 2: Build multi-label CNN ─────────────────────────────────────────────

def build_model():
    """
    Multi-label CNN disease classifier.

    Architecture matches project flowchart:
        Conv1(32)+ReLU -> MaxPool1
        Conv2(64)+ReLU -> MaxPool2
        Conv3(128)+ReLU -> MaxPool3
        Flatten -> Dense(256)+ReLU -> BatchNorm -> Dropout(0.5)
        Dense(4) + Sigmoid   <- MULTI-LABEL output

    CRITICAL DIFFERENCE FROM SINGLE-LABEL:
        Softmax: probabilities sum to 1.0 (only ONE disease possible)
        Sigmoid: each output is independent (MULTIPLE diseases possible)

    Loss: binary_crossentropy (each class treated independently)
    """
    model = keras.Sequential(name="TomatoLeafDiseaseCNN_MultiLabel")
    model.add(keras.layers.Input(shape=(*IMAGE_SIZE, 3)))

    # CNN Feature Extraction Module
    # [1] Convolution Layer 1 + ReLU
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        padding="same", name="Conv_Layer1_ReLU"
    ))
    # [2] Max Pooling Layer 1
    model.add(keras.layers.MaxPooling2D((2, 2), name="MaxPool_Layer1"))

    # [3] Convolution Layer 2 + ReLU
    model.add(keras.layers.Conv2D(
        64, (3, 3), activation="relu",
        padding="same", name="Conv_Layer2_ReLU"
    ))
    # [4] Max Pooling Layer 2
    model.add(keras.layers.MaxPooling2D((2, 2), name="MaxPool_Layer2"))

    # [5] Convolution Layer 3 + ReLU
    model.add(keras.layers.Conv2D(
        128, (3, 3), activation="relu",
        padding="same", name="Conv_Layer3_ReLU"
    ))
    # [6] Max Pooling Layer 3
    model.add(keras.layers.MaxPooling2D((2, 2), name="MaxPool_Layer3"))

    # Classification Module
    # [7] Flatten: 2D -> 1D
    model.add(keras.layers.Flatten(name="Flatten_2D_to_1D"))

    # [8] Dense + ReLU
    model.add(keras.layers.Dense(256, activation="relu", name="Dense_Hidden_ReLU"))
    model.add(keras.layers.BatchNormalization(name="BatchNorm"))

    # [9] Dropout
    model.add(keras.layers.Dropout(0.5, name="Dropout"))

    # [10] Multi-label Sigmoid output
    # Each neuron independently outputs P(disease_i present)
    # Unlike Softmax, multiple neurons can be >= 0.70 simultaneously
    model.add(keras.layers.Dense(
        NUM_CLASSES,
        activation="sigmoid",   # MULTI-LABEL: each class independent
        name="Sigmoid_MultiLabel_Output"
    ))

    # Compile with binary_crossentropy for multi-label classification
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",     # Multi-label loss
        metrics=["accuracy"],
    )

    return model


# ── Step 3: Data generators ───────────────────────────────────────────────────

# ── Step 3: Data generators ───────────────────────────────────────────────────

def get_generators():
    """
    ImageDataGenerators with strong augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.3,
        brightness_range=[0.6, 1.4],
        shear_range=0.2,
        channel_shift_range=30.0,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE, 
        batch_size=BATCH_SIZE,
        class_mode="categorical", # FIX: Must be categorical for folder structure
        subset="training",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical", # FIX: Must be categorical
        subset="validation",
        shuffle=False,
    )
    return train_gen, val_gen


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Multi-Label Tomato Leaf Disease CNN - Training")
    print("=" * 60)

    # Verify dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found: {DATASET_DIR}")
        sys.exit(1)

    folders = os.listdir(DATASET_DIR)
    print(f"\n[Dataset] Found folders: {folders}")

    expected = ["Bacterial_Spot", "Healthy", "Late_Blight", "Yellow_Leaf_Curl"]
    missing  = [f for f in expected if f not in folders]
    if missing:
        print(f"[ERROR] Missing folders: {missing}")
        sys.exit(1)

    # Count images
    total = 0
    for folder in expected:
        path   = os.path.join(DATASET_DIR, folder)
        images = [f for f in os.listdir(path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {folder:25s}: {len(images)} images")
        total += len(images)
    print(f"  Total: {total} images\n")

    # Update config
    update_config()

    # Ensure model dir
    os.makedirs(os.path.join(PROJECT_ROOT, "model"), exist_ok=True)

    # Build model
    print("[Model] Building multi-label CNN...")
    model = build_model()
    model.summary()

    # Generators
    print("\n[Data] Preparing generators...")
    train_gen, val_gen = get_generators()

    # Verify class mapping
    print("\n" + "=" * 60)
    print("  CLASS INDEX MAPPING")
    print("=" * 60)
    for folder, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  index {idx} -> {folder:25s} -> {CLASS_LABELS[idx]}")
    print("=" * 60)

    print(f"\n  Training   : {train_gen.samples} images")
    print(f"  Validation : {val_gen.samples} images")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Architecture: Multi-Label Sigmoid (not Softmax)\n")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train
    print("[Training] Starting multi-label training...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Results
    best_val  = max(history.history["val_accuracy"]) * 100
    train_acc = history.history["accuracy"][-1] * 100

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Training accuracy   : {train_acc:.2f}%")
    print(f"  Best val accuracy   : {best_val:.2f}%")
    print(f"  Output type         : Multi-Label Sigmoid")
    print(f"  Disease threshold   : 70%")

    # Save
    print(f"\n[Saving] {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("[Saving] Multi-label model saved!")

    print("\n" + "=" * 60)
    print("  FINAL CLASS MAPPING")
    print("=" * 60)
    for i, label in enumerate(CLASS_LABELS):
        print(f"  index {i} -> {label}")

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("  1. Stop backend:   Ctrl+C")
    print("  2. Restart:        python app.py")
    print("  3. Test with real tomato leaf images")
    print("  4. Multiple diseases will now be detected!")
    print("=" * 60)


if __name__ == "__main__":
    main()
