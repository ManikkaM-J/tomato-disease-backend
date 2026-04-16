"""
train_stage1_validator.py
--------------------------
STEP 2: Train Stage 1 Tomato Leaf Validator

Run AFTER build_dataset.py has built the Validator_Dataset.

Uses MobileNetV2 Transfer Learning:
- Pre-trained on 1.4 million ImageNet images
- Already knows shapes, textures, edges
- Fine-tuned to distinguish tomato vs non-tomato leaves
- Expected accuracy: 95-99%

TWO-PHASE TRAINING:
-------------------
Phase 1 (MobileNetV2 FROZEN):
    Only the classification head is trained.
    Fast convergence. Expected: 90-95% accuracy.

Phase 2 (MobileNetV2 FINE-TUNED):
    Last layers of MobileNetV2 unfrozen.
    Learns tomato-specific features.
    Expected: 95-99% accuracy.

DATASET USED:
-------------
Validator_Dataset/
    tomato_leaf/   <- 2000 tomato leaf images
    non_tomato/    <- 2000 non-tomato images

CLASS MAPPING (alphabetical):
    non_tomato  -> index 0 -> sigmoid near 0.0
    tomato_leaf -> index 1 -> sigmoid near 1.0
    sigmoid >= 0.5 = TOMATO LEAF confirmed

HOW TO RUN:
-----------
1. Run build_dataset.py first
2. Activate venv:   venv/Scripts/activate
3. Run:             python train_stage1_validator.py
4. Restart backend: python app.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import MobileNetV2                 # type: ignore

print(f"TensorFlow  : {tf.__version__}")
print(f"Project root: {PROJECT_ROOT}")

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_DIR          = os.path.join(PROJECT_ROOT, "Validator_Dataset")
VALIDATOR_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "tomato_leaf_validator.keras")

IMAGE_SIZE       = (224, 224)
BATCH_SIZE       = 16
VALIDATION_SPLIT = 0.2

# Phase 1 — train top layers (MobileNetV2 frozen)
PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3

# Phase 2 — fine-tune MobileNetV2 last layers
PHASE2_EPOCHS  = 15
PHASE2_LR      = 1e-5
FINE_TUNE_FROM = 100   # unfreeze layers after this index


# ── Verify dataset ────────────────────────────────────────────────────────────

def verify_dataset():
    """Check Validator_Dataset exists and has correct structure."""
    print(f"\n[Dataset] Checking: {DATASET_DIR}")

    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found: {DATASET_DIR}")
        print("Run build_dataset.py first!")
        sys.exit(1)

    tomato_dir     = os.path.join(DATASET_DIR, "tomato_leaf")
    non_tomato_dir = os.path.join(DATASET_DIR, "non_tomato")

    for d, name in [(tomato_dir, "tomato_leaf"), (non_tomato_dir, "non_tomato")]:
        if not os.path.exists(d):
            print(f"[ERROR] Missing folder: {d}")
            print("Run build_dataset.py first!")
            sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    tomato_count = sum(
        1 for _, _, files in os.walk(tomato_dir)
        for f in files if os.path.splitext(f)[1] in extensions
    )
    non_tomato_count = sum(
        1 for _, _, files in os.walk(non_tomato_dir)
        for f in files if os.path.splitext(f)[1] in extensions
    )

    print(f"[Dataset] tomato_leaf : {tomato_count} images")
    print(f"[Dataset] non_tomato  : {non_tomato_count} images")
    print(f"[Dataset] Total       : {tomato_count + non_tomato_count} images")

    if tomato_count < 50 or non_tomato_count < 50:
        print("[ERROR] Not enough images! Run build_dataset.py first.")
        sys.exit(1)

    # Class weights for imbalance
    total = tomato_count + non_tomato_count
    w_non = total / (2.0 * non_tomato_count)
    w_tom = total / (2.0 * tomato_count)

    print(f"\n[Dataset] Class weights (imbalance fix):")
    print(f"  non_tomato  (index 0): {w_non:.3f}")
    print(f"  tomato_leaf (index 1): {w_tom:.3f}")

    return {0: w_non, 1: w_tom}


# ── Build MobileNetV2 model ───────────────────────────────────────────────────

def build_model():
    """
    MobileNetV2 Transfer Learning model.

    Class mapping:
        non_tomato  -> index 0 -> sigmoid near 0.0
        tomato_leaf -> index 1 -> sigmoid near 1.0
        sigmoid >= 0.5 = tomato leaf CONFIRMED
    """
    print("\n[Model] Loading MobileNetV2 pre-trained on ImageNet...")

    base_model = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False
    print(f"[Model] MobileNetV2 loaded. Total layers: {len(base_model.layers)}")
    print("[Model] Phase 1: Base model FROZEN.")

    # Build model
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="GlobalAvgPool")(x)
    x = keras.layers.Dense(256, activation="relu", name="Dense1_ReLU")(x)
    x = keras.layers.BatchNormalization(name="BatchNorm1")(x)
    x = keras.layers.Dropout(0.5, name="Dropout1")(x)
    x = keras.layers.Dense(128, activation="relu", name="Dense2_ReLU")(x)
    x = keras.layers.Dropout(0.3, name="Dropout2")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="Sigmoid_Output")(x)

    model = keras.Model(inputs, outputs, name="TomatoValidator_MobileNetV2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


# ── Data generators ───────────────────────────────────────────────────────────

def get_generators():
    # CRITICAL FIX: No rescale here!
    # MobileNetV2 preprocess_input inside model handles [0,255] -> [-1,1].
    # rescale + preprocess_input = double normalization = 50% accuracy.
    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.3,
        brightness_range=[0.5, 1.5],
        shear_range=0.2,
        fill_mode="nearest",
        validation_split=VALIDATION_SPLIT,
    )
    val_datagen = ImageDataGenerator(
        validation_split=VALIDATION_SPLIT,
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Stage 1 Validator - MobileNetV2 Training")
    print("=" * 60)

    # Verify dataset
    class_weights = verify_dataset()

    # Ensure model directory
    os.makedirs(os.path.join(PROJECT_ROOT, "model"), exist_ok=True)

    # Build model
    model, base_model = build_model()
    model.summary()

    # Generators
    print("\n[Data] Preparing generators...")
    train_gen, val_gen = get_generators()

    # Verify class mapping
    print("\n" + "=" * 60)
    print("  CLASS MAPPING VERIFICATION")
    print("=" * 60)
    all_ok = True
    for folder, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        expected = "non_tomato" if idx == 0 else "tomato_leaf"
        status   = "OK" if folder == expected else "ERROR"
        if folder != expected:
            all_ok = False
        print(f"  [{status}] index {idx} -> {folder}")

    if not all_ok:
        print("\n[ERROR] Class mapping wrong!")
        sys.exit(1)

    print("=" * 60)
    print(f"\n  Training   : {train_gen.samples} images")
    print(f"  Validation : {val_gen.samples} images")

    # ── PHASE 1 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1: Training head only (MobileNetV2 frozen)")
    print(f"  Epochs: {PHASE1_EPOCHS} | LR: {PHASE1_LR}")
    print("=" * 60)

    p1_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=VALIDATOR_MODEL_PATH,
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=PHASE1_EPOCHS, callbacks=p1_callbacks,
        class_weight=class_weights, verbose=1,
    )
    phase1_best = max(h1.history["val_accuracy"]) * 100
    print(f"\n[Phase 1] Best val accuracy: {phase1_best:.2f}%")

    # Save Phase 1 best model before Phase 2 overwrites it
    import shutil as _shutil
    phase1_path = VALIDATOR_MODEL_PATH.replace(".keras", "_phase1_best.keras")
    _shutil.copy(VALIDATOR_MODEL_PATH, phase1_path)
    print(f"[Phase 1] Best model backed up to: {phase1_path}")

    # ── PHASE 2 ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: Fine-tuning MobileNetV2")
    print(f"  Unfreezing from layer {FINE_TUNE_FROM}")
    print(f"  Epochs: {PHASE2_EPOCHS} | LR: {PHASE2_LR}")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_FROM]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    p2_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-8, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=VALIDATOR_MODEL_PATH,
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
    ]

    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=PHASE2_EPOCHS, callbacks=p2_callbacks,
        class_weight=class_weights, verbose=1,
    )
    phase2_best = max(h2.history["val_accuracy"]) * 100

    # ── Results ───────────────────────────────────────────────────────────
    overall_best = max(phase1_best, phase2_best)

    # If Phase 1 was better, restore it
    if phase1_best >= phase2_best:
        import shutil
        shutil.copy(phase1_path, VALIDATOR_MODEL_PATH)
        print(f"[Info] Phase 1 was better ({phase1_best:.2f}% vs {phase2_best:.2f}%)")
        print(f"[Info] Restored Phase 1 model as final model.")

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Phase 1 best accuracy : {phase1_best:.2f}%")
    print(f"  Phase 2 best accuracy : {phase2_best:.2f}%")
    print(f"  Overall best accuracy : {overall_best:.2f}%")

    # Save
    print(f"\n[Saving] {VALIDATOR_MODEL_PATH}")
    model.save(VALIDATOR_MODEL_PATH)
    print("[Saving] MobileNetV2 validator saved!")

    print("\n" + "=" * 60)
    print("  FINAL CLASS MAPPING")
    print("=" * 60)
    for folder, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  index {idx} -> {folder}")

    print("\n" + "=" * 60)
    print("  NEXT STEPS")
    print("=" * 60)
    print("  1. Stop backend:    Ctrl+C")
    print("  2. Restart:         python app.py")
    print("  3. Test non-tomato  -> should be REJECTED")
    print("  4. Test tomato leaf -> should PASS + disease")
    print(f"  Expected accuracy   : {overall_best:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
