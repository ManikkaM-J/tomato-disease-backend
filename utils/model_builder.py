"""
utils/model_builder.py
-----------------------
Defines the exact CNN architecture required by the project flowchart.

ARCHITECTURE (matches flowchart exactly):
=========================================

    Input (224, 224, 3)
         |
         v
    +------------------------------------------+
    |      CNN FEATURE EXTRACTION MODULE       |
    |                                          |
    |  [1] Convolution Layer 1  (32 filters)   |
    |  [2] ReLU Activation                     |
    |  [3] Max Pooling Layer 1  (2x2)          |
    |  [4] Convolution Layer 2  (64 filters)   |
    |  [5] ReLU Activation                     |
    +------------------------------------------+
         |
         v
    +------------------------------------------+
    |         CLASSIFICATION MODULE            |
    |                                          |
    |  [6] Flatten  (2D feature map -> 1D)     |
    |  [7] Dense(128) + ReLU                   |
    |  [8] Dropout(0.5)                        |
    |  [9] Dense(4)  + Softmax                 |
    +------------------------------------------+
         |
         v
    Output: Healthy | Bacterial Spot | Yellow Leaf Curl | Late Blight

FIX 1: Removed the second MaxPooling2D that was not in the project flowchart.
       The flowchart specifies: Conv1->ReLU->MaxPool1->Conv2->ReLU->Flatten.
       There is NO second MaxPool between Conv2 and Flatten.

FIX 2: Import path uses `from tensorflow import keras` then `keras.layers`
       which works correctly across TF 2.x and TF 2.16 + Keras 3.x.

FIX 3: model_builder.py is designed to be run standalone as well as imported,
       so the `from config import ...` line must work in both contexts.
       The if-block at the bottom adds the project root to sys.path when the
       file is run directly (e.g. `python utils/model_builder.py`).
"""

import os
import sys

# Allow running this file directly from the utils/ subdirectory
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FIX 2: stable import path across TF versions
import tensorflow as tf
from tensorflow import keras

Sequential    = keras.Sequential
Conv2D        = keras.layers.Conv2D
MaxPooling2D  = keras.layers.MaxPooling2D
Flatten       = keras.layers.Flatten
Dense         = keras.layers.Dense
Dropout       = keras.layers.Dropout
Input         = keras.layers.Input
Adam          = keras.optimizers.Adam

from config import CLASS_LABELS, IMAGE_SIZE, IMAGE_CHANNELS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = len(CLASS_LABELS)                    # 4
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)        # (224, 224, 3)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model() -> keras.Sequential:
    """
    Build and compile the CNN model following the project flowchart exactly.

    Layer-by-layer breakdown
    ------------------------

    [1] Convolution Layer 1  (32 filters, 3x3 kernel, padding='same')
        - Scans the 224x224 RGB image with 32 learnable filters.
        - Each filter detects a specific low-level pattern:
          edges, color transitions, texture variations in the leaf.
        - Output shape: (224, 224, 32)

    [2] ReLU Activation  (activation='relu' inside Conv2D)
        - Rectified Linear Unit: f(x) = max(0, x)
        - Introduces non-linearity by zeroing out negative activations.
        - Allows the network to learn complex, non-linear disease patterns.

    [3] Max Pooling Layer 1  (2x2 pool)
        - Down-samples the feature map by taking the maximum value in each
          2x2 window, reducing spatial size by half.
        - Output shape: (112, 112, 32)
        - Reduces computation and provides local translational invariance.

    [4] Convolution Layer 2  (64 filters, 3x3 kernel, padding='same')
        - Deeper filters combine low-level features into higher-level ones:
          leaf spot shapes, lesion boundary patterns, color anomalies.
        - Output shape: (112, 112, 64)

    [5] ReLU Activation  (activation='relu' inside Conv2D)
        - Same non-linearity as layer [2], applied to the deeper features.

    [6] Flatten  (2D feature map -> 1D vector)
        - Reshapes the 3D tensor (112, 112, 64) into a flat 1D vector.
        - This is the bridge between the convolutional and dense sections.
        - Output shape: (802816,)

    [7] Dense(128) + ReLU  (fully-connected hidden layer)
        - Every flattened feature connects to 128 neurons.
        - Learns high-level combinations of features for classification.

    [8] Dropout(0.5)
        - During training, randomly sets 50% of neurons to zero each step.
        - Prevents overfitting by forcing the network not to rely on any
          single neuron.  Inactive during inference.

    [9] Dense(4) + Softmax  (output layer)
        - 4 neurons, one per disease class.
        - Softmax converts raw scores (logits) into a probability distribution
          that sums to 1.0.  The class with the highest probability is the
          predicted disease.

    Returns
    -------
    keras.Sequential
        Compiled model ready for model.fit() or model.predict().
    """

    model = Sequential(name="TomatoLeafDiseaseCNN")

    # ---- Input layer -------------------------------------------------------
    model.add(Input(shape=INPUT_SHAPE))

    # ===== CNN FEATURE EXTRACTION MODULE ====================================

    # [1] Convolution Layer 1  +  [2] ReLU Activation
    model.add(Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation="relu",      # ReLU Activation built in
        padding="same",
        name="Conv_Layer1_ReLU",
    ))

    # [3] Max Pooling Layer 1
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        name="MaxPool_Layer1",
    ))

    # [4] Convolution Layer 2  +  [5] ReLU Activation
    # NOTE: No second MaxPool here — not in the flowchart spec.
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu",      # ReLU Activation built in
        padding="same",
        name="Conv_Layer2_ReLU",
    ))

    # ===== CLASSIFICATION MODULE ============================================

    # [6] Flatten: 2D feature map -> 1D vector
    model.add(Flatten(name="Flatten_2D_to_1D"))

    # [7] Fully-connected hidden layer + ReLU
    model.add(Dense(128, activation="relu", name="Dense_Hidden_ReLU"))

    # [8] Dropout for regularisation
    model.add(Dropout(0.5, name="Dropout"))

    # [9] Output layer: Dense + Softmax over 4 disease classes
    model.add(Dense(
        NUM_CLASSES,
        activation="softmax",   # Softmax Activation
        name="Softmax_Output",
    ))

    # ---- Compile -----------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Quick self-test: python utils/model_builder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Building CNN model...")
    cnn = build_model()
    cnn.summary()
    print()
    print("Class labels :", CLASS_LABELS)
    print("Input shape  :", INPUT_SHAPE)
    print("Output classes:", NUM_CLASSES)
    print()
    print("Architecture (per flowchart):")
    print("  Conv_Layer1_ReLU -> MaxPool_Layer1 -> Conv_Layer2_ReLU")
    print("  -> Flatten_2D_to_1D -> Dense_Hidden_ReLU -> Dropout -> Softmax_Output")
