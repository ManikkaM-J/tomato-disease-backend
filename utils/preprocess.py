"""
utils/preprocess.py
--------------------
Implements the IMAGE PREPROCESSING stage of the CNN pipeline:

    Input Image
        |
        v
    +---------------------------+
    | Image Resizing            |  -> resize to 224 x 224
    | Normalization             |  -> pixel values / 255.0  -> [0.0, 1.0]
    | Data Augmentation         |  -> rotation, flip, zoom  (training only)
    +---------------------------+
        |
        v
    CNN Feature Extraction Module

FIX 1: `from config import ...` now works because app.py injects the project
        root into sys.path before any other import happens.

FIX 2: `tensorflow.keras.preprocessing.image.ImageDataGenerator` was
        deprecated in TF 2.13 and moved/removed in TF 2.16+.
        We import it through `tf.keras.preprocessing.image` which is the
        stable path across TF 2.x and Keras 3.x.

FIX 3: `Image.LANCZOS` is the correct constant for Pillow >= 9.1.0.
        A safe fallback is provided for older Pillow versions.
"""

import io
import os

import numpy as np
from PIL import Image

# FIX 2: stable import path for ImageDataGenerator across TF versions
import tensorflow as tf
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# FIX 1: config is importable because app.py sets sys.path first
from config import ALLOWED_EXTENSIONS, IMAGE_SIZE

# FIX 3: LANCZOS resampling filter — safe across Pillow versions
try:
    _RESAMPLE = Image.Resampling.LANCZOS   # Pillow >= 9.1.0
except AttributeError:
    _RESAMPLE = Image.LANCZOS              # Pillow < 9.1.0


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """
    Return True when *filename* carries an allowed image extension.

    Checks that:
    - The filename contains at least one dot.
    - The extension (after the last dot) is in ALLOWED_EXTENSIONS
      (case-insensitive comparison).

    Parameters
    ----------
    filename : str
        The original filename submitted by the client.

    Returns
    -------
    bool

    Examples
    --------
    >>> allowed_file("leaf.jpg")
    True
    >>> allowed_file("report.pdf")
    False
    >>> allowed_file("noextension")
    False
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# Inference preprocessing  (used by /predict endpoint)
# ---------------------------------------------------------------------------

def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single tomato leaf image for **inference**.

    Preprocessing pipeline
    ----------------------
    1. Load with Pillow and convert to RGB (removes alpha channel if present).
    2. Image Resizing  : resize to IMAGE_SIZE (224 x 224 pixels).
    3. Array conversion: PIL Image  ->  float32 NumPy array.
    4. Normalization   : divide all pixel values by 255.0  ->  range [0, 1].
    5. Batch dimension : shape (224, 224, 3)  ->  (1, 224, 224, 3).

    Parameters
    ----------
    image_path : str
        Path to the saved upload file on disk.

    Returns
    -------
    np.ndarray
        float32 array of shape (1, 224, 224, 3) ready for model.predict().

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *image_path*.
    OSError
        If Pillow cannot open the file (corrupt or unsupported format).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Step 1 - Load and ensure RGB color mode
    img = Image.open(image_path).convert("RGB")

    # Step 2 - Image Resizing: 224 x 224
    img = img.resize(IMAGE_SIZE, _RESAMPLE)

    # Step 3 - Convert to float32 NumPy array
    img_array = np.array(img, dtype=np.float32)

    # Step 4 - Normalization: [0, 255] -> [0.0, 1.0]
    img_array = img_array / 255.0

    # Step 5 - Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Same preprocessing pipeline as :func:`preprocess_image` but accepts
    raw image bytes instead of a file path.

    Useful when you want to skip writing a temporary file to disk:

        img_bytes = request.files["file"].read()
        img_array = preprocess_image_from_bytes(img_bytes)

    Parameters
    ----------
    image_bytes : bytes
        Raw image file content.

    Returns
    -------
    np.ndarray
        float32 array of shape (1, 224, 224, 3).
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE, _RESAMPLE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------------------------------------------------------------------------
# Training pipeline  (data augmentation generators)
# ---------------------------------------------------------------------------

def get_training_augmentation_generator(
    validation_split: float = 0.2,
) -> tuple:
    """
    Build Keras ImageDataGenerator objects for the **training pipeline**.

    Data Augmentation transforms applied only during training:
    - Rotation up to 20 degrees
    - Width and height shift up to 20 %
    - Horizontal flip
    - Zoom range +/- 20 %
    - Brightness variation +/- 20 %
    - Nearest-neighbor fill for newly created pixels

    Normalization (/ 255) is applied by both generators so that pixel
    values fed into the CNN are always in the [0.0, 1.0] range.

    Parameters
    ----------
    validation_split : float
        Fraction of the dataset to reserve for validation (default 0.2).

    Returns
    -------
    train_datagen : ImageDataGenerator
        Generator with augmentation + normalization for training.
    val_datagen : ImageDataGenerator
        Generator with normalization only for validation.

    Usage example
    -------------
    >>> train_gen, val_gen = get_training_augmentation_generator()
    >>> train_flow = train_gen.flow_from_directory(
    ...     "dataset/",
    ...     target_size=(224, 224),
    ...     batch_size=32,
    ...     class_mode="categorical",
    ...     subset="training",
    ... )
    >>> val_flow = val_gen.flow_from_directory(
    ...     "dataset/",
    ...     target_size=(224, 224),
    ...     batch_size=32,
    ...     class_mode="categorical",
    ...     subset="validation",
    ... )
    >>> model.fit(train_flow, validation_data=val_flow, epochs=20)
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,          # Normalization
        rotation_range=20,            # Augmentation: random rotation
        width_shift_range=0.2,        # Augmentation: horizontal shift
        height_shift_range=0.2,       # Augmentation: vertical shift
        horizontal_flip=True,         # Augmentation: mirror image
        vertical_flip=False,
        zoom_range=0.2,               # Augmentation: zoom in / out
        brightness_range=[0.8, 1.2],  # Augmentation: brightness variation
        fill_mode="nearest",
        validation_split=validation_split,
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,          # Normalization only — no augmentation
        validation_split=validation_split,
    )

    return train_datagen, val_datagen
