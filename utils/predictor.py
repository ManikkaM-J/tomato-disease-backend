"""
utils/predictor.py
-------------------
TWO-STAGE MULTI-LABEL DETECTION PIPELINE

STAGE 1 — Tomato Leaf Validator (binary sigmoid)
    Checks if image is a tomato leaf.
    non_tomato  -> index 0 -> sigmoid near 0.0
    tomato_leaf -> index 1 -> sigmoid near 1.0
    sigmoid >= 0.5 = tomato leaf confirmed

STAGE 2 — Multi-Label Disease Classifier (4x independent sigmoid)
    Each output neuron independently predicts one disease.
    A leaf CAN have multiple diseases simultaneously.

    output[0] = P(Bacterial Spot)   >= 0.70 -> detected
    output[1] = P(Healthy)          >= 0.70 -> detected
    output[2] = P(Late Blight)      >= 0.70 -> detected
    output[3] = P(Yellow Leaf Curl) >= 0.70 -> detected

RESPONSE STRUCTURE:
    primary_disease    -> disease with highest confidence
    secondary_diseases -> other diseases above 70% threshold
    all_scores         -> all 4 class probabilities
"""

import os
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

from config import CLASS_LABELS, MODEL_PATH, DISEASE_THRESHOLD

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATOR_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "tomato_leaf_validator.keras")

# ── Global singletons ─────────────────────────────────────────────────────────
_validator_model = None
_disease_model   = None

# Stage 1 threshold
TOMATO_THRESHOLD = 0.5


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_validator():
    """Load Stage 1 binary validator. No-op if already loaded."""
    global _validator_model
    if _validator_model is not None:
        return
    if not os.path.exists(VALIDATOR_MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 1 validator model not found:\n  {VALIDATOR_MODEL_PATH}\n"
            "Run: python train_stage1_validator.py"
        )
    print(f"[Predictor] Loading Stage 1 validator: {VALIDATOR_MODEL_PATH}")
    # Use custom_object_scope to handle MobileNetV2 preprocessing layers
    import tensorflow as tf
    _validator_model = tf.keras.models.load_model(
        VALIDATOR_MODEL_PATH,
        compile=False,
    )
    print("[Predictor] Stage 1 validator loaded.")


def _load_disease():
    """Load Stage 2 multi-label disease model. No-op if already loaded."""
    global _disease_model
    if _disease_model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Stage 2 disease model not found:\n  {MODEL_PATH}\n"
            "Run: python train_model.py"
        )
    print(f"[Predictor] Loading Stage 2 disease model: {MODEL_PATH}")
    _disease_model = load_model(MODEL_PATH)
    print(f"[Predictor] Stage 2 disease model loaded.")
    print(f"[Predictor] Classes     : {CLASS_LABELS}")
    print(f"[Predictor] Threshold   : {DISEASE_THRESHOLD * 100:.0f}%")
    print(f"[Predictor] Mode        : Multi-Label Sigmoid")


def get_model():
    """Pre-warm both models. Called at app startup."""
    _load_validator()
    _load_disease()
    return _disease_model


# ── Stage 1: Tomato Leaf Validation ──────────────────────────────────────────

def validate_tomato_leaf(img_array: np.ndarray) -> dict:
    """
    Binary classification: tomato_leaf vs non_tomato.

    flow_from_directory (alphabetical):
        non_tomato  -> index 0 -> sigmoid near 0.0
        tomato_leaf -> index 1 -> sigmoid near 1.0

    sigmoid >= 0.5 = tomato leaf CONFIRMED
    sigmoid <  0.5 = NOT a tomato leaf
    """
    _load_validator()

    # CRITICAL: MobileNetV2 preprocess_input inside the model expects
    # raw pixels [0, 255] — NOT normalized [0, 1].
    # preprocess.py divides by 255 so we must multiply back.
    raw_input = img_array * 255.0   # restore [0,1] -> [0,255]
    raw = float(_validator_model.predict(raw_input, verbose=0)[0][0])

    is_tomato        = raw >= TOMATO_THRESHOLD
    tomato_conf      = round(raw * 100.0, 2)
    non_tomato_conf  = round((1.0 - raw) * 100.0, 2)

    if is_tomato:
        msg = (
            f"Tomato leaf confirmed ({tomato_conf:.1f}% confidence). "
            f"Proceeding to disease analysis..."
        )
    else:
        msg = (
            f"This does not appear to be a tomato leaf "
            f"({non_tomato_conf:.1f}% confidence it is NOT a tomato leaf). "
            f"Please upload a clear photo of a tomato leaf."
        )

    return {
        "is_tomato":             is_tomato,
        "tomato_confidence":     tomato_conf,
        "non_tomato_confidence": non_tomato_conf,
        "validation_message":    msg,
    }


# ── Stage 2: Multi-Label Disease Detection ────────────────────────────────────

def predict_disease(img_array: np.ndarray) -> dict:
    """
    Full 2-stage multi-label pipeline.

    Stage 1: Validate tomato leaf.
    Stage 2: Multi-label disease detection.

    Multi-label logic:
    ------------------
    Each of the 4 sigmoid outputs is checked independently
    against DISEASE_THRESHOLD (0.70).

    - primary_disease   : disease with highest confidence score
    - secondary_diseases: other diseases >= 70% threshold
    - If only Healthy >= 70%: leaf is healthy
    - If disease(s) >= 70%: those diseases are detected

    Parameters
    ----------
    img_array : np.ndarray
        float32 shape (1, 224, 224, 3)

    Returns
    -------
    dict — full structured JSON response
    """

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    validation = validate_tomato_leaf(img_array)

    if not validation["is_tomato"]:
        return {
            "success":               False,
            "is_tomato_leaf":        False,
            "stage":                 1,
            "tomato_confidence":     validation["tomato_confidence"],
            "non_tomato_confidence": validation["non_tomato_confidence"],
            "error":                 "Not a tomato leaf",
            "message":               validation["validation_message"],
            "suggestion": (
                "Please upload a clear, well-lit photo of a tomato leaf. "
                "Make sure the leaf fills most of the image frame. "
                "Avoid uploading other plants, food, animals, or objects."
            ),
        }

    # ── STAGE 2: Multi-label inference ────────────────────────────────────────
    _load_disease()

    # predictions shape: (1, 4)
    # Each value is an INDEPENDENT sigmoid probability [0.0 to 1.0]
    predictions = _disease_model.predict(img_array, verbose=0)
    probs       = predictions[0]   # shape: (4,)

    # Build all_scores dict — percentage rounded to 2 d.p.
    all_scores = {
        label: round(float(prob) * 100.0, 2)
        for label, prob in zip(CLASS_LABELS, probs)
    }

    # ── Multi-label detection ─────────────────────────────────────────────────
    # Find all diseases at or above the 70% threshold
    detected = [
        {"disease": label, "confidence": round(float(prob) * 100.0, 2)}
        for label, prob in zip(CLASS_LABELS, probs)
        if float(prob) >= DISEASE_THRESHOLD
    ]

    # Sort detected diseases by confidence descending
    detected.sort(key=lambda x: x["confidence"], reverse=True)

    # Identify primary disease (highest confidence above threshold)
    # If nothing above threshold, take the highest scoring class
    if detected:
        primary         = detected[0]
        secondary_list  = detected[1:]
    else:
        # Nothing above threshold — take highest score as primary
        best_idx        = int(np.argmax(probs))
        primary         = {
            "disease":    CLASS_LABELS[best_idx],
            "confidence": round(float(probs[best_idx]) * 100.0, 2),
        }
        secondary_list = []

    primary_disease    = primary["disease"]
    primary_confidence = primary["confidence"]
    secondary_diseases = secondary_list

    # ── Build result message ──────────────────────────────────────────────────
    if primary_disease == "Healthy" and not secondary_diseases:
        # Leaf is healthy
        message = "No disease detected. The tomato leaf appears healthy."
        is_healthy = True

    elif primary_disease == "Healthy" and secondary_diseases:
        # Healthy but also showing some disease signs
        disease_names = [d["disease"] for d in secondary_diseases]
        message = (
            f"Leaf is mostly healthy but shows early signs of: "
            f"{', '.join(disease_names)}. Monitor closely."
        )
        is_healthy = False

    elif len(secondary_diseases) > 0:
        # Multiple diseases detected
        secondary_names = [d["disease"] for d in secondary_diseases]
        message = (
            f"Multiple diseases detected. "
            f"Primary: {primary_disease} ({primary_confidence:.1f}%). "
            f"Also detected: {', '.join(secondary_names)}."
        )
        is_healthy = False

    else:
        # Single disease detected
        message = (
            f"Disease detected. "
            f"The tomato leaf is affected by {primary_disease}."
        )
        is_healthy = False

    return {
        "success":             True,
        "is_tomato_leaf":      True,
        "stage":               2,
        "tomato_confidence":   validation["tomato_confidence"],
        "is_healthy":          is_healthy,
        "disease":             primary_disease,
        "confidence":          primary_confidence,
        "secondary_diseases":  secondary_diseases,
        "message":             message,
        "all_scores":          all_scores,
    }
