"""
routes/predict_routes.py
-------------------------
Flask Blueprint for Multi-Label Tomato Leaf Disease Detection API.

Endpoints:
    GET  /          -> API status
    GET  /health    -> both model health check
    POST /predict   -> 2-stage multi-label disease detection

RESPONSE FORMAT (Stage 2 success):
{
    "success": true,
    "is_tomato_leaf": true,
    "stage": 2,
    "tomato_confidence": 96.43,
    "is_healthy": false,
    "disease": "Late Blight",
    "confidence": 89.43,
    "secondary_diseases": [
        {"disease": "Bacterial Spot", "confidence": 74.21}
    ],
    "message": "Multiple diseases detected...",
    "all_scores": {
        "Bacterial Spot": 74.21,
        "Healthy": 3.12,
        "Late Blight": 89.43,
        "Yellow Leaf Curl": 12.33
    }
}
"""

import os
import uuid
import logging

from flask import Blueprint, jsonify, request, current_app

from utils.preprocess import allowed_file, preprocess_image
from utils.predictor  import get_model, predict_disease

logger     = logging.getLogger(__name__)
predict_bp = Blueprint("predict", __name__)


# ── GET / ─────────────────────────────────────────────────────────────────────

@predict_bp.route("/", methods=["GET"])
def index():
    return jsonify({
        "success": True,
        "message": "Multi-Label Tomato Leaf Disease Detection API is running",
        "version": "3.0.0",
        "pipeline": "Stage 1: Tomato Validator -> Stage 2: Multi-Label Disease Classifier",
        "detection_mode": "Multi-Label Sigmoid (multiple diseases per leaf)",
        "disease_threshold": "70% confidence",
        "endpoints": {
            "health":  "GET  /health",
            "predict": "POST /predict  (multipart/form-data, key: 'file')",
        },
    }), 200


# ── GET /health ───────────────────────────────────────────────────────────────

@predict_bp.route("/health", methods=["GET"])
def health():
    """Health check — verifies both models are loaded."""
    from utils.predictor import _validator_model, _disease_model

    stage1_loaded = _validator_model is not None
    stage2_loaded = _disease_model is not None
    model_info    = {}

    try:
        model         = get_model()
        stage1_loaded = True
        stage2_loaded = model is not None

        try:
            input_shape  = str(model.input_shape)
            output_shape = str(model.output_shape)
        except AttributeError:
            try:
                input_shape  = str(model.layers[0].input_shape)
                output_shape = str(model.layers[-1].output_shape)
            except Exception:
                input_shape  = "unknown"
                output_shape = "unknown"

        try:
            total_params = int(model.count_params())
        except Exception:
            total_params = -1

        model_info = {
            "stage1_validator":  "loaded",
            "stage2_disease":    "loaded",
            "detection_mode":    "Multi-Label Sigmoid",
            "disease_threshold": "70%",
            "input_shape":       input_shape,
            "output_shape":      output_shape,
            "total_params":      total_params,
        }

    except FileNotFoundError as exc:
        model_info = {"error": str(exc)}

    both_loaded = stage1_loaded and stage2_loaded

    return jsonify({
        "success":              True,
        "status":               "healthy" if both_loaded else "degraded",
        "stage1_loaded":        stage1_loaded,
        "stage2_loaded":        stage2_loaded,
        "model_info":           model_info,
        "upload_folder_exists": os.path.isdir(
            current_app.config.get("UPLOAD_FOLDER", "")
        ),
    }), 200


# ── POST /predict ─────────────────────────────────────────────────────────────

@predict_bp.route("/predict", methods=["POST"])
def predict():
    """
    Multi-Label 2-Stage Prediction Endpoint.

    Stage 1: Tomato Leaf Validator
        - Checks if uploaded image is a tomato leaf
        - NOT tomato leaf -> HTTP 422 with explanation

    Stage 2: Multi-Label Disease Classifier
        - Detects ALL diseases present above 70% threshold
        - Returns primary disease + secondary diseases
        - Returns all 4 class probability scores

    Request:
        POST /predict
        Content-Type: multipart/form-data
        Body: file=<image>  (png/jpg/jpeg/webp, max 10MB)

    Success HTTP 200:
    {
        "success": true,
        "is_tomato_leaf": true,
        "stage": 2,
        "tomato_confidence": 96.43,
        "is_healthy": false,
        "disease": "Late Blight",
        "confidence": 89.43,
        "secondary_diseases": [
            {"disease": "Bacterial Spot", "confidence": 74.21}
        ],
        "message": "Multiple diseases detected...",
        "all_scores": {...}
    }

    Rejection HTTP 422 (not a tomato leaf):
    {
        "success": false,
        "is_tomato_leaf": false,
        "error": "Not a tomato leaf",
        "message": "...",
        "suggestion": "..."
    }

    Error codes:
        400 -> no file / empty filename
        415 -> unsupported file type
        422 -> not a tomato leaf OR corrupt image
        500 -> unexpected server error
        503 -> model files missing
    """

    # Step 1: Verify file key
    if "file" not in request.files:
        logger.warning("POST /predict - 'file' key missing")
        return jsonify({
            "success": False,
            "error": (
                "No file uploaded. "
                "Send multipart/form-data with key 'file'."
            ),
        }), 400

    file = request.files["file"]

    # Step 2: Verify filename
    if not file.filename:
        logger.warning("POST /predict - empty filename")
        return jsonify({
            "success": False,
            "error": "No filename. Please select a valid image file.",
        }), 400

    # Step 3: Verify extension
    if not allowed_file(file.filename):
        logger.warning("POST /predict - unsupported type: %s", file.filename)
        return jsonify({
            "success": False,
            "error": (
                f"File type not supported: '{file.filename}'. "
                "Allowed: png, jpg, jpeg, webp."
            ),
        }), 415

    # Step 4: Save temp file
    ext       = file.filename.rsplit(".", 1)[1].lower()
    unique_fn = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_fn)

    try:
        file.save(save_path)
        logger.info("Upload saved: %s", save_path)
    except OSError as exc:
        logger.error("Save failed: %s", exc)
        return jsonify({
            "success": False,
            "error": "Server could not save the file. Please try again.",
        }), 500

    # Step 5: Preprocess
    try:
        img_array = preprocess_image(save_path)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        _cleanup(save_path)
        return jsonify({
            "success": False,
            "error": f"Image could not be processed: {exc}",
        }), 422

    # Step 6: 2-stage multi-label prediction
    try:
        result = predict_disease(img_array)
    except FileNotFoundError as exc:
        logger.error("Model missing: %s", exc)
        _cleanup(save_path)
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 503
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        _cleanup(save_path)
        return jsonify({
            "success": False,
            "error": f"Prediction failed: {exc}",
        }), 500

    # Step 7: Cleanup
    _cleanup(save_path)

    # Step 8: Return with correct HTTP status
    if not result.get("is_tomato_leaf", True):
        logger.warning(
            "Stage 1 REJECTED - tomato_conf=%.2f%% non_tomato_conf=%.2f%%",
            result.get("tomato_confidence", 0.0),
            result.get("non_tomato_confidence", 0.0),
        )
        return jsonify(result), 422

    # Log multi-label results
    secondary = result.get("secondary_diseases", [])
    if secondary:
        secondary_str = ", ".join(
            f"{d['disease']} ({d['confidence']}%)" for d in secondary
        )
        logger.info(
            "Stage 2 MULTI-LABEL - primary: %s (%.2f%%), secondary: %s",
            result.get("disease"),
            result.get("confidence", 0.0),
            secondary_str,
        )
    else:
        logger.info(
            "Stage 2 SINGLE - disease: %s (%.2f%%)",
            result.get("disease"),
            result.get("confidence", 0.0),
        )

    return jsonify(result), 200


# ── Helper ────────────────────────────────────────────────────────────────────

def _cleanup(filepath: str) -> None:
    """Silently remove temp upload file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError as exc:
        logger.warning("Could not remove '%s': %s", filepath, exc)
