"""
app.py
------
Entry point and application factory for the
Tomato Leaf Disease Detection API.

FIX 1: sys.path injection at the top ensures that config.py, utils/, and
        routes/ are always importable regardless of where the user runs
        `python app.py` from on Windows.

FIX 2: Logging format uses only ASCII dashes to avoid Windows console
        encoding errors (cp1252 / cp850 cannot render en-dash U+2013).

FIX 3: Blueprint import moved inside create_app() after sys.path is set
        so that the blueprint's own imports also resolve correctly.

Startup sequence
----------------
1. Inject project root into sys.path.
2. Create Flask app and load config.
3. Enable CORS.
4. Ensure the uploads directory exists.
5. Pre-warm the CNN model (loaded once; first request is instant).
6. Register Blueprint (/, /health, /predict).
7. Run development server.
"""

import os
import sys
import logging

# ── FIX 1: ensure project root is on sys.path ─────────────────────────────────
# This makes `import config`, `from utils.x import y`, and
# `from routes.x import y` work correctly on all platforms including Windows,
# no matter which directory the user is in when they run `python app.py`.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Now all project-level imports are safe ────────────────────────────────────
from flask import Flask          # noqa: E402
from flask_cors import CORS      # noqa: E402
import config                    # noqa: E402

# ── FIX 2: ASCII-only logging format (safe on Windows cp850/cp1252) ───────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Flask application factory.

    Returns a fully configured and ready-to-serve Flask application.
    Importing the blueprint here (not at module top) guarantees that
    sys.path is already set before any nested imports in the blueprint fire.
    """

    # ── FIX 3: import blueprint after sys.path is established ─────────────
    from routes.predict_routes import predict_bp
    from utils.predictor import get_model

    app = Flask(__name__)

    # ── Configuration ─────────────────────────────────────────────────────
    app.config["UPLOAD_FOLDER"]      = config.UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["DEBUG"]              = config.DEBUG

    # ── CORS ──────────────────────────────────────────────────────────────
    # Allow every origin so any frontend can call the API during development.
    CORS(app, resources={r"/*": {"origins": "*"}})
    logger.info("CORS enabled for all origins.")

    # ── Upload directory ──────────────────────────────────────────────────
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    logger.info("Upload folder ready: %s", app.config["UPLOAD_FOLDER"])

    # ── Pre-warm model ────────────────────────────────────────────────────
    # Loading takes a few seconds the first time; doing it at startup means
    # /predict responses are immediate.
    logger.info("Pre-warming CNN model...")
    try:
        get_model()
        logger.info("CNN model loaded and ready.")
    except FileNotFoundError as exc:
        logger.warning(
            "Model file not found at startup. "
            "Prediction will fail until the model is placed correctly.\n%s",
            exc,
        )

    # ── Register blueprint ────────────────────────────────────────────────
    app.register_blueprint(predict_bp)
    logger.info("Blueprint 'predict_bp' registered (routes: /, /health, /predict).")

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flask_app = create_app()
    logger.info(
        "Starting Tomato Leaf Disease Detection API at http://127.0.0.1:5000"
    )
    flask_app.run(
        host="0.0.0.0",
        port=5000,
        debug=config.DEBUG,
        # use_reloader=False prevents the model from loading twice in debug mode
        use_reloader=False,
    )
app = create_app()