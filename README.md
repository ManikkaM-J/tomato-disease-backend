# Tomato Leaf Disease Detection - Backend API

**College Conference Project**
CNN + Digital Image Processing | Flask REST API | TensorFlow / Keras

---

## Project Overview

This backend serves a Convolutional Neural Network (CNN) that analyses
uploaded tomato leaf images and classifies them into one of four categories:

| Index | Class            | Description                              |
|-------|------------------|------------------------------------------|
| 0     | Healthy          | No disease present                       |
| 1     | Bacterial Spot   | Small dark water-soaked lesions on leaf  |
| 2     | Yellow Leaf Curl | Yellowing and upward curling of leaves   |
| 3     | Late Blight      | Large irregular dark blotches on leaf    |

---

## Architecture Workflow

```
+----------------------------------------------------------+
|                      INPUT IMAGE                         |
|           Tomato leaf image uploaded by user             |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
|                  IMAGE PREPROCESSING                     |
|                                                          |
|  [1] Image Resizing    -> resize to 224 x 224 pixels     |
|  [2] Normalization     -> pixel values / 255.0           |
|                           values in range [0.0, 1.0]     |
|  [3] Data Augmentation -> rotation, flip, zoom           |
|                           (applied during training only) |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
|            CNN FEATURE EXTRACTION MODULE                 |
|                                                          |
|  [1] Convolution Layer 1  (32 filters, 3x3)             |
|  [2] ReLU Activation                                     |
|  [3] Max Pooling Layer 1  (2x2 pool)                    |
|  [4] Convolution Layer 2  (64 filters, 3x3)             |
|  [5] ReLU Activation                                     |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
|              CLASSIFICATION MODULE                       |
|                                                          |
|  [6] Flatten  (2D feature map -> 1D vector)              |
|  [7] Dense(128) + ReLU                                   |
|  [8] Dropout(0.5)                                        |
|  [9] Dense(4)  + Softmax Activation                      |
+----------------------------------------------------------+
                            |
                            v
+----------------------------------------------------------+
|              RESULT AND NOTIFICATION                     |
|                                                          |
|  - Disease present or not                                |
|  - Disease name                                          |
|  - Confidence score (percentage)                         |
|  - Full JSON response for frontend integration           |
+----------------------------------------------------------+
```

---

## Technologies Used

| Technology       | Version        | Purpose                          |
|------------------|----------------|----------------------------------|
| Python           | 3.9 - 3.11     | Core language                    |
| Flask            | 2.3+           | REST API framework               |
| flask-cors       | 4.0+           | Cross-Origin Resource Sharing    |
| TensorFlow/Keras | 2.13 - 2.16    | CNN model training and inference |
| NumPy            | 1.23+          | Numerical array operations       |
| Pillow           | 9.4+           | Image loading and resizing       |

> **Python version note:** TensorFlow 2.x supports Python 3.9, 3.10, and
> 3.11 on Windows. Python 3.12 is NOT yet fully supported by TensorFlow.
> Download Python 3.11 from https://www.python.org/downloads/ if needed.

---

## Folder Structure

```
tomato_leaf_backend/
|
+-- app.py                    <- Flask entry point and application factory
+-- config.py                 <- All constants: paths, image size, classes
+-- requirements.txt          <- Python dependencies
+-- README.md                 <- This file
+-- .gitignore
|
+-- model/
|   +-- PLACE_MODEL_HERE.txt  <- Read this, then place your .h5 file here
|   +-- tomato_leaf_disease_model.h5   (you add this)
|
+-- utils/
|   +-- __init__.py
|   +-- preprocess.py         <- Resize, normalize, augment pipeline
|   +-- predictor.py          <- Model singleton + inference function
|   +-- model_builder.py      <- CNN architecture definition
|
+-- routes/
|   +-- __init__.py
|   +-- predict_routes.py     <- Flask Blueprint: /, /health, /predict
|
+-- uploads/
    +-- .keep                 <- Keeps folder in git; uploads go here
```

---

## Setup Steps (Windows + VS Code)

### Step 1 — Check your Python version

Open VS Code terminal (`Ctrl + backtick`) and run:

```
python --version
```

You need Python 3.9, 3.10, or 3.11.
Download from: https://www.python.org/downloads/

---

### Step 2 — Open the project in VS Code

```
File -> Open Folder -> select the tomato_leaf_backend folder
```

---

### Step 3 — Create a virtual environment

In the VS Code terminal:

```
python -m venv venv
```

---

### Step 4 — Activate the virtual environment

```
venv\Scripts\activate
```

You should see `(venv)` at the start of your terminal prompt.

> If you get a PowerShell execution policy error, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> Then try activating again.

---

### Step 5 — Install dependencies

```
pip install -r requirements.txt
```

This installs Flask, TensorFlow, NumPy, Pillow, and flask-cors.
It may take 2-5 minutes depending on your internet speed.

---

### Step 6 — Place your trained model

Copy your trained model file into the `model/` folder:

```
tomato_leaf_backend/
    model/
        tomato_leaf_disease_model.h5    <- place it here
```

The model file must:
- Be a Keras HDF5 format (.h5)
- Accept input shape (None, 224, 224, 3)
- Have 4 Softmax output neurons
- Use the same class index order as CLASS_LABELS in config.py

See `model/PLACE_MODEL_HERE.txt` for full training instructions.

---

### Step 7 — Run the backend

```
python app.py
```

Expected terminal output:

```
2024-xx-xx [INFO] CORS enabled for all origins.
2024-xx-xx [INFO] Upload folder ready: C:\...\tomato_leaf_backend\uploads
2024-xx-xx [INFO] Pre-warming CNN model...
2024-xx-xx [INFO] CNN model loaded and ready.
2024-xx-xx [INFO] Blueprint 'predict_bp' registered (routes: /, /health, /predict).
2024-xx-xx [INFO] Starting Tomato Leaf Disease Detection API at http://127.0.0.1:5000
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

The API is now running at **http://localhost:5000**

Stop the server at any time with `Ctrl + C`.

---

## API Endpoints

### GET /

Returns a status message confirming the API is running.

```
http://localhost:5000/
```

Response:
```json
{
    "success": true,
    "message": "Tomato Leaf Disease Detection API is running",
    "version": "1.0.0",
    "endpoints": {
        "health":  "GET  /health",
        "predict": "POST /predict  (multipart/form-data, key: 'file')"
    }
}
```

---

### GET /health

Returns model load status and server health information.

```
http://localhost:5000/health
```

Response (model loaded):
```json
{
    "success": true,
    "status": "healthy",
    "model_loaded": true,
    "model_info": {
        "input_shape": "(None, 224, 224, 3)",
        "output_shape": "(None, 4)",
        "total_params": 46935620
    },
    "upload_folder_exists": true
}
```

Response (model missing):
```json
{
    "success": true,
    "status": "degraded",
    "model_loaded": false,
    "model_info": {
        "error": "Trained model not found at: ...\\model\\tomato_leaf_disease_model.h5"
    },
    "upload_folder_exists": true
}
```

---

### POST /predict

Upload a tomato leaf image and receive a disease prediction.

- **Method:** POST
- **Content-Type:** multipart/form-data
- **Field name:** `file`
- **Accepted types:** png, jpg, jpeg, webp
- **Max file size:** 10 MB

**Success response — diseased leaf:**
```json
{
    "success": true,
    "disease": "Late Blight",
    "confidence": 97.43,
    "message": "Disease detected successfully. The tomato leaf is affected by Late Blight.",
    "all_scores": {
        "Healthy": 0.23,
        "Bacterial Spot": 1.87,
        "Yellow Leaf Curl": 0.47,
        "Late Blight": 97.43
    }
}
```

**Success response — healthy leaf:**
```json
{
    "success": true,
    "disease": "Healthy",
    "confidence": 98.12,
    "message": "No disease detected. The tomato leaf appears healthy.",
    "all_scores": {
        "Healthy": 98.12,
        "Bacterial Spot": 0.73,
        "Yellow Leaf Curl": 0.81,
        "Late Blight": 0.34
    }
}
```

**Error — no file uploaded:**
```json
{
    "success": false,
    "error": "No file uploaded. Send a multipart/form-data request with the key 'file'."
}
```

**Error — unsupported file type:**
```json
{
    "success": false,
    "error": "File type not supported for 'document.pdf'. Allowed types: png, jpg, jpeg, webp."
}
```

**Error — model file missing:**
```json
{
    "success": false,
    "error": "Trained model not found at: ...\\model\\tomato_leaf_disease_model.h5"
}
```

---

## How to Test

### Option 1 — Browser

Open your browser and visit:

```
http://localhost:5000
http://localhost:5000/health
```

### Option 2 — Postman

1. Open Postman
2. Click **New Request**
3. Set method to **POST**
4. Set URL to `http://localhost:5000/predict`
5. Click **Body** tab
6. Select **form-data**
7. Add a key named `file`, change type to **File**
8. Click **Select Files** and choose a tomato leaf image
9. Click **Send**

### Option 3 — curl (Command Line)

**Windows Command Prompt:**
```
curl -X POST http://localhost:5000/predict -F "file=@C:\path\to\leaf.jpg"
```

**Windows PowerShell:**
```powershell
curl -X POST http://localhost:5000/predict -F "file=@C:\path\to\leaf.jpg"
```

**Git Bash / macOS / Linux:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@/path/to/leaf.jpg"
```

**Test the root and health endpoints:**
```
curl http://localhost:5000/
curl http://localhost:5000/health
```

---

## Frontend Integration

Send a `POST` request to `http://localhost:5000/predict` using the
browser `fetch` API:

```javascript
async function detectDisease(imageFile) {
    const formData = new FormData();
    formData.append("file", imageFile);

    const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
    });

    const result = await response.json();

    if (result.success) {
        console.log("Disease:", result.disease);
        console.log("Confidence:", result.confidence + "%");
        console.log("Message:", result.message);
        console.log("All scores:", result.all_scores);
    } else {
        console.error("Error:", result.error);
    }
}

// Usage with a file input element:
// const fileInput = document.getElementById("imageInput");
// detectDisease(fileInput.files[0]);
```

---

## Verify CNN Architecture (Optional)

Run this from the project root to confirm the model layers are correct:

```
python utils/model_builder.py
```

Expected layer output:

```
Layer (type)               Output Shape         Param #
=================================================================
Conv_Layer1_ReLU (Conv2D)  (None, 224, 224, 32) 896
MaxPool_Layer1 (MaxPool2D) (None, 112, 112, 32) 0
Conv_Layer2_ReLU (Conv2D)  (None, 112, 112, 64) 18496
Flatten_2D_to_1D (Flatten) (None, 802816)        0
Dense_Hidden_ReLU (Dense)  (None, 128)           102,760,576
Dropout (Dropout)          (None, 128)           0
Softmax_Output (Dense)     (None, 4)             516
=================================================================
```

---

## Troubleshooting

| Problem | Solution |
|---------|---------|
| `ModuleNotFoundError: No module named 'flask'` | Run `pip install -r requirements.txt` and ensure `(venv)` is active |
| `ModuleNotFoundError: No module named 'config'` | Always run `python app.py` from inside the project folder |
| `FileNotFoundError: model not found` | Place `tomato_leaf_disease_model.h5` inside the `model/` folder |
| PowerShell activation error | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` then retry |
| `Port 5000 already in use` | Change port in `app.py`: `flask_app.run(port=5001, ...)` |
| Slow first startup | Normal — TensorFlow initialises GPU/CPU kernels on first load |
| `Object is not JSON serializable` | Fixed in v2 — all predictions return plain Python float/int |
| `AttributeError: input_shape` | Fixed in v2 — /health uses try/except for Keras 3.x compatibility |

---

*Built for the college conference project:*
*"Tomato Leaf Disease Detection Using CNN and Digital Image Processing"*
