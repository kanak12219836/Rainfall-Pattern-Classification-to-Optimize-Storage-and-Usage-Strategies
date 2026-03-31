import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Rainfall Pattern Classification API")

MODEL_PATH = "rainfall_model.h5"

# Load model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Model loaded successfully")
except Exception as e:
    model = None
    print(f" Error loading model from {MODEL_PATH}: {e}")

# Class labels
CLASS_NAMES = ["Light", "Medium", "Heavy"]

# Image size (same as training)
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {e}")


@app.get("/")
def root():
    return {"message": "Rainfall Pattern Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model not loaded. Ensure {MODEL_PATH} exists in root directory."}
        )

    try:
        file_bytes = await file.read()

        # Preprocess image
        img_array = preprocess_image(file_bytes)

        # Prediction
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        predicted_label = CLASS_NAMES[class_idx]

        return {
            "predicted_class": predicted_label,
            "confidence": confidence,
            "class_index": class_idx,
            "classes": CLASS_NAMES,
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Prediction failed: {str(e)}"}
        )
