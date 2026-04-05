import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf

app = FastAPI(title="Rainfall Pattern Classification API (Xception)")

# 1. Model path (Xception ka best model yahan save karke upload karo)
MODEL_PATH = "xception_rainfall_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Xception model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# 2. Classes & image size (tumhare project ke hisaab se)
CLASS_NAMES = ["Light", "Medium", "Heavy"]
IMG_SIZE = (224, 224)  # Xception ke training me jo size use kiya tha, wahi rakho


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Bytes -> tensor
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)  # (1, H, W, 3)
    return img


@app.get("/")
def root():
    return {"message": "Rainfall Pattern Classification API is running (Xception model)"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model not loaded. Ensure {MODEL_PATH} is present and valid Xception model."},
        )

    try:
        content = await file.read()
        img = preprocess_image(content)
        preds = model.predict(img)
        idx = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        label = CLASS_NAMES[idx]

        return {
            "predicted_class": label,
            "confidence": conf,
            "class_index": idx,
            "classes": CLASS_NAMES,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
