import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Rainfall Pattern Classification API")

MODEL_PATH = "rainfall_model.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

CLASS_NAMES = ["Light", "Medium", "Heavy"]
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.get("/")
def root():
    return {"message": "Rainfall Pattern Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model not loaded. Ensure {MODEL_PATH} is present."},
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
