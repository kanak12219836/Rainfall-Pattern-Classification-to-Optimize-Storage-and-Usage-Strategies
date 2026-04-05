import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Rainfall Pattern Classification API")

# Load the best performing model (Xception)
MODEL_PATH = "models/xception_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[ERROR] Could not load model: {e}")

CLASS_NAMES = ["Light", "Medium", "Heavy"]  # Update according to your labels
IMG_SIZE = (224, 224)  # Xception input size

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image")
    image = image.resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

@app.get("/")
def root():
    return {"message": "Rainfall Pattern Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded."})

    try:
        content = await file.read()
        img = preprocess_image(content)
        preds = model.predict(img)

        if preds is None or preds.shape[0] == 0:
            raise ValueError("Prediction failed")

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
