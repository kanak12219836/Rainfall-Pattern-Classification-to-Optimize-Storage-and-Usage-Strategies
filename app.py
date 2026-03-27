import io
import uvicorn
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf

app = FastAPI(title="Rainfall Pattern Classification API")


MODEL_PATH = "rainfall_model.h5"

# Load the trained model once at startup
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

# Update to your actual class names in correct order
CLASS_NAMES = ["Light", "Medium", "Heavy"]

# Update to the image size used in your training script
IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def root():
    return {"message": "Rainfall Pattern Classification API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": f"Model not loaded. Check {MODEL_PATH} exists on server."}
        )

    try:
        file_bytes = await file.read()
        img_array = preprocess_image(file_bytes)
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
            content={"error": str(e)}
        )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
