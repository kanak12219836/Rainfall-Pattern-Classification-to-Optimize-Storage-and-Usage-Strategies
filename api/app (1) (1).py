import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import tensorflow as tf

# Config 
MODEL_PATH = "models_trained/xception_quant.tflite"
IMG_SIZE = (224, 224)

CLASS_NAMES = ['heavy', 'light', 'medium']

# Load Model 
model = tf.keras.models.load_model(MODEL_PATH)

# App 
app = FastAPI(title="Rainfall Classification API")

# Utils
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

# Routes
@app.get("/")
def root():
    return {"message": "Rainfall Classification API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        processed = preprocess_image(image)

        preds = model.predict(processed)
        pred_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        return JSONResponse({
            "class": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))