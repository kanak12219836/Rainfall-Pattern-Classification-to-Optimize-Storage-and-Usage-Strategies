import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# Initialize app
app = FastAPI()

# Load model
MODEL_PATH = "models_trained/xception_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (update if different)
CLASS_NAMES = ["heavy", "medium", "light"]

# Image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # same as training
    image = np.array(image) / 255.0   # normalize
    image = np.expand_dims(image, axis=0)
    return image

# Root endpoint
@app.get("/")
def home():
    return {"message": "Rainfall Classification API is running 🚀"}

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        processed_image = preprocess_image(image)

        # Predict
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        return {
            "prediction": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}
