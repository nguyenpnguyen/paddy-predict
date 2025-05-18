"""
FastAPI Application for Paddy Image Classification, Variety, and Age Prediction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse # Keep HTMLResponse for the main page, use JSONResponse for predictions
from fastapi.templating import Jinja2Templates # Keep Jinja2Templates for the main page
import numpy as np 
import tensorflow as tf
from tensorflow import keras
import pickle

import os

# --- Configuration ---
# Paths to your saved model files
# Make sure these paths are correct relative to where you run the FastAPI app
DISEASE_MODEL_PATH = 'saved_models/model_disease.keras'
DISEASE_ENCODER_PATH = 'saved_models/label_encoder_disease.pkl'
VARIETY_MODEL_PATH = 'saved_models/model_variety.keras'
VARIETY_ENCODER_PATH = 'saved_models/label_encoder_variety.pkl'
AGE_MODEL_PATH = 'saved_models/model_age.keras'


# Define the image size your models expect
IMAGE_SIZE = (128, 128) # Must match the target_size used during training

# Define the class names for classification tasks in the same order as your model's output
# DISEASE_CLASS_NAMES: From your Task 1 training script's train_generator.class_indices
DISEASE_CLASS_NAMES = sorted([
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro'
])

# VARIETY_CLASS_NAMES: You will need to determine these from your Task 2 data
VARIETY_CLASS_NAMES = sorted([
    'ADT45',
    'IR20',
    'KarnatakaPonni',
    'Onthanel',
    'Ponni',
    'Surya',
    'Zonal',
    'AndraPonni',
    'AtchayaPonni',
    'RR'
])


# --- Model Loading ---
# Load the trained models when the application starts
disease_model = None
variety_model = None
age_model = None

# Try loading models - add more robust error handling as needed
if os.path.exists(DISEASE_MODEL_PATH):
    disease_model = keras.models.load_model(DISEASE_MODEL_PATH)
    with open(DISEASE_ENCODER_PATH, "rb") as f:
        disease_encoder = pickle.load(f)
    print(f"Disease model loaded successfully from {DISEASE_MODEL_PATH}")
else:
    print(f"Disease model not found at {DISEASE_MODEL_PATH}. Prediction will use placeholder.")

if os.path.exists(VARIETY_MODEL_PATH):
    variety_model = keras.models.load_model(VARIETY_MODEL_PATH)
    with open(VARIETY_ENCODER_PATH, "rb") as f:
        variety_encoder = pickle.load(f)
    print(f"Variety model loaded successfully from {VARIETY_MODEL_PATH}")
else:
     print(f"Variety model not found at {VARIETY_MODEL_PATH}. Prediction will use placeholder.")

if os.path.exists(AGE_MODEL_PATH):
    age_model = keras.models.load_model(AGE_MODEL_PATH)
    print(f"Age model loaded successfully from {AGE_MODEL_PATH}")
else:
    print(f"Age model not found at {AGE_MODEL_PATH}. Prediction will use placeholder.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Paddy ML Tasks API",
    description="API for classifying paddy images (Disease, Variety, Age).",
    version="1.0.0",
)


templates = Jinja2Templates(directory="templates")

async def preprocess_image(file: UploadFile, target_size: tuple):
    """Reads, preprocesses, and returns an image as a numpy array."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"File type '{file.content_type}' is not supported. Please upload an image file.")

    try:
         image_bytes = await file.read()
         image = tf.io.decode_jpeg(image_bytes, channels=3)
         image = tf.image.resize(image, target_size)
         image_array = tf.image.convert_image_dtype(image, tf.float32).numpy()
         image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
         return image_array

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")


# Root endpoint to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for Disease Classification
@app.post("/predict/disease/", response_class=JSONResponse) # Change response_class back to JSONResponse
async def predict_disease(file: UploadFile = File(...)): # Remove request parameter
    """Receives an image and returns the predicted disease or 'normal' as JSON."""
    if disease_model is None:
         raise HTTPException(status_code=500, detail="Disease model not loaded.")

    try:
        image_array = await preprocess_image(file, IMAGE_SIZE)
        predictions = disease_model.predict(image_array, verbose=1)
        #disease_label = disease_encoder.inverse_transform(np.argmax(predictions, axis=1))
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = DISEASE_CLASS_NAMES[predicted_class_index]
        return JSONResponse(content={"predicted_class": predicted_class_name})

    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Error during disease prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


# Endpoint for Variety Classification (Placeholder)
@app.post("/predict/variety/", response_class=JSONResponse) # Change response_class back to JSONResponse
async def predict_variety(file: UploadFile = File(...)): # Remove request parameter
    if variety_model is None:
        return HTTPException(status_code=500, detail="Variety model not loaded.")

    try:
        image_array = await preprocess_image(file, IMAGE_SIZE)
        predictions = variety_model.predict(image_array)
        #variety_label = variety_encoder.inverse_transform(np.argmax(predictions, axis=1))
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_variety_name = VARIETY_CLASS_NAMES[predicted_class_index]
        return JSONResponse(content={"predicted_variety": predicted_variety_name})

    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Error during variety prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during variety prediction: {e}")


# Endpoint for Age Prediction (Placeholder)
@app.post("/predict/age/", response_class=JSONResponse) # Change response_class back to JSONResponse
async def predict_age(file: UploadFile = File(...)): # Remove request parameter
    if age_model is None:
        return HTTPException(status_code=500, detail="Age model not loaded.")

    try:
        image_array = await preprocess_image(file, IMAGE_SIZE)
        predicted_age_value = age_model.predict(image_array)[0][0] 
        return JSONResponse(content={"predicted_age": int(predicted_age_value)}) # Return as int for JSON

    except HTTPException as e:
         raise e
    except Exception as e:
        print(f"Error during age prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during age prediction: {e}")
