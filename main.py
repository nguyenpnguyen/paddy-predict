"""
FastAPI Application for Paddy Image Classification, Variety, and Age Prediction

To run this application:
Run the server: uvicorn main:app --reload (if this file is named main.py)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow import keras
from PIL import Image
import numpy as np
import io

# --- Configuration ---
# Paths to your saved model files
# Make sure these paths are correct relative to where you run the FastAPI app
DISEASE_MODEL_PATH = 'paddy_disease_model.h5'
VARIETY_MODEL_PATH = 'paddy_variety_model.h5'
AGE_MODEL_PATH = 'paddy_age_model.h5'


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

try:
    disease_model = keras.models.load_model(DISEASE_MODEL_PATH)
    print(f"Disease model loaded successfully from {DISEASE_MODEL_PATH}")
except Exception as e:
    print(f"Error loading disease model from {DISEASE_MODEL_PATH}: {e}")
    print("Please ensure the disease model file exists and is valid.")

try:
    # Load your trained variety classification model here
    # variety_model = keras.models.load_model(VARIETY_MODEL_PATH)
    print(f"Attempted to load variety model from {VARIETY_MODEL_PATH} (Placeholder)")
except Exception as e:
    print(f"Error loading variety model from {VARIETY_MODEL_PATH}: {e}")
    print("Please ensure the variety model file exists and is valid if implementing this task.")

try:
    # Load your trained age prediction model here
    # age_model = keras.models.load_model(AGE_MODEL_PATH)
    print(f"Attempted to load age model from {AGE_MODEL_PATH} (Placeholder)")
except Exception as e:
    print(f"Error loading age model from {AGE_MODEL_PATH}: {e}")
    print("Please ensure the age model file exists and is valid if implementing this task.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Paddy ML Tasks API",
    description="API for classifying paddy images (Disease, Variety, Age).",
    version="1.0.0",
)

# Configure Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# --- Helper Function for Image Preprocessing ---
async def preprocess_image(file: UploadFile, target_size: tuple):
    """Reads, preprocesses, and returns an image as a numpy array."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"File type '{file.content_type}' is not supported. Please upload an image file.")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

        return image_array

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")


# --- Endpoints ---

# Root endpoint to serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main index.html page."""
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for Disease Classification
@app.post("/predict/disease/")
async def predict_disease(file: UploadFile = File(...)):
    """Receives an image and returns the predicted disease or 'normal'."""
    if disease_model is None:
        raise HTTPException(status_code=500, detail="Disease model not loaded.")

    image_array = await preprocess_image(file, IMAGE_SIZE)

    try:
        predictions = disease_model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = DISEASE_CLASS_NAMES[predicted_class_index]

        return JSONResponse(content={"predicted_class": predicted_class_name})

    except Exception as e:
        print(f"Error during disease prediction: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


# Endpoint for Variety Classification (Placeholder)
@app.post("/predict/variety/")
async def predict_variety(file: UploadFile = File(...)):
    """Receives an image and returns the predicted paddy variety (Placeholder)."""
    if variety_model is None:
        # In a real implementation, load the variety model here or at startup
        # For now, return a placeholder response
        # raise HTTPException(status_code=500, detail="Variety model not loaded.")
        return JSONResponse(content={"predicted_variety": "Variety Prediction Placeholder"})

    # --- Add variety prediction logic here ---
    # image_array = await preprocess_image(file, IMAGE_SIZE)
    # predictions = variety_model.predict(image_array)
    # predicted_class_index = np.argmax(predictions, axis=1)[0]
    # predicted_variety_name = VARIETY_CLASS_NAMES[predicted_class_index]
    # return JSONResponse(content={"predicted_variety": predicted_variety_name})
    pass # Remove this pass when implementing


# Endpoint for Age Prediction (Placeholder)
@app.post("/predict/age/")
async def predict_age(file: UploadFile = File(...)):
    """Receives an image and returns the predicted paddy age (Placeholder)."""
    if age_model is None:
        # In a real implementation, load the age model here or at startup
        # For now, return a placeholder response
        # raise HTTPException(status_code=500, detail="Age model not loaded.")
         return JSONResponse(content={"predicted_age": "Age Prediction Placeholder"})

    # --- Add age prediction logic here ---
    # image_array = await preprocess_image(file, IMAGE_SIZE)
    # predicted_age_value = age_model.predict(image_array)[0][0] # Assuming regression model outputs a single value
    # return JSONResponse(content={"predicted_age": float(predicted_age_value)}) # Return as float for JSON

    pass # Remove this pass when implementing
