"""
FastAPI Application for Paddy Image Classification, Variety, and Age Prediction
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse # Keep HTMLResponse for the main page, use JSONResponse for predictions
from fastapi.templating import Jinja2Templates # Keep Jinja2Templates for the main page
from PIL import Image
import numpy as np
import io
import os
# from starlette.middleware.cors import CORSMiddleware # CORS might not be needed for same-origin

# --- Configuration ---
# Paths to your saved model files
# Make sure these paths are correct relative to where you run the FastAPI app
DISEASE_MODEL_PATH = 'paddy_disease_model.keras'
VARIETY_MODEL_PATH = 'paddy_variety_model.keras'
AGE_MODEL_PATH = 'paddy_age_model.keras'


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
try:
    # Import tensorflow here to avoid loading issues if not installed
    import tensorflow as tf
    from tensorflow import keras

    if os.path.exists(DISEASE_MODEL_PATH):
        disease_model = keras.models.load_model(DISEASE_MODEL_PATH)
        print(f"Disease model loaded successfully from {DISEASE_MODEL_PATH}")
    else:
        print(f"Disease model not found at {DISEASE_MODEL_PATH}. Prediction will use placeholder.")

    # Load your trained variety classification model here
    # if os.path.exists(VARIETY_MODEL_PATH):
    #     variety_model = keras.models.load_model(VARIETY_MODEL_PATH)
    #     print(f"Variety model loaded successfully from {VARIETY_MODEL_PATH}")
    # else:
    #      print(f"Variety model not found at {VARIETY_MODEL_PATH}. Prediction will use placeholder.")
    print(f"Attempted to load variety model from {VARIETY_MODEL_PATH} (Placeholder)")


    # Load your trained age prediction model here
    # if os.path.exists(AGE_MODEL_PATH):
    #     age_model = keras.models.load_model(AGE_MODEL_PATH)
    #     print(f"Age model loaded successfully from {AGE_MODEL_PATH}")
    # else:
    #     print(f"Age model not found at {AGE_MODEL_PATH}. Prediction will use placeholder.")
    print(f"Attempted to load age model from {AGE_MODEL_PATH} (Placeholder)")


except ImportError:
    print("Error: TensorFlow and Keras not installed. Cannot load models.")
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Paddy ML Tasks API",
    description="API for classifying paddy images (Disease, Variety, Age).",
    version="1.0.0",
)

# # CORS middleware might not be necessary if serving from the same origin (localhost)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Be more restrictive in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


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
        # Re-raise as HTTPException to be caught by FastAPI's error handling
        raise HTTPException(status_code=500, detail=f"An error occurred during image processing: {e}")


# --- Endpoints ---

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
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = DISEASE_CLASS_NAMES[predicted_class_index]

        # Return the result as JSON
        return JSONResponse(content={"predicted_class": predicted_class_name})

    except HTTPException as e:
         # Re-raise HTTPException from preprocess_image
         raise e
    except Exception as e:
        print(f"Error during disease prediction: {e}")
        # Return a general error message as JSON
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


# Endpoint for Variety Classification (Placeholder)
@app.post("/predict/variety/", response_class=JSONResponse) # Change response_class back to JSONResponse
async def predict_variety(file: UploadFile = File(...)): # Remove request parameter
    """Receives an image and returns the predicted paddy variety (Placeholder) as JSON."""
    if variety_model is None:
        # Return a placeholder response as JSON
        return JSONResponse(content={"predicted_variety": "Variety Prediction Placeholder (Model not loaded)"})

    try:
        # --- Add variety prediction logic here ---
        # image_array = await preprocess_image(file, IMAGE_SIZE)
        # predictions = variety_model.predict(image_array)
        # predicted_class_index = np.argmax(predictions, axis=1)[0]
        # predicted_variety_name = VARIETY_CLASS_NAMES[predicted_class_index]
        # # Return the result as JSON
        # return JSONResponse(content={"predicted_variety": predicted_variety_name})
        pass # Remove this pass when implementing the actual logic

    except HTTPException as e:
         # Re-raise HTTPException from preprocess_image
         raise e
    except Exception as e:
        print(f"Error during variety prediction: {e}")
        # Return a general error message as JSON
        raise HTTPException(status_code=500, detail=f"An error occurred during variety prediction: {e}")


# Endpoint for Age Prediction (Placeholder)
@app.post("/predict/age/", response_class=JSONResponse) # Change response_class back to JSONResponse
async def predict_age(file: UploadFile = File(...)): # Remove request parameter
    """Receives an image and returns the predicted paddy age (Placeholder) as JSON."""
    if age_model is None:
        # Return a placeholder response as JSON
         return JSONResponse(content={"predicted_age": "Age Prediction Placeholder (Model not loaded)"})

    try:
        # --- Add age prediction logic here ---
        # image_array = await preprocess_image(file, IMAGE_SIZE)
        # predicted_age_value = age_model.predict(image_array)[0][0] # Assuming regression model outputs a single value
        # # Return the result as JSON
        # return JSONResponse(content={"predicted_age": float(predicted_age_value)}) # Return as float for JSON

        pass # Remove this pass when implementing the actual logic

    except HTTPException as e:
         # Re-raise HTTPException from preprocess_image
         raise e
    except Exception as e:
        print(f"Error during age prediction: {e}")
        # Return a general error message as JSON
        raise HTTPException(status_code=500, detail=f"An error occurred during age prediction: {e}")
