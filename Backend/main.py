import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import tensorflow as tf
import numpy as np

# Import your modules
from weather_api import get_city_coordinates, get_daily_forecasts
from prediction_service import predict_weather

# Load models with error handling and compatibility
try:
    # Disable TensorFlow warnings and info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Load models with custom objects if needed
    feature_extractor = tf.keras.models.load_model(
        "models/feature_extractor.h5",
        compile=False  # Skip compilation for inference-only usage
    )
    print("✓ Feature extractor model loaded successfully")
except Exception as e:
    print(f"Error loading feature extractor: {e}")
    feature_extractor = None

try:
    with open("models/xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    print("✓ XGB model loaded successfully")
except Exception as e:
    print(f"Error loading XGB model: {e}")
    xgb_model = None

try:
    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    print("✓ Label encoder loaded successfully")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# === FastAPI App ===
app = FastAPI(title="Weather Prediction API")

# Get allowed origins from environment variable
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "🌤 Weather Prediction API is running!"}

@app.get("/health")
def health_check():
    models_loaded = all([
        feature_extractor is not None,
        xgb_model is not None,
        label_encoder is not None
    ])
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "python_version": os.environ.get("PYTHON_VERSION", "unknown")
    }

@app.get("/weather/predict/{city}")
def get_prediction(city: str):
    if None in [feature_extractor, xgb_model, label_encoder]:
        return {"error": "Models not loaded properly"}
    
    lat, lon = get_city_coordinates(city)
    if not lat or not lon:
        return {"error": "City not found"}

    forecast_data = get_daily_forecasts(lat, lon)
    if not forecast_data:
        return {"error": "Could not fetch forecast"}

    predictions = predict_weather(forecast_data, feature_extractor, xgb_model, label_encoder)
    return {"city": city.title(), "predictions": predictions}

# Add startup event to verify Python version
@app.on_event("startup")
async def startup_event():
    import sys
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")