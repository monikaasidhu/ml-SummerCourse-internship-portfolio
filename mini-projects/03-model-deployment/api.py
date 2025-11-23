"""
MODEL DEPLOYMENT API - FastAPI Application
Serves trained ML models via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import joblib
import numpy as np
import json

# Initialize FastAPI
app = FastAPI(
    title="ML Model Deployment API",
    description="Production API serving multiple ML models",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
models = {}
metadata = {}

def load_models():
    """Load all trained models"""
    try:
        print("Loading models...")
        
        # Load Iris model
        models['iris_model'] = joblib.load('iris_model.pkl')
        models['iris_scaler'] = joblib.load('iris_scaler.pkl')
        print(" Iris model loaded")
        
        # Load Wine model
        models['wine_model'] = joblib.load('wine_model.pkl')
        models['wine_scaler'] = joblib.load('wine_scaler.pkl')
        print(" Wine model loaded")
        
        # Load metadata
        with open('model_metadata.json', 'r') as f:
            global metadata
            metadata = json.load(f)
        print(" Metadata loaded")
        
        return True
    except Exception as e:
        print(f" Error loading models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    load_models()

# Request/Response Models
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10)
    sepal_width: float = Field(..., ge=0, le=10)
    petal_length: float = Field(..., ge=0, le=10)
    petal_width: float = Field(..., ge=0, le=10)
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float
    
    class Config:
        schema_extra = {
            "example": {
                "alcohol": 13.2,
                "malic_acid": 2.0,
                "ash": 2.4,
                "alcalinity_of_ash": 18.0,
                "magnesium": 100.0,
                "total_phenols": 2.5,
                "flavanoids": 2.8,
                "nonflavanoid_phenols": 0.3,
                "proanthocyanins": 1.8,
                "color_intensity": 5.0,
                "hue": 1.0,
                "od280_od315": 3.0,
                "proline": 1000.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    all_probabilities: Dict[str, float]
    model_used: str
    timestamp: str

# Endpoints
@app.get("/")
def root():
    return {
        "message": "ML Model Deployment API",
        "version": "1.0.0",
        "models_loaded": len(models) > 0,
        "available_endpoints": [
            "/predict/iris",
            "/predict/wine",
            "/model-info",
            "/health"
        ],
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "iris": "iris_model" in models,
            "wine": "wine_model" in models
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
def get_model_info():
    """Get information about loaded models"""
    return {
        "models": metadata,
        "total_models": len(metadata),
        "status": "active"
    }

@app.post("/predict/iris", response_model=PredictionResponse)
def predict_iris(data: IrisInput):
    """Predict Iris species"""
    
    if 'iris_model' not in models:
        raise HTTPException(status_code=503, detail="Iris model not loaded")
    
    try:
        # Prepare input
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])
        
        # Scale
        features_scaled = models['iris_scaler'].transform(features)
        
        # Predict
        prediction = models['iris_model'].predict(features_scaled)[0]
        probabilities = models['iris_model'].predict_proba(features_scaled)[0]
        
        # Get class names
        classes = metadata['iris_model']['classes']
        
        return PredictionResponse(
            prediction=classes[prediction],
            probability=float(probabilities[prediction]),
            all_probabilities={
                class_name: float(prob) 
                for class_name, prob in zip(classes, probabilities)
            },
            model_used="RandomForestClassifier",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/wine", response_model=PredictionResponse)
def predict_wine(data: WineInput):
    """Predict Wine type"""
    
    if 'wine_model' not in models:
        raise HTTPException(status_code=503, detail="Wine model not loaded")
    
    try:
        # Prepare input
        features = np.array([[
            data.alcohol, data.malic_acid, data.ash,
            data.alcalinity_of_ash, data.magnesium,
            data.total_phenols, data.flavanoids,
            data.nonflavanoid_phenols, data.proanthocyanins,
            data.color_intensity, data.hue,
            data.od280_od315, data.proline
        ]])
        
        # Scale
        features_scaled = models['wine_scaler'].transform(features)
        
        # Predict
        prediction = models['wine_model'].predict(features_scaled)[0]
        probabilities = models['wine_model'].predict_proba(features_scaled)[0]
        
        # Get class names
        classes = metadata['wine_model']['classes']
        
        return PredictionResponse(
            prediction=classes[prediction],
            probability=float(probabilities[prediction]),
            all_probabilities={
                class_name: float(prob) 
                for class_name, prob in zip(classes, probabilities)
            },
            model_used="LogisticRegression",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/stats")
def get_stats():
    """Get API statistics"""
    return {
        "total_models": len(models) // 2,  # Each model has a scaler
        "active_models": ["iris", "wine"],
        "api_uptime": "running",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" STARTING ML MODEL API SERVER")
    print("="*60)
    print("\n API Documentation: http://localhost:8000/docs")
    print(" Homepage: http://localhost:8000")
    print("\n Server running... Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
