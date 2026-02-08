"""
FastAPI server for food image recognition
Replaces external Hugging Face API with local model inference
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Optional
import logging

from model_loader import FoodRecognitionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Food Recognition API",
    description="Local food recognition service for Plate Party mobile app",
    version="1.0.0"
)

# Configure CORS for React Native app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model (lazy loading)
model: Optional[FoodRecognitionModel] = None

# ============================================================================
# Request/Response Models
# ============================================================================

class ImageRequest(BaseModel):
    """Request model for image analysis"""
    inputs: str  # Base64 data URL (data:image/jpeg;base64,...)

class Classification(BaseModel):
    """Single classification result"""
    label: str
    score: float

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    logger.info("Loading food recognition model...")
    try:
        model = FoodRecognitionModel()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Food Recognition API",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model.model_name if model else None
    }

@app.post("/analyze", response_model=List[Classification])
async def analyze_image(request: ImageRequest):
    """
    Analyze food image and return classifications

    Compatible with Hugging Face Inference API format:
    - Input: {"inputs": "data:image/jpeg;base64,..."}
    - Output: [{"label": "food_name", "score": 0.95}, ...]
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extract base64 data from data URL
        base64_data = request.inputs
        if not base64_data.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Invalid image format. Expected data URL format.")

        logger.info("Processing image analysis request...")

        # Run inference
        classifications = model.predict(base64_data)

        logger.info(f"Analysis complete. Found {len(classifications)} predictions")

        return classifications

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# ============================================================================
# Server Configuration
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=True,  # Enable hot reload for development
        log_level="info"
    )
