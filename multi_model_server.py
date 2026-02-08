"""
Enhanced FastAPI server with multi-model ensemble support
Provides improved food recognition accuracy through model combination
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import List, Optional, Dict, Any
import logging
from collections import defaultdict
import asyncio

from model_loader import FoodRecognitionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Food Recognition API",
    description="Multi-model food recognition service with ensemble predictions",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model Configuration
# ============================================================================

# Available models for ensemble
MODEL_CONFIGS = {
    "primary": {
        "name": "nateraw/food",  # Working food classification model
        "weight": 1.0,
        "enabled": True,
    },
    "secondary": {
        "name": "Kaludi/food-category-classification-v2.0",  # Alternative model
        "weight": 0.8,
        "enabled": True,
    },
}

# Model instances
models: Dict[str, Optional[FoodRecognitionModel]] = {
    "primary": None,
    "secondary": None,
}

# ============================================================================
# Request/Response Models
# ============================================================================

class ImageRequest(BaseModel):
    inputs: str  # Base64 data URL

class Classification(BaseModel):
    label: str
    score: float
    model: Optional[str] = None

class EnsembleClassification(BaseModel):
    label: str
    score: float
    models: List[str]
    individual_scores: Dict[str, float]

class AnalysisResponse(BaseModel):
    predictions: List[Classification]
    ensemble_predictions: Optional[List[EnsembleClassification]] = None
    model_status: Dict[str, bool]

# ============================================================================
# Model Loading
# ============================================================================

async def load_model_async(model_key: str, config: dict) -> Optional[FoodRecognitionModel]:
    """Load a model asynchronously"""
    if not config["enabled"]:
        return None

    try:
        logger.info(f"Loading {model_key} model: {config['name']}")
        model = FoodRecognitionModel(model_name=config["name"])
        logger.info(f"{model_key} model loaded successfully")
        return model
    except Exception as e:
        logger.warning(f"Failed to load {model_key} model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global models

    logger.info("Starting multi-model server...")

    # Load primary model (required)
    models["primary"] = await load_model_async("primary", MODEL_CONFIGS["primary"])

    if models["primary"] is None:
        logger.error("Primary model failed to load - server cannot start")
        raise RuntimeError("Primary model required")

    # Load secondary model (optional, for ensemble)
    try:
        models["secondary"] = await load_model_async("secondary", MODEL_CONFIGS["secondary"])
    except Exception as e:
        logger.warning(f"Secondary model not available: {e}")
        models["secondary"] = None

    loaded_count = sum(1 for m in models.values() if m is not None)
    logger.info(f"Server ready with {loaded_count} model(s)")

# ============================================================================
# Ensemble Logic
# ============================================================================

def merge_predictions(
    all_predictions: Dict[str, List[Dict[str, Any]]],
    model_weights: Dict[str, float]
) -> List[EnsembleClassification]:
    """
    Merge predictions from multiple models using weighted averaging

    Items predicted by multiple models get boosted confidence
    """
    # Aggregate scores by label
    label_data = defaultdict(lambda: {
        "total_weighted_score": 0.0,
        "models": [],
        "individual_scores": {},
    })

    for model_key, predictions in all_predictions.items():
        weight = model_weights.get(model_key, 1.0)

        for pred in predictions:
            label = pred["label"].lower().replace("_", " ")
            score = pred["score"] * weight

            data = label_data[label]
            data["total_weighted_score"] += score
            data["models"].append(model_key)
            data["individual_scores"][model_key] = pred["score"]

    # Build ensemble results
    ensemble_results = []

    for label, data in label_data.items():
        # Bonus for multi-model agreement
        agreement_bonus = 1.0 + (len(data["models"]) - 1) * 0.15

        avg_score = (data["total_weighted_score"] / len(data["models"])) * agreement_bonus
        avg_score = min(avg_score, 1.0)  # Cap at 1.0

        ensemble_results.append(EnsembleClassification(
            label=label.title(),
            score=avg_score,
            models=data["models"],
            individual_scores=data["individual_scores"],
        ))

    # Sort by score descending
    ensemble_results.sort(key=lambda x: x.score, reverse=True)

    return ensemble_results[:20]  # Return top 20

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "ok",
        "service": "Enhanced Food Recognition API",
        "version": "2.0.0",
        "models_loaded": {k: v is not None for k, v in models.items()},
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "models": {
            key: {
                "loaded": model is not None,
                "name": MODEL_CONFIGS[key]["name"] if model else None,
            }
            for key, model in models.items()
        },
    }

@app.get("/models")
async def list_models():
    """List available models and their status"""
    return {
        "models": {
            key: {
                "name": config["name"],
                "weight": config["weight"],
                "enabled": config["enabled"],
                "loaded": models.get(key) is not None,
            }
            for key, config in MODEL_CONFIGS.items()
        }
    }

@app.post("/analyze", response_model=List[Classification])
async def analyze_image(request: ImageRequest):
    """
    Analyze food image using primary model only

    Maintains backward compatibility with original API
    """
    if models["primary"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        if not request.inputs.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        logger.info("Processing single-model analysis...")
        classifications = models["primary"].predict(request.inputs)

        return [
            Classification(label=c["label"], score=c["score"], model="primary")
            for c in classifications
        ]

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-v2", response_model=List[Classification])
async def analyze_image_secondary(request: ImageRequest):
    """
    Analyze food image using secondary model

    Returns empty list if secondary model not available
    """
    if models["secondary"] is None:
        return []

    try:
        if not request.inputs.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        logger.info("Processing secondary model analysis...")
        classifications = models["secondary"].predict(request.inputs)

        return [
            Classification(label=c["label"], score=c["score"], model="secondary")
            for c in classifications
        ]

    except Exception as e:
        logger.warning(f"Secondary model error: {e}")
        return []

@app.post("/analyze-ensemble")
async def analyze_image_ensemble(request: ImageRequest) -> AnalysisResponse:
    """
    Analyze food image using all available models with ensemble merging

    Provides higher accuracy by combining predictions from multiple models
    """
    if models["primary"] is None:
        raise HTTPException(status_code=503, detail="No models available")

    try:
        if not request.inputs.startswith("data:image"):
            raise HTTPException(status_code=400, detail="Invalid image format")

        logger.info("Processing ensemble analysis...")

        # Collect predictions from all available models
        all_predictions: Dict[str, List[Dict[str, Any]]] = {}
        model_weights: Dict[str, float] = {}

        # Run models in parallel
        async def run_model(key: str, model: FoodRecognitionModel) -> tuple:
            try:
                preds = model.predict(request.inputs)
                return key, preds
            except Exception as e:
                logger.warning(f"Model {key} failed: {e}")
                return key, []

        tasks = []
        for key, model in models.items():
            if model is not None:
                tasks.append(run_model(key, model))
                model_weights[key] = MODEL_CONFIGS[key]["weight"]

        # Wait for all models
        results = await asyncio.gather(*[asyncio.to_thread(lambda k=k, m=m: (k, m.predict(request.inputs))) for k, m in models.items() if m is not None])

        for key, preds in results:
            if preds:
                all_predictions[key] = preds

        logger.info(f"Got predictions from {len(all_predictions)} models")

        # Merge predictions
        ensemble_predictions = merge_predictions(all_predictions, model_weights)

        # Also return primary predictions for backward compatibility
        primary_preds = all_predictions.get("primary", [])

        return AnalysisResponse(
            predictions=[
                Classification(label=p["label"], score=p["score"], model="primary")
                for p in primary_preds
            ],
            ensemble_predictions=ensemble_predictions,
            model_status={k: v is not None for k, v in models.items()},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ensemble analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-with-portions")
async def analyze_with_portions(request: ImageRequest):
    """
    Analyze food image and estimate portion sizes

    Uses image analysis to estimate relative portion sizes
    """
    # First get food predictions
    predictions = await analyze_image(request)

    # Add portion size estimates based on typical serving sizes
    TYPICAL_PORTIONS = {
        "protein": {"small": 75, "medium": 150, "large": 225},
        "grains": {"small": 75, "medium": 150, "large": 225},
        "vegetables": {"small": 50, "medium": 100, "large": 150},
        "fruits": {"small": 80, "medium": 150, "large": 200},
        "dairy": {"small": 30, "medium": 60, "large": 100},
        "default": {"small": 50, "medium": 100, "large": 150},
    }

    def get_category(label: str) -> str:
        label_lower = label.lower()
        if any(p in label_lower for p in ["chicken", "beef", "pork", "fish", "egg", "meat"]):
            return "protein"
        if any(g in label_lower for g in ["rice", "bread", "pasta", "noodle"]):
            return "grains"
        if any(v in label_lower for v in ["salad", "vegetable", "broccoli", "carrot"]):
            return "vegetables"
        if any(f in label_lower for f in ["apple", "banana", "fruit", "berry"]):
            return "fruits"
        if any(d in label_lower for d in ["cheese", "milk", "yogurt"]):
            return "dairy"
        return "default"

    enhanced_predictions = []
    for pred in predictions:
        category = get_category(pred.label)
        portions = TYPICAL_PORTIONS.get(category, TYPICAL_PORTIONS["default"])

        enhanced_predictions.append({
            "label": pred.label,
            "score": pred.score,
            "model": pred.model,
            "category": category,
            "estimated_portions": {
                "small": {"grams": portions["small"], "multiplier": 0.5},
                "medium": {"grams": portions["medium"], "multiplier": 1.0},
                "large": {"grams": portions["large"], "multiplier": 1.5},
            },
            "default_portion": "medium",
        })

    return {
        "predictions": enhanced_predictions,
        "model_status": {k: v is not None for k, v in models.items()},
    }

# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    # Disable reload in production (when PRODUCTION env var is set)
    is_production = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")

    logger.info(f"Starting enhanced server on {host}:{port} (production={is_production})")

    uvicorn.run(
        "multi_model_server:app",
        host=host,
        port=port,
        reload=not is_production,
        log_level="info"
    )
