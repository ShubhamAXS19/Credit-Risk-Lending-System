from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
from pathlib import Path
from loguru import logger
from src.app.api.v1.router import api_router
from src.config import PROJ_ROOT

# Define artifact paths (Update these if structure is different)
MODEL_PATH = PROJ_ROOT / "models" / "Project 1: Credit Default Prediction" / "tuned_model.pkl"
CALIBRATOR_PATH = PROJ_ROOT / "models" / "Project 1: Credit Default Prediction" / "calibrator.pkl"
FEATURES_PATH = PROJ_ROOT / "models" / "Project 1: Credit Default Prediction" / "features.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load artifacts on startup
    logger.info("Loading model artifacts...")
    try:
        app.state.model = joblib.load(MODEL_PATH)
        app.state.calibrator = joblib.load(CALIBRATOR_PATH)
        app.state.feature_names = joblib.load(FEATURES_PATH)
        logger.success("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        app.state.model = None
        app.state.calibrator = None
        app.state.feature_names = None
    
    yield
    
    # Clean up on shutdown
    app.state.model = None

app = FastAPI(title="Credit Risk & Lending System API", lifespan=lifespan)

# CORS middleware to allow cross-origin requests (useful if frontend runs on different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    status = "healthy" if app.state.model is not None else "degraded (model not loaded)"
    return {"status": status, "app_name": "Credit Risk & Lending System"}
