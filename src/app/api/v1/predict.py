
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib

router = APIRouter()

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    default_probability: float
    decision: str

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest):
    # Retrieve artifacts from app state
    model = request.app.state.model
    calibrator = request.app.state.calibrator
    feature_names = request.app.state.feature_names

    if not model or not feature_names:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Create DataFrame from input
    try:
        input_data = pd.DataFrame([payload.features])
        
        # Ensure all expected features are present, fill missing with 0 (or handle appropriately)
        # For this MVP, we assume the user provides all fields or we error out if strict.
        # Ideally, we align the columns.
        
        # Check for missing columns
        missing_cols = set(feature_names) - set(input_data.columns)
        if missing_cols:
             raise HTTPException(status_code=400, detail=f"Missing feature columns: {missing_cols}")

        # Reorder columns to match training data
        input_data = input_data[feature_names]
        
        # Predict probability
        # Note: The model pipeline might expect raw data. 
        # If the model handles preprocessing, we are good.
        # Based on the notebook, the model seems to be a pipeline or directly an estimator.
        # We will assume it's capable of handling the input provided it matches the schema.
        
        # Get raw probability from the base model
        # Using predict_proba if available, else decision function
        if hasattr(model, "predict_proba"):
            # Usually returns [prob_0, prob_1]
            prob_default = model.predict_proba(input_data)[:, 1][0]
        else:
            # Fallback if specific model type doesn't support predict_proba directly 
            # (though most classifiers do)
            prob_default = 0.0 # Placeholder/Error

        # Calibrate if calibrator exists
        if calibrator:
            # Calibrated probability
            # Calibrator expects 1D array of probabilities if it's IsotonicRegression on 1D input
            # Check how calibrator fits. In notebook: iso.fit(y_probs_val, y_val)
            # So it expects probabilities.
            prob_default = calibrator.predict([prob_default])[0]

        # Decision rule (e.g., threshold 0.10 from notebook)
        threshold = 0.10
        decision = "Reject" if prob_default > threshold else "Approve"

        return PredictionResponse(
            default_probability=float(prob_default),
            decision=decision
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
