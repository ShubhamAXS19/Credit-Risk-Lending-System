import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import shap
from loguru import logger

from src.config import PROJ_ROOT, MODELS_DIR
from src.features import engineer_features, FEATURE_COLUMNS

# Artifact paths
MODEL_PATH = PROJ_ROOT / "models" / "Project 1: Credit Default Prediction" / "tuned_model.pkl"
FEATURES_PATH = PROJ_ROOT / "models" / "Project 1: Credit Default Prediction" / "features.pkl"
OUTPUT_SHAP_DIR = PROJ_ROOT / "outputs" / "shap_plots"
OUTPUT_SHAP_DIR.mkdir(parents=True, exist_ok=True)

class SHAPScorer:
    """SHAP Explainability Scorer for Credit Risk Automated Decisions."""
    
    def __init__(self, model_path: Path = MODEL_PATH, features_path: Path = FEATURES_PATH):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_names = None
        self.explainer = None
        self._load_artifacts()
        
    def _load_artifacts(self):
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            # Fallback to general models dir
            alt_path = MODELS_DIR / "xgb_model.pkl"
            if alt_path.exists():
                self.model = joblib.load(alt_path)
            else:
                logger.warning("No pre-trained model found for SHAP; training default pipeline...")
                from src.modeling.train import train_and_evaluate_models
                train_and_evaluate_models()
                self.model = joblib.load(self.model_path)
                
        if self.features_path.exists():
            self.feature_names = joblib.load(self.features_path)
        else:
            self.feature_names = FEATURE_COLUMNS
            
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"TreeExplainer failed ({e}); falling back to Explainer...")
            self.explainer = shap.Explainer(self.model)
            
    def generate_shap_scorecard(self, applicant_id: str, input_features: dict) -> dict:
        """
        Generate feature attributions and waterfall plot for a single applicant.
        Returns:
            dict containing top_positive_factors, top_negative_factors, shap_values, waterfall_plot_path
        """
        # Ensure DataFrame input
        raw_df = pd.DataFrame([input_features])
        feat_df = engineer_features(raw_df)
        
        # Ensure all columns present
        for col in self.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
                
        X_input = feat_df[self.feature_names]
        
        # Compute SHAP values
        shap_res = self.explainer(X_input)
        
        if len(shap_res.values.shape) == 3: # Multi-class output
            vals = shap_res.values[0, :, 1]
            base_val = float(shap_res.base_values[0, 1])
        elif len(shap_res.values.shape) == 2:
            vals = shap_res.values[0]
            base_val = float(shap_res.base_values[0]) if isinstance(shap_res.base_values, (np.ndarray, list)) else float(shap_res.base_values)
        else:
            vals = shap_res.values
            base_val = float(shap_res.base_values)
            
        shap_dict = {feat: float(val) for feat, val in zip(self.feature_names, vals)}
        
        # Sort factors (negative SHAP reduces default risk / pushes approval; positive SHAP increases default risk / pushes rejection)
        # Positive factors for approval = lowest SHAP values (reduces default risk)
        # Negative factors (pushing rejection) = highest positive SHAP values (increases default risk)
        sorted_factors = sorted(shap_dict.items(), key=lambda x: x[1])
        
        top_positive = [{"feature": f, "shap_value": round(v, 4)} for f, v in sorted_factors[:5]]
        top_negative = [{"feature": f, "shap_value": round(v, 4)} for f, v in sorted_factors[-5:][::-1]]
        
        # Save Waterfall plot
        plot_path = OUTPUT_SHAP_DIR / f"{applicant_id}_waterfall.png"
        try:
            plt.figure(figsize=(8, 5))
            shap.plots.waterfall(shap_res[0], show=False)
            plt.title(f"SHAP Decision Attribution — Applicant {applicant_id}", fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Could not render SHAP waterfall plot ({e}); creating fallback plot...")
            plt.figure(figsize=(8, 4))
            top_features = list(shap_dict.keys())[:8]
            top_vals = [shap_dict[k] for k in top_features]
            colors = ['crimson' if v > 0 else 'forestgreen' for v in top_vals]
            plt.barh(top_features, top_vals, color=colors)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            plt.title(f"Feature Impact on Risk (SHAP) — {applicant_id}")
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        return {
            "applicant_id": applicant_id,
            "base_value": round(base_val, 4),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "shap_values": {k: round(v, 4) for k, v in shap_dict.items()},
            "waterfall_plot_path": str(plot_path)
        }

# Global singleton helper
_shap_scorer_instance = None

def get_shap_scorer():
    global _shap_scorer_instance
    if _shap_scorer_instance is None:
        _shap_scorer_instance = SHAPScorer()
    return _shap_scorer_instance

def generate_shap_scorecard(applicant_id: str, input_features: dict) -> dict:
    scorer = get_shap_scorer()
    return scorer.generate_shap_scorecard(applicant_id, input_features)
