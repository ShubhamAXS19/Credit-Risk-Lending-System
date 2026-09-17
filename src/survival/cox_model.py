import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter
from loguru import logger

from src.config import PROJ_ROOT, MODELS_DIR
from src.dataset import generate_credit_dataset
from src.features import engineer_features
from src.survival.data_prep import prepare_survival_data

MODEL_PATH = MODELS_DIR / "cox_ph_model.pkl"

class CreditCoxModel:
    """Cox Proportional Hazards Model for IFRS 9 Lifetime PD & Staging."""
    
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.cph = CoxPHFitter(penalizer=0.1)
        self.is_fitted = False
        self._load_or_train()
        
    def _load_or_train(self):
        if self.model_path.exists():
            try:
                self.cph = joblib.load(self.model_path)
                self.is_fitted = True
                return
            except Exception as e:
                logger.warning(f"Could not load Cox model ({e}); retraining...")
                
        self.train_cox_model()
        
    def train_cox_model(self, n_samples: int = 3000):
        logger.info("Training Cox Proportional Hazards Model...")
        raw_df = generate_credit_dataset(n_samples=n_samples, seed=42)
        feat_df = engineer_features(raw_df)
        surv_df = prepare_survival_data(feat_df)
        
        cox_cols = [
            'duration', 'event',
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'DebtRatio',
            'total_delinquency',
            'credit_stress_index'
        ]
        
        df_cox = surv_df[cox_cols].dropna().copy()
        
        # Standardize features
        for c in cox_cols[2:]:
            df_cox[c] = (df_cox[c] - df_cox[c].mean()) / (df_cox[c].std() + 1e-6)
            
        self.cph.fit(df_cox, duration_col='duration', event_col='event')
        self.is_fitted = True
        
        c_index = float(self.cph.concordance_index_)
        logger.success(f"Cox PH Model trained (C-index: {c_index:.4f})")
        
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.cph, self.model_path)
        
    def predict_lifetime_pd(self, input_features: dict, horizon_months: int = 36) -> dict:
        raw_df = pd.DataFrame([input_features])
        feat_df = engineer_features(raw_df)
        
        cox_cols = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'DebtRatio',
            'total_delinquency',
            'credit_stress_index'
        ]
        
        X_in = feat_df[cox_cols].copy()
        # Scale input
        for c in cox_cols:
            if c not in X_in.columns:
                X_in[c] = 0.0
            X_in[c] = (X_in[c] - X_in[c].mean()) / (X_in[c].std() + 1e-6)
            
        # Predict survival function S(t)
        surv_func = self.cph.predict_survival_function(X_in)
        months = list(range(1, max(horizon_months + 1, 37)))
        
        surv_probs = []
        for m in months:
            if m in surv_func.index:
                prob_s = float(surv_func.loc[m].iloc[0])
            else:
                closest_idx = surv_func.index[np.abs(surv_func.index - m).argmin()]
                prob_s = float(surv_func.loc[closest_idx].iloc[0])
            surv_probs.append((m, round(prob_s, 4)))
            
        # Lifetime Cumulative PD = 1 - S(t)
        pd_12m = round(1.0 - dict(surv_probs).get(12, 0.95), 4)
        pd_24m = round(1.0 - dict(surv_probs).get(24, 0.90), 4)
        pd_36m = round(1.0 - dict(surv_probs).get(36, 0.85), 4)
        
        # IFRS 9 Staging Logic
        if pd_12m < 0.05:
            stage = "Stage 1 (Performing)"
        elif pd_12m < 0.20:
            stage = "Stage 2 (Significant Increase in Credit Risk - SICR)"
        else:
            stage = "Stage 3 (Credit Impaired)"
            
        return {
            "c_index": round(float(self.cph.concordance_index_), 4),
            "pd_12m": pd_12m,
            "pd_24m": pd_24m,
            "pd_36m": pd_36m,
            "ifrs9_stage": stage,
            "survival_curve": surv_probs[:horizon_months]
        }

_cox_model_instance = None

def get_cox_model() -> CreditCoxModel:
    global _cox_model_instance
    if _cox_model_instance is None:
        _cox_model_instance = CreditCoxModel()
    return _cox_model_instance

def get_lifetime_pd(applicant_features: dict, horizon_months: int = 36) -> dict:
    model = get_cox_model()
    return model.predict_lifetime_pd(applicant_features, horizon_months=horizon_months)
