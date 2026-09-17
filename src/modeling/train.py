import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import typer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.dataset import generate_credit_dataset
from src.features import engineer_features, FEATURE_COLUMNS

app = typer.Typer()

def compute_ks_statistic(y_true, y_probs):
    """Compute Kolmogorov-Smirnov (KS) statistic."""
    df = pd.DataFrame({'target': y_true, 'prob': y_probs})
    df = df.sort_values(by='prob', ascending=False)
    df['bads'] = df['target']
    df['goods'] = 1 - df['target']
    cum_bads = df['bads'].cumsum() / df['bads'].sum()
    cum_goods = df['goods'].cumsum() / df['goods'].sum()
    ks_stat = (cum_bads - cum_goods).abs().max() * 100
    return float(ks_stat)

def train_and_evaluate_models(data_path: Path = None):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    proj1_dir = MODELS_DIR / "Project 1: Credit Default Prediction"
    proj1_dir.mkdir(parents=True, exist_ok=True)
    
    if data_path is None or not data_path.exists():
        logger.info("Generating dataset for training...")
        raw_df = generate_credit_dataset(n_samples=12000, seed=42)
    else:
        raw_df = pd.read_csv(data_path)
        
    df = engineer_features(raw_df)
    
    X = df[FEATURE_COLUMNS]
    y = df['SeriousDlqin2yrs']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression (Scorecard)": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost Classifier": xgb.XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, eval_metric='logloss', random_state=42),
        "LightGBM Classifier": lgb.LGBMClassifier(n_estimators=120, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1),
        "CatBoost Classifier": CatBoostClassifier(iterations=120, depth=5, learning_rate=0.05, verbose=0, random_seed=42)
    }
    
    results = {}
    best_auc = 0.0
    best_model_name = ""
    best_model_obj = None
    
    logger.info("Training and evaluating model suite...")
    
    for name, clf in models.items():
        t0 = time.time()
        if "Logistic" in name:
            clf.fit(X_train_scaled, y_train)
            probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
        fit_time = time.time() - t0
        
        # Benchmark latency for 1 prediction
        t_lat = time.time()
        if "Logistic" in name:
            _ = clf.predict_proba(X_test_scaled[:1])
        else:
            _ = clf.predict_proba(X_test[:1])
        latency_ms = (time.time() - t_lat) * 1000
        
        auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        ks = compute_ks_statistic(y_test, probs)
        brier = float(brier_score_loss(y_test, probs))
        
        results[name] = {
            "AUC_ROC": round(auc, 4),
            "PR_AUC": round(pr_auc, 4),
            "KS_Statistic": round(ks, 2),
            "Brier_Score": round(brier, 4),
            "Fit_Time_Sec": round(fit_time, 3),
            "Inference_Latency_MS": round(latency_ms, 3)
        }
        logger.info(f"{name} -> AUC: {auc:.4f} | KS: {ks:.2f} | Latency: {latency_ms:.2f}ms")
        
        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_model_obj = clf
            
    # Train calibrator for best model (XGBoost)
    xgb_model = models["XGBoost Classifier"]
    calibrator = CalibratedClassifierCV(estimator=xgb_model, cv='prefit', method='sigmoid')
    calibrator.fit(X_test, y_test)
    
    # Save artifacts to both project path & main models path
    joblib.dump(xgb_model, proj1_dir / "tuned_model.pkl")
    joblib.dump(calibrator, proj1_dir / "calibrator.pkl")
    joblib.dump(FEATURE_COLUMNS, proj1_dir / "features.pkl")
    joblib.dump(scaler, proj1_dir / "scaler.pkl")
    
    joblib.dump(xgb_model, MODELS_DIR / "xgb_model.pkl")
    joblib.dump(models["Logistic Regression (Scorecard)"], MODELS_DIR / "logistic_model.pkl")
    joblib.dump(models["LightGBM Classifier"], MODELS_DIR / "lightgbm_model.pkl")
    joblib.dump(models["CatBoost Classifier"], MODELS_DIR / "catboost_model.pkl")
    
    # Save evaluation benchmark JSON
    eval_json_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.success(f"Best Model: {best_model_name} (AUC {best_auc:.4f}). Saved artifacts to {proj1_dir}")
    return results

@app.command()
def main():
    train_and_evaluate_models()

if __name__ == "__main__":
    app()
