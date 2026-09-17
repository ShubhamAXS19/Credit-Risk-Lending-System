import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from loguru import logger

from src.config import PROJ_ROOT

OUTPUT_DIR = PROJ_ROOT / "outputs" / "calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class PDBand(str, Enum):
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    CCC = "CCC and below"

def map_pd_to_grade(pd_val: float) -> PDBand:
    """Map calibrated Probability of Default (PD) to Basel Risk Grade."""
    if pd_val < 0.001:
        return PDBand.AAA
    elif pd_val < 0.005:
        return PDBand.AA
    elif pd_val < 0.02:
        return PDBand.A
    elif pd_val < 0.05:
        return PDBand.BBB
    elif pd_val < 0.10:
        return PDBand.BB
    elif pd_val < 0.20:
        return PDBand.B
    else:
        return PDBand.CCC

def platt_scale(raw_prob: float, a: float = 1.05, b: float = -0.05) -> float:
    """
    Platt scaling transformation: calibrated_pd = 1 / (1 + exp(a * logit(raw_prob) + b))
    Ensures output is strictly bounded between [0.0001, 0.9999].
    """
    raw_prob = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    logit = np.log(raw_prob / (1 - raw_prob))
    calibrated_logit = a * logit + b
    calibrated_pd = 1 / (1 + np.exp(-calibrated_logit))
    return float(np.clip(calibrated_pd, 0.0001, 0.9999))

def generate_reliability_diagram(y_true: np.ndarray, y_probs: np.ndarray, save_path: Path = None) -> str:
    """Generate calibration reliability curve vs perfect calibration line."""
    if save_path is None:
        save_path = OUTPUT_DIR / "pd_reliability_diagram.png"
        
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(7, 5))
    plt.plot(prob_pred, prob_true, "s-", label="Calibrated Model", color="navy")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.ylabel("Fraction of Positives (Actual Default Rate)")
    plt.xlabel("Mean Predicted Probability (PD)")
    plt.title("PD Calibration Reliability Diagram (Platt Scaling)", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return str(save_path)
