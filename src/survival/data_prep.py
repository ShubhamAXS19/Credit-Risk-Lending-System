import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from loguru import logger

from src.config import PROJ_ROOT
from src.dataset import generate_credit_dataset

OUTPUT_DIR = PROJ_ROOT / "outputs" / "survival"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare survival analysis dataset:
    - duration: months from loan origination to default or censoring (1 to 60)
    - event: 1 if defaulted (SeriousDlqin2yrs == 1), 0 if active/paid off (censored)
    """
    df_surv = df.copy()
    
    if 'event' not in df_surv.columns:
        df_surv['event'] = df_surv['SeriousDlqin2yrs'].astype(int)
        
    if 'duration' not in df_surv.columns:
        # Generate realistic duration based on event status & term
        terms = df_surv.get('term_months', pd.Series(36, index=df_surv.index)).values
        np.random.seed(42)
        durations = np.zeros(len(df_surv))
        for i in range(len(df_surv)):
            if df_surv['event'].iloc[i] == 1:
                # Defaulted loans fail earlier (e.g. month 3 to 24)
                durations[i] = int(np.random.triangular(left=3, mode=12, right=terms[i]))
            else:
                # Active/Paid loans survive full term or up to current month
                durations[i] = int(terms[i])
        df_surv['duration'] = np.clip(durations, 1, 60)
        
    return df_surv

def plot_kaplan_meier_curves(df: pd.DataFrame, save_path: Path = None) -> str:
    """Fit Kaplan-Meier survival curves stratified by risk grade and perform log-rank test."""
    if save_path is None:
        save_path = OUTPUT_DIR / "kaplan_meier_survival_curves.png"
        
    df_surv = prepare_survival_data(df)
    
    if 'risk_grade' not in df_surv.columns:
        from src.basel.pd_model import map_pd_to_grade, platt_scale
        df_surv['pd'] = [platt_scale(p) for p in df_surv['SeriousDlqin2yrs'].astype(float).values]
        df_surv['risk_grade'] = df_surv['pd'].apply(lambda p: map_pd_to_grade(p).value)
        
    plt.figure(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    
    grades = df_surv['risk_grade'].unique()
    for g in sorted(grades):
        mask = (df_surv['risk_grade'] == g)
        if mask.sum() > 5:
            kmf.fit(df_surv.loc[mask, 'duration'], df_surv.loc[mask, 'event'], label=f"Grade {g}")
            kmf.plot_survival_function(ci_show=False)
            
    plt.title("Kaplan-Meier Loan Survival Curves Stratified by Basel Risk Grade", fontsize=12)
    plt.xlabel("Timeline (Months from Loan Origination)")
    plt.ylabel("Survival Probability (No Default)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Multivariate Log-Rank Test
    logrank_res = multivariate_logrank_test(
        event_durations=df_surv['duration'],
        groups=df_surv['risk_grade'],
        event_observed=df_surv['event']
    )
    logger.info(f"Log-Rank Test p-value: {logrank_res.p_value:.6f}")
    
    return str(save_path)
