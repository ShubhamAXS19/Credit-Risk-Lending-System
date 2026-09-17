import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any
from loguru import logger

from src.basel.pd_model import map_pd_to_grade, platt_scale
from src.basel.lgd_model import get_lgd
from src.basel.ead_model import calculate_ead, get_ccf

def calculate_expected_loss(pd_val: float, lgd_val: float, ead_val: float) -> float:
    """Calculate Expected Loss (EL) per loan: EL = PD * LGD * EAD."""
    return float(pd_val * lgd_val * ead_val)

def calculate_rwa(pd_val: float, lgd_val: float, ead_val: float, maturity: float = 2.5) -> float:
    """
    Calculate Risk-Weighted Assets (RWA) using the Basel II IRB Formula for corporate/retail exposures.
    """
    pd_bounded = float(np.clip(pd_val, 0.0001, 0.9999))
    lgd_bounded = float(np.clip(lgd_val, 0.01, 0.99))
    M = float(maturity)
    
    # Asset Correlation (R) for retail exposures
    R = 0.12 * (1 - np.exp(-50 * pd_bounded)) / (1 - np.exp(-50)) + 0.24 * (1 - (1 - np.exp(-50 * pd_bounded)) / (1 - np.exp(-50)))
    
    # Maturity Adjustment (b)
    b = (0.11852 - 0.05478 * np.log(pd_bounded)) ** 2
    
    # Normal inverse CDF G(p) and normal CDF N(x)
    G_pd = norm.ppf(pd_bounded)
    G_999 = norm.ppf(0.999)
    
    term1 = (1 - R) ** -0.5 * G_pd
    term2 = (R / (1 - R)) ** 0.5 * G_999
    N_arg = term1 + term2
    
    # Capital Requirement K
    K_raw = lgd_bounded * norm.cdf(N_arg) - pd_bounded * lgd_bounded
    K = max(K_raw, 0.0) * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    
    # Risk-Weighted Assets (RWA) = K * 12.5 * EAD
    rwa = K * 12.5 * ead_val
    return float(round(rwa, 2))

class PortfolioRiskAggregator:
    """Aggregates portfolio-level Expected Loss (EL), RWA, and Capital Adequacy Ratio (CAR)."""
    
    def __init__(self, available_capital: float = 5000000.0):
        self.available_capital = available_capital
        
    def evaluate_portfolio(self, loans_df: pd.DataFrame) -> Dict[str, Any]:
        df = loans_df.copy()
        
        # Calculate PD, LGD, EAD, EL, RWA per loan
        if 'pd' not in df.columns:
            if 'SeriousDlqin2yrs' in df.columns:
                raw_prob = df['SeriousDlqin2yrs'].astype(float).values
            else:
                raw_prob = np.full(len(df), 0.08)
            df['pd'] = [platt_scale(p) for p in raw_prob]
            
        df['risk_grade'] = df['pd'].apply(lambda p: map_pd_to_grade(p).value)
        
        if 'lgd' not in df.columns:
            df['lgd'] = df.apply(lambda r: get_lgd(r.get('applicant_id', '1'), r.get('collateral_type', 'UNSECURED'), r.get('ltv_ratio', 1.0)), axis=1)
            
        if 'ead' not in df.columns:
            drawn = df.get('drawn_amount', df.get('loan_amount', pd.Series(10000, index=df.index))).values
            undrawn = df.get('undrawn_amount', pd.Series(0, index=df.index)).values
            ltypes = df.get('loan_type', pd.Series('TERM_LOAN', index=df.index)).values
            df['ead'] = [calculate_ead(drawn[i], undrawn[i], ltypes[i]) for i in range(len(df))]
            
        df['expected_loss'] = df['pd'] * df['lgd'] * df['ead']
        df['rwa'] = [calculate_rwa(df['pd'].iloc[i], df['lgd'].iloc[i], df['ead'].iloc[i]) for i in range(len(df))]
        
        total_exposure = float(df['ead'].sum())
        total_el = float(df['expected_loss'].sum())
        total_rwa = float(df['rwa'].sum())
        
        # Basel III Minimum Capital (8% Tier 1, 10.5% Buffer)
        min_capital_req_8pct = total_rwa * 0.08
        min_capital_req_10_5pct = total_rwa * 0.105
        
        car_ratio = (self.available_capital / total_rwa) if total_rwa > 0 else 1.0
        car_breach = car_ratio < 0.08
        
        # Breakdowns
        el_by_grade = df.groupby('risk_grade')['expected_loss'].agg(['sum', 'count']).to_dict(orient='index')
        el_by_loan_type = df.groupby('loan_type')['expected_loss'].sum().to_dict()
        el_by_collateral = df.groupby('collateral_type')['expected_loss'].sum().to_dict()
        
        return {
            "total_loans": len(df),
            "total_exposure_ead": round(total_exposure, 2),
            "total_expected_loss": round(total_el, 2),
            "portfolio_el_rate": round((total_el / total_exposure * 100) if total_exposure > 0 else 0, 2),
            "total_rwa": round(total_rwa, 2),
            "available_capital": round(self.available_capital, 2),
            "car_ratio": round(car_ratio, 4),
            "car_breach_flag": car_breach,
            "min_capital_8pct": round(min_capital_req_8pct, 2),
            "min_capital_10_5pct": round(min_capital_req_10_5pct, 2),
            "el_by_grade": el_by_grade,
            "el_by_loan_type": el_by_loan_type,
            "el_by_collateral": el_by_collateral,
            "processed_df": df
        }
