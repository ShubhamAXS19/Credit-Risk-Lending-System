import numpy as np
import pandas as pd
from enum import Enum
from typing import Union
from sklearn.ensemble import GradientBoostingRegressor
from loguru import logger

class CollateralType(str, Enum):
    UNSECURED = "UNSECURED"
    RESIDENTIAL_PROPERTY = "RESIDENTIAL_PROPERTY"
    VEHICLE = "VEHICLE"
    FIXED_DEPOSIT = "FIXED_DEPOSIT"
    GOLD = "GOLD"

# Foundation IRB standard LGD values
FIRB_LGD_LOOKUP = {
    CollateralType.UNSECURED: 0.75,
    CollateralType.RESIDENTIAL_PROPERTY: 0.35,
    CollateralType.VEHICLE: 0.40,
    CollateralType.FIXED_DEPOSIT: 0.10,
    CollateralType.GOLD: 0.25,
}

class MLLGDModel:
    """Gradient Boosting Regression model for Machine Learning LGD estimation."""
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.is_trained = False
        self._train_synthetic_lgd_model()
        
    def _train_synthetic_lgd_model(self):
        np.random.seed(42)
        n = 2000
        ltv = np.random.uniform(0.3, 1.5, n)
        c_code = np.random.choice([0, 1, 2, 3, 4], n) # 0: unsec, 1: prop, 2: veh, 3: dep, 4: gold
        
        # Ground truth LGD with noise
        base_lgd = np.select(
            [c_code == 0, c_code == 1, c_code == 2, c_code == 3, c_code == 4],
            [0.75, 0.35, 0.40, 0.10, 0.25]
        )
        observed_lgd = (base_lgd + 0.15 * ltv + np.random.normal(0, 0.05, n)).clip(0.05, 0.95)
        
        X = pd.DataFrame({'c_code': c_code, 'ltv': ltv})
        self.model.fit(X, observed_lgd)
        self.is_trained = True
        
    def predict_lgd(self, collateral_type: str, ltv_ratio: float = 1.0) -> float:
        mapping = {"UNSECURED": 0, "RESIDENTIAL_PROPERTY": 1, "VEHICLE": 2, "FIXED_DEPOSIT": 3, "GOLD": 4}
        code = mapping.get(str(collateral_type).upper(), 0)
        X_in = pd.DataFrame({'c_code': [code], 'ltv': [ltv_ratio]})
        pred = self.model.predict(X_in)[0]
        return float(np.clip(pred, 0.05, 0.95))

_ml_lgd_instance = None

def get_ml_lgd_model() -> MLLGDModel:
    global _ml_lgd_instance
    if _ml_lgd_instance is None:
        _ml_lgd_instance = MLLGDModel()
    return _ml_lgd_instance

def get_lgd(
    loan_id: str = "LOAN-001",
    collateral_type: Union[CollateralType, str] = CollateralType.UNSECURED,
    ltv_ratio: float = 1.0,
    use_ml: bool = False
) -> float:
    """
    Retrieve LGD estimate via Foundation IRB lookup or ML model.
    Falls back to UNSECURED (0.75) if collateral type is invalid/missing.
    """
    # Normalize string collateral input
    ctype_str = str(collateral_type).upper()
    try:
        ctype_enum = CollateralType(ctype_str)
    except ValueError:
        ctype_enum = CollateralType.UNSECURED
        
    if use_ml:
        ml_model = get_ml_lgd_model()
        return ml_model.predict_lgd(ctype_enum.value, ltv_ratio=ltv_ratio)
    else:
        return FIRB_LGD_LOOKUP.get(ctype_enum, 0.75)

def compare_lgd_approaches(loans_df: pd.DataFrame) -> pd.DataFrame:
    """Compare Foundation IRB LGD vs ML LGD across a portfolio DataFrame."""
    results = loans_df.copy()
    results['firb_lgd'] = results.apply(lambda r: get_lgd(r.get('applicant_id', '1'), r.get('collateral_type', 'UNSECURED'), r.get('ltv_ratio', 1.0), use_ml=False), axis=1)
    results['ml_lgd'] = results.apply(lambda r: get_lgd(r.get('applicant_id', '1'), r.get('collateral_type', 'UNSECURED'), r.get('ltv_ratio', 1.0), use_ml=True), axis=1)
    results['lgd_diff'] = results['ml_lgd'] - results['firb_lgd']
    return results[['applicant_id', 'collateral_type', 'ltv_ratio', 'firb_lgd', 'ml_lgd', 'lgd_diff']]
