import numpy as np
import pandas as pd
from enum import Enum
from typing import Union

class LoanType(str, Enum):
    TERM_LOAN = "TERM_LOAN"
    REVOLVING_CREDIT = "REVOLVING_CREDIT"
    CREDIT_CARD = "CREDIT_CARD"
    OVERDRAFT = "OVERDRAFT"

def get_ccf(loan_type: Union[LoanType, str], horizon_months: int = 12) -> float:
    """Retrieve Basel II Credit Conversion Factor (CCF)."""
    ltype_str = str(loan_type).upper()
    
    if ltype_str == LoanType.TERM_LOAN.value:
        return 1.0
    elif ltype_str == LoanType.REVOLVING_CREDIT.value:
        return 0.20 if horizon_months <= 12 else 0.50
    elif ltype_str == LoanType.CREDIT_CARD.value:
        return 0.75
    elif ltype_str == LoanType.OVERDRAFT.value:
        return 0.75
    else:
        return 1.0 # Default fallback

def calculate_ead(
    outstanding_balance: float,
    undrawn_commitment: float = 0.0,
    loan_type: Union[LoanType, str] = LoanType.TERM_LOAN,
    horizon_months: int = 12
) -> float:
    """Calculate Exposure at Default: EAD = Outstanding Balance + (Undrawn Commitment * CCF)."""
    ccf = get_ccf(loan_type, horizon_months=horizon_months)
    ead = float(outstanding_balance + undrawn_commitment * ccf)
    return round(ead, 2)

def calculate_ead_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized EAD calculation for 10K+ portfolio DataFrame rows."""
    drawn = df['drawn_amount'].fillna(df.get('loan_amount', 0.0)).values
    undrawn = df.get('undrawn_amount', pd.Series(0, index=df.index)).fillna(0.0).values
    loan_types = df.get('loan_type', pd.Series('TERM_LOAN', index=df.index)).astype(str).str.upper().values
    horizons = df.get('term_months', pd.Series(12, index=df.index)).fillna(12).values
    
    ccfs = np.ones(len(df))
    for i in range(len(df)):
        lt = loan_types[i]
        hz = horizons[i]
        if lt == 'TERM_LOAN':
            ccfs[i] = 1.0
        elif lt == 'REVOLVING_CREDIT':
            ccfs[i] = 0.20 if hz <= 12 else 0.50
        elif lt in ['CREDIT_CARD', 'OVERDRAFT']:
            ccfs[i] = 0.75
        else:
            ccfs[i] = 1.0
            
    ead_series = drawn + undrawn * ccfs
    return pd.Series(ead_series, index=df.index)
