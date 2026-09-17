import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import typer
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

FEATURE_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberOfTimes90DaysLate',
    'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfDependents',
    'total_delinquency',
    'income_per_person',
    'credit_stress_index',
    'ltv_ratio'
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw credit portfolio dataset into engineered feature set."""
    df_out = df.copy()
    
    # Feature 1: Total delinquency count
    df_out['total_delinquency'] = (
        df_out['NumberOfTime30-59DaysPastDueNotWorse'].fillna(0) +
        df_out['NumberOfTime60-89DaysPastDueNotWorse'].fillna(0) +
        df_out['NumberOfTimes90DaysLate'].fillna(0)
    )
    
    # Feature 2: Income per person
    dependents = df_out['NumberOfDependents'].fillna(0) + 1.0
    income = df_out['MonthlyIncome'].fillna(df_out['MonthlyIncome'].median())
    df_out['income_per_person'] = income / dependents
    
    # Feature 3: Credit Stress Index (Utilization x Debt Ratio)
    util = df_out['RevolvingUtilizationOfUnsecuredLines'].fillna(0)
    dtr = df_out['DebtRatio'].fillna(0)
    df_out['credit_stress_index'] = util * dtr
    
    # Feature 4: LTV (Loan-To-Value) Ratio
    c_val = df_out.get('collateral_value', pd.Series(0, index=df_out.index)).fillna(0)
    l_amt = df_out.get('loan_amount', pd.Series(10000, index=df_out.index)).fillna(10000)
    df_out['ltv_ratio'] = np.where(c_val > 0, l_amt / c_val, 1.0)
    
    return df_out

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "credit_portfolio.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv"
):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        from src.dataset import generate_credit_dataset
        logger.info("Raw dataset missing; generating new dataset...")
        raw_df = generate_credit_dataset()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(input_path, index=False)
    else:
        raw_df = pd.read_csv(input_path)
        
    logger.info("Engineering features...")
    processed_df = engineer_features(raw_df)
    processed_df.to_csv(output_path, index=False)
    logger.success(f"Saved engineered features to {output_path}")

if __name__ == "__main__":
    app()
