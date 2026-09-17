import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import typer
from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def generate_credit_dataset(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic 10k+ credit lending dataset with bureau, collateral, and macro features."""
    np.random.seed(seed)
    
    # Applicant IDs
    applicant_ids = [f"LOAN-{10000 + i}" for i in range(n_samples)]
    
    # Demographics & Bureau features
    age = np.random.randint(21, 75, size=n_samples)
    monthly_income = np.random.lognormal(mean=8.5, sigma=0.6, size=n_samples).clip(1500, 45000)
    num_dependents = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    revolving_util = np.random.beta(a=0.5, b=1.5, size=n_samples).clip(0.0, 1.5)
    debt_ratio = np.random.beta(a=1.0, b=3.0, size=n_samples).clip(0.05, 3.0)
    
    times_30_59_late = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.75, 0.12, 0.06, 0.04, 0.02, 0.01])
    times_60_89_late = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.85, 0.09, 0.04, 0.02])
    times_90_late = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.88, 0.07, 0.03, 0.02])
    
    open_lines = np.random.poisson(lam=7, size=n_samples).clip(1, 30)
    real_estate_lines = np.random.poisson(lam=1, size=n_samples).clip(0, 8)
    
    # Loan & Collateral specifications (Basel II/III)
    loan_types = np.random.choice(['TERM_LOAN', 'REVOLVING_CREDIT', 'CREDIT_CARD', 'OVERDRAFT'], size=n_samples, p=[0.45, 0.30, 0.15, 0.10])
    collateral_types = np.random.choice(['UNSECURED', 'RESIDENTIAL_PROPERTY', 'VEHICLE', 'FIXED_DEPOSIT', 'GOLD'], size=n_samples, p=[0.40, 0.25, 0.20, 0.10, 0.05])
    
    loan_amount = np.random.uniform(5000, 100000, size=n_samples)
    drawn_amount = loan_amount * np.random.uniform(0.5, 1.0, size=n_samples)
    undrawn_amount = loan_amount - drawn_amount
    
    # Collateral valuations based on collateral type
    collateral_value = np.zeros(n_samples)
    for i in range(n_samples):
        ctype = collateral_types[i]
        lamt = loan_amount[i]
        if ctype == 'RESIDENTIAL_PROPERTY':
            collateral_value[i] = lamt * np.random.uniform(1.2, 2.5)
        elif ctype == 'VEHICLE':
            collateral_value[i] = lamt * np.random.uniform(1.0, 1.5)
        elif ctype in ['FIXED_DEPOSIT', 'GOLD']:
            collateral_value[i] = lamt * np.random.uniform(1.05, 1.3)
        else:
            collateral_value[i] = 0.0
            
    term_months = np.random.choice([12, 24, 36, 48, 60], size=n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    interest_rate = np.random.uniform(0.06, 0.24, size=n_samples)
    vintage_year = np.random.choice([2021, 2022, 2023, 2024, 2025], size=n_samples)
    
    # Macroeconomic context
    unemployment_rate = np.random.normal(loc=0.055, scale=0.015, size=n_samples).clip(0.03, 0.12)
    interest_rate_macro = np.random.normal(loc=0.045, scale=0.01, size=n_samples).clip(0.02, 0.08)
    gdp_growth = np.random.normal(loc=0.025, scale=0.01, size=n_samples).clip(-0.02, 0.06)
    
    # Synthetic default probability calculation (Ground Truth Logit)
    logit = (
        -2.2
        + 1.8 * revolving_util
        + 1.1 * times_90_late
        + 0.6 * times_30_59_late
        + 0.5 * debt_ratio
        - 0.02 * (age - 40)
        - 0.00008 * (monthly_income - 5000)
        + 5.0 * (unemployment_rate - 0.05)
        + np.random.normal(loc=0, scale=0.5, size=n_samples)
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default_label = (np.random.rand(n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'applicant_id': applicant_ids,
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': times_30_59_late,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLinesAndLoans': open_lines,
        'NumberOfTimes90DaysLate': times_90_late,
        'NumberRealEstateLoansOrLines': real_estate_lines,
        'NumberOfTime60-89DaysPastDueNotWorse': times_60_89_late,
        'NumberOfDependents': num_dependents,
        'loan_amount': loan_amount,
        'drawn_amount': drawn_amount,
        'undrawn_amount': undrawn_amount,
        'loan_type': loan_types,
        'collateral_type': collateral_types,
        'collateral_value': collateral_value,
        'term_months': term_months,
        'interest_rate': interest_rate,
        'vintage_year': vintage_year,
        'unemployment_rate': unemployment_rate,
        'interest_rate_macro': interest_rate_macro,
        'gdp_growth': gdp_growth,
        'SeriousDlqin2yrs': default_label
    })
    
    return df

@app.command()
def main(
    output_path: Path = RAW_DATA_DIR / "credit_portfolio.csv",
    n_samples: int = 10000
):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating synthetic credit portfolio with {n_samples} samples...")
    df = generate_credit_dataset(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    logger.success(f"Saved dataset to {output_path} (Default rate: {df['SeriousDlqin2yrs'].mean():.2%})")

if __name__ == "__main__":
    app()
