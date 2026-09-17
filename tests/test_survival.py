import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.survival.cox_model import get_lifetime_pd
from src.survival.data_prep import prepare_survival_data
from src.dataset import generate_credit_dataset

client = TestClient(app)

@pytest.fixture
def sample_features():
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.35,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.40,
        "MonthlyIncome": 6000.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }

def test_prepare_survival_data():
    raw_df = generate_credit_dataset(n_samples=100, seed=42)
    surv_df = prepare_survival_data(raw_df)
    assert "duration" in surv_df.columns
    assert "event" in surv_df.columns
    assert (surv_df["duration"] > 0).all()

def test_cox_lifetime_pd(sample_features):
    res = get_lifetime_pd(sample_features, horizon_months=36)
    assert res["c_index"] > 0.50
    assert res["pd_12m"] <= res["pd_24m"] <= res["pd_36m"] # Monotonic cumulative PD
    assert "Stage" in res["ifrs9_stage"]
    assert len(res["survival_curve"]) == 36

def test_lifetime_pd_endpoint(sample_features):
    response = client.post(
        "/api/v1/lifetime-pd",
        json={
            "features": sample_features,
            "horizon_months": 36
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "concordance_index" in data
    assert data["pd_12m"] <= data["pd_36m"]
    assert len(data["survival_curve"]) == 36
