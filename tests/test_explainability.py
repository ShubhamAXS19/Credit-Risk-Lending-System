import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.explainability.shap_scorer import generate_shap_scorecard

client = TestClient(app)

@pytest.fixture
def sample_applicant_features():
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.25,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.35,
        "MonthlyIncome": 7500.0,
        "NumberOfOpenCreditLinesAndLoans": 8,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1
    }

def test_generate_shap_scorecard(sample_applicant_features):
    res = generate_shap_scorecard("LOAN-TEST-01", sample_applicant_features)
    assert res["applicant_id"] == "LOAN-TEST-01"
    assert "top_positive_factors" in res
    assert "top_negative_factors" in res
    assert len(res["top_positive_factors"]) == 5
    assert len(res["top_negative_factors"]) == 5
    assert "waterfall_plot_path" in res

def test_explain_endpoint(sample_applicant_features):
    response = client.post(
        "/api/v1/explain/LOAN-TEST-01",
        json={"features": sample_applicant_features}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["applicant_id"] == "LOAN-TEST-01"
    assert "default_probability" in data
    assert data["decision"] in ["Approve", "Reject"]
    assert len(data["top_positive_factors"]) == 5
    assert len(data["top_negative_factors"]) == 5
