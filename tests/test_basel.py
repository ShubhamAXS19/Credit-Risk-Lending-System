import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.basel.pd_model import map_pd_to_grade, PDBand, platt_scale
from src.basel.lgd_model import get_lgd, CollateralType
from src.basel.ead_model import calculate_ead, LoanType
from src.basel.expected_loss import calculate_expected_loss, calculate_rwa, PortfolioRiskAggregator
from src.dataset import generate_credit_dataset

client = TestClient(app)

def test_pd_grade_boundaries():
    assert map_pd_to_grade(0.0005) == PDBand.AAA
    assert map_pd_to_grade(0.003) == PDBand.AA
    assert map_pd_to_grade(0.015) == PDBand.A
    assert map_pd_to_grade(0.035) == PDBand.BBB
    assert map_pd_to_grade(0.075) == PDBand.BB
    assert map_pd_to_grade(0.15) == PDBand.B
    assert map_pd_to_grade(0.25) == PDBand.CCC

def test_lgd_lookup():
    assert get_lgd(collateral_type="UNSECURED") == 0.75
    assert get_lgd(collateral_type="RESIDENTIAL_PROPERTY") == 0.35
    assert get_lgd(collateral_type="VEHICLE") == 0.40
    assert get_lgd(collateral_type="FIXED_DEPOSIT") == 0.10
    assert get_lgd(collateral_type="GOLD") == 0.25
    assert get_lgd(collateral_type="INVALID_TYPE") == 0.75 # Fallback

def test_ead_calculation():
    # Term loan: CCF = 1.0
    assert calculate_ead(10000, 5000, "TERM_LOAN") == 15000.0
    # Revolving loan < 12m: CCF = 0.20
    assert calculate_ead(10000, 5000, "REVOLVING_CREDIT", horizon_months=6) == 11000.0
    # Revolving loan > 12m: CCF = 0.50
    assert calculate_ead(10000, 5000, "REVOLVING_CREDIT", horizon_months=24) == 12500.0
    # Credit Card: CCF = 0.75
    assert calculate_ead(10000, 4000, "CREDIT_CARD") == 13000.0

def test_expected_loss_and_rwa():
    el = calculate_expected_loss(pd_val=0.05, lgd_val=0.45, ead_val=20000)
    assert el == 450.0
    
    rwa = calculate_rwa(pd_val=0.05, lgd_val=0.45, ead_val=20000)
    assert rwa > 0

def test_portfolio_aggregator():
    df = generate_credit_dataset(n_samples=500, seed=123)
    aggregator = PortfolioRiskAggregator(available_capital=1000000.0)
    res = aggregator.evaluate_portfolio(df)
    
    assert res["total_loans"] == 500
    assert res["total_expected_loss"] > 0
    assert res["total_rwa"] > 0
    assert "car_ratio" in res

def test_regulatory_endpoint():
    response = client.post(
        "/api/v1/regulatory",
        json={
            "applicant_id": "LOAN-TEST-REG",
            "raw_default_prob": 0.08,
            "drawn_amount": 20000.0,
            "undrawn_amount": 5000.0,
            "loan_type": "CREDIT_CARD",
            "collateral_type": "UNSECURED"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["applicant_id"] == "LOAN-TEST-REG"
    assert data["expected_loss"] > 0
    assert data["rwa"] > 0
    assert data["tier1_capital_required_8pct"] > 0
