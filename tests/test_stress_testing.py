import pytest
from fastapi.testclient import TestClient
from src.app.main import app
from src.stress_testing.macro_model import StressTestEngine, STRESS_SCENARIOS
from src.dataset import generate_credit_dataset

client = TestClient(app)

def test_stress_test_engine_scenarios():
    df = generate_credit_dataset(n_samples=300, seed=42)
    engine = StressTestEngine(available_capital=2000000.0)
    
    base_res = engine.run_stress_test(df, scenario="baseline")
    mod_res = engine.run_stress_test(df, scenario="moderate_stress")
    sev_res = engine.run_stress_test(df, scenario="severe_stress")
    
    # Verify monotonic increase of Stressed EL: Baseline <= Moderate <= Severe
    assert base_res["stressed_metrics"]["total_el"] <= mod_res["stressed_metrics"]["total_el"]
    assert mod_res["stressed_metrics"]["total_el"] <= sev_res["stressed_metrics"]["total_el"]
    assert sev_res["stressed_metrics"]["el_delta"] >= 0

def test_stress_endpoint():
    response = client.post(
        "/api/v1/stress-test",
        json={
            "scenario_name": "severe_stress",
            "available_capital": 3000000.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_key"] == "severe_stress"
    assert "baseline_metrics" in data
    assert "stressed_metrics" in data
    assert data["stressed_metrics"]["el_delta"] >= 0
