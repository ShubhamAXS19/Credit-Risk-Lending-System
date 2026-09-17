from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from src.dataset import generate_credit_dataset
from src.stress_testing.macro_model import StressTestEngine, STRESS_SCENARIOS

router = APIRouter()

class StressTestRequest(BaseModel):
    scenario_name: str = Field("severe_stress", example="severe_stress")
    unemployment_delta: Optional[float] = Field(0.05, ge=0.0, le=0.25)
    rate_hike_bps: Optional[float] = Field(250.0, ge=0.0, le=1000.0)
    gdp_growth: Optional[float] = Field(-0.02, ge=-0.15, le=0.15)
    available_capital: float = Field(5000000.0, ge=10000.0)

class ScenarioMetrics(BaseModel):
    total_el: float
    portfolio_el_rate: float
    total_rwa: float
    car_ratio: float
    car_breach: bool

class StressedMetrics(ScenarioMetrics):
    el_delta: float
    el_pct_increase: float
    capital_consumed_pct: float

class StressTestResponse(BaseModel):
    scenario_key: str
    scenario_name: str
    macro_inputs: Dict[str, float]
    baseline_metrics: ScenarioMetrics
    stressed_metrics: StressedMetrics

@router.post(
    "/stress-test",
    response_model=StressTestResponse,
    summary="Macroeconomic Portfolio Stress Testing (RBI/CCAR Scenarios)",
    description="Simulate portfolio Expected Loss and Risk-Weighted Asset shifts under macroeconomic shocks (Unemployment spikes, Rate hikes, GDP contraction)."
)
async def run_portfolio_stress_test(payload: StressTestRequest):
    try:
        engine = StressTestEngine(available_capital=payload.available_capital)
        
        # Load test portfolio sample
        df_sample = generate_credit_dataset(n_samples=1000, seed=42)
        
        if payload.scenario_name in STRESS_SCENARIOS:
            scen_input = payload.scenario_name
        else:
            scen_input = {
                "name": "Custom Macro Scenario",
                "unemployment_delta": payload.unemployment_delta,
                "rate_hike_bps": payload.rate_hike_bps,
                "gdp_growth": payload.gdp_growth
            }
            
        res = engine.run_stress_test(df_sample, scenario=scen_input)
        
        return StressTestResponse(
            scenario_key=res["scenario_key"],
            scenario_name=res["scenario_name"],
            macro_inputs=res["macro_inputs"],
            baseline_metrics=ScenarioMetrics(**res["baseline_metrics"]),
            stressed_metrics=StressedMetrics(**res["stressed_metrics"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
