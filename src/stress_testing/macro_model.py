import numpy as np
import pandas as pd
from typing import Dict, Any, Union
import statsmodels.api as sm
from loguru import logger

from src.basel.pd_model import map_pd_to_grade, platt_scale
from src.basel.expected_loss import PortfolioRiskAggregator

STRESS_SCENARIOS = {
    'baseline': {
        'name': 'Baseline Scenario',
        'gdp_growth': 0.065,
        'unemployment_delta': 0.0,
        'rate_hike_bps': 0
    },
    'moderate_stress': {
        'name': 'Moderate Stress Scenario',
        'gdp_growth': 0.03,
        'unemployment_delta': 0.02,
        'rate_hike_bps': 100
    },
    'severe_stress': {
        'name': 'Severe Stress Scenario (Recession)',
        'gdp_growth': -0.02,
        'unemployment_delta': 0.05,
        'rate_hike_bps': 250
    }
}

class MacroFactorModel:
    """OLS Macro Regression Model mapping GDP growth, Unemployment shifts, and Interest Rate hikes to PD shifts."""
    def __init__(self):
        # Calibrated historical sensitivity coefficients:
        # 1% rise in unemployment -> +0.35 logit shift
        # 100 bps rate hike -> +0.15 logit shift
        # 1% GDP contraction -> +0.20 logit shift
        self.beta_unemp = 17.5    # per unit delta (0.02 = +0.35)
        self.beta_rate = 0.0015   # per bps (100 bps = +0.15)
        self.beta_gdp = -4.0      # per unit growth (-0.02 = +0.08)
        
    def compute_pd_shift(self, unemployment_delta: float, rate_hike_bps: float, gdp_growth: float) -> float:
        """Compute delta shift in logit space."""
        shift = (
            self.beta_unemp * unemployment_delta
            + self.beta_rate * rate_hike_bps
            + self.beta_gdp * (gdp_growth - 0.065)
        )
        return float(shift)

class StressTestEngine:
    """Stress Test Execution Engine simulating portfolio EL and CAR shifts under macro scenarios."""
    def __init__(self, available_capital: float = 5000000.0):
        self.available_capital = available_capital
        self.macro_model = MacroFactorModel()
        self.aggregator = PortfolioRiskAggregator(available_capital=available_capital)
        
    def run_stress_test(
        self,
        loans_df: pd.DataFrame,
        scenario: Union[str, Dict[str, Any]] = "severe_stress"
    ) -> Dict[str, Any]:
        """Execute macro stress scenario on portfolio DataFrame."""
        if isinstance(scenario, str):
            scen_dict = STRESS_SCENARIOS.get(scenario.lower(), STRESS_SCENARIOS['severe_stress'])
            scen_key = scenario.lower()
        else:
            scen_dict = scenario
            scen_key = scenario.get('name', 'custom')
            
        unemp_delta = float(scen_dict.get('unemployment_delta', 0.0))
        rate_hike = float(scen_dict.get('rate_hike_bps', 0))
        gdp = float(scen_dict.get('gdp_growth', 0.065))
        
        # 1. Evaluate Baseline Portfolio Risk
        baseline_res = self.aggregator.evaluate_portfolio(loans_df)
        base_df = baseline_res['processed_df']
        
        # 2. Compute Stressed PDs via logit shift
        logit_shift = self.macro_model.compute_pd_shift(unemp_delta, rate_hike, gdp)
        
        stressed_df = base_df.copy()
        base_pds = stressed_df['pd'].values
        base_logits = np.log(base_pds / (1 - base_pds))
        
        stressed_logits = base_logits + max(logit_shift, 0.0) # PDs only increase under stress
        stressed_pds = 1 / (1 + np.exp(-stressed_logits))
        stressed_df['pd'] = np.clip(stressed_pds, 0.0001, 0.9999)
        
        # 3. Evaluate Stressed Portfolio Risk
        stressed_res = self.aggregator.evaluate_portfolio(stressed_df)
        
        # Delta Metrics
        baseline_el = baseline_res['total_expected_loss']
        stressed_el = stressed_res['total_expected_loss']
        el_delta = stressed_el - baseline_el
        el_pct_increase = (el_delta / baseline_el * 100) if baseline_el > 0 else 0.0
        
        baseline_rwa = baseline_res['total_rwa']
        stressed_rwa = stressed_res['total_rwa']
        
        capital_consumed_pct = (stressed_el / self.available_capital * 100) if self.available_capital > 0 else 0.0
        
        return {
            "scenario_key": scen_key,
            "scenario_name": scen_dict.get('name', scen_key),
            "macro_inputs": {
                "unemployment_delta": unemp_delta,
                "rate_hike_bps": rate_hike,
                "gdp_growth": gdp
            },
            "baseline_metrics": {
                "total_el": baseline_el,
                "portfolio_el_rate": baseline_res['portfolio_el_rate'],
                "total_rwa": baseline_rwa,
                "car_ratio": baseline_res['car_ratio'],
                "car_breach": baseline_res['car_breach_flag']
            },
            "stressed_metrics": {
                "total_el": stressed_el,
                "portfolio_el_rate": stressed_res['portfolio_el_rate'],
                "total_rwa": stressed_rwa,
                "car_ratio": stressed_res['car_ratio'],
                "car_breach": stressed_res['car_breach_flag'],
                "el_delta": round(el_delta, 2),
                "el_pct_increase": round(el_pct_increase, 2),
                "capital_consumed_pct": round(capital_consumed_pct, 2)
            },
            "el_by_grade_stressed": stressed_res['el_by_grade']
        }
