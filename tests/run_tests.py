import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.basel.pd_model import map_pd_to_grade, PDBand, platt_scale
from src.basel.lgd_model import get_lgd
from src.basel.ead_model import calculate_ead
from src.basel.expected_loss import calculate_expected_loss, calculate_rwa, PortfolioRiskAggregator
from src.stress_testing.macro_model import StressTestEngine
from src.survival.data_prep import prepare_survival_data
from src.dataset import generate_credit_dataset

class TestBaselEngine(unittest.TestCase):
    def test_pd_mapping(self):
        self.assertEqual(map_pd_to_grade(0.0005), PDBand.AAA)
        self.assertEqual(map_pd_to_grade(0.003), PDBand.AA)
        self.assertEqual(map_pd_to_grade(0.015), PDBand.A)
        self.assertEqual(map_pd_to_grade(0.035), PDBand.BBB)
        self.assertEqual(map_pd_to_grade(0.075), PDBand.BB)
        self.assertEqual(map_pd_to_grade(0.15), PDBand.B)
        self.assertEqual(map_pd_to_grade(0.25), PDBand.CCC)

    def test_lgd_lookup(self):
        self.assertEqual(get_lgd(collateral_type="UNSECURED"), 0.75)
        self.assertEqual(get_lgd(collateral_type="RESIDENTIAL_PROPERTY"), 0.35)
        self.assertEqual(get_lgd(collateral_type="VEHICLE"), 0.40)

    def test_ead_calculation(self):
        self.assertEqual(calculate_ead(10000, 5000, "TERM_LOAN"), 15000.0)
        self.assertEqual(calculate_ead(10000, 5000, "REVOLVING_CREDIT", horizon_months=6), 11000.0)

    def test_expected_loss(self):
        el = calculate_expected_loss(0.05, 0.45, 20000)
        self.assertEqual(el, 450.0)
        rwa = calculate_rwa(0.05, 0.45, 20000)
        self.assertGreater(rwa, 0)

class TestMacroStressEngine(unittest.TestCase):
    def test_stress_test(self):
        df = generate_credit_dataset(n_samples=200, seed=42)
        engine = StressTestEngine(available_capital=1000000.0)
        res_base = engine.run_stress_test(df, scenario="baseline")
        res_sev = engine.run_stress_test(df, scenario="severe_stress")
        self.assertGreaterEqual(res_sev["stressed_metrics"]["total_el"], res_base["stressed_metrics"]["total_el"])

class TestSurvivalData(unittest.TestCase):
    def test_survival_prep(self):
        df = generate_credit_dataset(n_samples=100, seed=42)
        surv_df = prepare_survival_data(df)
        self.assertIn("duration", surv_df.columns)
        self.assertIn("event", surv_df.columns)

if __name__ == "__main__":
    unittest.main()
