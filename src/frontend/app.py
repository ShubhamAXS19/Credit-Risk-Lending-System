"""
Credit Risk & Lending System — Enterprise Regulatory Dashboard
================================================================
Modules:
  - 📊 Executive Portfolio Overview (Expected Loss, RWA, Capital Adequacy CAR%)
  - ⚖️ Multi-Model Evaluation Benchmark (Logistic, XGBoost, LightGBM, CatBoost)
  - 🏛️ Basel II/III Risk Engine (PD, LGD by Collateral, EAD with CCFs, EL, RWA)
  - ⚡ Macro Stress Testing Dashboard (RBI/CCAR Scenario Shock Sensitivity)
  - 🔍 SHAP Underwriting Explainer (Feature Attributions & Reason Codes)
  - ⏳ IFRS 9 Lifetime PD & Cox PH Survival Analysis

Run:
    streamlit run src/frontend/app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.config import PROJ_ROOT, MODELS_DIR
from src.dataset import generate_credit_dataset
from src.features import engineer_features
from src.basel.pd_model import map_pd_to_grade, platt_scale
from src.basel.lgd_model import get_lgd, CollateralType
from src.basel.ead_model import calculate_ead, get_ccf
from src.basel.expected_loss import PortfolioRiskAggregator, calculate_expected_loss, calculate_rwa
from src.stress_testing.macro_model import StressTestEngine, STRESS_SCENARIOS
from src.explainability.shap_scorer import generate_shap_scorecard
from src.survival.cox_model import get_lifetime_pd

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Credit Risk & Lending System — Basel III",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif !important; background-color: #121410 !important; color: #d4d0c4 !important; }
.stApp { background-color: #121410; }
[data-testid="stSidebar"] { background-color: #0c0e0a !important; border-right: 0.5px solid #2e2e28; }
.sidebar-title { font-family: 'IBM Plex Mono', monospace; font-size: 14px; font-weight: 600; color: #4caf78 !important; letter-spacing: 0.08em; padding-bottom: 12px; border-bottom: 0.5px solid #2e2e28; }
.metric-box { background: #1a1c17; border: 0.5px solid #2e2e28; border-radius: 10px; padding: 16px; text-align: center; }
.metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 24px; font-weight: 600; margin-top: 4px; }
.metric-val.good { color: #4caf78; }
.metric-val.warn { color: #d4a843; }
.metric-val.bad { color: #e05a4a; }
.metric-sub { font-size: 11px; color: #7a7a6e; }
.card { background: #161813; border: 0.5px solid #2e2e28; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Cache synthetic portfolio data
@st.cache_data
def load_portfolio_data():
    raw_df = generate_credit_dataset(n_samples=2500, seed=42)
    return engineer_features(raw_df)

portfolio_df = load_portfolio_data()

# Sidebar Navigation
with st.sidebar:
    st.markdown('<p class="sidebar-title">🏦 BASEL RISK ENGINE</p>', unsafe_allow_html=True)
    module = st.radio(
        label="Module Navigation",
        options=[
            "📊 Executive Portfolio Overview",
            "🏛️ Basel II/III Risk Engine",
            "⚡ Macro Stress Testing",
            "🔍 SHAP Underwriting Explainer",
            "⏳ IFRS 9 Survival Analysis",
            "⚖️ Multi-Model Benchmark"
        ],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:#7a7a6e">Framework: Basel II/III IRB<br>Models: Logistic, XGBoost, LightGBM, CatBoost, Cox PH<br>API Status: Active (<a href="http://localhost:8000/docs" target="_blank" style="color:#4caf78">Swagger Docs</a>)</p>', unsafe_allow_html=True)

# =============================================================================
# MODULE 1: EXECUTIVE PORTFOLIO OVERVIEW
# =============================================================================
if module == "📊 Executive Portfolio Overview":
    st.title("📊 Executive Credit Portfolio Overview")
    st.caption("Basel II/III Portfolio Expected Loss, Risk-Weighted Assets (RWA), and Capital Adequacy Monitoring.")
    
    aggregator = PortfolioRiskAggregator(available_capital=15000000.0)
    eval_res = aggregator.evaluate_portfolio(portfolio_df)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div style="font-size:12px;color:#7a7a6e">Total Exposure (EAD)</div><div class="metric-val">${eval_res["total_exposure_ead"]:,.0f}</div><div class="metric-sub">{eval_res["total_loans"]:,} Loan Applications</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div style="font-size:12px;color:#7a7a6e">Portfolio Expected Loss (EL)</div><div class="metric-val bad">${eval_res["total_expected_loss"]:,.0f}</div><div class="metric-sub">{eval_res["portfolio_el_rate"]:.2f}% EL Rate</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div style="font-size:12px;color:#7a7a6e">Risk-Weighted Assets (RWA)</div><div class="metric-val warn">${eval_res["total_rwa"]:,.0f}</div><div class="metric-sub">Basel IRB Formula</div></div>', unsafe_allow_html=True)
    with m4:
        car_cls = "good" if not eval_res["car_breach_flag"] else "bad"
        st.markdown(f'<div class="metric-box"><div style="font-size:12px;color:#7a7a6e">Capital Adequacy Ratio (CAR)</div><div class="metric-val {car_cls}">{eval_res["car_ratio"]*100:.2f}%</div><div class="metric-sub">Min Required: 8.00%</div></div>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk Grade Exposure Distribution")
        grade_df = pd.DataFrame(eval_res["el_by_grade"]).T.reset_index().rename(columns={'index': 'Risk Grade'})
        fig = px.bar(grade_df, x='Risk Grade', y='sum', title="Expected Loss by Basel Risk Grade", labels={'sum': 'Expected Loss ($)'}, color='sum', color_continuous_scale='Reds')
        fig.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("Expected Loss by Collateral Type")
        col_df = pd.DataFrame(list(eval_res["el_by_collateral"].items()), columns=['Collateral', 'Expected Loss'])
        fig_pie = px.pie(col_df, names='Collateral', values='Expected Loss', title="EL Concentration by Collateral Class", hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
        st.plotly_chart(fig_pie, use_container_width=True)

# =============================================================================
# MODULE 2: BASEL II/III RISK ENGINE
# =============================================================================
elif module == "🏛️ Basel II/III Risk Engine":
    st.title("🏛️ Basel II/III Regulatory Risk Calculator")
    st.caption("Calculate loan-level PD, Collateral-Segmented LGD, EAD with CCFs, Expected Loss, and RWA.")
    
    with st.form("basel_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Probability of Default (PD)**")
            raw_pd = st.slider("Raw Default Probability", 0.001, 0.50, 0.08, step=0.005)
            loan_amt = st.number_input("Drawn Outstanding Amount ($)", value=25000.0, step=1000.0)
            undrawn_amt = st.number_input("Undrawn Line Credit ($)", value=10000.0, step=1000.0)
            
        with col2:
            st.markdown("**LGD & Collateral**")
            collateral = st.selectbox("Collateral Type", ["UNSECURED", "RESIDENTIAL_PROPERTY", "VEHICLE", "FIXED_DEPOSIT", "GOLD"])
            collateral_val = st.number_input("Collateral Valuation ($)", value=35000.0, step=1000.0)
            use_ml_lgd = st.checkbox("Use ML LGD Regressor", value=False)
            
        with col3:
            st.markdown("**EAD & Loan Structure**")
            loan_type = st.selectbox("Facility Type", ["TERM_LOAN", "REVOLVING_CREDIT", "CREDIT_CARD", "OVERDRAFT"])
            horizon = st.slider("Credit Horizon (Months)", 6, 60, 24, step=6)
            
        submitted = st.form_submit_button("⚡ Compute Basel Risk Metrics")
        
    if submitted:
        calib_pd = platt_scale(raw_pd)
        grade = map_pd_to_grade(calib_pd).value
        ltv = loan_amt / collateral_val if collateral_val > 0 else 1.0
        lgd_val = get_lgd("LOAN-LIVE", collateral_type=collateral, ltv_ratio=ltv, use_ml=use_ml_lgd)
        ead_val = calculate_ead(loan_amt, undrawn_amt, loan_type=loan_type, horizon_months=horizon)
        el_val = calculate_expected_loss(calib_pd, lgd_val, ead_val)
        rwa_val = calculate_rwa(calib_pd, lgd_val, ead_val)
        
        st.markdown("---")
        st.subheader("Calculation Breakdown")
        
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1: st.metric("Calibrated PD", f"{calib_pd*100:.2f}%", f"Grade: {grade}")
        with b2: st.metric("Loss Given Default (LGD)", f"{lgd_val*100:.1f}%", f"LTV: {ltv*100:.0f}%")
        with b3: st.metric("Exposure at Default (EAD)", f"${ead_val:,.0f}", f"CCF Applied")
        with b4: st.metric("Expected Loss (EL)", f"${el_val:,.2f}", f"PD × LGD × EAD")
        with b5: st.metric("Risk-Weighted Asset", f"${rwa_val:,.2f}", f"Tier 1 Cap: ${rwa_val*0.08:,.2f}")

# =============================================================================
# MODULE 3: MACRO STRESS TESTING
# =============================================================================
elif module == "⚡ Macro Stress Testing":
    st.title("⚡ Macroeconomic Stress Testing (RBI/CCAR Scenarios)")
    st.caption("Simulate portfolio Expected Loss and Capital Adequacy sensitivity under adverse macroeconomic shocks.")
    
    scen_choice = st.selectbox("Select Macroeconomic Scenario", ["baseline", "moderate_stress", "severe_stress", "custom"])
    
    if scen_choice == "custom":
        c1, c2, c3 = st.columns(3)
        with c1: unemp = st.slider("Unemployment Rate Rise (%)", 0.0, 10.0, 4.0) / 100.0
        with c2: rate_hike = st.slider("Interest Rate Hike (bps)", 0, 500, 200)
        with c3: gdp = st.slider("GDP Growth Rate (%)", -5.0, 6.0, -1.0) / 100.0
        scen_payload = {"name": "Custom Shock", "unemployment_delta": unemp, "rate_hike_bps": rate_hike, "gdp_growth": gdp}
    else:
        scen_payload = scen_choice
        
    cap_val = st.number_input("Portfolio Tier 1 Available Capital ($)", value=8000000.0, step=500000.0)
    
    engine = StressTestEngine(available_capital=cap_val)
    st_res = engine.run_stress_test(portfolio_df, scenario=scen_payload)
    
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Stressed Expected Loss", f"${st_res['stressed_metrics']['total_el']:,.0f}", f"+${st_res['stressed_metrics']['el_delta']:,.0f} Shift")
    with s2: st.metric("Stressed EL Rate", f"{st_res['stressed_metrics']['portfolio_el_rate']:.2f}%", f"+{st_res['stressed_metrics']['el_pct_increase']:.1f}% Increase")
    with s3: st.metric("Stressed RWA", f"${st_res['stressed_metrics']['total_rwa']:,.0f}")
    with s4:
        car_val = st_res['stressed_metrics']['car_ratio'] * 100
        st.metric("Stressed CAR %", f"{car_val:.2f}%", "BREACH <8%" if st_res['stressed_metrics']['car_breach'] else "Adequate")
        
    st.markdown("---")
    st.subheader("EL Sensitivity across Scenarios")
    
    scen_comparison = []
    for k in ["baseline", "moderate_stress", "severe_stress"]:
        r = engine.run_stress_test(portfolio_df, scenario=k)
        scen_comparison.append({
            "Scenario": STRESS_SCENARIOS[k]['name'],
            "Expected Loss ($)": r['stressed_metrics']['total_el'],
            "CAR (%)": r['stressed_metrics']['car_ratio'] * 100
        })
    df_scen = pd.DataFrame(scen_comparison)
    fig_st = px.bar(df_scen, x='Scenario', y='Expected Loss ($)', title="Portfolio EL Shift Across Stress Scenarios", color='CAR (%)', color_continuous_scale='Reds_r')
    fig_st.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
    st.plotly_chart(fig_st, use_container_width=True)

# =============================================================================
# MODULE 4: SHAP UNDERWRITING EXPLAINER
# =============================================================================
elif module == "🔍 SHAP Underwriting Explainer":
    st.title("🔍 SHAP Regulatory Underwriting Decision Explainer")
    st.caption("Instance-level feature attribution and reason codes for automated credit decisioning (SR 11-7 guidelines).")
    
    applicant_id = st.text_input("Applicant ID", value="LOAN-10042")
    
    col1, col2 = st.columns(2)
    with col1:
        revol_util = st.slider("Revolving Utilization", 0.0, 1.5, 0.45)
        age = st.slider("Borrower Age", 18, 80, 38)
        times_late = st.number_input("90+ Days Late Count", value=1, min_value=0)
        debt_ratio = st.slider("Debt-to-Income Ratio", 0.0, 2.0, 0.40)
    with col2:
        income = st.number_input("Monthly Income ($)", value=6500.0)
        lines = st.number_input("Open Credit Lines", value=7)
        dependents = st.number_input("Dependents", value=1)
        
    input_dict = {
        "RevolvingUtilizationOfUnsecuredLines": revol_util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": lines,
        "NumberOfTimes90DaysLate": times_late,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": dependents
    }
    
    shap_res = generate_shap_scorecard(applicant_id, input_dict)
    
    st.markdown("---")
    st.subheader(f"Decision Explanation — {applicant_id}")
    
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("### ✅ Top Positive Factors (Approving Drivers)")
        for item in shap_res['top_positive_factors']:
            st.success(f"**{item['feature']}**: SHAP {item['shap_value']:+.4f}")
            
    with p2:
        st.markdown("### ❌ Top Adverse Factors (Rejection Drivers)")
        for item in shap_res['top_negative_factors']:
            st.error(f"**{item['feature']}**: SHAP {item['shap_value']:+.4f}")
            
    if Path(shap_res['waterfall_plot_path']).exists():
        st.image(shap_res['waterfall_plot_path'], caption="SHAP Waterfall Attribution Chart")

# =============================================================================
# MODULE 5: IFRS 9 SURVIVAL ANALYSIS
# =============================================================================
elif module == "⏳ IFRS 9 Survival Analysis":
    st.title("⏳ IFRS 9 Lifetime PD & Cox Proportional Hazards")
    st.caption("Estimate cumulative lifetime default probabilities over 12, 24, and 36 months for IFRS 9 staging.")
    
    util = st.slider("Revolving Utilization Rate", 0.0, 1.5, 0.35)
    dti = st.slider("Debt Ratio", 0.0, 2.0, 0.45)
    lates = st.number_input("Total Delinquency Count", value=1, min_value=0)
    
    feat_input = {
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": 40,
        "DebtRatio": dti,
        "MonthlyIncome": 5500.0,
        "total_delinquency": lates,
        "credit_stress_index": util * dti
    }
    
    surv_res = get_lifetime_pd(feat_input, horizon_months=36)
    
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Concordance Index (C-Index)", f"{surv_res['c_index']:.4f}")
    with s2: st.metric("12-Month PD (Stage 1)", f"{surv_res['pd_12m']*100:.2f}%")
    with s3: st.metric("24-Month Cumulative PD", f"{surv_res['pd_24m']*100:.2f}%")
    with s4: st.metric("36-Month Cumulative PD", f"{surv_res['pd_36m']*100:.2f}%")
    
    st.info(f"**IFRS 9 Classification**: {surv_res['ifrs9_stage']}")
    
    # Plot survival curve
    curve_df = pd.DataFrame(surv_res['survival_curve'], columns=['Month', 'Survival_Probability'])
    fig_surv = px.line(curve_df, x='Month', y='Survival_Probability', title="Cox PH Survival Curve S(t)", markers=True)
    fig_surv.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
    st.plotly_chart(fig_surv, use_container_width=True)

# =============================================================================
# MODULE 6: MULTI-MODEL BENCHMARK
# =============================================================================
elif module == "⚖️ Multi-Model Benchmark":
    st.title("⚖️ Multi-Model Evaluation Benchmark Suite")
    st.caption("Performance metrics comparison across Logistic Regression, XGBoost, LightGBM, CatBoost, and Random Forest.")
    
    eval_json_path = MODELS_DIR / "evaluation_results.json"
    if not eval_json_path.exists():
        from src.modeling.train import train_and_evaluate_models
        with st.spinner("Training model suite..."):
            res_dict = train_and_evaluate_models()
    else:
        with open(eval_json_path, "r") as f:
            res_dict = json.load(f)
            
    df_metrics = pd.DataFrame(res_dict).T.reset_index().rename(columns={'index': 'Model'})
    
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['AUC_ROC', 'KS_Statistic', 'PR_AUC'], color='#1e3a29'), use_container_width=True)
    
    f1, f2 = st.columns(2)
    with f1:
        fig_auc = px.bar(df_metrics, x='Model', y='AUC_ROC', title="AUC-ROC Score Comparison", color='AUC_ROC', color_continuous_scale='Greens')
        fig_auc.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
        st.plotly_chart(fig_auc, use_container_width=True)
        
    with f2:
        fig_ks = px.bar(df_metrics, x='Model', y='KS_Statistic', title="KS-Statistic Comparison", color='KS_Statistic', color_continuous_scale='Blues')
        fig_ks.update_layout(template="plotly_dark", plot_bgcolor="#161813", paper_bgcolor="#161813")
        st.plotly_chart(fig_ks, use_container_width=True)