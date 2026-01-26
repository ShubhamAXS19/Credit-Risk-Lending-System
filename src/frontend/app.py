import streamlit as st

# Page config
st.set_page_config(
    page_title="Credit Risk & Lending System",
    page_icon="🏦",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
project_selection = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Project 1: Credit Default Prediction",
        "Project 2: Loan Amount Prediction",
        "Project 3: Early Payment Prediction",
        "Project 4: Credit Score Prediction",
        "Project 5: Loan Recovery Prediction"
    ]
)

# Main Content Area
if project_selection == "Home":
    st.title("🏦 Credit Risk & Lending System")
    st.markdown("""
    Welcome to the Comprehensive Credit Risk & Lending System. 
    This platform integrates multiple machine learning models to address critical aspects of the lending lifecycle.
    
    ### Available Modules:
    
    1. **Credit Default Prediction**: Predict the likelihood of a borrower defaulting.
    2. **Loan Amount Prediction**: Estimate the appropriate loan amount for a borrower.
    3. **Early Payment Prediction**: Forecast prepayment risk.
    4. **Credit Score Prediction**: AI-driven credit scoring using alternative data.
    5. **Loan Recovery Prediction**: Estimate recovery amounts for defaulted loans.
    
    👈 Select a project from the sidebar to get started.
    """)


elif project_selection == "Project 1: Credit Default Prediction":
    st.header("Project 1: Credit Default Prediction")
    st.markdown("### Borrower Information")
    st.info("Enter the key financial details below. Other features will be set to default values for this test.")

    import joblib
    import requests
    import pandas as pd
    from pathlib import Path
    
    # Load features list to ensure correct order
    try:
        features_path = Path("models/Project 1: Credit Default Prediction/features.pkl")
        if not features_path.exists():
             st.error(f"Features file not found at {features_path}. Please check path.")
             st.stop()
        feature_names = joblib.load(features_path)
    except Exception as e:
        st.error(f"Error loading feature definitions: {e}")
        st.stop()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amt_income_total = st.number_input("Total Income", value=150000.0, step=1000.0)
            amt_credit = st.number_input("Credit Amount", value=500000.0, step=1000.0)
            amt_annuity = st.number_input("Loan Annuity", value=25000.0, step=500.0)
            
        with col2:
            # Days are negative in this dataset usually, but let's ask for positive years/days and convert
            age_years = st.number_input("Age (Years)", value=30, step=1)
            employed_years = st.number_input("Years Employed", value=5, step=1)
            
            # External Sources are normalized 0-1 scores
            ext_source_2 = st.slider("External Source 2 Score", 0.0, 1.0, 0.5)
            ext_source_3 = st.slider("External Source 3 Score", 0.0, 1.0, 0.5)

        # Hidden logic for derived features
        days_birth = -1 * age_years * 365
        days_employed = -1 * employed_years * 365
        
        submitted = st.form_submit_button("Predict Risk")
        
    if submitted:
        # 1. Initialize full feature vector with defaults (0)
        # In a real scenario, median imputation from training set is best.
        input_data = {f: 0.0 for f in feature_names}
        
        # 2. Map Inputs
        input_data['AMT_INCOME_TOTAL'] = amt_income_total
        input_data['AMT_CREDIT'] = amt_credit
        input_data['AMT_ANNUITY'] = amt_annuity
        input_data['DAYS_BIRTH'] = days_birth
        input_data['DAYS_EMPLOYED'] = days_employed
        input_data['EXT_SOURCE_2'] = ext_source_2
        input_data['EXT_SOURCE_3'] = ext_source_3
        
        # 3. Calculate Domain Features (important for model performance)
        # CREDIT_INCOME_RATIO
        if amt_income_total > 0:
            input_data['CREDIT_INCOME_RATIO'] = amt_credit / amt_income_total
            input_data['ANNUITY_INCOME_RATIO'] = amt_annuity / amt_income_total
        
        # CREDIT_TERM
        if amt_annuity > 0:
            input_data['CREDIT_TERM'] = amt_credit / amt_annuity
            
        # DAYS_EMPLOYED_RATIO
        if days_birth != 0:
            input_data['DAYS_EMPLOYED_RATIO'] = days_employed / days_birth

        # Send to API
        api_url = "http://localhost:8000/api/v1/predict"
        payload = {"features": input_data}
        
        try:
            with st.spinner("Scoring..."):
                response = requests.post(api_url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prob = result["default_probability"]
                decision = result["decision"]
                
                st.divider()
                st.subheader("Results")
                c1, c2, c3 = st.columns(3)
                c1.metric("Default Probability", f"{prob:.2%}")
                c2.metric("Decision", decision)
                
                if decision == "Reject":
                    st.error("High risk detected based on calibrated model.")
                else:
                    st.success("Application approved based on current thresholds.")
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Connection failed: {e}")



elif project_selection == "Project 2: Loan Amount Prediction":
    st.header("Project 2: Loan Amount Prediction")
    st.info("Module under construction 🚧")

elif project_selection == "Project 3: Early Payment Prediction":
    st.header("Project 3: Early Payment Prediction")
    st.info("Module under construction 🚧")

elif project_selection == "Project 4: Credit Score Prediction":
    st.header("Project 4: Credit Score Prediction")
    st.info("Module under construction 🚧")

elif project_selection == "Project 5: Loan Recovery Prediction":
    st.header("Project 5: Loan Recovery Prediction")
    st.info("Module under construction 🚧")
