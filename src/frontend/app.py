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
    st.info("Module under construction 🚧")

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
