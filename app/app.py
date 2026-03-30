import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Risk Evaluator", page_icon="🏦", layout="centered")

@st.cache_resource
def load_ml_components():
    base_path = os.path.dirname(__file__)
    imputer = joblib.load(os.path.join(base_path, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
    model = joblib.load(os.path.join(base_path, 'xgb_model.joblib'))
    expected_cols = joblib.load(os.path.join(base_path, 'expected_columns.joblib'))
    # NEW: Load the baseline "Average Joe" profile
    baseline = joblib.load(os.path.join(base_path, 'baseline_profile.joblib'))
    return imputer, scaler, model, expected_cols, baseline

imputer, scaler, model, expected_cols, baseline_profile = load_ml_components()

st.title("🏦 Corporate Credit Risk Evaluator")
st.markdown("Enter the applicant's financial details below to generate a live Probability of Default (PoD) using our XGBoost model.")
st.divider()

st.subheader("Applicant Information")
col1, col2 = st.columns(2)

with col1:
    loan_amnt = st.number_input("Loan Amount Requested ($)", min_value=1000, max_value=40000, value=10000, step=500)
    annual_inc = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=60000, step=1000)
    emp_length = st.selectbox("Employment Length", ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"])

with col2:
    int_rate = st.number_input("Proposed Interest Rate (%)", min_value=5.0, max_value=30.0, value=10.5, step=0.1)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    home_ownership = st.selectbox("Home Ownership Status", ["RENT", "OWN", "MORTGAGE", "ANY"])

st.divider()

if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing profile..."):
        
        # A. Start with the "Average Joe" baseline instead of zeros!
        input_data = pd.DataFrame([baseline_profile])
        
        # B. Overwrite the baseline with the user's specific inputs
        input_data['loan_amnt'] = loan_amnt
        input_data['annual_inc'] = annual_inc
        input_data['int_rate'] = int_rate
        input_data['dti'] = dti
        
        # C. Handle Categorical inputs carefully
        # First, wipe the baseline clean for these specific categories
        for col in expected_cols:
            if col.startswith("home_ownership_") or col.startswith("emp_length_"):
                input_data[col] = 0
                
        # Then apply the user's explicit choices
        home_col = f"home_ownership_{home_ownership}"
        if home_col in expected_cols:
            input_data[home_col] = 1
            
        emp_col = f"emp_length_{emp_length}"
        if emp_col in expected_cols:
            input_data[emp_col] = 1

        # D. Predict
        imputed_data = imputer.transform(input_data)
        scaled_data = scaler.transform(imputed_data)
        probability_of_default = model.predict_proba(scaled_data)[0][1]
        
        # E. Display
        st.subheader("Evaluation Results")
        THRESHOLD = 0.35 
        
        if probability_of_default > THRESHOLD:
            st.error(f"⚠️ HIGH RISK: The Probability of Default is {probability_of_default * 100:.2f}%")
            st.markdown(f"**Recommendation:** Reject or request manual underwriter review. Risk exceeds the {THRESHOLD*100}% corporate threshold.")
        else:
            st.success(f"✅ LOW RISK: The Probability of Default is {probability_of_default * 100:.2f}%")
            st.markdown(f"**Recommendation:** Approve. Profile meets automated risk criteria.")