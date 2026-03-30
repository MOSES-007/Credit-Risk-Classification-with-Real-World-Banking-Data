import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Credit Risk Evaluator", page_icon="🏦", layout="centered")

# --- 2. LOAD ML ASSETS ---
# @st.cache_resource ensures we only load these large files once when the app starts
@st.cache_resource
def load_ml_components():
    # Assuming app.py is inside the 'app' folder alongside the joblib files
    base_path = os.path.dirname(__file__)
    imputer = joblib.load(os.path.join(base_path, 'imputer.joblib'))
    scaler = joblib.load(os.path.join(base_path, 'scaler.joblib'))
    model = joblib.load(os.path.join(base_path, 'xgb_model.joblib'))
    expected_cols = joblib.load(os.path.join(base_path, 'expected_columns.joblib'))
    return imputer, scaler, model, expected_cols

imputer, scaler, model, expected_cols = load_ml_components()

# --- 3. APP HEADER ---
st.title("🏦 Corporate Credit Risk Evaluator")
st.markdown("Enter the applicant's financial details below to generate a live Probability of Default (PoD) using our XGBoost model.")
st.divider()

# --- 4. USER INPUT FORM ---
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

# --- 5. PREDICTION LOGIC ---
if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
    with st.spinner("Analyzing profile..."):
        
        # A. Create a completely empty dataframe with the 158 columns our model expects
        input_data = pd.DataFrame(0, index=[0], columns=expected_cols)
        
        # B. Slot the numerical inputs into their correct columns
        input_data['loan_amnt'] = loan_amnt
        input_data['annual_inc'] = annual_inc
        input_data['int_rate'] = int_rate
        input_data['dti'] = dti
        
        # C. Handle the Categorical inputs (One-Hot Encoding alignment)
        # We construct the column name exactly as get_dummies did (e.g., 'home_ownership_RENT')
        home_col = f"home_ownership_{home_ownership}"
        if home_col in expected_cols:
            input_data[home_col] = 1
            
        emp_col = f"emp_length_{emp_length}"
        if emp_col in expected_cols:
            input_data[emp_col] = 1

        # D. Push the data through the pipeline (Impute -> Scale -> Predict)
        imputed_data = imputer.transform(input_data)
        scaled_data = scaler.transform(imputed_data)
        
        # We want the probability of Class 1 (Default)
        probability_of_default = model.predict_proba(scaled_data)[0][1]
        
        # --- 6. DISPLAY RESULTS ---
        st.subheader("Evaluation Results")
        
        # Set our business threshold (e.g., 35% based on our earlier curve)
        THRESHOLD = 0.35 
        
        if probability_of_default > THRESHOLD:
            st.error(f"⚠️ HIGH RISK: The Probability of Default is {probability_of_default * 100:.2f}%")
            st.markdown(f"**Recommendation:** Reject or request manual underwriter review. Risk exceeds the {THRESHOLD*100}% corporate threshold.")
        else:
            st.success(f"✅ LOW RISK: The Probability of Default is {probability_of_default * 100:.2f}%")
            st.markdown(f"**Recommendation:** Approve. Profile meets automated risk criteria.")