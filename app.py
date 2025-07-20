import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("tuned_random_forest_model.joblib")
label_encoders = joblib.load("label_encoders.pkl")

st.title("Credit Risk Prediction App")
st.write("Enter details to predict creditworthiness using Random Forest.")

with st.form("input_form"):
    status = st.selectbox("Status", ['< 0 DM', '0 <= ... < 200 DM', '>= 200 DM', 'no checking account'])
    duration = st.number_input("Duration (months)", min_value=1, max_value=72, value=24)
    credit_history = st.selectbox("Credit History", ['no credits/all paid', 'all paid', 'existing paid', 'delay', 'critical/other'])
    purpose = st.selectbox("Purpose", ['car (new)', 'car (used)', 'furniture', 'radio/TV', 'appliance',
                                       'repairs', 'education', 'vacation', 'retraining', 'business', 'other'])
    credit_amount = st.number_input("Credit Amount", min_value=0, max_value=100000, value=2500)
    savings = st.selectbox("Savings", ['< 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM', '>= 1000 DM', 'unknown'])
    employment = st.selectbox("Employment", ['unemployed', '< 1 yr', '1 <= ... < 4 yrs', '4 <= ... < 7 yrs', '>= 7 yrs'])
    installment_rate = st.slider("Installment Rate", 1, 4, 2)
    personal_status = st.selectbox("Personal Status", ['male div/sep', 'female div/sep/mar', 'male single', 'male mar/wid', 'female single'])
    other_debtors = st.selectbox("Other Debtors", ['none', 'co-applicant', 'guarantor'])
    residence_since = st.slider("Years at Current Residence", 1, 4, 2)
    property_ = st.selectbox("Property", ['real estate', 'insurance', 'car/other', 'unknown'])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    other_installment = st.selectbox("Other Installment Plans", ['bank', 'stores', 'none'])
    housing = st.selectbox("Housing", ['rent', 'own', 'free'])
    number_credits = st.slider("Existing Credits", 1, 4, 1)
    job = st.selectbox("Job", ['unskilled non-res', 'unskilled res', 'skilled', 'high qual/self'])
    people_liable = st.selectbox("Number of People Liable", [1, 2])
    telephone = st.selectbox("Telephone", ['none', 'yes'])
    foreign_worker = st.selectbox("Foreign Worker", ['yes', 'no'])

    submit = st.form_submit_button("Predict")

if submit:
    try:
        input_dict = {
            'Status': status,
            'Duration': duration,
            'CreditHistory': credit_history,
            'Purpose': purpose,
            'CreditAmount': credit_amount,
            'Savings': savings,
            'EmploymentSince': employment,
            'InstallmentRate': installment_rate,
            'PersonalStatusSex': personal_status,
            'DebtorsGuarantors': other_debtors,
            'ResidenceSince': residence_since,
            'Property': property_,
            'Age': age,
            'OtherInstallmentPlans': other_installment,
            'Housing': housing,
            'ExistingCredits': number_credits,
            'Job': job,
            'NumPeopleLiable': people_liable,
            'Telephone': telephone,
            'ForeignWorker': foreign_worker
        }

        input_df = pd.DataFrame([input_dict])

        # Apply label encoders
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][prediction]
        result = "Good Credit Risk" if prediction == 0 else "Bad Credit Risk"
        st.success(f"Prediction: **{result}**")
        st.info(f"Probability: {prob:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
