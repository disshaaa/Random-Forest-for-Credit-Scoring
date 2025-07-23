import streamlit as st
import pandas as pd
import joblib
import warnings

# --- Page Configuration ---
st.set_page_config(
    page_title="German Credit Risk Predictor",
    page_icon="🇩🇪",
    layout="wide"
)

# Suppress the scikit-learn version mismatch warning
warnings.filterwarnings("ignore", category=UserWarning)

# --- Mappings for UI Dropdowns (Human-Readable Labels) ---
status_map = {'A11': '< 0 DM', 'A12': '0 <= ... < 200 DM', 'A13': '>= 200 DM / salary assignments for at least 1 year', 'A14': 'no checking account'}
history_map = {'A30': 'no credits taken/ all credits paid back duly', 'A31': 'all credits at this bank paid back duly', 'A32': 'existing credits paid back duly till now', 'A33': 'delay in paying off in the past', 'A34': 'critical account/ other credits existing (not at this bank)'}
purpose_map = {'A40': 'car (new)', 'A41': 'car (used)', 'A42': 'furniture/equipment', 'A43': 'radio/television', 'A44': 'domestic appliances', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'}
savings_map = {'A61': '< 100 DM', 'A62': '100 <= ... < 500 DM', 'A63': '500 <= ... < 1000 DM', 'A64': '>= 1000 DM', 'A65': 'unknown/ no savings account'}
employment_map = {'A71': 'unemployed', 'A72': '< 1 year', 'A73': '1 <= ... < 4 years', 'A74': '4 <= ... < 7 years', 'A75': '>= 7 years'}
personal_status_map = {'A91': 'male : divorced/separated', 'A92': 'female : divorced/separated/married', 'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'male : single'}
other_debtors_map = {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'}
property_map = {'A121': 'real estate', 'A122': 'if not A121 : building society savings agreement/ life insurance', 'A123': 'if not A121/A122 : car or other', 'A124': 'unknown / no property'}
other_plans_map = {'A141': 'bank', 'A142': 'stores', 'A143': 'none'}
housing_map = {'A151': 'rent', 'A152': 'own', 'A153': 'for free'}
job_map = {'A171': 'unemployed/ unskilled - non-resident', 'A172': 'unskilled - resident', 'A173': 'skilled employee / official', 'A174': 'management/ self-employed/ highly qualified employee/ officer'}
telephone_map = {'A191': 'none', 'A192': 'yes, registered under the customers name'}
foreign_worker_map = {'A201': 'yes', 'A202': 'no'}

# --- Mappings to Replicate Label Encoding ---
status_encoding = {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}
history_encoding = {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4}
purpose_encoding = {'A40': 0, 'A41': 1, 'A410': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10}
savings_encoding = {'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}
employment_encoding = {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}
personal_status_encoding = {'A91': 0, 'A92': 1, 'A93': 2, 'A94': 3, 'A95': 4}
debtors_encoding = {'A101': 0, 'A102': 1, 'A103': 2}
property_encoding = {'A121': 0, 'A122': 1, 'A123': 2, 'A124': 3}
other_plans_encoding = {'A141': 0, 'A142': 1, 'A143': 2}
housing_encoding = {'A151': 0, 'A152': 1, 'A153': 2}
job_encoding = {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}
telephone_encoding = {'A191': 0, 'A192': 1}
foreign_worker_encoding = {'A201': 0, 'A202': 1}

# Helper to get key from value for UI maps
def get_key_from_value(d, val):
    for key, value in d.items():
        if value == val:
            return key
    return None

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load('tuned_random_forest_model.joblib')
        return model
    except FileNotFoundError:
        return None

model = load_model()

# --- Sidebar for User Inputs ---
st.sidebar.header("Applicant Information")

def get_user_input():
    # Get human-readable labels from user
    status_label = st.sidebar.selectbox('Checking Account Status', options=list(status_map.values()))
    history_label = st.sidebar.selectbox('Credit History', options=list(history_map.values()))
    purpose_label = st.sidebar.selectbox('Purpose', options=list(purpose_map.values()))
    savings_label = st.sidebar.selectbox('Savings Account/Bonds', options=list(savings_map.values()))
    employment_label = st.sidebar.selectbox('Present Employment Since', options=list(employment_map.values()))
    personal_status_label = st.sidebar.selectbox('Personal Status and Sex', options=list(personal_status_map.values()))
    other_debtors_label = st.sidebar.selectbox('Other Debtors / Guarantors', options=list(other_debtors_map.values()))
    property_label = st.sidebar.selectbox('Property', options=list(property_map.values()))
    other_plans_label = st.sidebar.selectbox('Other Installment Plans', options=list(other_plans_map.values()))
    housing_label = st.sidebar.selectbox('Housing', options=list(housing_map.values()))
    job_label = st.sidebar.selectbox('Job', options=list(job_map.values()))
    telephone_label = st.sidebar.selectbox('Telephone', options=list(telephone_map.values()))
    foreign_worker_label = st.sidebar.selectbox('Foreign Worker', options=list(foreign_worker_map.values()))
    
    # Get numeric values from user
    duration = st.sidebar.slider('Duration in Month', 4, 72, 24)
    credit_amount = st.sidebar.number_input('Credit Amount (in DM)', min_value=250, max_value=20000, value=2500, step=100)
    installment_rate = st.sidebar.slider('Installment Rate in % of Disposable Income', 1, 4, 3)
    residence_since = st.sidebar.slider('Present Residence Since', 1, 4, 2)
    age = st.sidebar.slider('Age in Years', 18, 75, 35)
    existing_credits = st.sidebar.slider('Number of Existing Credits at this Bank', 1, 4, 1)
    num_dependents = st.sidebar.slider('Number of People Liable to Provide Maintenance For', 1, 2, 1)

    # Convert labels to codes, then to integers.
    # **CRITICAL FIX**: Keys now match the exact column names from the training error.
    input_dict = {
        'Status': status_encoding[get_key_from_value(status_map, status_label)],
        'CreditHistory': history_encoding[get_key_from_value(history_map, history_label)],
        'Purpose': purpose_encoding[get_key_from_value(purpose_map, purpose_label)],
        'Savings': savings_encoding[get_key_from_value(savings_map, savings_label)],
        'EmploymentSince': employment_encoding[get_key_from_value(employment_map, employment_label)],
        'PersonalStatusSex': personal_status_encoding[get_key_from_value(personal_status_map, personal_status_label)], # FIX
        'DebtorsGuarantors': debtors_encoding[get_key_from_value(other_debtors_map, other_debtors_label)],
        'Property': property_encoding[get_key_from_value(property_map, property_label)],
        'OtherInstallmentPlans': other_plans_encoding[get_key_from_value(other_plans_map, other_plans_label)],
        'Housing': housing_encoding[get_key_from_value(housing_map, housing_label)],
        'Job': job_encoding[get_key_from_value(job_map, job_label)],
        'Telephone': telephone_encoding[get_key_from_value(telephone_map, telephone_label)],
        'ForeignWorker': foreign_worker_encoding[get_key_from_value(foreign_worker_map, foreign_worker_label)],
        'Duration': duration,
        'CreditAmount': credit_amount,
        'InstallmentRate': installment_rate,
        'ResidenceSince': residence_since, # FIX
        'Age': age,
        'ExistingCredits': existing_credits,
        'NumPeopleLiable': num_dependents, # FIX
    }
    return pd.DataFrame([input_dict])

user_data = get_user_input()

# --- Main Page for Displaying Results ---
st.title("German Credit Risk Prediction")
st.write("This app predicts credit risk based on applicant information. Please fill out the details in the sidebar.")

if model is None:
    st.error("Model file not found. Please ensure `tuned_random_forest_model.joblib` is in the same directory.")
else:
    if st.sidebar.button('Predict Credit Risk'):
        # **CRITICAL FIX**: Column list now matches the exact names and order the model expects.
        model_columns = [
            'Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 'Savings', 
            'EmploymentSince', 'InstallmentRate', 'PersonalStatusSex', 'DebtorsGuarantors', 
            'ResidenceSince', 'Property', 'Age', 'OtherInstallmentPlans', 'Housing', 
            'ExistingCredits', 'Job', 'NumPeopleLiable', 'Telephone', 'ForeignWorker'
        ]

        # Reorder the user_data DataFrame to match the model's expected input
        processed_data = user_data[model_columns]

        # Make prediction
        try:
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)

            st.subheader("Prediction Result")
            # Your model outputs 1 for 'Good' and 2 for 'Bad'
            if prediction[0] == 1:
                st.success("✅ Good Credit Risk")
                st.write(f"**Confidence Score:** {prediction_proba[0][0]:.2%}")
            else:
                st.error("❌ Bad Credit Risk")
                st.write(f"**Confidence Score:** {prediction_proba[0][1]:.2%}")
            
            with st.expander("See Prediction Details"):
                st.write("The following processed data was used for the prediction:")
                st.dataframe(processed_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.write("App built with Streamlit for the German Credit dataset.")
