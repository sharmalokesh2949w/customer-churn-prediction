import streamlit as st
import pickle
import pandas as pd

# Load model & columns
model = pickle.load(open('model.sav', 'rb'))
cols = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction")

# 🔹 Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
tenure = st.number_input("Tenure", min_value=0)
monthly = st.number_input("Monthly Charges", min_value=0.0)
total = st.number_input("Total Charges", min_value=0.0)

# 🔹 Predict
if st.button("Predict"):

    # Create base input dictionary
    input_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': online_backup,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'tenure': tenure,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply get_dummies
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=cols, fill_value=0)

    # Prediction + Probability
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)

    churn_prob = probability[0][1] * 100

    # Output
    if prediction[0] == 1:
        st.error(f"Customer will churn ❌ ({churn_prob:.2f}%)")
    else:
        st.success(f"Customer will stay ✅ ({100 - churn_prob:.2f}%)")


