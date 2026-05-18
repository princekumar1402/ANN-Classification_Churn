# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# =========================
# Load Trained Model
# =========================

model = tf.keras.models.load_model("models/ann_model.h5")

# =========================
# Load Encoders and Scaler
# =========================

with open("models/label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("models/one_hot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# =========================
# Streamlit UI
# =========================

st.title("Customer Churn Prediction Using ANN")

st.write("Enter customer details below:")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider("Age", 18, 100, 35)

tenure = st.slider("Tenure", 0, 10, 5)

balance = st.number_input("Balance", value=50000.0)

num_of_products = st.slider("Number of Products", 1, 4, 1)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

estimated_salary = st.number_input(
    "Estimated Salary",
    value=50000.0
)

# =========================
# Prediction Button
# =========================

if st.button("Predict"):

    # Create Input DataFrame
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # Label Encode Gender
    input_data["Gender"] = label_encoder_gender.transform(
        input_data["Gender"]
    )

    # One Hot Encode Geography
    input_encoded = onehot_encoder_geo.transform(input_data)

    # Scale Data
    input_scaled = scaler.transform(input_encoded)

    # Prediction
    prediction = model.predict(input_scaled)

    prediction_probability = prediction[0][0]

    # Display Probability
    st.write(f"Churn Probability: {prediction_probability * 100:.2f}%")

    # Final Result
    if prediction_probability > 0.5:
        st.error("Customer is likely to leave the bank.")
    else:
        st.success("Customer is likely to stay with the bank.")