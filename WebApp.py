import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("linear_regression_model.pkl")

# Define the Streamlit app
st.title("Advertising Budget vs Sales Prediction")
st.write("Enter the advertising budget details to predict sales.")

# Input fields for user to enter advertising budget
TV_budget = st.number_input("TV Ad Budget ($)", min_value=0.0, format="%.2f")
Radio_budget = st.number_input("Radio Ad Budget ($)", min_value=0.0, format="%.2f")
Newspaper_budget = st.number_input("Newspaper Ad Budget ($)", min_value=0.0, format="%.2f")

# Predict sales on button click
if st.button("Predict Sales"):
    # Prepare input data
    input_data = np.array([[TV_budget, Radio_budget, Newspaper_budget]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"Predicted Sales: ${prediction[0]:.2f}")
