import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Fuel Scarcity Predictor", page_icon="⛽", layout="wide")
st.title("⛽ Fuel Scarcity Forecasting Model (MVP)")
st.write("Predicting the likelihood of fuel scarcity 1 week in advance using time-series indicators.")

# Define feature names matching exact model training sequence
feature_names = [
    'Petrol_Price_USD_per_Liter', 'Diesel_Price_USD_per_Liter',
    'Natural_Gas_Price_USD_per_MMBtu', 'Crude_Oil_Price_USD_per_Barrel',
    'Inflation_Rate (%)', 'Exchange_Rate_vs_USD', 'GDP_Growth (%)',
    'Supply_Index', 'Demand_Index', 'Geopolitical_Risk_Index', 'Event_Flag',
    'Currency_Devaluation (%)', 'Tax_Rate_on_Fuel (%)', 'Petrol_7d_MA',
    'Petrol_28d_vol', 'Petrol_lag_1', 'Crude_lag_1', 'Year', 'Month',
    'Quarter', 'Week', 'Day_Of_Week', 'Supply_Index_lag_1',
    'Demand_Index_lag_1', 'Supply_Demand_Ratio_lag_1', 'Subsidy_Level_Encoded'
]

# Load trained model artifact
@st.cache_resource
def load_artifacts():
    model = joblib.load('fuel_scarcity_model.pkl')
    return model

try:
    model = load_artifacts()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("📊 Market Indicators")

supply_lag = st.sidebar.slider(
    "Supply Index", 
    min_value=10.0, max_value=100.0, value=27.0, step=1.0,
    help="Values below 40 indicate severe supply contraction."
)

demand_lag = st.sidebar.slider(
    "Demand Index", 
    min_value=10.0, max_value=100.0, value=82.0, step=1.0
)

# Automatically compute ratio
ratio_lag = round(supply_lag / (demand_lag + 1e-5), 4)

fx_rate = st.sidebar.number_input("Exchange Rate vs USD (₦/$)", 0.0, 5000.0, 1200.0, 10.0)

# Petrol Price in Naira Input & Automatic USD Conversion
petrol_ngn = st.sidebar.number_input(
    "Petrol Price (₦ / Liter)", 
    min_value=0.0, max_value=5000.0, value=1250.0, step=10.0,
    help="Enter pump price in Naira."
)

petrol_usd = round(petrol_ngn / fx_rate, 4)


# diesel Price in Naira Input & Automatic USD Conversion
diesel_ngn = st.sidebar.number_input(
    "diesel Price (₦ / Liter)", 
    min_value=0.0, max_value=5000.0, value=1750.0, step=10.0,
    help="Enter pump price in Naira."
)

diesel_usd = round(diesel_ngn / fx_rate, 4)


crude_lag = st.sidebar.number_input("Crude Oil Price ($/barrel)", 0.0, 200.0, 75.0, 1.0)
inflation = st.sidebar.number_input("Inflation Rate (%)", 0.0, 50.0, 25.0, 0.5)
subsidy = st.sidebar.selectbox("Subsidy Level", options=["Low", "Medium", "High"], index=2)

subsidy_map = {"Low": 0, "Medium": 1, "High": 2}

# --- Dynamic Feature Calculations---
# Active deficit condition check
is_severe_deficit = (demand_lag > supply_lag) and (supply_lag < 40)

# Volatility and risk flags react dynamically to supply deficits & macro stress
estimated_volatility = 0.14 if is_severe_deficit else 0.065
geopolitical_risk = 35.0 if is_severe_deficit else 26.0
event_flag = 1 if is_severe_deficit else 0
devaluation = 5.0 if inflation > 20.0 else 0.0

# --- Prediction Block ---
if st.button("🚀 Run Scarcity Forecast"):
    input_data = pd.DataFrame([{
        'Petrol_Price_USD_per_Liter': petrol_usd,
        'Diesel_Price_USD_per_Liter': diesel_usd,
        'Natural_Gas_Price_USD_per_MMBtu': 1.68,
        'Crude_Oil_Price_USD_per_Barrel': crude_lag,
        'Inflation_Rate (%)': inflation,
        'Exchange_Rate_vs_USD': fx_rate,
        'GDP_Growth (%)': 2.0,
        'Supply_Index': supply_lag,
        'Demand_Index': demand_lag,
        'Geopolitical_Risk_Index': geopolitical_risk,
        'Event_Flag': event_flag,
        'Currency_Devaluation (%)': devaluation,
        'Tax_Rate_on_Fuel (%)': 22.5,
        'Petrol_7d_MA': petrol_usd,            # Converted USD price
        'Petrol_28d_vol': estimated_volatility,  # Dynamic volatility
        'Petrol_lag_1': petrol_usd,            # Converted USD price
        'Crude_lag_1': crude_lag,
        'Year': 2026,
        'Month': 8,
        'Quarter': 3,
        'Week': 52,
        'Day_Of_Week': 0,
        'Supply_Index_lag_1': supply_lag,
        'Demand_Index_lag_1': demand_lag,
        'Supply_Demand_Ratio_lag_1': ratio_lag,
        'Subsidy_Level_Encoded': subsidy_map[subsidy]
    }])[feature_names]

    # Predict Scarcity Probability
    prob = model.predict_proba(input_data)[0, 1]
    is_scarcity = prob >= 0.40

    st.markdown("---")
    st.subheader("📊 Forecast Summary")
    st.metric(label="Predicted Scarcity Probability", value=f"{prob:.1%}")

    if is_scarcity:
        st.error("⚠️ **HIGH RISK:** Fuel Scarcity predicted for next week (Exceeds 40% Decision Threshold).")
    else:
        st.success("✅ **LOW RISK:** Supply levels expected to remain normal next week.")