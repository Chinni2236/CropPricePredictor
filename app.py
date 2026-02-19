import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Crop Price Prediction",
    page_icon="🌾",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("xgb_crop_price_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, features = load_model()

st.title("🌾 Crop Price Prediction System")
st.caption("AI-powered crop price forecasting")

st.sidebar.title("🌱 Crop Conditions")

input_data = {}

with st.sidebar.expander("📅 Time & Climate", expanded=True):
    input_data["year"] = st.number_input("Year", min_value=2015, max_value=2035, value=2025)

    input_data["rainfall_mm"] = st.slider(
        "Rainfall (mm)", 0, 3000, 800
    )

    input_data["rainfall_deviation_pct"] = st.slider(
        "Rainfall Deviation (%)", -100, 100, 0
    )

    input_data["avg_temp_c"] = st.slider(
        "Average Temperature (°C)", 10, 45, 30
    )

with st.sidebar.expander("🌾 Farm & Soil", expanded=True):
    input_data["yield_qtl_per_acre"] = st.slider(
        "Yield (quintal/acre)", 0.0, 50.0, 15.0
    )

    input_data["soil_fertility_index"] = st.slider(
        "Soil Fertility Index", 0.0, 1.0, 0.6
    )

    input_data["irrigated"] = st.radio(
        "Irrigation Available?", ["No", "Yes"]
    )
    input_data["irrigated"] = 1 if input_data["irrigated"] == "Yes" else 0

with st.sidebar.expander("💰 Cost Factors", expanded=True):
    input_data["fertilizer_cost_rs_per_acre"] = st.slider(
        "Fertilizer Cost (₹/acre)", 0, 20000, 5000
    )

    input_data["labor_cost_rs_per_acre"] = st.slider(
        "Labor Cost (₹/acre)", 0, 30000, 8000
    )

    input_data["diesel_price_rs_per_l"] = st.slider(
        "Diesel Price (₹/L)", 60, 120, 90
    )

input_df = pd.DataFrame([input_data])

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Crop Price: ₹ {prediction:,.2f}")

    fig = px.bar(
        x=features,
        y=input_df.iloc[0].values,
        labels={"x": "Feature", "y": "Value"},
        title="Feature Values Used in Prediction"
    )
    st.plotly_chart(fig, use_container_width=True)
