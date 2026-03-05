import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Crop Price Intelligence",
    layout="wide"
)

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🌾 AI Agricultural Market Intelligence Platform")
st.caption("Smart Crop Price Forecasting using Machine Learning")

st.sidebar.header("Farm Information")

crop = st.sidebar.selectbox(
    "Crop Type",
    ["Paddy","Maize","Cotton","Turmeric","Chilli"]
)

rainfall = st.sidebar.slider(
    "Rainfall This Season (mm)",
    0,1500,800
)

temperature = st.sidebar.slider(
    "Average Temperature (°C)",
    10,45,30
)

soil_type = st.sidebar.selectbox(
    "Soil Type",
    ["Poor","Average","Fertile"]
)

irrigation = st.sidebar.selectbox(
    "Irrigation Availability",
    ["Low","Medium","High"]
)

farm_size = st.sidebar.slider(
    "Farm Size (Acres)",
    1,50,10
)

market_demand = st.sidebar.selectbox(
    "Market Demand",
    ["Low","Medium","High"]
)

soil_map = {
    "Poor":0.4,
    "Average":0.6,
    "Fertile":0.8
}

soil = soil_map[soil_type]

irrigation_map = {
    "Low":0.45,
    "Medium":0.6,
    "High":0.75
}

ndvi = irrigation_map[irrigation]

demand_map = {
    "Low":0.3,
    "Medium":0.55,
    "High":0.8
}

demand = demand_map[market_demand]

yield_q = (
    rainfall*0.015 +
    ndvi*15 +
    soil*10
)

arrivals = farm_size * yield_q * 20

input_data = {
    "rainfall_mm":rainfall,
    "avg_temp_c":temperature,
    "yield_qtl_per_acre":yield_q,
    "ndvi_satellite_index":ndvi,
    "export_demand_index":demand,
    "mandi_arrivals_qtl":arrivals,
    "soil_fertility_index":soil
}

for c in ["Paddy","Maize","Cotton","Turmeric","Chilli"]:
    input_data["crop_"+c] = 1 if crop==c else 0

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=features,fill_value=0)

prediction = model.predict(input_df)[0]

best_case = prediction*1.08
worst_case = prediction*0.92

c1,c2,c3 = st.columns(3)

c1.metric("Predicted Price ₹/Qtl",f"{prediction:.0f}")
c2.metric("Best Case ₹",f"{best_case:.0f}")
c3.metric("Worst Case ₹",f"{worst_case:.0f}")

st.subheader("Farm Condition Indicators")

col1,col2,col3 = st.columns(3)

gauge1 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=soil*100,
    title={'text':"Soil Quality"},
    gauge={'axis':{'range':[0,100]}}
))

gauge2 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=ndvi*100,
    title={'text':"Crop Health"},
    gauge={'axis':{'range':[0,100]}}
))

gauge3 = go.Figure(go.Indicator(
    mode="gauge+number",
    value=demand*100,
    title={'text':"Market Demand"},
    gauge={'axis':{'range':[0,100]}}
))

col1.plotly_chart(gauge1,use_container_width=True)
col2.plotly_chart(gauge2,use_container_width=True)
col3.plotly_chart(gauge3,use_container_width=True)

st.subheader("Model Feature Influence")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

fig = px.bar(
    importance,
    orientation="h",
    title="Feature Importance"
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("30 Day Price Forecast")

days = np.arange(1,31)

future = []

for d in days:

    temp = input_df.copy()

    temp["export_demand_index"] = demand + np.random.normal(0,0.05)
    temp["mandi_arrivals_qtl"] = arrivals + np.random.normal(0,300)

    future.append(temp)

future_df = pd.concat(future)

prices = model.predict(future_df)

forecast = pd.DataFrame({
    "Day":days,
    "Predicted Price":prices
})

fig2 = px.line(
    forecast,
    x="Day",
    y="Predicted Price",
    markers=True,
    title="Price Trend Forecast"
)

st.plotly_chart(fig2,use_container_width=True)
