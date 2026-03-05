import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Crop Market Intelligence", page_icon="🌾", layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.markdown("""
<style>
.main-title{
font-size:42px;
font-weight:700;
}
.card{
background-color:#111827;
padding:25px;
border-radius:15px;
}
.metric{
font-size:30px;
font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌾 AI Agricultural Market Intelligence</div>', unsafe_allow_html=True)
st.caption("Smart Crop Price Prediction & Farm Profit Insights")

st.sidebar.header("Farm Information")

crop = st.sidebar.selectbox("Crop", ["Paddy","Maize","Cotton","Turmeric","Chilli"])

rainfall = st.sidebar.slider("Season Rainfall (mm)", 0, 1500, 800)

temperature = st.sidebar.slider("Temperature (°C)", 10, 45, 30)

soil_type = st.sidebar.selectbox("Soil Quality", ["Poor","Average","Fertile"])

irrigation = st.sidebar.selectbox("Irrigation Level", ["Low","Medium","High"])

farm_size = st.sidebar.slider("Farm Size (Acres)", 1, 50, 10)

market_demand = st.sidebar.selectbox("Market Demand", ["Low","Medium","High"])

soil_map = {
"Poor":0.4,
"Average":0.6,
"Fertile":0.85
}

soil = soil_map[soil_type]

irrigation_map = {
"Low":0.45,
"Medium":0.6,
"High":0.8
}

ndvi = irrigation_map[irrigation]

demand_map = {
"Low":0.35,
"Medium":0.6,
"High":0.85
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
input_df = input_df.reindex(columns=features, fill_value=0)

prediction = model.predict(input_df)[0]

best = prediction*1.1
worst = prediction*0.9

col1,col2,col3 = st.columns(3)

col1.metric("Predicted Market Price ₹/Qtl", f"{prediction:,.0f}")
col2.metric("Best Case Market Price ₹", f"{best:,.0f}")
col3.metric("Worst Case Market Price ₹", f"{worst:,.0f}")

st.markdown("---")

st.subheader("Farm Condition Indicators")

c1,c2,c3 = st.columns(3)

soil_gauge = go.Figure(go.Indicator(
mode="gauge+number",
value=soil*100,
title={'text':"Soil Quality"},
gauge={'axis':{'range':[0,100]}}
))

health_gauge = go.Figure(go.Indicator(
mode="gauge+number",
value=ndvi*100,
title={'text':"Crop Health"},
gauge={'axis':{'range':[0,100]}}
))

demand_gauge = go.Figure(go.Indicator(
mode="gauge+number",
value=demand*100,
title={'text':"Market Demand"},
gauge={'axis':{'range':[0,100]}}
))

c1.plotly_chart(soil_gauge, use_container_width=True)
c2.plotly_chart(health_gauge, use_container_width=True)
c3.plotly_chart(demand_gauge, use_container_width=True)

st.markdown("---")

st.subheader("Farm Production Estimate")

production = yield_q * farm_size
revenue = production * prediction

p1,p2,p3 = st.columns(3)

p1.metric("Expected Yield (Qtl)", f"{production:,.1f}")
p2.metric("Farm Revenue ₹", f"{revenue:,.0f}")
p3.metric("Revenue per Acre ₹", f"{revenue/farm_size:,.0f}")

st.markdown("---")

st.subheader("30 Day Market Forecast")

days = np.arange(1,31)

future = []

for d in days:
    temp = input_df.copy()
    temp["export_demand_index"] = demand + np.random.normal(0,0.04)
    temp["mandi_arrivals_qtl"] = arrivals + np.random.normal(0,300)
    future.append(temp)

future_df = pd.concat(future)

prices = model.predict(future_df)

forecast = pd.DataFrame({
"Day":days,
"Predicted Price":prices
})

fig = px.line(
forecast,
x="Day",
y="Predicted Price",
markers=True
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.subheader("Model Feature Influence")

importance = pd.Series(
model.feature_importances_,
index=features
).sort_values()

fig2 = px.bar(
importance,
orientation="h"
)

st.plotly_chart(fig2, use_container_width=True)
