import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="AI Crop Price Intelligence",layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("AI Agricultural Market Intelligence Platform")

st.sidebar.header("Farm Conditions")

crop = st.sidebar.selectbox(
    "Crop Type",
    ["Paddy","Maize","Cotton","Turmeric","Chilli"]
)

rainfall = st.sidebar.slider("Rainfall (mm)",0,1500,800)
temperature = st.sidebar.slider("Temperature (°C)",10,45,30)
ndvi = st.sidebar.slider("NDVI Vegetation Index",0.3,0.9,0.6)
soil = st.sidebar.slider("Soil Fertility Index",0.3,0.9,0.6)
demand = st.sidebar.slider("Export Demand Index",0.1,1.0,0.5)
arrivals = st.sidebar.slider("Mandi Arrivals (qtl)",2000,10000,5000)

yield_q = (
    rainfall*0.015 +
    ndvi*15 +
    soil*10
)

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
    col = "crop_"+c
    input_data[col] = 1 if crop==c else 0

input_df = pd.DataFrame([input_data])

input_df = input_df.reindex(columns=features,fill_value=0)

prediction = model.predict(input_df)[0]

best_case = prediction*1.08
worst_case = prediction*0.92

c1,c2,c3 = st.columns(3)

c1.metric("Predicted Price ₹/Qtl",f"{prediction:.2f}")
c2.metric("Best Case ₹",f"{best_case:.2f}")
c3.metric("Worst Case ₹",f"{worst_case:.2f}")

st.subheader("Feature Importance")

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values()

fig = px.bar(
    importance,
    orientation="h"
)

st.plotly_chart(fig,use_container_width=True)

st.subheader("30 Day Price Forecast")

days = np.arange(1,31)

future = []

for d in days:

    temp = input_df.copy()

    temp["export_demand_index"] = demand + np.random.normal(0,0.05)
    temp["mandi_arrivals_qtl"] = arrivals + np.random.normal(0,400)

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
    markers=True
)

st.plotly_chart(fig2,use_container_width=True)
