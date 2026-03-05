import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from api_weather import get_weather
from api_market import get_mandi_prices
from map_visualization import create_market_map

st.set_page_config(page_title="AI Agricultural Market Intelligence", layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("AI Agricultural Market Intelligence Platform")
st.caption("Live Data + Machine Learning Crop Price Forecasting")

st.sidebar.header("Farm Inputs")

crop = st.sidebar.selectbox(
    "Crop",
    ["Paddy","Maize","Cotton","Turmeric","Chilli"]
)

city = st.sidebar.selectbox(
    "District Weather Source",
    ["Hyderabad","Warangal","Karimnagar","Khammam","Nizamabad"]
)

weather = get_weather(city) or {
    "temperature":30,
    "rainfall":0,
    "humidity":60
}

rainfall = weather["rainfall"]
temperature = weather["temperature"]

yield_q = st.sidebar.slider("Yield (quintal/acre)",5,40,18)
ndvi = st.sidebar.slider("NDVI Vegetation Index",0.3,0.9,0.6)
demand = st.sidebar.slider("Export Demand Index",0.1,1.0,0.5)
arrivals = st.sidebar.slider("Mandi Arrivals (qtl)",2000,9000,5000)

neighbor_price = st.sidebar.slider(
    "Neighbor State Price (₹/qtl)",
    1200,2500,1800
)

price_lag = st.sidebar.slider(
    "Last Week Price (₹/qtl)",
    1200,2500,1700
)

soil = st.sidebar.slider("Soil Fertility Index",0.3,0.9,0.6)

st.subheader("Live Weather Conditions")

c1,c2,c3 = st.columns(3)

c1.metric("Temperature °C",round(temperature,2))
c2.metric("Rainfall mm",round(rainfall,2))
c3.metric("Humidity %",weather["humidity"])

input_data = {f:0 for f in features}

if "rainfall_mm" in input_data:
    input_data["rainfall_mm"]=rainfall

if "avg_temp_c" in input_data:
    input_data["avg_temp_c"]=temperature

if "yield_qtl_per_acre" in input_data:
    input_data["yield_qtl_per_acre"]=yield_q

if "ndvi_satellite_index" in input_data:
    input_data["ndvi_satellite_index"]=ndvi

if "export_demand_index" in input_data:
    input_data["export_demand_index"]=demand

if "mandi_arrivals_qtl" in input_data:
    input_data["mandi_arrivals_qtl"]=arrivals

if "neighbor_state_price_rs" in input_data:
    input_data["neighbor_state_price_rs"]=neighbor_price

if "price_lag_7d" in input_data:
    input_data["price_lag_7d"]=price_lag

if "soil_fertility_index" in input_data:
    input_data["soil_fertility_index"]=soil

crop_feature="crop_"+crop.lower()

if crop_feature in input_data:
    input_data[crop_feature]=1

input_df=pd.DataFrame([input_data])
input_df=input_df.reindex(columns=features,fill_value=0)

if st.button("Predict Crop Price"):

    prediction=model.predict(input_df)[0]

    best=prediction*1.08
    worst=prediction*0.92

    m1,m2,m3=st.columns(3)

    m1.metric("Predicted Price ₹/qtl",round(prediction,2))
    m2.metric("Best Case ₹",round(best,2))
    m3.metric("Worst Case ₹",round(worst,2))

    st.subheader("Scenario Simulation")

    scenario_df=pd.DataFrame({
        "Scenario":[
            "High Demand",
            "Low Arrivals",
            "Rainfall Shock",
            "Base Case"
        ],
        "Price":[
            prediction*1.1,
            prediction*1.12,
            prediction*0.95,
            prediction
        ]
    })

    fig=px.bar(scenario_df,x="Scenario",y="Price")

    st.plotly_chart(fig,use_container_width=True)

st.subheader("Live Mandi Prices")

mandi_df=get_mandi_prices()

st.dataframe(
    mandi_df[
        ["commodity","state","district","market","modal_price"]
    ].head(20)
)

st.subheader("District Market Map")

deck=create_market_map()

st.pydeck_chart(deck)

st.subheader("AI Model Explanation")

explainer=shap.TreeExplainer(model)

shap_values=explainer.shap_values(input_df)

exp=shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=input_df.iloc[0],
    feature_names=input_df.columns
)

plt.figure()

shap.plots.waterfall(exp,show=False)

st.pyplot(plt.gcf())
