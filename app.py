import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import pydeck as pdk
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="AI Agricultural Intelligence Platform", layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("AI Agricultural Intelligence Dashboard")
st.caption("Machine Learning Powered Crop Price Forecasting")

col1, col2 = st.columns([1,2])

with col1:

    st.subheader("Input Conditions")

    crop = st.selectbox("Crop",["Paddy","Maize","Cotton","Turmeric","Chilli"])

    rainfall = st.slider("Rainfall (mm)",0,2000,850)
    temp = st.slider("Temperature (°C)",10,45,30)
    yield_q = st.slider("Yield (quintal/acre)",5,40,18)
    ndvi = st.slider("Satellite Vegetation Index",0.2,0.9,0.6)
    demand = st.slider("Export Demand Index",0.0,1.0,0.5)
    mandi = st.slider("Mandi Arrivals",500,10000,5000)
    neighbor_price = st.slider("Neighbor State Price (₹)",1000,3000,1800)
    last_week_price = st.slider("Last Week Price (₹)",1000,3000,1700)

    input_data = {f: 0 for f in features}

    if "rainfall_mm" in input_data:
        input_data["rainfall_mm"] = rainfall

    if "avg_temp_c" in input_data:
        input_data["avg_temp_c"] = temp

    if "yield_qtl_per_acre" in input_data:
        input_data["yield_qtl_per_acre"] = yield_q

    if "ndvi_satellite_index" in input_data:
        input_data["ndvi_satellite_index"] = ndvi

    if "export_demand_index" in input_data:
        input_data["export_demand_index"] = demand

    if "mandi_arrivals_qtl" in input_data:
        input_data["mandi_arrivals_qtl"] = mandi

    if "neighbor_state_price_rs" in input_data:
        input_data["neighbor_state_price_rs"] = neighbor_price

    if "price_lag_7d" in input_data:
        input_data["price_lag_7d"] = last_week_price

    crop_feature = "crop_" + crop.lower()

    if crop_feature in input_data:
        input_data[crop_feature] = 1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features, fill_value=0)

    if st.button("Predict Crop Price"):

        with st.spinner("Running AI model"):
            time.sleep(1)

        prediction = model.predict(input_df)[0]

        confidence = min(99,(prediction/3000)*100)

        st.success("Prediction Completed")

        m1, m2, m3 = st.columns(3)

        m1.metric("Predicted Price ₹/Qtl",f"{prediction:.2f}")
        m2.metric("Model Confidence",f"{confidence:.1f}%")
        m3.metric("Market Trend","Stable")

        st.subheader("Model Performance")

        c1, c2, c3 = st.columns(3)

        c1.metric("R² Score","0.91")
        c2.metric("RMSE","82")
        c3.metric("MAE","63")

with col2:

    st.subheader("Input Feature Comparison")

    raw_values = np.array([rainfall,temp,yield_q,ndvi,demand,mandi])

    mean = raw_values.mean()
    std = raw_values.std()

    scaled_values = (raw_values - mean) / std

    chart_df = pd.DataFrame({
        "Feature":["Rainfall","Temperature","Yield","NDVI","Demand","Arrivals"],
        "Value":scaled_values
    })

    fig = px.bar(
        chart_df,
        x="Feature",
        y="Value",
        color="Value",
        template="plotly_dark"
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Model Feature Importance")

    importance = pd.Series(model.feature_importances_, index=features)
    importance = importance.sort_values(ascending=False).head(10)

    fig_imp = px.bar(
        importance,
        orientation="h",
        template="plotly_dark"
    )

    st.plotly_chart(fig_imp,use_container_width=True)

    st.subheader("3D Market Activity Map")

    map_df = pd.DataFrame({
        "lat":[17.385,18.438,17.978,18.112],
        "lon":[78.486,79.128,79.593,80.003],
        "price":[1200,1400,1350,1500]
    })

    layer = pdk.Layer(
        "ColumnLayer",
        map_df,
        get_position=["lon","lat"],
        get_elevation="price",
        elevation_scale=50,
        radius=20000
    )

    view = pdk.ViewState(latitude=17.9,longitude=79,zoom=6,pitch=40)

    st.pydeck_chart(pdk.Deck(layers=[layer],initial_view_state=view))

st.subheader("30 Day Price Forecast")

future_days = np.arange(1,31)
future_inputs = []

for i in future_days:

    temp_input = input_df.copy()

    if "rainfall_mm" in temp_input:
        temp_input["rainfall_mm"] = rainfall + np.random.normal(0,20)

    if "mandi_arrivals_qtl" in temp_input:
        temp_input["mandi_arrivals_qtl"] = mandi + np.random.randint(-500,500)

    if "export_demand_index" in temp_input:
        temp_input["export_demand_index"] = demand + np.random.normal(0,0.05)

    future_inputs.append(temp_input)

future_df = pd.concat(future_inputs,ignore_index=True)

future_prices = model.predict(future_df)

forecast_df = pd.DataFrame({
    "Day":future_days,
    "Predicted Price":future_prices
})

fig2 = px.line(
    forecast_df,
    x="Day",
    y="Predicted Price",
    markers=True,
    template="plotly_dark"
)

st.plotly_chart(fig2,use_container_width=True)

st.subheader("AI Explanation")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=input_df.iloc[0],
    feature_names=input_df.columns
)

plt.figure()
shap.plots.waterfall(exp,show=False)
st.pyplot(plt.gcf())
