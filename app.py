import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
import pydeck as pdk

st.set_page_config(page_title="AI Agriculture Intelligence",layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🌾 AI Agricultural Intelligence Platform")

st.write("Real-time crop price forecasting using machine learning")

col1,col2 = st.columns([1,2])

with col1:

    crop = st.selectbox("Crop",["Paddy","Maize","Cotton","Turmeric","Chilli"])

    rainfall = st.slider("Rainfall (mm)",0,2000,850)

    temp = st.slider("Temperature (°C)",10,45,30)

    yield_q = st.slider("Yield (quintal/acre)",5,40,18)

    ndvi = st.slider("Satellite NDVI Index",0.2,0.9,0.6)

    demand = st.slider("Export Demand Index",0.0,1.0,0.5)

    mandi = st.slider("Mandi Arrivals",500,10000,5000)

    input_data = {f:0 for f in features}

    input_data["rainfall_mm"]=rainfall
    input_data["avg_temp_c"]=temp
    input_data["yield_qtl_per_acre"]=yield_q
    input_data["ndvi_satellite_index"]=ndvi
    input_data["export_demand_index"]=demand
    input_data["mandi_arrivals_qtl"]=mandi

    crop_feature="crop_"+crop.lower()

    if crop_feature in input_data:
        input_data[crop_feature]=1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features,fill_value=0)

    if st.button("Predict Crop Price"):

        price = model.predict(input_df)[0]

        st.metric("Predicted Price ₹ / Quintal",f"{price:.2f}")

with col2:

    st.subheader("Feature Interaction")

    chart_df = pd.DataFrame({
        "Feature":["Rainfall","Temperature","Yield","NDVI","Demand","Arrivals"],
        "Value":[rainfall,temp,yield_q,ndvi*100,demand*100,mandi]
    })

    fig = px.bar(chart_df,x="Feature",y="Value",color="Value")

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("3D Market Map")

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

st.subheader("30-Day Price Forecast")

future_days = np.arange(1,31)

future_prices = model.predict(
    pd.concat([input_df]*30,ignore_index=True)
)

forecast_df = pd.DataFrame({
    "Day":future_days,
    "Predicted Price":future_prices
})

fig2 = px.line(forecast_df,x="Day",y="Predicted Price")

st.plotly_chart(fig2,use_container_width=True)

st.subheader("AI Explanation")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(input_df)

fig3 = shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
)

st.pyplot(fig3)
