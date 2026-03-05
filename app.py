import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import pydeck as pdk
import numpy as np

st.set_page_config(page_title="AI Crop Price Forecast",layout="wide")

model = joblib.load("xgb_crop_price_model.pkl")
features = joblib.load("model_features.pkl")

st.markdown("""
<style>
body{
background-image:url("https://images.unsplash.com/photo-1500382017468-9049fed747ef");
background-size:cover;
}
.big{
font-size:40px;
font-weight:bold;
color:white;
}
</style>
""",unsafe_allow_html=True)

st.markdown('<p class="big">🌾 AI Crop Price Forecast Platform</p>',unsafe_allow_html=True)

col1,col2=st.columns([1,2])

with col1:

    crop = st.selectbox("Crop",["Paddy","Cotton","Maize","Turmeric","Chilli"])
    rainfall = st.slider("Rainfall",0,2000,800)
    temp = st.slider("Temperature",10,45,30)
    yield_q = st.slider("Yield",5,40,15)
    demand = st.slider("Export Demand",0.0,1.0,0.5)
    mandi = st.slider("Mandi Arrivals",500,10000,5000)

    input_data = {f:0 for f in features}

    input_data["rainfall_mm"]=rainfall
    input_data["avg_temp_c"]=temp
    input_data["yield_qtl_per_acre"]=yield_q
    input_data["export_demand_index"]=demand
    input_data["mandi_arrivals_qtl"]=mandi

    crop_feature = "crop_"+crop.lower()
    if crop_feature in input_data:
        input_data[crop_feature]=1

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=features,fill_value=0)

    if st.button("Predict Price"):

        price = model.predict(input_df)[0]

        st.metric("Predicted Price ₹/quintal",f"{price:.2f}")

with col2:

    st.subheader("Feature Interaction")

    data = pd.DataFrame({
    "Feature":["Rainfall","Temperature","Yield","Demand","Arrivals"],
    "Value":[rainfall,temp,yield_q,demand*100,mandi]
    })

    fig = px.bar(data,x="Feature",y="Value",color="Value")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("3D Agricultural Map")

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
