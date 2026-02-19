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

st.sidebar.header("Input Parameters")

input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

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
