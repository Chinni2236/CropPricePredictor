import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Crop Price Prediction",
    page_icon="🌾",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_crop_price_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, features = load_model()

# -------------------- SIDEBAR STYLE --------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f172a;
}
</style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🌾 Crop Price Prediction System")
st.caption("AI-powered crop price forecasting for Telangana")

st.divider()

# -------------------- INITIALIZE INPUT DATA --------------------
input_data = {feature: 0 for feature in features}

# -------------------- 🌾 CROP SELECTION (FIRST) --------------------
crop_features = [f for f in features if f.startswith("crop_")]

# Actual crop names from dataset (verified)
all_crops = ["Paddy", "Maize", "Chilli", "Turmeric", "Cotton"]

selected_crop = st.sidebar.selectbox(
    "🌾 Crop Type",
    all_crops
)

# Reset all crop_* columns
for f in crop_features:
    input_data[f] = 0

# Handle drop_first=True safely
selected_feature = "crop_" + selected_crop.lower()

if selected_feature in input_data:
    input_data[selected_feature] = 1
# else: baseline crop (dropped during get_dummies)

st.sidebar.divider()

# -------------------- 📅 TIME & CLIMATE --------------------
with st.sidebar.expander("📅 Time & Climate", expanded=True):
    if "year" in input_data:
        input_data["year"] = st.number_input("Year", 2015, 2035, 2025)

    if "rainfall_mm" in input_data:
        input_data["rainfall_mm"] = st.slider("Rainfall (mm)", 0, 3000, 800)

    if "rainfall_deviation_pct" in input_data:
        input_data["rainfall_deviation_pct"] = st.slider(
            "Rainfall Deviation (%)", -100, 100, 0
        )

    if "avg_temp_c" in input_data:
        input_data["avg_temp_c"] = st.slider(
            "Average Temperature (°C)", 10, 45, 30
        )

# -------------------- 🌾 FARM & SOIL --------------------
with st.sidebar.expander("🌾 Farm & Soil", expanded=True):
    if "yield_qtl_per_acre" in input_data:
        input_data["yield_qtl_per_acre"] = st.slider(
            "Yield (quintal/acre)", 0.0, 50.0, 15.0
        )

    if "soil_fertility_index" in input_data:
        input_data["soil_fertility_index"] = st.slider(
            "Soil Fertility Index", 0.0, 1.0, 0.6
        )

    if "irrigated" in input_data:
        irrigated = st.radio("Irrigation Available?", ["No", "Yes"])
        input_data["irrigated"] = 1 if irrigated == "Yes" else 0

# -------------------- 💰 COST FACTORS --------------------
with st.sidebar.expander("💰 Cost Factors", expanded=True):
    if "fertilizer_cost_rs_per_acre" in input_data:
        input_data["fertilizer_cost_rs_per_acre"] = st.slider(
            "Fertilizer Cost (₹/acre)", 0, 20000, 5000
        )

    if "labor_cost_rs_per_acre" in input_data:
        input_data["labor_cost_rs_per_acre"] = st.slider(
            "Labor Cost (₹/acre)", 0, 30000, 8000
        )

    if "diesel_price_rs_per_l" in input_data:
        input_data["diesel_price_rs_per_l"] = st.slider(
            "Diesel Price (₹/L)", 60, 120, 90
        )

# -------------------- CREATE MODEL INPUT --------------------
input_df = pd.DataFrame(
    [[input_data[f] for f in features]],
    columns=features
)

# -------------------- PREDICTION --------------------
if st.button("📊 Predict Price"):
    prediction = model.predict(input_df)[0]

    st.metric(
        label=f"💰 Predicted {selected_crop} Price (₹ per quintal)",
        value=f"{prediction:,.2f}"
    )

    st.divider()

    feature_df = pd.DataFrame({
        "Feature": features,
        "Value": input_df.iloc[0].values
    })

    fig = px.bar(
        feature_df,
        x="Feature",
        y="Value",
        title="Input Feature Values Used for Prediction",
        color="Value"
    )

    st.plotly_chart(fig, use_container_width=True)

# -------------------- FOOTER --------------------
st.divider()
st.caption("Built with ❤️ using Machine Learning")
