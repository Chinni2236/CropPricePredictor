import pydeck as pdk
import pandas as pd

def create_market_map():

    df = pd.DataFrame({
        "district": ["Warangal","Karimnagar","Khammam","Nizamabad"],
        "lat": [17.978,18.438,17.247,18.672],
        "lon": [79.594,79.128,80.151,78.100],
        "price": [1750,1800,1700,1850],
        "rain": [720,750,690,780],
        "temp": [29,30,31,28],
        "yield": [18,20,17,19]
    })

    layer = pdk.Layer(
        "ColumnLayer",
        df,
        get_position=["lon","lat"],
        get_elevation="price",
        radius=25000,
        elevation_scale=40,
        pickable=True
    )

    view = pdk.ViewState(
        latitude=17.9,
        longitude=79.2,
        zoom=6,
        pitch=45
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={
            "text":
            "District: {district}\n"
            "Price: ₹{price}\n"
            "Rainfall: {rain} mm\n"
            "Temp: {temp} °C\n"
            "Yield: {yield} qtl"
        }
    )

    return deck
