import pydeck as pdk
import pandas as pd

def create_market_map():

    df=pd.DataFrame({
        "district":["Warangal","Karimnagar","Khammam","Nizamabad"],
        "lat":[17.978,18.438,17.247,18.672],
        "lon":[79.594,79.128,80.151,78.100],
        "price":[1750,1800,1700,1850],
        "rain":[750,720,810,690],
        "temp":[29,30,31,28]
    })

    layer=pdk.Layer(
        "ColumnLayer",
        df,
        get_position=["lon","lat"],
        get_elevation="price",
        radius=20000,
        elevation_scale=40,
        pickable=True
    )

    view=pdk.ViewState(
        latitude=17.9,
        longitude=79.3,
        zoom=6,
        pitch=45
    )

    deck=pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={
            "text":"District: {district}\nPrice: ₹{price}\nRainfall: {rain} mm\nTemp: {temp} °C"
        }
    )

    return deck
