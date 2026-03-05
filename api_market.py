import pandas as pd

def get_mandi_prices():

    url = "https://raw.githubusercontent.com/plotly/datasets/master/india_agmarknet_prices.csv"

    df = pd.read_csv(url)

    df = df.rename(columns={
        "Commodity": "commodity",
        "State": "state",
        "District": "district",
        "Market": "market",
        "Modal Price": "modal_price"
    })

    return df
