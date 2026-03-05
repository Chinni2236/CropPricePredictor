import requests
import pandas as pd

API_KEY="YOUR_DATA_GOV_API_KEY"

def get_mandi_prices():

    url=f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&limit=100"

    r=requests.get(url)

    data=r.json()

    records=data["records"]

    df=pd.DataFrame(records)

    df["modal_price"]=df["modal_price"].astype(float)

    return df
