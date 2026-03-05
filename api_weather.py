import requests

API_KEY = "YOUR_OPENWEATHER_API_KEY"

def get_weather(city):

    url=f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    r=requests.get(url)

    data=r.json()

    weather={
        "temperature":data["main"]["temp"],
        "humidity":data["main"]["humidity"],
        "rainfall":data.get("rain",{}).get("1h",0),
        "wind":data["wind"]["speed"]
    }

    return weather
