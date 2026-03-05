import requests

city_coords = {
    "Hyderabad": (17.385, 78.486),
    "Warangal": (17.978, 79.594),
    "Karimnagar": (18.438, 79.128),
    "Khammam": (17.247, 80.151),
    "Nizamabad": (18.672, 78.100)
}

def get_weather(city):

    lat, lon = city_coords[city]

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    r = requests.get(url)

    data = r.json()

    weather = {
        "temperature": data["current_weather"]["temperature"],
        "wind": data["current_weather"]["windspeed"],
        "rainfall": 0,
        "humidity": 60
    }

    return weather
