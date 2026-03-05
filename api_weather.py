import requests

city_coords = {
    "Hyderabad": (17.385, 78.486),
    "Warangal": (17.978, 79.594),
    "Karimnagar": (18.438, 79.128),
    "Khammam": (17.247, 80.151),
    "Nizamabad": (18.672, 78.100)
}

def get_weather(city):

    lat, lon = city_coords.get(city, (17.385, 78.486))

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

    try:

        response = requests.get(url, timeout=10)
        data = response.json()

        current = data.get("current_weather", {})

        weather = {
            "temperature": current.get("temperature", 30),
            "rainfall": 0,
            "humidity": 60,
            "wind": current.get("windspeed", 5)
        }

        return weather

    except Exception:

        return {
            "temperature": 30,
            "rainfall": 0,
            "humidity": 60,
            "wind": 5
        }
