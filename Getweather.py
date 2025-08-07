import requests

weather_codes = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Drizzle: Light",
    61: "Rain: Slight", 63: "Rain: Moderate", 65: "Rain: Heavy",
    71: "Snow fall: Slight", 73: "Snow fall: Moderate", 75: "Snow fall: Heavy",
    95: "Thunderstorm: Slight or moderate"
}

def get_weather_info():
    loc = requests.get("https://ipinfo.io/json").json()
    lat, lon = loc["loc"].split(",")
    city, country = loc["city"], loc["country"]
    weather = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
    ).json()["current_weather"]
    desc = weather_codes.get(weather["weathercode"], "Unknown")
    return city, country, desc, weather["temperature"], "Day" if weather["is_day"] else "Night"

def build_prompt():
    city, country, desc, temp, tod = get_weather_info()
    return f"A {desc} landscape in {city}, {country} during {tod}, temperature {temp}°C, oil painting, 8K masterpiece"

if __name__ == "__main__":
    print(build_prompt())