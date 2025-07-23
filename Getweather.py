import requests

# Weather code mapping
weather_codes = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog", 51: "Drizzle: Light", 53: "Drizzle: Moderate",
    55: "Drizzle: Dense intensity", 56: "Freezing Drizzle: Light", 57: "Freezing Drizzle: Dense",
    61: "Rain: Slight", 63: "Rain: Moderate", 65: "Rain: Heavy",
    66: "Freezing Rain: Light", 67: "Freezing Rain: Heavy",
    71: "Snow fall: Slight", 73: "Snow fall: Moderate", 75: "Snow fall: Heavy",
    77: "Snow grains", 80: "Rain showers: Slight", 81: "Rain showers: Moderate",
    82: "Rain showers: Violent", 85: "Snow showers: Slight", 86: "Snow showers: Heavy",
    95: "Thunderstorm: Slight or moderate", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

def get_weather_description(code):
    return weather_codes.get(code, "Unknown weather code")

def get_location():
    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    loc = data["loc"].split(",")
    return {
        "latitude": float(loc[0]),
        "longitude": float(loc[1]),
        "city": data.get("city", "Unknown"),
        "region": data.get("region", ""),
        "country": data.get("country", "")}

def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True}
    

    response = requests.get(url, params=params)
    data = response.json()
    current_weather = data["current_weather"]
    description = get_weather_description(current_weather["weathercode"])
    day_status = "Day" if current_weather["is_day"] == 1 else "Night"
    return {
        "temperature": current_weather["temperature"],
        "windspeed": current_weather["windspeed"],
        "description": description,
        "time_of_day": day_status}

def get_weather_info():
    loc = get_location()
    weather = get_weather(loc["latitude"], loc["longitude"])
    return {
        "city": loc["city"],
        "country": loc["country"],
        "description": weather["description"],
        "temperature": weather["temperature"],
        "windspeed": weather["windspeed"],
        "time_of_day": weather["time_of_day"]
    }

if __name__ == "__main__":
    info = get_weather_info()
    print(info)
