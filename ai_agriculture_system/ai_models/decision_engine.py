def decide_irrigation(sensor, weather, plant_health):

    irrigation = 0.5  # base

    # 💧 moisture influence
    if sensor["soil_moisture"] < 0.3:
        irrigation += 0.3

    # 🌡️ temperature influence
    if weather["temperature"] > 30:
        irrigation += 0.2

    # 🌱 plant health influence
    if plant_health < 60:
        irrigation += 0.2

    # 🌧️ rain influence
    if weather["rain"] > 0:
        irrigation -= 0.5

    # 🔮 future awareness
    if "predicted_moisture" in sensor and sensor["predicted_moisture"] < 0.3:
       irrigation += 0.2

    return round(max(0, min(irrigation, 1)), 2)