def nutrient_prediction(plant_health):
    if plant_health == "CRITICAL":
        return "LOW_N"
    elif plant_health == "STRESSED":
        return "MEDIUM_N"
    else:
        return "SUFFICIENT"


def water_forecasting(current_moisture):
    return current_moisture + 5


def vision_ai_simulation(plant_health):
    if plant_health == "CRITICAL":
        return "DISEASED"
    elif plant_health == "STRESSED":
        return "WEAK"
    else:
        return "GOOD"