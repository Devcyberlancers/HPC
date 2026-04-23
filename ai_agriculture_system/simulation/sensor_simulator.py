import random
import time

# GLOBAL STATE
soil_state = {
    "moisture": 0.5,
    "N": 80.0,
    "P": 50.0,
    "K": 60.0
}

def get_sensor_data(weather=None, irrigation=0):
    global soil_state

    # TIME-BASED STATE SWITCH (EVERY 5 SEC)
    t = int(time.time() / 5) % 3

    # WEATHER
    temp = weather["temperature"] if weather else 30
    humidity = weather["humidity"] if weather else 50
    rain = weather["rain"] if weather else 0

    # FORCE STATES
    if t == 0:
        # HEALTHY
        soil_state["moisture"] = 0.7
        soil_state["N"] = 90
        soil_state["P"] = 70
        soil_state["K"] = 80

    elif t == 1:
        # MODERATE
        soil_state["moisture"] = 0.5
        soil_state["N"] = 60
        soil_state["P"] = 40
        soil_state["K"] = 50

    else:
        # CRITICAL
        soil_state["moisture"] = 0.2
        soil_state["N"] = 30
        soil_state["P"] = 20
        soil_state["K"] = 25

    # pH variation
    pH = 6.5 + random.uniform(-0.3, 0.3)

    # EC
    EC = 1.5 + (soil_state["N"] + soil_state["P"] + soil_state["K"]) / 300.0

    return {
        "soil_moisture": round(soil_state["moisture"], 3),
        "pH": round(pH, 2),
        "EC": round(EC, 2),
        "N": soil_state["N"],
        "P": soil_state["P"],
        "K": soil_state["K"]
    }