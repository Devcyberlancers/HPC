from simulation.sensor_simulator import get_sensor_data
from backend.weather_service import get_weather

from physics.evapotranspiration import compute_et
from ai_models.rl_agent import get_action
from ai_models.soil_model import soil_health
from ai_models.decision_engine import final_decision

def run_pipeline():

    sensor = get_sensor_data()
    weather = get_weather(12.97, 77.59)

    et = compute_et(weather["temperature"], weather["humidity"])

    action = get_action(sensor)

    soil_score = soil_health(
        sensor["pH"], sensor["EC"],
        sensor["N"], sensor["P"], sensor["K"]
    )

    final = final_decision(action, weather, soil_score)

    return {
        "sensor": sensor,
        "weather": weather,
        "irrigation": final,
        "soil_health": soil_score
    }