from fastapi import FastAPI
import asyncio

from backend.config import SENSORS
from simulation.sensor_simulator import get_sensor_data
from backend.weather_service import get_weather
from ai_models.plant_health import plant_health
from genai.explanation_engine import generate_explanation
from ai_models.rl_agent import get_actions  
from genai.rag_engine import refine_decision

app = FastAPI()


@app.get("/status")
async def status():

    states = []
    zone_data = []

    # 🔹 STEP 1 — Collect data from all sensors
    for sensor in SENSORS:

        weather = get_weather(sensor["lat"], sensor["lon"])

        # initial sensor reading (no irrigation yet)
        data = get_sensor_data(weather, irrigation=0)
        
        # 🔥 Build RL state
        state = [
            data["soil_moisture"],
            weather["temperature"],
            weather["humidity"],
            data["N"],
            data["P"],
            data["K"]
        ]

        states.append(state)

        zone_data.append({
            "id": sensor["id"],
            "sensor": data,
            "weather": weather
        })

    # 🔥 STEP 2 — RL decision (batch)
    actions = get_actions(states)

    print("RL OUTPUT:", actions)  # ✅ debug log

    results = []

    for i, zone in enumerate(zone_data):

        rl_irrigation = float(actions[i])

    # ✅ FIRST: simulate irrigation
        updated_data = get_sensor_data(zone["weather"], rl_irrigation)

    # ✅ SECOND: compute plant health
        health = plant_health(
            updated_data["soil_moisture"],
            updated_data["pH"],
            updated_data["EC"],
            updated_data["N"],
            updated_data["P"],
            updated_data["K"]
        )

    # ✅ THIRD: apply RAG
        try:
           irrigation, context = refine_decision(
            updated_data,
            zone["weather"],
            health,
            rl_irrigation
        )
        except Exception as e:
            print("RAG FAILED:", e)
            irrigation = rl_irrigation
            context = []

    # ✅ explanation
        explanation = generate_explanation(
            updated_data,
            zone["weather"],
            health,
            irrigation,
            context
        )

        results.append({
            "id": zone["id"],
            "sensor": updated_data,
            "weather": zone["weather"],
            "plant_health": health,
            "irrigation": irrigation,
            "explanation": explanation,
            "rag_context": context
        })
    return {
        "zones": results
    }