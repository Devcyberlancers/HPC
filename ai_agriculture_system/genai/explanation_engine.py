import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GENAI_API_KEY")


def generate_explanation(sensor, weather, plant_health, irrigation, context):

    prompt = f"""
You are an AI agriculture expert.

Analyze the situation and explain the irrigation decision.

DATA:
- Soil Moisture: {sensor['soil_moisture']}
- Temperature: {weather['temperature']}
- Humidity: {weather['humidity']}
- Plant Health: {plant_health}
- Irrigation: {irrigation}

KNOWLEDGE:
{context}

Explain clearly WHY this irrigation is recommended.
Keep it simple and human-friendly.
"""

    try:
        print("🔥 Calling GenAI...")

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "AI Agriculture System"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
        )

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        else:
            print("API ERROR:", result)
            raise Exception("Invalid response")

    except Exception as e:
        print("⚠️ GenAI failed:", e)

        # 🔄 FALLBACK (rule-based)
        reasons = []

        if sensor["soil_moisture"] < 0.3:
            reasons.append("low soil moisture")

        if weather["temperature"] > 30:
            reasons.append("high temperature")

        if plant_health < 60:
            reasons.append("poor plant health")

        if weather["rain"] > 0:
            reasons.append("rain detected")

        if not reasons:
            return "Conditions are stable. Moderate irrigation applied."

        explanation = " + ".join(reasons)

        return f"{explanation} → irrigation adjusted to {round(irrigation, 2)}"