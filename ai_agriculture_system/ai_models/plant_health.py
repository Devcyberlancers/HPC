def plant_health(soil_moisture, pH, EC, N, P, K):

    moisture_score = soil_moisture

    ph_score = 1 - abs(pH - 7) / 7

    ec_score = 1 - (EC / 5)

    nutrient_score = (N + P + K) / 300  # normalize

    health = (
        moisture_score * 0.3 +
        ph_score * 0.2 +
        ec_score * 0.2 +
        nutrient_score * 0.3
    )

    return round(health * 100, 2)