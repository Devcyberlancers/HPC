NUM_SENSORS = 3

SENSORS = [
    {"id": i, "lat": 12.97 + i*0.01, "lon": 77.59 + i*0.01}
    for i in range(NUM_SENSORS)
]