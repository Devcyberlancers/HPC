def predict_moisture(current, et):
    return current - et

def predict_moisture(current_moisture, temperature, humidity):

    # 🌡️ evaporation factor
    evaporation = 0.05 * (temperature / 30)

    # 💧 humidity reduces evaporation
    humidity_factor = (100 - humidity) / 100

    predicted = current_moisture - (evaporation * humidity_factor)

    return round(max(0, predicted), 3)