def compute_et(temp, humidity):
    return max(0, 0.1 * (temp - humidity/100))