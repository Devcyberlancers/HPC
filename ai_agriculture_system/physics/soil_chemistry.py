def ph_factor(ph):
    return 1 - abs(ph - 7) / 7

def ec_penalty(ec):
    return ec / 5