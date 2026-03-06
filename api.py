from flask import Flask, jsonify
import random
import time

app = Flask(__name__)

FIELDS = ["f1", "f2", "f3", "f4"]
NPK_TYPES = ["nob", "npk", "soil", "leaf"]


def r_bool(p_true=0.5) -> bool:
    return random.random() < p_true


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gen_state():
    tank_level = round(random.uniform(0, 100), 1)

    fields = {}
    for f in FIELDS:
        water_level = round(random.uniform(0, 10), 1)
        moisture = round(random.uniform(0, 100), 1)
        ph = round(random.uniform(5.5, 8.5), 2)

        irrigation = r_bool(0.5)
        drain = r_bool(0.2) if not irrigation else r_bool(0.1)

        acid = ph > 7.6 and r_bool(0.3)
        base = ph < 6.2 and r_bool(0.3)
        if acid and base:
            base = False

        fields[f] = {
            "water_level": water_level,
            "moisture": moisture,
            "ph": ph,
            "irrigation": irrigation,
            "drain": drain,
            "acid": acid,
            "base": base,
        }

    chosen_field = random.choice(FIELDS)
    ts_ms = int(time.time() * 1000)

    npk_data = {
        "N": random.randint(0, 255),
        "P": random.randint(0, 255),
        "K": random.randint(0, 255),
        "T": random.randint(-10, 60),
        "H": random.randint(0, 100),
        "PH": random.randint(0, 14),
        "R": random.randint(0, 1000),
    }

    payload = {
        "ok": True,
        "last": {
            "tank": {
                "name": "main",
                "level": tank_level,
                "motor": r_bool(0.4),
                "rain": r_bool(0.2),
            },
            "fields": fields,
            "npk": {
                "type": random.choice(NPK_TYPES),
                "field": chosen_field,
                "ts": ts_ms,
                "data": npk_data,
            },
        },
    }
    return payload


@app.get("/state")
def state():
    return jsonify(gen_state())


if __name__ == "__main__":
    print("[API] Serving random field state on http://127.0.0.1:18085/state")
    app.run(host="127.0.0.1", port=18085, debug=False)
