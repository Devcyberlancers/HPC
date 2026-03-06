import os
import threading
import time
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from flask import Flask, Response, jsonify, render_template_string

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:18085/state"
API_TIMEOUT_SEC = 1.5

FIELDS_TO_USE = ["f1", "f2", "f3", "f4"]
FIELD_NAMES = ["F1", "F2", "F3", "F4"]

POLL_EVERY_SEC = 1.0
RENDER_EVERY_SEC = 1.0
MAX_POINTS = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIRTUAL_FIELDS_PER_REAL = 25000
MC_RUNS = 64
INNER_STEPS = 20

MOISTURE_NOISE_STD = 4.0
PH_NOISE_STD = 0.20

MOISTURE_STEP_NOISE_STD = 0.12
PH_STEP_NOISE_STD = 0.01

theta_wp = 0.12
theta_optL = 0.20
theta_optH = 0.35
theta_sat = 0.45

ph_min = 5.0
ph_optL = 6.0
ph_optH = 7.5
ph_max = 8.5

session = requests.Session()
lock = threading.Lock()
shared: Dict[str, Any] = {
    "src": "INIT",
    "updated_at": 0.0,
    "meta": {},
    "m": [50.0, 50.0, 50.0, 50.0],
    "ph": [6.8, 6.8, 6.8, 6.8],
    "mean_S": [0.0, 0.0, 0.0, 0.0],
    "p10_S": [0.0, 0.0, 0.0, 0.0],
    "p90_S": [0.0, 0.0, 0.0, 0.0],
    "t_hist": [],
    "mean_hist": [[], [], [], []],
    "p10_hist": [[], [], [], []],
    "p90_hist": [[], [], [], []],
    "png": b"",
    "perf": {
        "device": DEVICE,
        "last_compute_sec": 0.0,
        "virtual_fields_total": 4 * VIRTUAL_FIELDS_PER_REAL,
        "mc_runs": MC_RUNS,
        "inner_steps": INNER_STEPS,
    },
}


def waterlevel_to_moisture_percent(wl: float) -> float:
    return float(np.clip(wl * 10.0, 0.0, 100.0))


def moisture_percent_to_theta_torch(m_percent: torch.Tensor) -> torch.Tensor:
    return 0.05 + (m_percent / 100.0) * (0.45 - 0.05)


def f_moist_torch(th: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(th)

    m1 = (th > theta_wp) & (th < theta_optL)
    out[m1] = (th[m1] - theta_wp) / (theta_optL - theta_wp)

    m2 = (th >= theta_optL) & (th <= theta_optH)
    out[m2] = 1.0

    m3 = (th > theta_optH) & (th < theta_sat)
    out[m3] = (theta_sat - th[m3]) / (theta_sat - theta_optH)

    return out.clamp(0.0, 1.0)


def f_ph_torch(ph: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(ph)

    m1 = (ph > ph_min) & (ph < ph_optL)
    out[m1] = (ph[m1] - ph_min) / (ph_optL - ph_min)

    m2 = (ph >= ph_optL) & (ph <= ph_optH)
    out[m2] = 1.0

    m3 = (ph > ph_optH) & (ph < ph_max)
    out[m3] = (ph_max - ph[m3]) / (ph_max - ph_optH)

    return out.clamp(0.0, 1.0)


def fetch_fields():
    r = session.get(API_URL, timeout=API_TIMEOUT_SEC)
    r.raise_for_status()
    data = r.json()

    last = data.get("last", {})
    fields = last.get("fields", {})
    tank = last.get("tank", {})

    m_list, ph_list = [], []
    for k in FIELDS_TO_USE:
        f = fields.get(k, {})
        wl = float(f.get("water_level", 0.0))
        phv = float(f.get("ph", 6.8))
        m_list.append(waterlevel_to_moisture_percent(wl))
        ph_list.append(float(np.clip(phv, 4.0, 9.0)))

    meta = {
        "tank_name": tank.get("name"),
        "tank_level": tank.get("level"),
        "tank_motor": tank.get("motor"),
        "tank_rain": tank.get("rain"),
    }
    return m_list, ph_list, meta


def fig_to_png_bytes(fig):
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def render_stress_graph_png(
    t_hist: List[float],
    mean_hist: List[List[float]],
    p10_hist: List[List[float]],
    p90_hist: List[List[float]],
):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)

    for i, name in enumerate(FIELD_NAMES):
        t = np.asarray(t_hist, dtype=np.float32)
        mean_y = np.asarray(mean_hist[i], dtype=np.float32)
        p10_y = np.asarray(p10_hist[i], dtype=np.float32)
        p90_y = np.asarray(p90_hist[i], dtype=np.float32)

        ax.plot(t, mean_y, linewidth=2, label=f"{name} mean")
        ax.fill_between(t, p10_y, p90_y, alpha=0.15)

    ax.set_title("HPC Live Stress Graph (Mean with P10-P90 Band)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("stress (0..1)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    return fig_to_png_bytes(fig)


def compute_stress_hpc(m_base_list: List[float], ph_base_list: List[float]):
    t0 = time.time()

    m_base = torch.tensor(m_base_list, dtype=torch.float32, device=DEVICE).view(4, 1)
    ph_base = torch.tensor(ph_base_list, dtype=torch.float32, device=DEVICE).view(4, 1)

    V = VIRTUAL_FIELDS_PER_REAL
    M = MC_RUNS

    m_virtual = m_base.repeat(1, V)
    ph_virtual = ph_base.repeat(1, V)

    m_virtual = (m_virtual + torch.randn((4, V), device=DEVICE) * MOISTURE_NOISE_STD).clamp(0.0, 100.0)
    ph_virtual = (ph_virtual + torch.randn((4, V), device=DEVICE) * PH_NOISE_STD).clamp(4.0, 9.0)

    m_mc = m_virtual.unsqueeze(-1).repeat(1, 1, M)
    ph_mc = ph_virtual.unsqueeze(-1).repeat(1, 1, M)

    m_mc = (m_mc + torch.randn_like(m_mc) * 1.5).clamp(0.0, 100.0)
    ph_mc = (ph_mc + torch.randn_like(ph_mc) * 0.05).clamp(4.0, 9.0)

    for _ in range(INNER_STEPS):
        m_mc = m_mc - 0.03 + torch.randn_like(m_mc) * MOISTURE_STEP_NOISE_STD
        ph_mc = ph_mc + torch.randn_like(ph_mc) * PH_STEP_NOISE_STD

        m_mc.clamp_(0.0, 100.0)
        ph_mc.clamp_(4.0, 9.0)

    theta_mc = moisture_percent_to_theta_torch(m_mc)
    stress_mc = f_moist_torch(theta_mc) * f_ph_torch(ph_mc)

    stress_flat = stress_mc.reshape(4, -1)

    mean_stress = stress_flat.mean(dim=1)
    p10 = torch.quantile(stress_flat, 0.10, dim=1)
    p90 = torch.quantile(stress_flat, 0.90, dim=1)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0

    return (
        mean_stress.detach().cpu().numpy().tolist(),
        p10.detach().cpu().numpy().tolist(),
        p90.detach().cpu().numpy().tolist(),
        elapsed,
    )


def worker_loop():
    start = time.time()
    last_poll = 0.0
    last_render = 0.0

    while True:
        now = time.time()

        if now - last_poll >= POLL_EVERY_SEC:
            try:
                m_list, ph_list, meta = fetch_fields()
                src = "API"
            except Exception:
                with lock:
                    m_list = list(shared["m"])
                    ph_list = list(shared["ph"])
                meta = {}
                src = "LAST_KNOWN"

            mean_S, p10_S, p90_S, compute_sec = compute_stress_hpc(m_list, ph_list)

            with lock:
                shared["src"] = src
                shared["updated_at"] = now
                shared["meta"] = meta
                shared["m"] = list(m_list)
                shared["ph"] = list(ph_list)
                shared["mean_S"] = list(mean_S)
                shared["p10_S"] = list(p10_S)
                shared["p90_S"] = list(p90_S)

                t = now - start
                shared["t_hist"].append(float(t))
                for i in range(4):
                    shared["mean_hist"][i].append(float(mean_S[i]))
                    shared["p10_hist"][i].append(float(p10_S[i]))
                    shared["p90_hist"][i].append(float(p90_S[i]))

                if len(shared["t_hist"]) > MAX_POINTS:
                    shared["t_hist"] = shared["t_hist"][-MAX_POINTS:]
                    for i in range(4):
                        shared["mean_hist"][i] = shared["mean_hist"][i][-MAX_POINTS:]
                        shared["p10_hist"][i] = shared["p10_hist"][i][-MAX_POINTS:]
                        shared["p90_hist"][i] = shared["p90_hist"][i][-MAX_POINTS:]

                shared["perf"]["last_compute_sec"] = compute_sec

            last_poll = now

        if now - last_render >= RENDER_EVERY_SEC:
            with lock:
                png = render_stress_graph_png(
                    shared["t_hist"],
                    shared["mean_hist"],
                    shared["p10_hist"],
                    shared["p90_hist"],
                )
                shared["png"] = png
            last_render = now

        time.sleep(0.05)


app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>HPC Stress Graph</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 18px; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 10px; }
    pre { background: #f6f6f6; padding: 10px; border-radius: 8px; }
  </style>
</head>
<body>
  <h2>HPC Stress Graph</h2>
  <img id="img" src="/stress.png"/>
  <pre id="info">loading...</pre>
  <script>
    function refresh(){
      document.getElementById("img").src = "/stress.png?t=" + Date.now();
      fetch("/status.json")
        .then(r => r.json())
        .then(s => {
          document.getElementById("info").textContent = JSON.stringify(s, null, 2);
        });
    }
    setInterval(refresh, 1200);
    refresh();
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(HTML)


@app.get("/stress.png")
def stress_png():
    with lock:
        data = shared["png"]
    return Response(data if data else b"", mimetype="image/png")


@app.get("/status.json")
def status_json():
    with lock:
        out = {
            "src": shared["src"],
            "updated_at": shared["updated_at"],
            "meta": shared["meta"],
            "performance": shared["perf"],
            "fields": [
                {
                    "name": FIELD_NAMES[i],
                    "moisture_percent": shared["m"][i],
                    "ph": shared["ph"][i],
                    "mean_stress": shared["mean_S"][i],
                    "p10_stress": shared["p10_S"][i],
                    "p90_stress": shared["p90_S"][i],
                }
                for i in range(4)
            ],
        }
    return jsonify(out)


if __name__ == "__main__":
    print("========================================")
    print("HPC Stress Graph Server")
    print(f"Device: {DEVICE}")
    print(f"VIRTUAL_FIELDS_PER_REAL: {VIRTUAL_FIELDS_PER_REAL}")
    print(f"MC_RUNS: {MC_RUNS}")
    print(f"INNER_STEPS: {INNER_STEPS}")
    print(f"Total virtual fields: {4 * VIRTUAL_FIELDS_PER_REAL}")
    print("========================================")

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    PORT = int(os.environ.get("PORT", "8080"))
    print(f"[ROOT_STRESS] Open: http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
