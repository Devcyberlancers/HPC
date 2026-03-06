import io
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from flask import Flask, Response

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:18085/state"
FIELDS = ["f1", "f2", "f3", "f4"]

HOST = "0.0.0.0"
PORT = 8088

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DT = 0.10
H, W = 206, 206

API_TIMEOUT_SEC = 1.5
API_POLL_EVERY_SEC = 2.0

D_DIFF = 0.18
GAMMA = 0.15

BETA_BASE = 0.03
BETA_VULN = 0.55
BETA_HUM = 0.25
BETA_TEMP = 0.22
BETA_NUT = 0.15

CROSS_MIX = 0.02

# =========================
# Helpers
# =========================
def waterlevel_to_moisture_percent(wl: float) -> float:
    return float(np.clip(wl * 10.0, 0.0, 100.0))


def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def rand_walk(x: torch.Tensor, sigma: float, lo: float, hi: float) -> torch.Tensor:
    return (x + sigma * torch.randn_like(x)).clamp(lo, hi)


def stress_from_moisture(m_percent_t: torch.Tensor) -> torch.Tensor:
    wp, optL, optH, sat = 20.0, 45.0, 80.0, 95.0
    m = m_percent_t
    out = torch.zeros_like(m)

    m1 = (m > wp) & (m < optL)
    out[m1] = (m[m1] - wp) / (optL - wp)

    m2 = (m >= optL) & (m <= optH)
    out[m2] = 1.0

    m3 = (m > optH) & (m < sat)
    out[m3] = (sat - m[m3]) / (sat - optH)

    return out.clamp(0.0, 1.0)


def stress_from_pH(ph_t: torch.Tensor) -> torch.Tensor:
    ph_min, optL, optH, ph_max = 5.0, 6.0, 7.5, 8.5
    ph = ph_t
    out = torch.zeros_like(ph)

    m1 = (ph > ph_min) & (ph < optL)
    out[m1] = (ph[m1] - ph_min) / (optL - ph_min)

    m2 = (ph >= optL) & (ph <= optH)
    out[m2] = 1.0

    m3 = (ph > optH) & (ph < ph_max)
    out[m3] = (ph_max - ph[m3]) / (ph_max - optH)

    return out.clamp(0.0, 1.0)


def plant_health(m_percent: torch.Tensor, ph: torch.Tensor) -> torch.Tensor:
    return clamp01(stress_from_moisture(m_percent) * stress_from_pH(ph))


def pathogen_temp_factor(temp_c: torch.Tensor, t_opt=28.0, sigma=6.0) -> torch.Tensor:
    return torch.exp(-0.5 * ((temp_c - t_opt) / sigma) ** 2).clamp(0.0, 1.0)


def nutrient_imbalance(N: torch.Tensor, P: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    mean = (N + P + K) / 3.0
    var = ((N - mean) ** 2 + (P - mean) ** 2 + (K - mean) ** 2) / 3.0
    return (var / 0.08).clamp(0.0, 1.0)


latest_png = None
state_lock = threading.Lock()
session = requests.Session()


def fetch_api():
    try:
        r = session.get(API_URL, timeout=API_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()
        fields = data.get("last", {}).get("fields", {})

        m_list, ph_list = [], []
        for key in FIELDS:
            f = fields.get(key, {})
            wl = float(f.get("water_level", 0.0))
            phv = float(f.get("ph", 7.0))
            m_list.append(waterlevel_to_moisture_percent(wl))
            ph_list.append(phv)

        m_t = torch.tensor(m_list, device=DEVICE, dtype=torch.float32)
        ph_t = torch.tensor(ph_list, device=DEVICE, dtype=torch.float32).clamp(4.0, 9.0)
        return m_t, ph_t, True
    except Exception:
        return None, None, False


def render_heatmap_png(D_tensor: torch.Tensor, per_field: dict) -> bytes:
    d_cpu = D_tensor.detach().float().cpu().numpy()[:, 0, :, :]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    titles = ["F1", "F2", "F3", "F4"]

    last_im = None
    for i in range(4):
        ax = axes[i]
        last_im = ax.imshow(d_cpu[i], vmin=0.0, vmax=1.0, origin="lower")
        ax.set_title(titles[i])
        ax.set_xticks([])
        ax.set_yticks([])

        txt = (
            f"Moist: {per_field['moist'][i]:5.1f}%\n"
            f"pH:    {per_field['ph'][i]:4.2f}\n"
            f"T:     {per_field['temp'][i]:4.1f}°C\n"
            f"Hum:   {per_field['hum'][i]:4.2f}\n"
            f"Dμ:    {per_field['D_mean'][i]:.3f}\n"
            f"Dmax:  {per_field['D_max'][i]:.3f}"
        )

        ax.text(
            1.02,
            0.50,
            txt,
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
            clip_on=False,
        )

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Disease density D (0..1)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def sim_loop():
    global latest_png

    torch.manual_seed(7)
    np.random.seed(7)

    lap_k = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=DEVICE
    ).view(1, 1, 3, 3)

    D = torch.zeros((4, 1, H, W), device=DEVICE)

    yy = torch.arange(H, device=DEVICE).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=DEVICE).view(1, W).expand(H, W)

    def seed_patch(field_idx: int, cx: int, cy: int, r: int, val: float):
        mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
        D[field_idx, 0][mask] = torch.maximum(
            D[field_idx, 0][mask], torch.tensor(val, device=DEVICE)
        )

    seed_patch(1, W // 2, H // 2, 14, 0.35)
    seed_patch(0, W // 3, H // 3, 10, 0.18)
    seed_patch(2, 2 * W // 3, H // 3, 9, 0.12)

    m_percent = torch.full((4,), 50.0, device=DEVICE)
    ph = torch.full((4,), 7.0, device=DEVICE)

    hum = torch.full((4,), 0.55, device=DEVICE)
    temp = torch.full((4,), 28.0, device=DEVICE)
    N = torch.full((4,), 0.60, device=DEVICE)
    P = torch.full((4,), 0.55, device=DEVICE)
    K = torch.full((4,), 0.50, device=DEVICE)

    last_api_t = 0.0
    step = 0

    while True:
        now = time.time()

        if now - last_api_t >= API_POLL_EVERY_SEC:
            m_new, ph_new, ok = fetch_api()
            if ok:
                m_percent = m_new
                ph = ph_new
            else:
                m_percent = rand_walk(m_percent, sigma=0.8, lo=0.0, hi=100.0)
                ph = rand_walk(ph, sigma=0.02, lo=4.0, hi=9.0)
            last_api_t = now

        hum = rand_walk(hum, sigma=0.02, lo=0.20, hi=0.98)
        temp = rand_walk(temp, sigma=0.20, lo=18.0, hi=42.0)
        N = rand_walk(N, sigma=0.02, lo=0.05, hi=0.95)
        P = rand_walk(P, sigma=0.02, lo=0.05, hi=0.95)
        K = rand_walk(K, sigma=0.02, lo=0.05, hi=0.95)

        health = plant_health(m_percent, ph)
        vuln = (1.0 - health).clamp(0.0, 1.0)
        tf = pathogen_temp_factor(temp)
        imb = nutrient_imbalance(N, P, K)

        growth_driver = (
            BETA_BASE + BETA_VULN * vuln + BETA_HUM * hum + BETA_TEMP * tf + BETA_NUT * imb
        ).clamp(0.0, 2.0)

        g = growth_driver.view(4, 1, 1, 1)

        lap_list = []
        for i in range(4):
            lap_i = F.conv2d(D[i : i + 1], lap_k, padding=1)
            lap_list.append(lap_i)
        lap = torch.cat(lap_list, dim=0)

        diffusion = D_DIFF * lap
        growth = g * D * (1.0 - D)
        decay = -GAMMA * D

        if CROSS_MIX > 0:
            means = D.mean(dim=(2, 3), keepdim=True)
            global_mean = means.mean(dim=0, keepdim=True)
            cross = CROSS_MIX * (global_mean - means)
        else:
            cross = 0.0

        D = (D + DT * (diffusion + growth + decay + cross)).clamp(0.0, 1.0)
        step += 1

        if step % 10 == 0:
            d_mean_vec = D.mean(dim=(2, 3)).view(-1)
            d_max_vec = D.amax(dim=(2, 3)).view(-1)

            per_field = {
                "moist": [float(x) for x in m_percent.detach().cpu().view(-1).tolist()],
                "ph": [float(x) for x in ph.detach().cpu().view(-1).tolist()],
                "temp": [float(x) for x in temp.detach().cpu().view(-1).tolist()],
                "hum": [float(x) for x in hum.detach().cpu().view(-1).tolist()],
                "D_mean": [float(x) for x in d_mean_vec.detach().cpu().tolist()],
                "D_max": [float(x) for x in d_max_vec.detach().cpu().tolist()],
            }

            png_bytes = render_heatmap_png(D, per_field)
            with state_lock:
                latest_png = png_bytes

        time.sleep(0.02)


app = Flask(__name__)


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Disease Heatmap Live</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 16px; }
    img  { border: 1px solid #ddd; max-width: 100%; height: auto; }
  </style>
</head>
<body>
  <h2>Live Disease Heatmap (F1–F4)</h2>
  <img id="hm" src="/heatmap.png" />
<script>
  function refresh() {
    document.getElementById("hm").src = "/heatmap.png?t=" + Date.now();
  }
  setInterval(refresh, 1000);
  refresh();
</script>
</body>
</html>
"""


@app.get("/heatmap.png")
def heatmap_png():
    with state_lock:
        if latest_png is None:
            return Response(b"", mimetype="image/png")
        return Response(latest_png, mimetype="image/png")


if __name__ == "__main__":
    print(f"[DISEASE] Device: {DEVICE}")
    print(f"[DISEASE] API: {API_URL}")
    print(f"[DISEASE] Open: http://127.0.0.1:{PORT}/")

    th = threading.Thread(target=sim_loop, daemon=True)
    th.start()

    app.run(host=HOST, port=PORT, debug=False, threaded=True)
