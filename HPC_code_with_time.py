import io
import time
import random
import requests
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Roboto", "DejaVu Sans", "Arial"]

import matplotlib.pyplot as plt
from flask import Flask, Response, render_template_string


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:18085/state"
API_TIMEOUT = 2.0

HOST = "0.0.0.0"
PORT = 8088

FIELDS_TO_USE = ["f1", "f2", "f3", "f4"]

MC_RUNS_PER_GPU = 256
NX = 96
NZ = 96
DT = 0.025
ROOT_ZONE_LAYERS = 20
DIFFUSION_SUBSTEPS = 12
STEPS_PER_CYCLE = 40
API_REFRESH_SEC = 2.0

SEED = 42

app = Flask(__name__)
shared = None


# ============================================================
# CROP CONFIG
# ============================================================

@dataclass
class CropProfile:
    name: str
    moisture_opt_low: float
    moisture_opt_high: float
    ph_opt_low: float
    ph_opt_high: float
    nutrient_opt_low: float
    nutrient_opt_high: float
    root_uptake_scale: float
    nutrient_uptake_scale: float
    evap_scale: float


CROPS: Dict[int, CropProfile] = {
    0: CropProfile("rice",      0.28, 0.45, 5.5, 7.0, 0.40, 0.90, 1.30, 1.10, 0.70),
    1: CropProfile("wheat",     0.18, 0.32, 6.0, 7.5, 0.35, 0.80, 1.00, 1.00, 1.00),
    2: CropProfile("maize",     0.20, 0.34, 5.8, 7.2, 0.38, 0.85, 1.10, 1.05, 1.10),
    3: CropProfile("tomato",    0.22, 0.36, 5.8, 6.8, 0.42, 0.88, 1.05, 1.15, 1.15),
    4: CropProfile("cotton",    0.16, 0.28, 5.8, 8.0, 0.30, 0.75, 0.85, 0.90, 1.20),
    5: CropProfile("sugarcane", 0.24, 0.40, 6.0, 7.8, 0.45, 0.92, 1.20, 1.10, 1.00),
}


# ============================================================
# HELPERS
# ============================================================

def clamp(x, lo=None, hi=None):
    if lo is not None:
        if torch.is_tensor(lo):
            x = torch.maximum(x, lo)
        else:
            x = torch.clamp(x, min=lo)
    if hi is not None:
        if torch.is_tensor(hi):
            x = torch.minimum(x, hi)
        else:
            x = torch.clamp(x, max=hi)
    return x

def now_sec():
    return time.perf_counter()

def crop_tensor_from_dict(field_crop_ids: torch.Tensor, attr: str, device: str, dtype: torch.dtype):
    vals = [getattr(CROPS[int(i)], attr) for i in field_crop_ids.detach().cpu().tolist()]
    return torch.tensor(vals, device=device, dtype=torch.float32).view(-1, 1, 1, 1).to(dtype)


def fetch_api_state() -> Dict[str, Any]:
    r = requests.get(API_URL, timeout=API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def extract_field_inputs(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    fields = payload.get("last", {}).get("fields", {})

    for fname in FIELDS_TO_USE:
        f = fields.get(fname, {})

        water_level = float(f.get("water_level", 0.0))
        ph = float(f.get("ph", 7.0))

        moisture_percent = max(0.0, min(100.0, water_level * 10.0))
        moisture_frac = max(0.02, min(0.98, water_level / 10.0))

        result[fname] = {
            "water_level": water_level,
            "moisture_percent": moisture_percent,
            "moisture_frac": moisture_frac,
            "ph": ph,
        }

    return result


def render_heatmap_png(data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, cbar_label: str) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Horizontal Soil Grid (X)")
    ax.set_ylabel("Depth Layer (Z)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# GPU ENGINE
# ============================================================

class HeavyEngine:
    def __init__(self, gpu_id: int):
        # CUDA is initialized ONLY here, inside child process
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.dtype = torch.float16
        self.n_fields = len(FIELDS_TO_USE)

        torch.manual_seed(SEED + gpu_id)
        random.seed(SEED + gpu_id)

        crop_ids = [0, 1, 2, 3]
        self.crop_id = torch.tensor(crop_ids[:self.n_fields], device=self.device)

        self.moisture_opt_low = crop_tensor_from_dict(self.crop_id, "moisture_opt_low", self.device, self.dtype)
        self.moisture_opt_high = crop_tensor_from_dict(self.crop_id, "moisture_opt_high", self.device, self.dtype)
        self.ph_opt_low = crop_tensor_from_dict(self.crop_id, "ph_opt_low", self.device, self.dtype)
        self.ph_opt_high = crop_tensor_from_dict(self.crop_id, "ph_opt_high", self.device, self.dtype)
        self.nutrient_opt_low = crop_tensor_from_dict(self.crop_id, "nutrient_opt_low", self.device, self.dtype)
        self.nutrient_opt_high = crop_tensor_from_dict(self.crop_id, "nutrient_opt_high", self.device, self.dtype)
        self.root_uptake_scale = crop_tensor_from_dict(self.crop_id, "root_uptake_scale", self.device, self.dtype)
        self.nutrient_uptake_scale = crop_tensor_from_dict(self.crop_id, "nutrient_uptake_scale", self.device, self.dtype)
        self.evap_scale = crop_tensor_from_dict(self.crop_id, "evap_scale", self.device, self.dtype)

        shape4 = (self.n_fields, MC_RUNS_PER_GPU, NX, NZ)

        self.ph = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(6.0, 7.2)
        self.temp = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(20.0, 34.0)

        self.theta_sat = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(0.42, 0.56)
        self.theta_fc = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(0.24, 0.34)
        self.theta_wp = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(0.08, 0.15)
        self.k_sat = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(0.015, 0.06)
        self.nutrient_diff = torch.empty((self.n_fields, MC_RUNS_PER_GPU, 1, 1), device=self.device, dtype=self.dtype).uniform_(0.002, 0.012)

        base_theta = torch.empty(shape4, device=self.device, dtype=self.dtype).uniform_(0.16, 0.28)
        depth_profile = torch.linspace(1.00, 0.85, NZ, device=self.device, dtype=self.dtype).view(1, 1, 1, NZ)
        x_profile = torch.linspace(0.95, 1.05, NX, device=self.device, dtype=self.dtype).view(1, 1, NX, 1)
        self.theta = clamp(base_theta * depth_profile * x_profile, 0.05, self.theta_sat)

        self.fert = torch.zeros(shape4, device=self.device, dtype=self.dtype)
        self.fert[:, :, :, :8] = torch.empty(
            (self.n_fields, MC_RUNS_PER_GPU, NX, 8),
            device=self.device,
            dtype=self.dtype
        ).uniform_(0.2, 0.7)

        z = torch.arange(NZ, device=self.device, dtype=self.dtype)
        root_depth = torch.zeros(NZ, device=self.device, dtype=self.dtype)
        root_depth[:ROOT_ZONE_LAYERS] = 1.0 - (z[:ROOT_ZONE_LAYERS] / max(1.0, float(ROOT_ZONE_LAYERS)))
        root_depth = root_depth / (root_depth.sum() + 1e-8)
        self.root_profile = root_depth.view(1, 1, 1, NZ)

        lap_kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            device=self.device,
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.lap_kernel = lap_kernel.to(self.dtype)

        self.stream_main = torch.cuda.Stream(device=gpu_id)
        self.sim_time = 0.0

    def ingest_api_values(self, field_inputs: Dict[str, Dict[str, float]]):
        for idx, fname in enumerate(FIELDS_TO_USE):
            vals = field_inputs.get(fname, {})
            moisture_frac = float(vals.get("moisture_frac", 0.20))
            ph = float(vals.get("ph", 7.0))

            moisture_frac = max(0.02, min(0.98, moisture_frac))
            ph = max(4.8, min(8.5, ph))

            top = moisture_frac
            bottom = max(0.02, moisture_frac * 0.75)

            profile_z = torch.linspace(top, bottom, NZ, device=self.device, dtype=self.dtype).view(1, 1, NZ)
            profile_2d = profile_z.repeat(MC_RUNS_PER_GPU, NX, 1)

            noise = torch.empty((MC_RUNS_PER_GPU, NX, NZ), device=self.device, dtype=self.dtype).uniform_(-0.02, 0.02)
            seeded_theta = clamp(profile_2d + noise, 0.02, self.theta_sat[idx:idx+1])

            self.theta[idx] = seeded_theta
            self.ph[idx, :, :, :] = ph

            fert_base = 0.15 + 0.5 * moisture_frac
            fert_noise = torch.empty((MC_RUNS_PER_GPU, NX, 8), device=self.device, dtype=self.dtype).uniform_(-0.08, 0.08)
            fert_top = clamp(fert_base + fert_noise, 0.0, 2.0)
            self.fert[idx, :, :, :8] = fert_top

    def moisture_conductivity(self):
        rel = (self.theta - self.theta_wp) / (self.theta_sat - self.theta_wp + 1e-6)
        rel = clamp(rel, 0.0, 1.0)
        return self.k_sat * (rel ** 3.0)

    def moisture_stress_factor(self):
        root_theta = self.theta[:, :, :, :ROOT_ZONE_LAYERS].mean(dim=(2, 3), keepdim=True)
        dry_denom = torch.clamp(self.moisture_opt_low - self.theta_wp, min=1e-6)
        wet_denom = torch.clamp(self.theta_sat - self.moisture_opt_high, min=1e-6)
        too_dry = clamp((root_theta - self.theta_wp) / dry_denom, 0.0, 1.0)
        too_wet = clamp((self.theta_sat - root_theta) / wet_denom, 0.0, 1.0)
        return torch.minimum(too_dry, too_wet)

    def ph_stress_factor(self):
        center = (self.ph_opt_low + self.ph_opt_high) * 0.5
        half_width = (self.ph_opt_high - self.ph_opt_low) * 0.5 + 1e-6
        dist = torch.abs(self.ph - center)
        return clamp(1.0 - (dist / half_width), 0.0, 1.0)

    def nutrient_stress_factor(self):
        root_fert = self.fert[:, :, :, :ROOT_ZONE_LAYERS].mean(dim=(2, 3), keepdim=True)
        below = clamp(root_fert / (self.nutrient_opt_low + 1e-6), 0.0, 1.0)
        above = clamp(self.nutrient_opt_high / (root_fert + 1e-6), 0.0, 1.0)
        return torch.minimum(below, above)

    def evapotranspiration_loss(self):
        temp_factor = clamp((self.temp - 18.0) / 18.0, 0.6, 1.8)
        return 0.0008 * self.evap_scale * temp_factor

    def laplacian_2d(self, x):
        n, m, nx, nz = x.shape
        y = x.reshape(n * m, 1, nx, nz)
        y = F.conv2d(y, self.lap_kernel, padding=1)
        return y.reshape(n, m, nx, nz)

    def step_moisture(self):
        K = self.moisture_conductivity()
        moisture_sf = self.moisture_stress_factor()
        ph_sf = self.ph_stress_factor()
        nutrient_sf = self.nutrient_stress_factor()
        combined_growth = clamp(0.45 * moisture_sf + 0.25 * ph_sf + 0.30 * nutrient_sf, 0.0, 1.0)

        for _ in range(DIFFUSION_SUBSTEPS):
            lap_theta = self.laplacian_2d(self.theta)
            self.theta = self.theta + DT * 0.08 * K * lap_theta

            grad_z = self.theta[:, :, :, :-1] - self.theta[:, :, :, 1:]
            flux_down = 0.25 * K[:, :, :, :-1] * grad_z + 0.05 * K[:, :, :, :-1]

            self.theta[:, :, :, :-1] -= DT * flux_down
            self.theta[:, :, :, 1:] += DT * flux_down

            uptake = 0.012 * self.root_uptake_scale * combined_growth * self.root_profile
            self.theta -= DT * uptake

            et = self.evapotranspiration_loss()
            self.theta[:, :, :, :4] -= DT * et

            excess_bottom = clamp(self.theta[:, :, :, -1:] - self.theta_fc, 0.0, None)
            self.theta[:, :, :, -1:] -= DT * 0.20 * excess_bottom

            self.theta = clamp(self.theta, 0.02, self.theta_sat)

    def step_fertilizer(self):
        moisture_sf = self.moisture_stress_factor()
        ph_sf = self.ph_stress_factor()
        uptake_factor = clamp(0.55 * moisture_sf + 0.45 * ph_sf, 0.0, 1.0)
        K = self.moisture_conductivity()

        for _ in range(DIFFUSION_SUBSTEPS):
            lap_fert = self.laplacian_2d(self.fert)
            self.fert = self.fert + DT * self.nutrient_diff * lap_fert

            adv = 0.14 * K[:, :, :, :-1] * self.fert[:, :, :, :-1]
            self.fert[:, :, :, :-1] -= DT * adv
            self.fert[:, :, :, 1:] += DT * adv

            nutrient_uptake = 0.022 * self.nutrient_uptake_scale * uptake_factor * self.root_profile * self.fert
            self.fert -= DT * nutrient_uptake

            leach = 0.05 * K[:, :, :, -1:] * self.fert[:, :, :, -1:]
            self.fert[:, :, :, -1:] -= DT * leach

            self.fert = clamp(self.fert, 0.0, 3.0)

    def step(self):
        self.temp += torch.empty_like(self.temp).uniform_(-0.02, 0.02)
        self.temp = clamp(self.temp, 15.0, 42.0)

        self.ph += torch.empty_like(self.ph).uniform_(-0.001, 0.001)
        self.ph = clamp(self.ph, 4.8, 8.5)

        self.step_moisture()
        self.step_fertilizer()
        self.sim_time += DT

    def root_stress_index(self):
        ms = self.moisture_stress_factor()
        ps = self.ph_stress_factor()
        ns = self.nutrient_stress_factor()
        healthy = clamp(0.45 * ms + 0.20 * ps + 0.35 * ns, 0.0, 1.0)
        return 1.0 - healthy

    def reduced_maps(self, field_idx: int):
        moisture = self.theta[field_idx].float().mean(dim=0).detach().cpu().numpy()
        fert = self.fert[field_idx].float().mean(dim=0).detach().cpu().numpy()

        theta_local = self.theta[field_idx].float().mean(dim=0)
        theta_root_mean = self.theta[field_idx, :, :, :ROOT_ZONE_LAYERS].float().mean()
        ph_val = self.ph[field_idx].float().mean()
        fert_root = self.fert[field_idx, :, :, :ROOT_ZONE_LAYERS].float().mean()

        moist_term = torch.abs(theta_local - theta_root_mean)
        moist_term = moist_term / (moist_term.max() + 1e-6)
        ph_term = torch.abs(ph_val - 6.8) / 2.0
        fert_term = torch.abs(fert_root - 0.7) / 1.5

        stress_map = 0.70 * moist_term + 0.15 * ph_term + 0.15 * fert_term
        stress_map = torch.clamp(stress_map, 0.0, 1.0)

        return moisture, fert, stress_map.detach().cpu().numpy()


# ============================================================
# GPU WORKER
# ============================================================

def gpu_worker(gpu_id: int, shared_dict):
    try:
        if not torch.cuda.is_available():
            shared_dict[f"gpu{gpu_id}_warn"] = "CUDA not available in worker"
            return

        if gpu_id >= torch.cuda.device_count():
            shared_dict[f"gpu{gpu_id}_warn"] = f"GPU {gpu_id} not available"
            return

        torch.cuda.set_device(gpu_id)
        torch.backends.cudnn.benchmark = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        engine = HeavyEngine(gpu_id)
        last_api_refresh = 0.0

        while True:
            cycle_t0 = now_sec()

            # -----------------------------
            # 1. API INGEST TIMER
            # -----------------------------
            api_t0 = now_sec()
            now = time.time()

            if now - last_api_refresh >= API_REFRESH_SEC:
                try:
                    payload = fetch_api_state()
                    field_inputs = extract_field_inputs(payload)
                    engine.ingest_api_values(field_inputs)
                    shared_dict[f"gpu{gpu_id}_api"] = field_inputs
                    shared_dict[f"gpu{gpu_id}_warn"] = ""
                    last_api_refresh = now
                except Exception as e:
                    shared_dict[f"gpu{gpu_id}_warn"] = f"API fetch failed: {e}"

            api_sec = now_sec() - api_t0
            shared_dict[f"gpu{gpu_id}_api_sec"] = api_sec

            # -----------------------------
            # 2. PURE COMPUTE TIMER
            # -----------------------------
            compute_t0 = now_sec()

            with torch.cuda.stream(engine.stream_main):
                for _ in range(STEPS_PER_CYCLE):
                    engine.step()

            torch.cuda.synchronize(gpu_id)
            compute_sec = now_sec() - compute_t0

            shared_dict[f"gpu{gpu_id}_compute_sec"] = compute_sec
            shared_dict[f"gpu{gpu_id}_sim_time"] = engine.sim_time

            # -----------------------------
            # 3. STRESS METRICS TIMER
            # -----------------------------
            metrics_t0 = now_sec()

            stress = engine.root_stress_index().float()
            shared_dict[f"gpu{gpu_id}_stress_avg"] = float(stress.mean().item())
            shared_dict[f"gpu{gpu_id}_stress_min"] = float(stress.min().item())
            shared_dict[f"gpu{gpu_id}_stress_max"] = float(stress.max().item())

            metrics_sec = now_sec() - metrics_t0
            shared_dict[f"gpu{gpu_id}_metrics_sec"] = metrics_sec

            # -----------------------------
            # 4. MAP REDUCTION TIMER
            # -----------------------------
            reduce_t0 = now_sec()

            for field_idx, fname in enumerate(FIELDS_TO_USE):
                moisture, fert, stress_map = engine.reduced_maps(field_idx)
                shared_dict[f"gpu{gpu_id}_{fname}_moisture"] = moisture.astype(np.float32)
                shared_dict[f"gpu{gpu_id}_{fname}_fert"] = fert.astype(np.float32)
                shared_dict[f"gpu{gpu_id}_{fname}_stress"] = stress_map.astype(np.float32)

            reduce_sec = now_sec() - reduce_t0
            shared_dict[f"gpu{gpu_id}_reduce_sec"] = reduce_sec

            # -----------------------------
            # 5. TOTAL CYCLE TIMER
            # -----------------------------
            cycle_total_sec = now_sec() - cycle_t0
            shared_dict[f"gpu{gpu_id}_cycle_total_sec"] = cycle_total_sec

            # -----------------------------
            # 6. THROUGHPUT
            # -----------------------------
            total_cell_updates = len(FIELDS_TO_USE) * MC_RUNS_PER_GPU * NX * NZ * STEPS_PER_CYCLE
            shared_dict[f"gpu{gpu_id}_cell_updates_total"] = float(total_cell_updates)
            shared_dict[f"gpu{gpu_id}_cell_updates_per_sec"] = float(total_cell_updates / max(compute_sec, 1e-8))
            shared_dict[f"gpu{gpu_id}_throughput_mcells_sec"] = float(total_cell_updates / max(compute_sec, 1e-8) / 1e6)
            shared_dict[f"gpu{gpu_id}_fps_equivalent"] = float(1.0 / max(cycle_total_sec, 1e-8))

    except Exception as e:
        shared_dict[f"gpu{gpu_id}_warn"] = f"Worker crashed: {e}"


# ============================================================
# DASHBOARD HELPERS
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Dual GPU Agriculture HPC Dashboard</title>
    <meta http-equiv="refresh" content="5">
    <style>
        body {
            font-family: Roboto, Arial, sans-serif;
            background: #0b1220;
            color: #e5e7eb;
            margin: 0;
            padding: 20px;
        }
        h1, h2, h3 { margin: 0 0 10px 0; }
        .meta, .gpu-box, .card {
            background: #111827;
            border-radius: 12px;
            padding: 14px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.25);
        }
        .meta { margin-bottom: 18px; }
        .gpu-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 18px;
        }
        .field-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(320px, 1fr));
            gap: 18px;
        }
        .triple {
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }
        img {
            width: 100%;
            border-radius: 10px;
            background: #fff;
        }
        .small {
            color: #9ca3af;
            font-size: 14px;
            line-height: 1.5;
        }
        .ok { color: #86efac; }
        .warn { color: #fca5a5; }
    </style>
</head>
<body>
    <h1>Dual GPU Agriculture HPC Dashboard</h1>

    <div class="meta">
        <div><strong>API:</strong> {{ api_url }}</div>
        <div><strong>Server:</strong> {{ host }}:{{ port }}</div>
        <div><strong>Per-GPU Tensor Shape:</strong> [4, {{ mc_runs }}, {{ nx }}, {{ nz }}]</div>
        <div><strong>Total GPUs Targeted:</strong> 2</div>
    </div>

    <div class="gpu-grid">
        {% for g in gpus %}
        <div class="gpu-box">
            <h3>GPU {{ g.id }}</h3>
            <div class="small">API ingest time: {{ g.api_sec }}</div>
            <div class="small">Pure compute time: {{ g.compute_sec }}</div>
            <div class="small">Metrics time: {{ g.metrics_sec }}</div>
            <div class="small">Reduction time: {{ g.reduce_sec }}</div>
            <div class="small">Full cycle time: {{ g.cycle_total_sec }}</div>
            <div class="small">Simulation time: {{ g.sim_time }}</div>
            <div class="small">Cell updates/sec: {{ g.cell_updates }}</div>
            <div class="small">Throughput: {{ g.throughput_mcells_sec }} MCells/sec</div>
            <div class="small">Cycle rate: {{ g.fps_equivalent }} cycles/sec</div>
            <div class="small">Stress avg: {{ g.stress_avg }}</div>
            <div class="small">Stress min/max: {{ g.stress_min }} / {{ g.stress_max }}</div>
            <div class="small {% if g.warn %}warn{% else %}ok{% endif %}">
            {{ g.warn if g.warn else "Worker running normally" }}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="field-grid">
        {% for field in fields %}
        <div class="card">
            <h2>{{ field.name }}</h2>
            <div class="small">
                water_level={{ field.water_level }},
                moisture={{ field.moisture_percent }}%,
                pH={{ field.ph }}
            </div>
            <div class="triple">
                <div>
                    <p>Combined Moisture Heatmap</p>
                    <img src="/heatmap/moisture/{{ field.index }}?t={{ ts }}" alt="Moisture heatmap">
                </div>
                <div>
                    <p>Combined Fertilizer Heatmap</p>
                    <img src="/heatmap/fertilizer/{{ field.index }}?t={{ ts }}" alt="Fertilizer heatmap">
                </div>
                <div>
                    <p>Combined Stress Heatmap</p>
                    <img src="/heatmap/stress/{{ field.index }}?t={{ ts }}" alt="Stress heatmap">
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


def get_latest_api_values():
    api0 = shared.get("gpu0_api", {}) if shared is not None else {}
    api1 = shared.get("gpu1_api", {}) if shared is not None else {}
    return api0 if api0 else api1


def combine_maps(field_name: str, kind: str):
    arr0 = shared.get(f"gpu0_{field_name}_{kind}", None)
    arr1 = shared.get(f"gpu1_{field_name}_{kind}", None)

    if arr0 is None and arr1 is None:
        return np.zeros((NX, NZ), dtype=np.float32)
    if arr0 is None:
        return np.array(arr1, dtype=np.float32)
    if arr1 is None:
        return np.array(arr0, dtype=np.float32)

    arr0 = np.array(arr0, dtype=np.float32)
    arr1 = np.array(arr1, dtype=np.float32)
    return (arr0 + arr1) / 2.0


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    api_vals = get_latest_api_values()

    fields = []
    for i, fname in enumerate(FIELDS_TO_USE):
        vals = api_vals.get(fname, {})
        fields.append({
            "index": i,
            "name": fname,
            "water_level": f"{vals.get('water_level', 0.0):.2f}",
            "moisture_percent": f"{vals.get('moisture_percent', 0.0):.2f}",
            "ph": f"{vals.get('ph', 7.0):.2f}",
        })

    gpus = []
    for gid in [0, 1]:
        gpus.append({
    "id": gid,
    "compute_sec": f"{shared.get(f'gpu{gid}_compute_sec', 0.0):.4f} sec" if shared else "0.0000 sec",
    "api_sec": f"{shared.get(f'gpu{gid}_api_sec', 0.0):.4f} sec" if shared else "0.0000 sec",
    "metrics_sec": f"{shared.get(f'gpu{gid}_metrics_sec', 0.0):.4f} sec" if shared else "0.0000 sec",
    "reduce_sec": f"{shared.get(f'gpu{gid}_reduce_sec', 0.0):.4f} sec" if shared else "0.0000 sec",
    "cycle_total_sec": f"{shared.get(f'gpu{gid}_cycle_total_sec', 0.0):.4f} sec" if shared else "0.0000 sec",
    "sim_time": f"{shared.get(f'gpu{gid}_sim_time', 0.0):.2f}" if shared else "0.00",
    "cell_updates": f"{shared.get(f'gpu{gid}_cell_updates_per_sec', 0.0):,.0f}" if shared else "0",
    "throughput_mcells_sec": f"{shared.get(f'gpu{gid}_throughput_mcells_sec', 0.0):.2f}" if shared else "0.00",
    "fps_equivalent": f"{shared.get(f'gpu{gid}_fps_equivalent', 0.0):.2f}" if shared else "0.00",
    "stress_avg": f"{shared.get(f'gpu{gid}_stress_avg', 0.0):.4f}" if shared else "0.0000",
    "stress_min": f"{shared.get(f'gpu{gid}_stress_min', 0.0):.4f}" if shared else "0.0000",
    "stress_max": f"{shared.get(f'gpu{gid}_stress_max', 0.0):.4f}" if shared else "0.0000",
    "warn": shared.get(f"gpu{gid}_warn", "") if shared else "",
})

    return render_template_string(
        HTML,
        api_url=API_URL,
        host=HOST,
        port=PORT,
        mc_runs=MC_RUNS_PER_GPU,
        nx=NX,
        nz=NZ,
        fields=fields,
        gpus=gpus,
        ts=int(time.time()),
    )


@app.route("/heatmap/<kind>/<int:field_idx>")
def heatmap(kind: str, field_idx: int):
    if kind not in {"moisture", "fertilizer", "stress"}:
        return Response("Invalid heatmap type", status=400)

    if field_idx < 0 or field_idx >= len(FIELDS_TO_USE):
        return Response("Invalid field index", status=400)

    fname = FIELDS_TO_USE[field_idx]

    if kind == "moisture":
        data = combine_maps(fname, "moisture")
        png = render_heatmap_png(data, f"{fname} Combined Soil Moisture Heatmap", "Blues", 0.0, 1.0, "Moisture")
    elif kind == "fertilizer":
        data = combine_maps(fname, "fert")
        png = render_heatmap_png(data, f"{fname} Combined Fertilizer Heatmap", "YlGn", 0.0, 3.0, "Fertilizer")
    else:
        data = combine_maps(fname, "stress")
        png = render_heatmap_png(data, f"{fname} Combined Stress Heatmap", "hot", 0.0, 1.0, "Stress")

    return Response(png, mimetype="image/png")


# ============================================================
# MAIN
# ============================================================

def main():
    global shared

    print("Starting spawn-safe dual-GPU agriculture heatmap server")
    print(f"API URL : {API_URL}")
    print(f"Open    : http://<server-ip>:{PORT}")

    mp.freeze_support()
    ctx = mp.get_context("spawn")

    manager = ctx.Manager()
    shared = manager.dict()

    p0 = ctx.Process(target=gpu_worker, args=(0, shared), daemon=True)
    p1 = ctx.Process(target=gpu_worker, args=(1, shared), daemon=True)

    p0.start()
    p1.start()

    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
