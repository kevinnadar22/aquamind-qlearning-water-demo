"""AquaMind — Q-learning + SB3 PPO on Jowitt & Xu via WNTR (Negm 2024 setup)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from network_sim import (
    BASELINE_SETTING,
    CRITICAL_NODE,
    SETPOINT_LEVELS,
    decode_action,
    network_topology,
    run_episode,
    snapshot,
    step_reward,
)
from ppo_agent import CONFIG_PATH, load_ppo, run_episode_ppo

ROOT = Path(__file__).resolve().parent
SEED = 42
HOURS = 24
N_STATES = 24
N_ACTIONS = 27
TRAIN_EPISODES = 180
SCENARIOS = ("background", "burst")
SCENARIO_ALIASES = {
    "normal": "background",
    "drought": "background",
    "leak": "burst",
}


def normalize_scenario(name: str) -> str:
    return SCENARIO_ALIASES.get(name, name)


NEGM_BACKGROUND = [
    {"algorithm": "NM", "reward": 965.7, "water_saved_pct": 66.0, "violations": 49},
    {"algorithm": "PSO", "reward": 994.5, "water_saved_pct": 65.9, "violations": 48},
    {"algorithm": "DE", "reward": 995.2, "water_saved_pct": 65.9, "violations": 46},
    {"algorithm": "ARS", "reward": 943.5, "water_saved_pct": 65.5, "violations": 78},
    {"algorithm": "SAC", "reward": 975.0, "water_saved_pct": 64.2, "violations": 49},
    {"algorithm": "TQC", "reward": 976.3, "water_saved_pct": 65.2, "violations": 50},
    {"algorithm": "TRPO", "reward": 800.5, "water_saved_pct": 73.2, "violations": 264},
    {"algorithm": "PPO", "reward": 780.0, "water_saved_pct": 73.4, "violations": 304},
    {"algorithm": "Recurrent PPO", "reward": 798.4, "water_saved_pct": 73.4, "violations": 300},
    {"algorithm": "DDPG", "reward": 525.0, "water_saved_pct": 44.1, "violations": 48},
    {"algorithm": "A2C", "reward": 822.5, "water_saved_pct": 73.2, "violations": 300},
]

NEGM_BURST = [
    {"algorithm": "NM", "reward": 6.791, "water_saved_pct": 12.68, "violations": 62},
    {"algorithm": "PSO", "reward": 17.23, "water_saved_pct": 29.16, "violations": 112},
    {"algorithm": "DE", "reward": 15.41, "water_saved_pct": 32.52, "violations": 203},
    {"algorithm": "ARS", "reward": 41.79, "water_saved_pct": 47.62, "violations": 69},
    {"algorithm": "SAC", "reward": 21.82, "water_saved_pct": 31.88, "violations": 126},
    {"algorithm": "TQC", "reward": 24.26, "water_saved_pct": 43.16, "violations": 241},
    {"algorithm": "TRPO", "reward": 33.29, "water_saved_pct": 47.01, "violations": 244},
    {"algorithm": "PPO", "reward": 20.43, "water_saved_pct": 40.10, "violations": 339},
    {"algorithm": "Recurrent PPO", "reward": 41.00, "water_saved_pct": 42.28, "violations": 58},
    {"algorithm": "DDPG", "reward": 18.54, "water_saved_pct": 38.57, "violations": 293},
    {"algorithm": "A2C", "reward": 42.15, "water_saved_pct": 58.46, "violations": 67},
]

Q_TABLES: dict[str, np.ndarray] = {}
TRAIN_CURVES: dict[str, list[float]] = {}
PPO_MODELS: dict[str, object] = {}


def state_from_hour(hour: int, _base_snapshot: dict) -> int:
    return hour % N_STATES


def warmup_cache() -> None:
    for scenario in SCENARIOS:
        for hour in range(HOURS):
            snapshot(scenario, hour, BASELINE_SETTING)
            for a in range(N_ACTIONS):
                snapshot(scenario, hour, decode_action(a))


def train_q(scenario: str) -> tuple[np.ndarray, list[float]]:
    random.seed(SEED)
    np.random.seed(SEED)
    q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
    alpha, gamma = 0.18, 0.95
    eps = 1.0
    curve: list[float] = []

    for _ in range(TRAIN_EPISODES):
        ep_reward = 0.0
        for hour in range(HOURS):
            base = snapshot(scenario, hour, BASELINE_SETTING)
            st = state_from_hour(hour, base)
            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
            else:
                a = int(np.argmax(q[st]))
            settings = decode_action(a)
            r = step_reward(scenario, hour, settings)
            ep_reward += r
            st2 = state_from_hour((hour + 1) % HOURS, base)
            q[st, a] += alpha * (r + gamma * float(np.max(q[st2])) - q[st, a])
        curve.append(round(ep_reward, 3))
        eps = max(0.05, eps * 0.97)
    return q, curve


def train_all() -> None:
    warmup_cache()
    for name in SCENARIOS:
        q, curve = train_q(name)
        Q_TABLES[name] = q
        TRAIN_CURVES[name] = curve


def load_ppo_models() -> None:
    for name in SCENARIOS:
        PPO_MODELS[name] = load_ppo(name)


def kpi_from_run(run: dict) -> dict:
    return {
        "total_leakage_ls": round(run["total_leakage"], 4),
        "water_saved_pct": round(run["water_saved_pct"], 2),
        "violations": run["violations"],
        "mean_pressure_m": round(run["mean_pressure"], 2),
        "min_pressure_m": round(run["min_pressure"], 2),
        "critical_min_pressure_m": round(run["critical_min_pressure"], 2),
    }


train_all()
load_ppo_models()

app = FastAPI(title="AquaMind", version="3.0.0")
TOPOLOGY = network_topology()


def series_from_run(run: dict) -> dict:
    hourly = run["hourly"]
    return {
        "leakage_total": [round(h["total_leakage"], 5) for h in hourly],
        "pressure_mean": [round(h["mean_pressure"], 3) for h in hourly],
        "pressure_min": [round(h["min_pressure"], 3) for h in hourly],
        "violations": [h["violations"] for h in hourly],
        "prv01": [h["prv_settings"]["PRV01"] for h in hourly],
        "prv31": [h["prv_settings"]["PRV31"] for h in hourly],
        "prv25": [h["prv_settings"]["PRV25"] for h in hourly],
        "critical_pressure": [round(h["pressures"][CRITICAL_NODE], 3) for h in hourly],
        "pressures": hourly[-1]["pressures"],
        "leakage_nodes": hourly[-1]["leakage"],
        "pipe_flow": hourly[-1]["pipe_flow"],
        "hourly_detail": hourly,
    }


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    ppo_loaded = {s: PPO_MODELS.get(s) is not None for s in SCENARIOS}
    return {
        "status": "ok",
        "engine": "wntr",
        "inp": "jowitt_negm.inp",
        "ppo_loaded": ppo_loaded,
    }


@app.get("/api/topology")
def topology():
    return TOPOLOGY


@app.get("/api/simulate")
def simulate(scenario: str = Query("background")):
    scenario = normalize_scenario(scenario)
    if scenario not in SCENARIOS:
        return JSONResponse(
            {"error": "unknown scenario", "valid": list(SCENARIOS)},
            status_code=400,
        )

    baseline = run_episode(scenario, "baseline", None, N_STATES, N_ACTIONS, state_from_hour, SEED)
    q_run = run_episode(
        scenario, "agent", Q_TABLES[scenario], N_STATES, N_ACTIONS, state_from_hour, SEED
    )

    ppo_run = None
    ppo_model = PPO_MODELS.get(scenario)
    if ppo_model is not None:
        ppo_run = run_episode_ppo(scenario, ppo_model)

    negm_rows = NEGM_BACKGROUND if scenario == "background" else NEGM_BURST
    ppo_note = ""
    if CONFIG_PATH.exists():
        ppo_note = json.loads(CONFIG_PATH.read_text()).get("hparams", {})
    comparison_note = (
        "Measured on this repo (seed 42, 39-pipe jowitt_negm.inp). Q-learning: tabular, "
        f"PRV levels {list(SETPOINT_LEVELS)} m. PPO: SB3 MlpPolicy, continuous [10,70] m; "
        f"training hparams in models/ppo_config.json ({ppo_note}). Negm thesis rows are reference only."
    )

    base_series = series_from_run(baseline)
    q_series = series_from_run(q_run)
    ppo_series = series_from_run(ppo_run) if ppo_run else None

    charts_network = {
        "baseline": base_series,
        "agent": q_series,
    }
    charts_leak = {"baseline": base_series["leakage_total"], "agent": q_series["leakage_total"]}
    charts_press = {"baseline": base_series["pressure_mean"], "agent": q_series["pressure_mean"]}
    charts_vio = {"baseline": base_series["violations"], "agent": q_series["violations"]}
    if ppo_series:
        charts_network["ppo"] = ppo_series
        charts_leak["ppo"] = ppo_series["leakage_total"]
        charts_press["ppo"] = ppo_series["pressure_mean"]
        charts_vio["ppo"] = ppo_series["violations"]

    active = q_run
    return {
        "scenario": scenario,
        "hours": HOURS,
        "topology": TOPOLOGY,
        "meta": {
            "baseline_prv_m": list(BASELINE_SETTING),
            "pressure_limits_m": [10, 70],
            "prv_action_levels_m": list(SETPOINT_LEVELS),
            "ppo_available": ppo_run is not None,
            "critical_node_assumption": {
                "node": CRITICAL_NODE,
                "label": "Critical node (assumed high-demand / hospital proxy — not part of benchmark)",
            },
            "seed": SEED,
        },
        "kpis": {
            "baseline": kpi_from_run(baseline),
            "agent": kpi_from_run(q_run),
            "ppo": kpi_from_run(ppo_run) if ppo_run else None,
            "delta": {
                "leakage_reduction_pct": round(q_run["water_saved_pct"], 2),
                "violation_delta": q_run["violations"] - baseline["violations"],
            },
        },
        "benchmark_comparison": {
            "source": "Negm (2024) — Table 5-4 (background) / Table 6-2 (burst)",
            "citation": "https://eprints.lancs.ac.uk/id/eprint/217610/",
            "note": comparison_note,
            "negm": negm_rows,
            "aquamind_qlearning": {
                "water_saved_pct": round(q_run["water_saved_pct"], 2),
                "violations": q_run["violations"],
            },
            "aquamind_ppo": (
                {
                    "water_saved_pct": round(ppo_run["water_saved_pct"], 2),
                    "violations": ppo_run["violations"],
                }
                if ppo_run
                else None
            ),
        },
        "charts": {
            "leakage": charts_leak,
            "pressure": charts_press,
            "violations": charts_vio,
            "training_reward": TRAIN_CURVES[scenario],
            "network": charts_network,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)
