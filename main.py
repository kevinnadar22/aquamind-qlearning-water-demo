"""AquaMind — tabular Q-learning on Jowitt & Xu via WNTR (Negm 2024 setup)."""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent
SEED = 42
HOURS = 24
N_STATES = 24  # hour of day
N_ACTIONS = 27  # 3 PRV setpoints cubed
TRAIN_EPISODES = 180
SCENARIOS = ("background", "burst")

# Negm (2024) Table 5-4 — Jowitt & Xu background leakage (p.131 / eprint Ch.5).
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

# Negm (2024) Table 6-2 — Jowitt & Xu burst case (p.156 / eprint Ch.6); burst nodes 4, 9, 11.
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


train_all()

app = FastAPI(title="AquaMind", version="2.0.0")
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
    return {"status": "ok", "engine": "wntr", "inp": "jowitt_negm.inp"}


@app.get("/api/topology")
def topology():
    return TOPOLOGY


@app.get("/api/simulate")
def simulate(scenario: str = Query("background", pattern="^(background|burst)$")):
    if scenario not in SCENARIOS:
        return JSONResponse({"error": "unknown scenario"}, status_code=400)

    q = Q_TABLES[scenario]
    baseline = run_episode(scenario, "baseline", None, N_STATES, N_ACTIONS, state_from_hour, SEED)
    agent = run_episode(scenario, "agent", q, N_STATES, N_ACTIONS, state_from_hour, SEED)

    negm_rows = NEGM_BACKGROUND if scenario == "background" else NEGM_BURST
    comparison_note = (
        "indicative — see README Differences from Negm (2024); tabular Q-learning, "
        f"PRV levels {list(SETPOINT_LEVELS)} m, {TRAIN_EPISODES} training episodes vs thesis DRL/benchmarks."
    )

    base_series = series_from_run(baseline)
    agent_series = series_from_run(agent)

    return {
        "scenario": scenario,
        "hours": HOURS,
        "topology": TOPOLOGY,
        "meta": {
            "baseline_prv_m": list(BASELINE_SETTING),
            "pressure_limits_m": [10, 70],
            "prv_action_levels_m": list(SETPOINT_LEVELS),
            "critical_node_assumption": {
                "node": CRITICAL_NODE,
                "label": "Critical node (assumed high-demand / hospital proxy — not part of benchmark)",
            },
            "seed": SEED,
        },
        "kpis": {
            "baseline": {
                "total_leakage_ls": round(baseline["total_leakage"], 4),
                "water_saved_pct": 0.0,
                "violations": baseline["violations"],
                "mean_pressure_m": round(baseline["mean_pressure"], 2),
                "min_pressure_m": round(baseline["min_pressure"], 2),
                "critical_min_pressure_m": round(baseline["critical_min_pressure"], 2),
            },
            "agent": {
                "total_leakage_ls": round(agent["total_leakage"], 4),
                "water_saved_pct": round(agent["water_saved_pct"], 2),
                "violations": agent["violations"],
                "mean_pressure_m": round(agent["mean_pressure"], 2),
                "min_pressure_m": round(agent["min_pressure"], 2),
                "critical_min_pressure_m": round(agent["critical_min_pressure"], 2),
            },
            "delta": {
                "leakage_reduction_pct": round(agent["water_saved_pct"], 2),
                "violation_delta": agent["violations"] - baseline["violations"],
            },
        },
        "benchmark_comparison": {
            "source": "Negm (2024) Lancaster PhD thesis — Table 5-4 (background) / Table 6-2 (burst)",
            "citation": "https://eprints.lancs.ac.uk/id/eprint/217610/",
            "note": comparison_note,
            "negm": negm_rows,
            "aquamind_qlearning": {
                "water_saved_pct": round(agent["water_saved_pct"], 2),
                "violations": agent["violations"],
            },
        },
        "charts": {
            "leakage": {
                "baseline": base_series["leakage_total"],
                "agent": agent_series["leakage_total"],
            },
            "pressure": {
                "baseline": base_series["pressure_mean"],
                "agent": agent_series["pressure_mean"],
            },
            "violations": {
                "baseline": base_series["violations"],
                "agent": agent_series["violations"],
            },
            "training_reward": TRAIN_CURVES[scenario],
            "network": {
                "baseline": base_series,
                "agent": agent_series,
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)
