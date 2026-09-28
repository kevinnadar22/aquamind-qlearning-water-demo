"""AquaMind — minimal Q-learning water network demo (FastAPI + static HTML)."""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parent
HOURS = 720  # 30-day horizon for reporting
TRAIN_HOURS = 168
TRAIN_EPISODES = 120
RES_CAPACITY = 100.0
RES_INIT = 82.0
CRITICAL_RES = 10.0
OFFPEAK_PRESS_LIMIT = 25.0
FAVAD_EXP = 1.18

ZONES = ["North", "Central", "Industrial", "Residential", "Hospital"]
VALVE = np.array([0.35, 0.65, 0.95])
BASE_K = np.array([0.018, 0.022, 0.020, 0.019, 0.015])
ZONE_WEIGHT = np.array([1.0, 1.2, 1.4, 0.9, 0.55])
HOSPITAL_IDX = 4

N_RES, N_TOD, N_DEM = 5, 6, 3
N_STATES = N_RES * N_TOD * N_DEM
N_ACTIONS = 9  # hospital valve (3) × other zones valve (3)

SCENARIOS = {
    "normal": {"inflow_mult": 1.0, "leak_zone": None, "leak_mult": 1.0},
    "drought": {"inflow_mult": 0.58, "leak_zone": None, "leak_mult": 1.0},
    "leak": {"inflow_mult": 1.0, "leak_zone": 2, "leak_mult": 3.2},
}

Q = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)
BASELINE_VALVES = (2, 2)  # static high-aperture SCADA profile


def demand_profile(hour: int) -> float:
    t = hour % 24
    base = 0.55 + 0.45 * math.sin((t - 6) * math.pi / 12) ** 2
    morning = 0.35 * math.exp(-((t - 8) ** 2) / 8)
    evening = 0.45 * math.exp(-((t - 19) ** 2) / 10)
    return base + morning + evening


def decode_action(a: int) -> tuple[int, int]:
    return a // 3, a % 3


def valve_vector(hosp_v: int, other_v: int) -> np.ndarray:
    v = np.full(5, VALVE[other_v])
    v[HOSPITAL_IDX] = VALVE[hosp_v]
    return v


def discretize_state(res_level: float, hour: int, total_demand: float) -> int:
    res_b = min(N_RES - 1, int(res_level / RES_CAPACITY * N_RES))
    tod_b = min(N_TOD - 1, (hour % 24) // 4)
    dem_b = 0 if total_demand < 2.8 else (1 if total_demand < 4.2 else 2)
    return res_b * N_TOD * N_DEM + tod_b * N_DEM + dem_b


def run_sim(
    scenario: str,
    policy: str,
    hours: int = HOURS,
    seed: int = 7,
) -> dict:
    cfg = SCENARIOS[scenario]
    random.seed(seed)
    np.random.seed(seed)

    res = RES_INIT
    base_inflow = 4.8 * cfg["inflow_mult"]
    k = BASE_K.copy()
    if cfg["leak_zone"] is not None:
        k[cfg["leak_zone"]] *= cfg["leak_mult"]

    series = {
        "reservoir": [],
        "leakage_total": [],
        "pressure_mean": [],
        "pressure_baseline": [],
        "pressure_agent": [],
        "hospital_met": [],
    }
    total_leak = total_delivered = 0.0
    hospital_demand = hospital_met = 0.0
    offpeak_pressures: list[float] = []
    autonomy_hit: int | None = None

    for h in range(hours):
        prof = demand_profile(h)
        noise = 1.0 + random.uniform(-0.08, 0.08)
        demands = ZONE_WEIGHT * prof * noise * 0.95

        if policy == "agent":
            st = discretize_state(res, h, float(demands.sum()))
            a = int(np.argmax(Q[st]))
            hv, ov = decode_action(a)
            valves = valve_vector(hv, ov)
        elif policy == "baseline":
            valves = valve_vector(*BASELINE_VALVES)

        head = 8.0 + 42.0 * (res / RES_CAPACITY)
        pressures = head * valves / np.sqrt(0.6 + demands / 6.0)
        pressures = np.clip(pressures, 5.0, 55.0)

        leak = k * (pressures ** FAVAD_EXP)
        supply_cap = valves * (res / RES_CAPACITY) * 6.5
        delivered = np.minimum(demands, supply_cap)
        res = res + base_inflow - float(delivered.sum()) - float(leak.sum())
        res = max(0.0, min(RES_CAPACITY, res))

        total_leak += float(leak.sum())
        total_delivered += float(delivered.sum())
        hospital_demand += float(demands[HOSPITAL_IDX])
        hospital_met += float(delivered[HOSPITAL_IDX])

        is_offpeak = not (6 <= (h % 24) <= 22)
        if is_offpeak:
            offpeak_pressures.append(float(pressures.mean()))

        if autonomy_hit is None and res <= CRITICAL_RES:
            autonomy_hit = h

        series["reservoir"].append(round(res, 3))
        series["leakage_total"].append(round(float(leak.sum()), 4))
        series["pressure_mean"].append(round(float(pressures.mean()), 3))
        if policy == "baseline":
            series["pressure_baseline"].append(round(float(pressures.mean()), 3))
        else:
            series["pressure_agent"].append(round(float(pressures.mean()), 3))

    autonomy_days = (autonomy_hit or hours) / 24.0
    hosp_rel = 100.0 * hospital_met / hospital_demand if hospital_demand else 0.0
    mean_offpeak = float(np.mean(offpeak_pressures)) if offpeak_pressures else 0.0

    return {
        "total_leakage": total_leak,
        "hospital_reliability_pct": round(hosp_rel, 2),
        "mean_offpeak_pressure_m": round(mean_offpeak, 2),
        "autonomy_days": round(autonomy_days, 1),
        "series": series,
    }


def step_reward(
    res: float,
    hour: int,
    demands: np.ndarray,
    delivered: np.ndarray,
    pressures: np.ndarray,
    leak: np.ndarray,
) -> float:
    leak_pen = -0.35 * float(leak.sum())
    hosp_short = float(max(0.0, demands[HOSPITAL_IDX] - delivered[HOSPITAL_IDX]))
    hosp_pen = -18.0 * hosp_short
    t = hour % 24
    offpeak = t < 6 or t > 22
    press_pen = 0.0
    if offpeak:
        excess = max(0.0, float(pressures.mean()) - OFFPEAK_PRESS_LIMIT)
        press_pen = -2.5 * excess
    reserve_bonus = 0.22 * (res / RES_CAPACITY)
    return leak_pen + hosp_pen + press_pen + reserve_bonus


def train_q(scenario: str = "normal") -> list[float]:
    alpha, gamma = 0.22, 0.92
    eps = 1.0
    rewards_curve: list[float] = []
    cfg = SCENARIOS[scenario]
    k_base = BASE_K.copy()

    for ep in range(TRAIN_EPISODES):
        res = RES_INIT + random.uniform(-6, 6)
        base_inflow = 4.8 * cfg["inflow_mult"]
        k = k_base.copy()
        if cfg["leak_zone"] is not None:
            k[cfg["leak_zone"]] *= cfg["leak_mult"]
        ep_reward = 0.0

        for h in range(TRAIN_HOURS):
            prof = demand_profile(h)
            noise = 1.0 + random.uniform(-0.08, 0.08)
            demands = ZONE_WEIGHT * prof * noise * 0.95
            st = discretize_state(res, h, float(demands.sum()))

            if random.random() < eps:
                a = random.randint(0, N_ACTIONS - 1)
            else:
                a = int(np.argmax(Q[st]))

            hv, ov = decode_action(a)
            valves = valve_vector(hv, ov)
            head = 8.0 + 42.0 * (res / RES_CAPACITY)
            pressures = head * valves / np.sqrt(0.6 + demands / 6.0)
            pressures = np.clip(pressures, 5.0, 55.0)
            leak = k * (pressures ** FAVAD_EXP)
            supply_cap = valves * (res / RES_CAPACITY) * 6.5
            delivered = np.minimum(demands, supply_cap)

            res = res + base_inflow - float(delivered.sum()) - float(leak.sum())
            res = max(0.0, min(RES_CAPACITY, res))

            r = step_reward(res, h, demands, delivered, pressures, leak)
            ep_reward += r

            st2 = discretize_state(res, h + 1, float(demands.sum()))
            best = float(np.max(Q[st2]))
            Q[st, a] += alpha * (r + gamma * best - Q[st, a])

        rewards_curve.append(round(ep_reward, 2))
        eps = max(0.04, eps * 0.965)

    return rewards_curve


def train_all_scenario_tables():
    curves: dict[str, list[float]] = {}
    tables: dict[str, np.ndarray] = {}
    for name in SCENARIOS:
        Q[:] = 0
        curves[name] = train_q(name)
        tables[name] = Q.copy()
    Q[:] = tables["normal"]
    return {"curves": curves, "tables": tables}


TRAINING = train_all_scenario_tables()

app = FastAPI(title="AquaMind", version="1.0.0")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "product": "AquaMind"}


@app.get("/api/simulate")
def simulate(scenario: str = Query("normal", pattern="^(normal|drought|leak)$")):
    global Q
    if scenario not in SCENARIOS:
        return JSONResponse({"error": "unknown scenario"}, status_code=400)

    Q[:] = TRAINING["tables"][scenario]
    baseline = run_sim(scenario, "baseline", HOURS, seed=11)
    agent = run_sim(scenario, "agent", HOURS, seed=11)

    base_leak = baseline["total_leakage"]
    agent_leak = agent["total_leakage"]
    reduction = 0.0 if base_leak <= 0 else 100.0 * (base_leak - agent_leak) / base_leak

    # merge pressure series for chart
    pressure_chart = {
        "labels": list(range(HOURS)),
        "baseline": baseline["series"]["pressure_baseline"],
        "agent": agent["series"]["pressure_agent"],
    }

    return {
        "scenario": scenario,
        "hours": HOURS,
        "kpis": {
            "baseline": {
                "water_loss_m3": round(base_leak, 2),
                "hospital_reliability_pct": baseline["hospital_reliability_pct"],
                "mean_offpeak_pressure_m": baseline["mean_offpeak_pressure_m"],
                "autonomy_days": baseline["autonomy_days"],
            },
            "agent": {
                "water_loss_m3": round(agent_leak, 2),
                "hospital_reliability_pct": agent["hospital_reliability_pct"],
                "mean_offpeak_pressure_m": agent["mean_offpeak_pressure_m"],
                "autonomy_days": agent["autonomy_days"],
            },
            "delta": {
                "water_loss_reduction_pct": round(reduction, 2),
                "hospital_reliability_gain_pct": round(
                    agent["hospital_reliability_pct"] - baseline["hospital_reliability_pct"], 2
                ),
            },
        },
        "charts": {
            "reservoir": {
                "baseline": baseline["series"]["reservoir"],
                "agent": agent["series"]["reservoir"],
            },
            "leakage": {
                "baseline": baseline["series"]["leakage_total"],
                "agent": agent["series"]["leakage_total"],
            },
            "pressure": pressure_chart,
            "training_reward": TRAINING["curves"][scenario],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
