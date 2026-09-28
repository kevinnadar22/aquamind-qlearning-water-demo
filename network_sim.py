"""Minimal WNTR/EPANET 2.2 wrapper for Jowitt & Xu (Negm PRV layout)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import wntr

ROOT = Path(__file__).resolve().parent
INP = ROOT / "jowitt_negm.inp"

PRV_IDS = ("PRV01", "PRV31", "PRV25")
JUNCTIONS = tuple(str(i) for i in range(1, 23))
RESERVOIRS = ("23", "24", "25")
PRESS_MIN, PRESS_MAX = 10.0, 70.0
BASELINE_SETTING = (40.0, 40.0, 40.0)
BURST_NODES = ("4", "9", "11")
BURST_COEFF = 3.0
CRITICAL_NODE = "5"  # highest base demand — labelled as assumed critical in UI

SETPOINT_LEVELS = (25.0, 40.0, 55.0)


def decode_action(action: int) -> tuple[float, float, float]:
    a0 = action // 9
    a1 = (action // 3) % 3
    a2 = action % 3
    return (
        SETPOINT_LEVELS[a0],
        SETPOINT_LEVELS[a1],
        SETPOINT_LEVELS[a2],
    )


def _load_model() -> wntr.network.WaterNetworkModel:
    return wntr.network.WaterNetworkModel(str(INP))


def _apply_scenario(wn: wntr.network.WaterNetworkModel, scenario: str) -> None:
    if scenario == "background":
        return
    if scenario != "burst":
        raise ValueError(f"unknown scenario: {scenario}")
    for j in JUNCTIONS:
        wn.get_node(j).emitter_coefficient = 0.0
    for j in BURST_NODES:
        wn.get_node(j).emitter_coefficient = BURST_COEFF


def _set_prvs(wn: wntr.network.WaterNetworkModel, settings: tuple[float, float, float]) -> None:
    for vid, sp in zip(PRV_IDS, settings):
        wn.get_link(vid).initial_setting = float(sp)


def _hour_snapshot(
    scenario: str,
    hour: int,
    settings: tuple[float, float, float],
) -> dict:
    wn = _load_model()
    _apply_scenario(wn, scenario)
    _set_prvs(wn, settings)
    wn.options.time.duration = max((hour + 1) * 3600, 3600)
    wn.options.time.hydraulic_timestep = 3600
    results = wntr.sim.EpanetSimulator(wn).run_sim()
    t = hour * 3600
    press = results.node["pressure"].loc[t]
    exp = wn.options.hydraulic.emitter_exponent

    per_press = {j: float(press[j]) for j in JUNCTIONS}
    per_leak: dict[str, float] = {}
    total_leak = 0.0
    vio = 0
    for j in JUNCTIONS:
        k = float(wn.get_node(j).emitter_coefficient or 0.0)
        p = max(per_press[j], 0.0)
        q = k * (p**exp)
        per_leak[j] = q
        total_leak += q
        if p < PRESS_MIN or p > PRESS_MAX:
            vio += 1

    pipe_flow: dict[str, float] = {}
    fr = results.link["flowrate"].loc[t]
    for name in fr.index:
        if name in wn.pipe_name_list or name in wn.valve_name_list:
            pipe_flow[str(name)] = abs(float(fr[name]))

    return {
        "hour": hour,
        "settings": settings,
        "pressures": per_press,
        "leakage": per_leak,
        "total_leakage": total_leak,
        "violations": vio,
        "mean_pressure": sum(per_press.values()) / len(per_press),
        "min_pressure": min(per_press.values()),
        "pipe_flow": pipe_flow,
        "prv_settings": {PRV_IDS[i]: settings[i] for i in range(3)},
    }


@lru_cache(maxsize=None)
def snapshot(scenario: str, hour: int, settings: tuple[float, float, float]) -> dict:
    return _hour_snapshot(scenario, hour, settings)


def step_reward(
    scenario: str,
    hour: int,
    after_settings: tuple[float, float, float],
    scale_leak: float = 3.0,
    scale_vio: float = 1.0,
) -> float:
    before = snapshot(scenario, hour, BASELINE_SETTING)
    after = snapshot(scenario, hour, after_settings)
    r = 0.0
    for j in JUNCTIONS:
        lb = before["leakage"][j]
        la = after["leakage"][j]
        if lb > 1e-12:
            r += scale_leak * (lb - la) / lb
        pb = before["pressures"][j]
        pa = after["pressures"][j]
        vb = int(pb < PRESS_MIN or pb > PRESS_MAX)
        va = int(pa < PRESS_MIN or pa > PRESS_MAX)
        r += scale_vio * (vb - va)
    return r


def run_episode(
    scenario: str,
    policy: str,
    q_table,
    n_states: int,
    n_actions: int,
    state_fn,
    seed: int = 0,
) -> dict:
    import random

    random.seed(seed)
    settings = BASELINE_SETTING
    hourly: list[dict] = []
    total_leak = 0.0
    total_violations = 0
    baseline_leak_sum = 0.0

    for hour in range(24):
        base = snapshot(scenario, hour, BASELINE_SETTING)
        baseline_leak_sum += base["total_leakage"]

        if policy == "agent":
            st = state_fn(hour, base)
            if q_table is not None:
                a = int(max(range(n_actions), key=lambda i: q_table[st, i]))
            else:
                a = 0
            settings = decode_action(a)
        else:
            settings = BASELINE_SETTING

        cur = snapshot(scenario, hour, settings)
        total_leak += cur["total_leakage"]
        total_violations += cur["violations"]
        hourly.append({**cur, "baseline_leak": base["total_leakage"]})

    water_saved_pct = 0.0
    if baseline_leak_sum > 0:
        water_saved_pct = 100.0 * (baseline_leak_sum - total_leak) / baseline_leak_sum

    crit_mins = [h["pressures"][CRITICAL_NODE] for h in hourly]
    return {
        "total_leakage": total_leak,
        "baseline_leakage": baseline_leak_sum,
        "water_saved_pct": water_saved_pct,
        "violations": total_violations,
        "mean_pressure": sum(h["mean_pressure"] for h in hourly) / 24,
        "min_pressure": min(h["min_pressure"] for h in hourly),
        "critical_min_pressure": min(crit_mins),
        "hourly": hourly,
    }


def network_topology() -> dict:
    wn = _load_model()
    coords = {}
    for name, node in wn.nodes():
        if hasattr(node, "coordinates") and node.coordinates:
            coords[str(name)] = {"x": float(node.coordinates[0]), "y": float(node.coordinates[1])}
    pipes = []
    for name in wn.pipe_name_list:
        link = wn.get_link(name)
        pipes.append({"id": name, "n1": str(link.start_node_name), "n2": str(link.end_node_name)})
    for name in wn.valve_name_list:
        link = wn.get_link(name)
        pipes.append(
            {
                "id": name,
                "n1": str(link.start_node_name),
                "n2": str(link.end_node_name),
                "valve": True,
            }
        )
    prv_nodes = {"PRV01": "P01_m", "PRV31": "P31_m", "PRV25": "P25_m"}
    return {
        "coordinates": coords,
        "pipes": pipes,
        "junctions": list(JUNCTIONS),
        "reservoirs": list(RESERVOIRS),
        "prvs": [{"id": k, "node": v} for k, v in prv_nodes.items()],
        "critical_node": CRITICAL_NODE,
        "burst_nodes": list(BURST_NODES),
    }
