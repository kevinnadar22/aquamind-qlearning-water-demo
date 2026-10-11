#!/usr/bin/env python3
"""Train SB3 PPO per scenario; light penalty sweep on background, then full train."""
from __future__ import annotations

import json
import sys

from ppo_agent import (
    DEFAULT_PPO_HPARAMS,
    MODELS_DIR,
    load_ppo,
    model_path,
    run_episode_ppo,
    save_config,
    train_ppo,
)

SCENARIOS = ("background", "burst")


def kpi_row(run: dict) -> dict:
    return {
        "total_leakage": round(run["total_leakage"], 4),
        "water_saved_pct": round(run["water_saved_pct"], 2),
        "violations": run["violations"],
        "mean_pressure_m": round(run["mean_pressure"], 2),
        "min_pressure_m": round(run["min_pressure"], 2),
        "critical_min_pressure_m": round(run["critical_min_pressure"], 2),
    }


def pick_penalty() -> float:
    target_saved, target_vio = 66.0, 46.0
    best_penalty = DEFAULT_PPO_HPARAMS["vio_penalty"]
    best_score = float("inf")
    for pen in (4.0, 6.0, 8.0):
        hp = {**DEFAULT_PPO_HPARAMS, "vio_penalty": pen, "total_timesteps": 3000}
        print(f"Background sweep vio_penalty={pen} …", flush=True)
        train_ppo("background", hp)
        model = load_ppo("background")
        assert model is not None
        run = run_episode_ppo("background", model)
        score = abs(run["water_saved_pct"] - target_saved) + 0.4 * abs(
            run["violations"] - target_vio
        )
        print(
            f"  saved={run['water_saved_pct']:.2f}% viol={run['violations']} score={score:.2f}",
            flush=True
        )
        if score < best_score:
            best_score = score
            best_penalty = pen
    return best_penalty


def main() -> None:
    import main as app_main  # noqa: F401 — Q-learning warmup/train

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("Penalty sweep (background, 6k steps each)…", flush=True)
    best_penalty = pick_penalty()

    hparams = {
        **DEFAULT_PPO_HPARAMS,
        "vio_penalty": best_penalty,
        "total_timesteps": 8000,
    }
    cfg = {"hparams": hparams, "scenarios": {}, "eval": {}}

    for scenario in SCENARIOS:
        print(f"Training {scenario} ({hparams['total_timesteps']} steps)…", flush=True)
        train_ppo(scenario, hparams)
        model = load_ppo(scenario)
        assert model is not None
        baseline = app_main.run_episode(
            scenario,
            "baseline",
            None,
            app_main.N_STATES,
            app_main.N_ACTIONS,
            app_main.state_from_hour,
            app_main.SEED,
        )
        q_run = app_main.run_episode(
            scenario,
            "agent",
            app_main.Q_TABLES[scenario],
            app_main.N_STATES,
            app_main.N_ACTIONS,
            app_main.state_from_hour,
            app_main.SEED,
        )
        ppo_run = run_episode_ppo(scenario, model)
        cfg["scenarios"][scenario] = str(model_path(scenario))
        cfg["eval"][scenario] = {
            "baseline": kpi_row(baseline),
            "qlearning": kpi_row(q_run),
            "ppo": kpi_row(ppo_run),
        }
        print(json.dumps(cfg["eval"][scenario], indent=2), flush=True)

    save_config(cfg)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
