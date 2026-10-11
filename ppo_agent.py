"""Stable-Baselines3 PPO training, load, and deterministic evaluation."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from network_sim import BASELINE_SETTING, CRITICAL_NODE, snapshot
from ppo_env import JowittPRVEnv, action_to_settings, build_obs

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
CONFIG_PATH = MODELS_DIR / "ppo_config.json"

# Tuned on background (penalty sweep); burst uses same hyperparameters.
DEFAULT_PPO_HPARAMS = {
    "leak_scale": 3.0,
    "vio_penalty": 1.5,
    "learning_rate": 3e-4,
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "total_timesteps": 18000,
    "seed": 42,
}


def model_path(scenario: str) -> Path:
    return MODELS_DIR / f"ppo_{scenario}.zip"


def _warmup_baseline(scenario: str) -> None:
    for hour in range(24):
        snapshot(scenario, hour, BASELINE_SETTING)


def train_ppo(scenario: str, hparams: dict | None = None) -> PPO:
    hp = {**DEFAULT_PPO_HPARAMS, **(hparams or {})}
    _warmup_baseline(scenario)

    def _make():
        return JowittPRVEnv(
            scenario,
            leak_scale=hp["leak_scale"],
            vio_penalty=hp["vio_penalty"],
        )

    env = DummyVecEnv([_make])
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        seed=hp["seed"],
        verbose=0,
    )
    model.learn(total_timesteps=int(hp["total_timesteps"]))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path(scenario)))
    return model


def _materialize_zip(scenario: str) -> Path | None:
    path = model_path(scenario)
    if path.exists():
        return path
    b64_path = path.with_suffix(path.suffix + ".b64")
    chunks: list[str] = []
    if b64_path.exists():
        chunks.append(b64_path.read_text().strip())
    else:
        i = 0
        while True:
            part = path.with_name(f"{path.name}.b64.part{i}")
            if not part.exists():
                break
            chunks.append(part.read_text())
            i += 1
    if not chunks:
        return None
    path.write_bytes(base64.b64decode("".join(chunks)))
    return path


def load_ppo(scenario: str) -> PPO | None:
    path = _materialize_zip(scenario)
    if path is None:
        return None
    return PPO.load(str(path))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {"hparams": DEFAULT_PPO_HPARAMS, "scenarios": {}}


def save_config(cfg: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def run_episode_ppo(scenario: str, model: PPO) -> dict:
    """Deterministic 24 h roll-out; KPIs match run_episode layout."""
    hourly: list[dict] = []
    total_leak = 0.0
    total_violations = 0
    baseline_leak_sum = 0.0
    settings = BASELINE_SETTING

    snap0 = snapshot(scenario, 0, BASELINE_SETTING)
    obs = build_obs(0, snap0, settings)

    for hour in range(24):
        base = snapshot(scenario, hour, BASELINE_SETTING)
        baseline_leak_sum += base["total_leakage"]

        action, _ = model.predict(obs, deterministic=True)
        settings = action_to_settings(action)
        cur = snapshot(scenario, hour, settings)
        total_leak += cur["total_leakage"]
        total_violations += cur["violations"]
        hourly.append({**cur, "baseline_leak": base["total_leakage"]})

        if hour + 1 < 24:
            obs = build_obs(hour + 1, cur, settings)

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


def ensure_models(training: bool = True) -> dict[str, PPO | None]:
    """Load PPO zip files; optionally train missing scenarios."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models: dict[str, PPO | None] = {}
    cfg = load_config()
    for scenario in ("background", "burst"):
        m = load_ppo(scenario)
        if m is None and training:
            print(f"Training PPO for {scenario}…")
            m = train_ppo(scenario, cfg.get("hparams"))
            cfg.setdefault("scenarios", {})[scenario] = {
                "model": str(model_path(scenario)),
            }
            save_config(cfg)
        models[scenario] = m
    return models
