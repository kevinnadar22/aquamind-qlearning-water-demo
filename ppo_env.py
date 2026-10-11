"""Gymnasium env for Jowitt & Xu PRV control (WNTR hourly steps)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from network_sim import (
    BASELINE_SETTING,
    JUNCTIONS,
    PRESS_MAX,
    PRESS_MIN,
    snapshot,
)

PRV_LOW, PRV_HIGH = 10.0, 70.0
N_JUNCTIONS = len(JUNCTIONS)
OBS_DIM = 1 + N_JUNCTIONS + 3  # hour norm + pressures + PRV setpoints


def action_to_settings(action: np.ndarray) -> tuple[float, float, float]:
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
    sp = PRV_LOW + (a + 1.0) * 0.5 * (PRV_HIGH - PRV_LOW)
    return (float(sp[0]), float(sp[1]), float(sp[2]))


def build_obs(hour: int, snap: dict, settings: tuple[float, float, float]) -> np.ndarray:
    hour_norm = float(hour) / 23.0
    pressures = [float(snap["pressures"][j]) / PRESS_MAX for j in JUNCTIONS]
    sp_norm = [float(s) / PRESS_MAX for s in settings]
    return np.array([hour_norm] + pressures + sp_norm, dtype=np.float32)


def step_reward_ppo(
    scenario: str,
    hour: int,
    settings: tuple[float, float, float],
    leak_scale: float = 3.0,
    vio_penalty: float = 5.0,
) -> float:
    base = snapshot(scenario, hour, BASELINE_SETTING)
    after = snapshot(scenario, hour, settings)
    r = 0.0
    for j in JUNCTIONS:
        lb = base["leakage"][j]
        la = after["leakage"][j]
        if lb > 1e-12:
            r += leak_scale * (lb - la) / lb
    r -= vio_penalty * float(after["violations"])
    return r


class JowittPRVEnv(gym.Env):
    """24 hourly steps; action = 3 PRV setpoints in [10, 70] m."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: str,
        leak_scale: float = 3.0,
        vio_penalty: float = 5.0,
    ):
        super().__init__()
        self.scenario = scenario
        self.leak_scale = leak_scale
        self.vio_penalty = vio_penalty
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.5, shape=(OBS_DIM,), dtype=np.float32
        )
        self.hour = 0
        self.settings = BASELINE_SETTING
        self._last_snap: dict | None = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.hour = 0
        self.settings = BASELINE_SETTING
        self._last_snap = snapshot(self.scenario, 0, BASELINE_SETTING)
        return build_obs(0, self._last_snap, self.settings), {}

    def step(self, action):
        self.settings = action_to_settings(action)
        cur = snapshot(self.scenario, self.hour, self.settings)
        self._last_snap = cur
        reward = step_reward_ppo(
            self.scenario,
            self.hour,
            self.settings,
            self.leak_scale,
            self.vio_penalty,
        )
        self.hour += 1
        terminated = self.hour >= 24
        if terminated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = build_obs(self.hour, cur, self.settings)
        return obs, reward, terminated, False, {}
