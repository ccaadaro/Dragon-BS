"""Unsupervised entropy environments from the original DRAGON-BS scripts.

The three profiles retain their original action spaces, observations and rewards.
Labels are deliberately absent from this module.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.stats import entropy


def band_statistics(x, bins=256):
    """Normalized histogram entropy and absolute Pearson band correlations."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Spectra must be a finite (pixels, bands) array with at least 2 of each.")
    ent = np.array([entropy(np.histogram(col, bins=bins)[0] + 1e-12, base=2)
                    for col in x.T])
    span = np.ptp(ent)
    ent = (ent - ent.min()) / span if span > 0 else np.zeros_like(ent)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.abs(np.corrcoef(x, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    return ent, corr


class BandSelectionEnv(gym.Env):
    """Algorithm-specific environment, preserving the historical reward profiles.

    A2C: mask + entropy + budget; IE - 0.5 * max(|corr|), with a final mean-IE bonus.
    DQN: mask + entropy; IE - 0.2 * mean(|corr|).
    PPO: mask; continuous preference vector, selected bands masked before argmax,
         IE reward and the original extra terminal transition with reward 1.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, x=None, *, algo="dqn", n_bands=30, alpha=None, statistics=None):
        super().__init__()
        if algo not in ("a2c", "dqn", "ppo"):
            raise ValueError(f"Unknown algorithm: {algo}")
        if statistics is None:
            statistics = band_statistics(x)
        self.band_entropy, self.band_correlations = statistics
        self.nb_bands = len(self.band_entropy)
        if not 1 <= n_bands <= self.nb_bands:
            raise ValueError(f"n_bands must be between 1 and {self.nb_bands}.")
        self.nb_exp_bands = int(n_bands)
        self.algo = algo
        self.alpha = {"a2c": 0.5, "dqn": 0.2, "ppo": 0.0}[algo] if alpha is None else alpha
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        if algo == "ppo" and self.alpha != 0:
            raise ValueError("The historical PPO profile uses entropy only (alpha=0).")
        self.action_space = (spaces.Box(0, 1, shape=(self.nb_bands,), dtype=np.float32)
                             if algo == "ppo" else spaces.Discrete(self.nb_bands))
        width = {"a2c": 2 * self.nb_bands + 1,
                 "dqn": 2 * self.nb_bands, "ppo": self.nb_bands}[algo]
        self.observation_space = spaces.Box(0, 1, shape=(width,), dtype=np.float32)
        self.state = np.zeros(self.nb_bands, dtype=np.float32)
        self.selected_bands = []
        self.current_step = 0
        self._terminated = False

    def _get_obs(self):
        blocks = [self.state]
        if self.algo != "ppo":
            blocks.append(self.band_entropy)
        if self.algo == "a2c":
            blocks.append([(self.nb_exp_bands - len(self.selected_bands)) / self.nb_exp_bands])
        return np.concatenate(blocks).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.nb_bands, dtype=np.float32)
        self.selected_bands = []
        self.current_step = 0
        self._terminated = False
        return self._get_obs(), {}

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Call reset() after the episode terminates.")
        if self.algo == "ppo":
            scores = np.array(action, dtype=np.float64, copy=True)
            if scores.shape != (self.nb_bands,) or not np.isfinite(scores).all():
                raise ValueError("PPO action must contain one finite preference per band.")
            if len(self.selected_bands) == self.nb_exp_bands:
                self._terminated = True
                return self._get_obs(), 1.0, True, False, self._info()
            scores[self.state == 1] = -np.inf
            band = int(np.argmax(scores))
        else:
            if not self.action_space.contains(action):
                raise ValueError(f"Invalid band action: {action}")
            band = int(action)

        reward = -1.0
        if self.state[band] == 0:
            previous = self.selected_bands
            correlations = self.band_correlations[band, previous]
            redundancy = 0.0 if not previous else (
                float(correlations.max()) if self.algo == "a2c" else float(correlations.mean()))
            reward = float(self.band_entropy[band] - self.alpha * redundancy)
            self.state[band] = 1.0
            self.selected_bands.append(band)
        self.current_step += 1
        self._terminated = self.algo != "ppo" and len(self.selected_bands) == self.nb_exp_bands
        if self._terminated and self.algo == "a2c":
            reward += float(np.mean(self.band_entropy[self.selected_bands]))
        return self._get_obs(), float(reward), self._terminated, False, self._info()

    def _info(self):
        return {"selected_bands": self.selected_bands.copy()}

    def render(self):
        print(f"Step {self.current_step}: {self.selected_bands}")
