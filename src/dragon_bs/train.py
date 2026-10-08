"""Train the original entropy-based DRAGON-BS algorithm profiles."""

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import scipy.io as sio

from .data import DATASETS, load_dataset
from .envs import BandSelectionEnv, band_statistics
from .models import DEFAULT_TIMESTEPS, action_scores, create_model, policy_kwargs


def select_bands(model, env):
    """Choose the highest scoring available band at every step, without resampling."""
    obs, _ = env.reset()
    states, actions = [], []
    reward = 0.0
    for _ in range(env.nb_exp_bands):
        scores = action_scores(model, obs, env.algo)
        scores[env.state == 1] = -np.inf
        band = int(np.argmax(scores))
        states.append(obs.copy())
        actions.append(band)
        action = (np.eye(1, env.nb_bands, band, dtype=np.float32)[0]
                  if env.algo == "ppo" else band)
        obs, r, _, _, _ = env.step(action)
        reward += r
    if env.algo == "ppo":
        _, r, _, _, _ = env.step(np.zeros(env.nb_bands, dtype=np.float32))
        reward += r
    return {"bands": actions, "reward": float(reward), "states": np.asarray(states)}


def versions():
    result = {}
    for name in ("dragon-bs", "numpy", "scipy", "gymnasium", "torch", "gbrl",
                 "stable-baselines3", "sb3-contrib", "tensorboard"):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", required=True, choices=list(DEFAULT_TIMESTEPS))
    p.add_argument("--dataset", default="indian_pines", choices=list(DATASETS))
    p.add_argument("--data-dir", type=Path, default=Path("data4drl"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, help="Default: A2C 16M, DQN 600k, PPO 200k")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-bands", type=int, default=30)
    p.add_argument("--alpha", type=float, help="Redundancy weight; A2C 0.5, DQN 0.2, PPO 0")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                   help="Actor/critic leaf updates. SGD is an explicit alternative to the original Adam.")
    p.add_argument("--rollout-steps", type=int, help="A2C/PPO steps per environment")
    p.add_argument("--batch-size", type=int, help="DQN/PPO batch size")
    p.add_argument("--learning-starts", type=int, default=10_000, help="DQN warm-up")
    p.add_argument("--max-episode-steps", type=int,
                   help="Default: 10 * n_bands, to bound repeated discrete actions")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.timesteps is None:
        args.timesteps = DEFAULT_TIMESTEPS[args.algo]
    for name in ("timesteps", "n_envs", "n_bands", "rollout_steps", "batch_size",
                 "max_episode_steps"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_starts < 0:
        p.error("--learning-starts must be nonnegative")
    args.output_dir = args.output_dir / f"{args.algo}_{args.dataset}_seed{args.seed}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error(f"Run directory already contains files: {args.output_dir}. Choose a new --output-dir.")
    try:
        import torch
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        p.error('Install training dependencies with pip install -e ".[train]"')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        import gbrl
        if not torch.cuda.is_available() or not gbrl.cuda_available():
            p.error("CUDA is unavailable to Torch or GBRL. Use --device cpu or a CUDA GBRL build.")

    data = load_dataset(args.dataset, data_dir=args.data_dir)
    statistics = band_statistics(data["x"])
    kwargs = dict(algo=args.algo, n_bands=args.n_bands, alpha=args.alpha, statistics=statistics)

    def make_env():
        return gym.wrappers.TimeLimit(BandSelectionEnv(**kwargs),
                                     max_episode_steps=args.max_episode_steps or 10 * args.n_bands)

    env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed, vec_env_cls=DummyVecEnv)
    if args.algo == "a2c":
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    args.output_dir.mkdir(parents=True)

    class EntropyDecay(BaseCallback):
        def _on_step(self):
            self.model.ent_coef = 0.02 * max(0.0, 1 - self.num_timesteps / args.timesteps)
            return True

    try:
        model = create_model(env, args)
        model.learn(total_timesteps=args.timesteps,
                    callback=EntropyDecay() if args.algo == "a2c" else None,
                    log_interval=10)
        model.save(str(args.output_dir / "model"))
        if args.algo == "a2c":
            env.save(str(args.output_dir / "vecnormalize.pkl"))
        evaluation_env = BandSelectionEnv(**kwargs)
        result = select_bands(model, evaluation_env)
        sio.savemat(args.output_dir / "bands.mat", {
            "selected_bands": np.asarray(result["bands"], dtype=np.int32),
            "band_indexing": "0-based", "env_reward": result["reward"],
        })
        np.savez_compressed(args.output_dir / "selection_trace.npz",
                            states=result["states"], bands=result["bands"])
        configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration, "versions": versions(),
            "input": {"file": data["source"], "shape": list(data["x"].shape)},
            "policy_kwargs": policy_kwargs(args.algo, data["nb_bands"],
                                            args.device, args.optimizer),
            "actual_timesteps": model.num_timesteps,
            "bands": result["bands"], "band_indexing": "0-based",
            "selection": "masked-greedy", "env_reward": result["reward"],
        }
        (args.output_dir / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Selected bands (0-based): {result['bands']}")
        print(f"Saved run to {args.output_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
