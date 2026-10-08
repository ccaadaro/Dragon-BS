"""Portable access to the patched GBRL backend used by the research scripts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile


DEFAULT_TIMESTEPS = {"a2c": 16_000_000, "dqn": 600_000, "ppo": 200_000}


def model_class(algo):
    # Lazy imports keep dataset utilities and CLI help usable without the training extra.
    try:
        if algo == "a2c":
            from ._vendor.gbrl_sb3.algos.a2c import A2C_GBRL
            return A2C_GBRL
        if algo == "dqn":
            from ._vendor.gbrl_sb3.algos.dqn import DQN_GBRL
            return DQN_GBRL
        if algo == "ppo":
            from ._vendor.gbrl_sb3.algos.ppo import PPO_GBRL
            return PPO_GBRL
    except ImportError as exc:
        raise ImportError('Install the training dependencies: pip install -e ".[train]"') from exc
    raise ValueError(f"Unknown algorithm: {algo}")


def policy_kwargs(algo, nb_bands, device="cpu", optimizer="adam"):
    tree = {"max_depth": 5, "n_bins": 256, "min_data_in_leaf": 0, "par_th": 2}
    params = {"split_score_func": "Cosine", "generator_type": "Quantile"}
    opts = {"params": params, "device": device}
    result = {"tree_struct": tree, "tree_optimizer": opts}
    if algo == "dqn":
        opts["critic_optimizer"] = {"start_idx": 0, "stop_idx": nb_bands}
    else:
        result["shared_tree_struct"] = algo == "ppo"
        tree["min_data_in_leaf"] = 5 if algo == "a2c" else 0
        rule = "Adam" if optimizer == "adam" else "SGD"
        opts["policy_optimizer"] = {"policy_algo": rule, "policy_lr": 1e-4,
                                     "policy_shrinkage": 0}
        opts["value_optimizer"] = {"value_algo": rule, "value_lr": 0.005038987598789402,
                                    "value_shrinkage": 0}
        if algo == "ppo":
            params["control_variates"] = False
    return result


def create_model(env, args):
    cls = model_class(args.algo)
    common = dict(env=env, seed=args.seed, device=args.device, verbose=1,
                  policy_kwargs=policy_kwargs(args.algo, env.action_space.n
                                              if args.algo != "ppo" else env.action_space.shape[0],
                                              args.device, args.optimizer),
                  tensorboard_log=str(args.output_dir / "tensorboard"))
    if args.algo == "a2c":
        return cls("MlpPolicy", **common, n_steps=args.rollout_steps or 40,
                   gamma=0.99, gae_lambda=0.98, vf_coef=1.0, ent_coef=0.02,
                   learning_rate=3e-4, total_n_steps=args.timesteps)
    if args.algo == "dqn":
        return cls("MlpPolicy", **common, buffer_size=100_000,
                   learning_starts=args.learning_starts, batch_size=args.batch_size or 64,
                   target_update_interval=1000, train_freq=4, gradient_steps=1,
                   max_q_grad_norm=1.0, normalize_q_grads=True,
                   exploration_fraction=0.2, exploration_initial_eps=1.0,
                   exploration_final_eps=0.1)
    return cls(**common, n_steps=args.rollout_steps or 8 * args.n_envs,
               batch_size=args.batch_size or 64, n_epochs=10, gamma=0.99,
               ent_coef=0.0, vf_coef=0.5, clip_range=0.2, gae_lambda=0.95,
               total_n_steps=args.timesteps)


def action_scores(model, observation, algo):
    """Unclipped action preferences / Q values for deterministic masked decoding."""
    import numpy as np
    obs = np.asarray(observation, dtype=np.float32)[None, :]
    if algo == "dqn":
        values = model.q_model(obs, requires_grad=False, tensor=False)
    else:
        values = model.policy.model.predict_policy(obs, requires_grad=False, tensor=False)
    if isinstance(values, tuple):
        values = values[0]
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    values = np.asarray(values).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Model produced non-finite action scores.")
    return deepcopy(values)


def load_run(run_dir, device="cpu"):
    """Load prediction trees from a run without deserializing pickled SB3 metadata."""
    from gbrl.models.actor_critic import ActorCritic
    from gbrl.models.critic import DiscreteCritic

    run_dir = Path(run_dir)
    report = json.loads((run_dir / "run.json").read_text())
    algo = report["configuration"]["algo"]
    allowed = {"model.gbrl_model", "model_policy.gbrl_model", "model_value.gbrl_model",
               "model.gbrl_meta", "model_policy.gbrl_meta", "model_value.gbrl_meta"}
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(run_dir / "model.zip") as archive:
            extracted = []
            for name in archive.namelist():
                if name in allowed:
                    (Path(temp) / name).write_bytes(archive.read(name))
                    extracted.append(name)
        if not any(name.endswith(".gbrl_model") for name in extracted):
            raise ValueError("Model archive contains no GBRL prediction trees.")
        prefix = str(Path(temp) / "model")
        if algo == "dqn":
            model = SimpleNamespace(q_model=DiscreteCritic.load_learner(prefix, device))
        else:
            model = SimpleNamespace(policy=SimpleNamespace(
                model=ActorCritic.load_learner(prefix, device)))
    return model, report
