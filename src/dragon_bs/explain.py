"""Tree SHAP for exported decision traces, with held-out reconstruction diagnostics."""

import argparse
import json
from pathlib import Path

import numpy as np

from .models import action_scores, load_run


def calibrate_shap(shap, predictions, *, atol=1e-2, rtol=1e-3):
    """Fit output scale/base on alternating states; validate on the other states.

    Some GBRL versions expose unscaled leaf-gradient SHAP. Any fitted sign/scale
    is explicitly reported. Passing this check is empirical reconstruction on
    these states, not a guarantee about arbitrary inputs or causal importance.
    """
    shap = np.asarray(shap, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if shap.ndim != 3 or shap.shape[0] != len(predictions):
        raise ValueError("Expected SHAP shape (states, features, outputs).")
    shap = shap[:, :, :predictions.shape[1]]
    if shap.shape[2] != predictions.shape[1] or len(shap) < 6:
        raise ValueError("Need at least six states and matching SHAP/output dimensions.")
    if not np.isfinite(shap).all() or not np.isfinite(predictions).all():
        raise ValueError("SHAP or model predictions contain non-finite values.")
    total = shap.sum(axis=1)
    training = np.arange(len(shap)) % 2 == 0
    contributions = np.zeros_like(shap)
    bases, scales, errors, verified = [], [], [], []
    for output in range(predictions.shape[1]):
        values, target = total[:, output], predictions[:, output]
        if np.ptp(values[training]) < 1e-12:
            scale, base = 0.0, float(target[training].mean())
            # A zero SHAP sum does not certify canceling nonzero contributions.
            identifiable = bool(np.max(np.abs(shap[:, :, output])) < 1e-12)
        else:
            scale, base = np.linalg.lstsq(
                np.column_stack([values[training], np.ones(training.sum())]),
                target[training], rcond=None)[0]
            identifiable = True
        reconstruction = scale * values[~training] + base
        error = float(np.max(np.abs(reconstruction - target[~training])))
        threshold = atol + rtol * float(np.max(np.abs(target[~training])))
        contributions[:, :, output] = scale * shap[:, :, output]
        bases.append(float(base))
        scales.append(float(scale))
        errors.append(error)
        verified.append(bool(identifiable and error <= threshold))
    report = {"complete": all(verified), "n_outputs": len(verified),
              "n_outputs_passed": sum(verified), "max_held_out_error": max(errors),
              "atol": atol, "rtol": rtol, "scales": scales, "bases": bases,
              "per_output_errors": errors, "per_output_passed": verified,
              "validation": "scale/base fitted on even states, tested on odd states"}
    return contributions, report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path, help="Run folder containing model.zip and selection_trace.npz")
    p.add_argument("--output-dir", type=Path, help="Default: RUN_DIR/explanation")
    args = p.parse_args(argv)
    output = args.output_dir or args.run_dir / "explanation"
    try:
        model, run = load_run(args.run_dir)
        trace = np.load(args.run_dir / "selection_trace.npz", allow_pickle=False)
        states, bands = trace["states"], trace["bands"]
        algo = run["configuration"]["algo"]
        n_bands = run["input"]["shape"][1]
        if algo == "dqn":
            learner = model.q_model.learner
        else:
            ac = model.policy.model.learner
            learner = ac if algo == "ppo" else ac.actor_learner
        raw = learner.shap(states)
        predictions = np.asarray([action_scores(model, state, algo) for state in states])
        contributions, report = calibrate_shap(raw, predictions[:, :n_bands])
    except (ValueError, FileNotFoundError, ImportError) as exc:
        p.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    (output / "reconstruction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(output / "attributions.npz", raw=raw, contributions=contributions,
                        predictions=predictions, bands=bands)
    if not report["complete"]:
        p.exit(1, f"SHAP reconstruction failed. Diagnostics saved to {output}.\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own = contributions[np.arange(len(states)), :, bands]
    fig, ax = plt.subplots(figsize=(11, 5))
    image = ax.imshow(own, aspect="auto", cmap="RdBu_r")
    fig.colorbar(image, ax=ax, label="Calibrated contribution to chosen output")
    ax.set(xlabel="Observation feature", ylabel="Selection step",
           title=f"{algo.upper()} decision trace — {run['configuration']['dataset']}")
    fig.tight_layout()
    fig.savefig(output / "decision_trace.png", dpi=160)
    plt.close(fig)
    print(f"Held-out reconstruction passed; saved explanation to {output}")


if __name__ == "__main__":
    main()
