# DRAGON-BS

Code accompanying **DRAGON-BS: A Unified Gradient-Boosted Reinforcement Learning
Framework for Interpretable Band Selection**, published in *IEEE Transactions on
Geoscience and Remote Sensing* (2026).
[Paper: 10.1109/TGRS.2026.3741200](https://doi.org/10.1109/TGRS.2026.3741200).

DRAGON-BS selects hyperspectral bands with gradient-boosted A2C, DQN and PPO
agents. Band selection uses spectral entropy and, in the A2C/DQN profiles,
spectral redundancy. Ground-truth labels are used only for downstream evaluation.

This release organizes the original entropy-based implementations into an
installable package. The three historical environment profiles are preserved;
their differences and the limits of exact paper reproduction are recorded in
[the reproducibility notes](docs/REPRODUCIBILITY.md).

## Install

Use Python 3.10–3.12. Training has been checked on Linux with Python 3.11 and CPU
GBRL execution.

```bash
git clone https://github.com/ccaadaro/Dragon-BS.git
cd Dragon-BS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,analysis]"
```

The required patched GBRL/SB3 backend is included under
`src/dragon_bs/_vendor/`; no separate checkout or Git submodule is needed.
See [third-party notices](THIRD_PARTY_NOTICES.md). For the exact runtime versions
used in local validation, see [requirements-tested.txt](requirements-tested.txt).

## Prepare data

Obtain the datasets separately and place the MATLAB files in `data4drl/` and
`data4classification/`. [Dataset instructions](docs/DATASETS.md) specify the
filenames, MATLAB keys, pixel order and original providers. These directories are
ignored by Git. Training needs only the spectral data, not ground truth.

## Train

```bash
python -m dragon_bs.train --algo dqn --dataset indian_pines --seed 42
python -m dragon_bs.train --algo a2c --dataset pavia --seed 42
python -m dragon_bs.train --algo ppo --dataset gulfport --seed 42
```

Run a short check before a full experiment:

```bash
python -m dragon_bs.train --algo dqn --dataset indian_pines \
  --timesteps 80 --learning-starts 16 --batch-size 16 --n-envs 1 \
  --n-bands 5 --output-dir outputs/smoke
```

Each run creates `outputs/ALGO_DATASET_seedSEED/` containing:

- `model.zip`: saved model and GBRL prediction trees.
- `bands.mat`: selected bands with explicit **0-based** indexing.
- `selection_trace.npz`: visited observations and selected actions.
- `run.json`: configuration, versions, input shape, selection order and reward.
- TensorBoard logs, plus A2C reward-normalization statistics.

Exported selections use masked greedy decoding: at each step the highest scoring
available band is chosen. Selection order is retained. Existing run directories
are protected from accidental overwriting. Use `--output-dir` for independent
experiments and `--data-dir` when data lives elsewhere. Full options:

```bash
python -m dragon_bs.train --help
```

For a batch of five explicitly specified method seeds:

```bash
DATASET=indian_pines DEVICE=cpu bash scripts/run_experiments.sh
```

## Evaluate

Classification splits are independent of the RL seed and shared across methods.
The evaluator provides 5-NN and RBF SVM (training-only five-fold parameter search),
and saves OA, AA and Cohen's kappa for every split plus their mean and standard
deviation.

```bash
python -m dragon_bs.evaluate outputs/dqn_indian_pines_seed42/run.json \
  --dataset indian_pines --classifier knn --train-size 0.1 \
  --split-seeds 0 1 2 3 4 --output outputs/evaluation_dqn_ip.json
```

This is a documented evaluation protocol; it is not a claim that the original
paper's split files or classification tables have been reconstructed. See
[the reproducibility notes](docs/REPRODUCIBILITY.md) before comparing numbers.

## Explain decisions

```bash
python -m dragon_bs.explain outputs/ppo_indian_pines_seed42
```

The tool computes tree SHAP on the saved decision trace and checks reconstruction
on held-out states. It records any fitted output sign/scale explicitly. It writes
a plot only when every output passes the check; failed checks produce diagnostics
and exit with a nonzero status. Use at least six selected bands. Adam-trained
models may fail; `--optimizer sgd` is an explicit alternative training setting.
Passing a trace check establishes empirical reconstruction on that trace, not
causal importance or a guarantee for all possible states.

## Repository layout

```text
src/dragon_bs/       Data loading, environments, training, evaluation, SHAP
src/dragon_bs/_vendor/  Required patched GBRL/SB3 backend and its license
docs/               Dataset and reproducibility instructions
scripts/            Batch runner
tests/              Environment, alignment, indexing and analysis checks
results/reference/  Historical band-selection CSVs with provenance notes
```

Reference CSVs are preserved as historical artifacts, with their original values
and mixed experiment labels. They are not certified reproductions of the paper
results. Read their [provenance notes](results/reference/README.md).

## Development

```bash
python -m pip install -e ".[analysis,dev]"
ruff check src/dragon_bs tests
pytest -q
```

GitHub Actions runs these checks on Python 3.10, 3.11 and 3.12 without benchmark
datasets or CUDA. Full scientific experiments are separate from these checks.

## Citation and acknowledgments

```bibtex
@article{garciaflores2026dragonbs,
  title={DRAGON-BS: A Unified Gradient-Boosted Reinforcement Learning Framework for Interpretable Band Selection},
  author={García-Flores, María B. and Cañada-Rostro, Carlos and Paoletti, Mercedes E. and Haut, Juan M.},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  doi={10.1109/TGRS.2026.3741200}
}
```

Machine-readable citation metadata is in [CITATION.cff](CITATION.cff). This work
builds on [DRL4BS](https://github.com/lcmou/DRL4BS),
[GBRL](https://github.com/NVlabs/gbrl),
[GBRL/SB3](https://github.com/NVlabs/gbrl_sb3) and
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
The repository retains its GPL-3.0 license; vendored NVIDIA code keeps its separate
[NVIDIA Source Code License-NC](src/dragon_bs/_vendor/gbrl_sb3/LICENSE).
