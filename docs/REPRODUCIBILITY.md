# Implementation provenance and reproduction scope

This release is a refactor of the entropy-based scripts in the author's local
`RL_BandSelection` working tree. The pre-cleanup Git HEAD was
`fd6e653`; the working tree also contained unpublished edits. Rewards and
observations were extracted from these working-tree files:

| Profile | Original source | Action | Observation | Reward |
| --- | --- | --- | --- | --- |
| A2C | `A2C/GBRL_A2C.py` | discrete band | mask, entropy, remaining budget | IE − 0.5 × maximum absolute correlation; final mean-IE bonus |
| DQN | `DQN/GBRL_DQN.py` | discrete band | mask, entropy | IE − 0.2 × mean absolute correlation |
| PPO | `PPO/GBRL_ppo_sb3.py` | continuous preferences | mask | IE of masked argmax band; extra terminal reward of 1 |

A2C/DQN repeated actions receive −1 and leave the subset unchanged. PPO masks
selected preferences before argmax. The original PPO terminal transition occurs
one step after the band budget is filled; that behavior is preserved.

The paper describes mean-information-entropy gain and entropy minus **average**
correlation. The retained DQN profile corresponds to the latter reward. The A2C
maximum-correlation/terminal-bonus profile and PPO preference action space are
implementation variants present in the local code. This release does not silently
replace them with a new method or advertise an implementation of every equation.

Later local experiments used a label-dependent Fisher separability reward. They
are research extensions, not the unsupervised method described by the paper, and
were preserved in the local archive rather than included in the public runner.

## Training configuration

Defaults come from the retained training scripts:

| Setting | A2C | DQN | PPO |
| --- | ---: | ---: | ---: |
| Requested timesteps | 16,000,000 | 600,000 | 200,000 |
| Environments | 4 | 4 | 4 |
| Rollout steps per environment | 40 | — | 32 |
| Batch size | — | 64 | 64 |
| Tree depth | 5 | 5 | 5 |
| Histogram bins | 256 | 256 | 256 |
| Minimum leaf data | 5 | 0 | 0 |
| Actor leaf learning rate | 0.0001 | — | 0.0001 |
| Value leaf learning rate | 0.005038987598789402 | backend default | 0.005038987598789402 |
| Actor/value leaf optimizer | Adam | backend default | Adam |
| Shared actor/value trees | no | — | yes |
| Reward normalization | yes | no | no |

Algorithm kwargs and actual elapsed timesteps are recorded in `run.json`.
A2C's entropy coefficient starts at 0.02 and decays linearly; PPO's is 0.
DQN retains a 100,000-transition buffer, 10,000-step warm-up, training frequency
4, and epsilon decreasing from 1 to 0.1 over 20% of training.

The CPU default replaces the inconsistent hard-coded CPU/CUDA settings in the
original scripts. The patched backend is based on NVIDIA GBRL/SB3 commit
`fa86a07eaf1b44704fc4a10d4d75da1915d5437f` plus the local compatibility/save
changes. Internal imports have been namespaced to prevent collisions. Its
separate license and provenance are documented in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Explicit changes in the runnable release

- Dataset and output locations are CLI arguments; there are no personal paths.
- Constant-band entropy/correlation calculations produce finite values.
- Episodes with repeated discrete actions are bounded by a configurable time
  limit, defaulting to ten times the selection budget.
- Exports use masked greedy decoding of raw policy preferences/Q values and
  always contain the requested number of distinct bands. The old scripts used
  stochastic policy predictions or random replacement of duplicate actions.
- Model, trace, versions, configuration and index convention are saved together.
- A2C/PPO use their matching algorithm classes and setup occurs once.
- Method and classification-split seeds are separate.

These changes improve execution and auditability. In particular, decoding and
time limits can change a selected subset relative to a historical run.

## Classification and paper tables

The public evaluator uses common stratified split seeds, standardizes from the
training pixels only, and supports 5-NN and RBF SVM with the paper's C/gamma grid.
Reported OA and AA are fractions, kappa is unscaled, and standard deviation uses
`ddof=0`. The paper's MLP, complete baseline suite, exact original split assets,
classification maps and figure-generation pipeline are not reproduced by this
evaluator. Archived CSVs contain mixed experiments and were not certified against
the final paper tables.

The paper's Pavia/Gulfport band counts (93/72) differ from the local files and
historical selections (103/64). Dataset-specific train/test proportions in the
paper also do not define one unambiguous set of reproducible split files.
Exact table reproduction therefore requires author confirmation of preprocessing,
original split indices, final run configurations and the figure/table artifacts.
This repository does not invent those missing mappings or results.

## SHAP

SHAP is computed from saved trees on saved observations. Sign/scale and base
values are calibrated using alternating states and validated on the held-out
states, for **every** output. Failed reconstruction still saves diagnostics but
produces no interpretation plot. At least six states are required. A passing
trace check is empirical evidence for that trace; it does not establish causality
or universal model completeness. SGD is an explicit alternate optimizer, not the
historical default.

## Validation of this release

The release was checked with:

- 18 automated checks for rewards, duplicate behavior, Gymnasium contracts,
  constant spectra, MATLAB pixel alignment, explicit band indexing,
  classification and held-out SHAP reconstruction.
- Direct comparisons of rewards, observations and termination against the three
  pre-cleanup environment classes on the same action sequences.
- Short CPU training runs for A2C, DQN and PPO on local Indian Pines spectra,
  including model/band/metadata export.
- Reloading all three saved tree models and reproducing their decoded decisions.
- A short downstream Indian Pines classification evaluation.
- A saved PPO/SGD model's tree SHAP reconstruction and explanation plot.

Short runs verify the software path and are not evidence of converged accuracy
or reproduction of the paper's quantitative results.
