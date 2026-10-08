# Third-party notices

The project retains its existing GPL-3.0 license in `LICENSE`.

The files under `src/dragon_bs/_vendor/gbrl_sb3/` are derived from
[NVIDIA GBRL/SB3](https://github.com/NVlabs/gbrl_sb3), copyright 2024 NVIDIA
Corporation, under the **NVIDIA Source Code License-NC**. Their license is
preserved in that directory; these files are not relicensed under GPL-3.0.

The source snapshot was taken from the backend used by the local research
scripts: upstream commit `fa86a07eaf1b44704fc4a10d4d75da1915d5437f`, with local
edits to A2C/DQN/PPO and model archive I/O. Only the required algorithm, policy,
buffer and I/O modules are included. Import paths were changed to the private
`dragon_bs._vendor.gbrl_sb3` namespace; model archive debug output was removed.

The external [GBRL library](https://github.com/NVlabs/gbrl) is installed as a
training dependency and retains its own license. Stable-Baselines3 and
sb3-contrib retain their upstream licenses as installed dependencies.

The original project builds on the band-selection problem formulation and code
of [DRL4BS](https://github.com/lcmou/DRL4BS). Cite the original work where its
contribution is used in addition to the DRAGON-BS paper.

Dataset providers and required references are linked in
[docs/DATASETS.md](docs/DATASETS.md). The paper PDF, raw datasets and local model
checkpoints are not included in this release.
