# wuji-openpi

[中文版](README_zh.md)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-openpi)](https://github.com/wuji-technology/wuji-openpi/releases)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.5%2B-9cf?logo=google&logoColor=white)](https://github.com/jax-ml/jax)
[![CUDA](https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Stars](https://img.shields.io/github/stars/wuji-technology/wuji-openpi?style=social)](https://github.com/wuji-technology/wuji-openpi/stargazers)

> A fork of [openpi](https://github.com/Physical-Intelligence/openpi) for SFT (supervised fine-tuning) and on-robot deployment of pi0 / pi0.5 VLA policies on **any dual-arm + Wuji Hand configuration**. It adds Wuji-specific data processing, training configs, a dimension-agnostic action pipeline, and a complete ROS2 deployment stack on top of upstream openpi.

<p align="center">
  <img src="docs/assets/demo.gif" width="80%" alt="wuji-openpi dual-arm + Wuji Hand demo" />
</p>

<p align="center">
  <sub>⏩ Full demo shown above at 2× speed. ▶️ Original speed with audio: <a href="docs/assets/demo.mp4">docs/assets/demo.mp4</a></sub>
</p>

## Configurations

The pipeline is **morphology-agnostic**: any dual-arm + Wuji Hand setup is supported, and the action dimensionality is driven entirely by config. The table below lists the reference config shipped in this repo.

| Embodiment | Reference config | Action dim | Deploy example |
|---|---|---|---|
| Dual arm + dual Wuji Hand | `pi05_wuji_multi_54d` | 54 (7+7 arms, 20+20 hands) | [`examples/wuji/`](examples/wuji/) |

> **Adapting to a different morphology.** You don't need this exact 54-dim layout. Set `arm_mode` (`single_left` / `single_right` / `dual`), `arm_dof`, and `hand_dof` in [`examples/wuji/config/deploy.yaml`](examples/wuji/config/deploy.yaml), and the matching `action_dim` in the training config — the same data, training, and deployment path then runs SFT at **any dimensionality**. See [Adapting to a new morphology](#adapting-to-a-new-morphology) below.

## Repository layout

```text
wuji-openpi/
├── examples/wuji/                    # dual-arm + Wuji Hand deployment example
│   ├── config/deploy.yaml            # deploy config (arm/hand DOF, ROS2 topics, broker, control rate)
│   ├── core/                         # ROS2 interface, timestamp sync, utils
│   ├── deploy/                       # deployment entry (main.py / ros_env.py)
│   └── README.md                     # detailed deployment doc
├── src/openpi/
│   ├── policies/wuji_policy.py       # WujiInputs / WujiOutputs data mapping
│   └── training/config.py            # pi05_wuji_multi_54d and other training configs
├── scripts/
│   ├── compute_norm_stats.py         # norm-stats computation
│   ├── train.py                      # training entry
│   └── serve_policy.py               # policy server entry
└── docs/                             # upstream openpi docs
```

## Requirements

- Linux x86_64
- NVIDIA GPU, CUDA 12
- Python 3.11+ with [uv](https://docs.astral.sh/uv/) — the supported installer
- For deployment: ROS2 Humble (Python 3.10) and a ROS2 topic namespace that matches [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git)

## Installation

Dependencies match upstream openpi and are managed with [uv](https://docs.astral.sh/uv/):

```bash
# 1. clone
git clone https://github.com/wuji-technology/wuji-openpi
cd wuji-openpi

# 2. resolve environment (skip LFS smudge so large weights aren't pulled eagerly)
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## Workflow

End-to-end path from teleoperation capture to closed-loop on-robot inference:

```text
┌────────────────────┐    ROS2 mcap     ┌──────────────────┐    LeRobot v2.1    ┌──────────────┐
│  wuji-hand-teleop  │ ───────────────▶ │   Conversion     │ ─────────────────▶ │  SFT Training│
│  (teleop capture)  │                  │ (mcap → lerobot) │                    │  (this repo) │
└────────────────────┘                  └──────────────────┘                    └──────┬───────┘
                                                                                       │
                                                                                       ▼
                                                            ┌──────────────────────────────┐
                                                            │  Policy Server (this repo)   │
                                                            │  WebSocket / ROS2 inference  │
                                                            └──────────────┬───────────────┘
                                                                           │ ROS2 obs/action
                                                                           ▼
                                                            ┌──────────────────────────────┐
                                                            │  dual-arm + Wuji Hand (real) │
                                                            │  ROS2 bridge: wuji-hand-...  │
                                                            └──────────────────────────────┘
```

### Step 1 — Collect teleoperation data

Use [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) to teleoperate the robot, recording synchronized trajectories for both arms + both dexterous hands together with the head camera and the left/right wrist cameras. Data is saved as **ROS2 mcap** bags.

### Step 2 — Convert to a LeRobot v2.1 dataset

Convert the ROS2 mcap recordings to LeRobot v2.1 datasets and place them in your training data directory. Dataset fields must line up with `WujiInputs` (see [`src/openpi/policies/wuji_policy.py`](src/openpi/policies/wuji_policy.py)).

Expected fields after conversion (shown for the reference 54-dim layout):

- `observation.state`: state vector (54-dim = 14 from both arms + 40 from both dexterous hands)
- `observation.images.cam_high`: head camera
- `observation.images.cam_left_wrist`: left wrist camera
- `observation.images.cam_right_wrist`: right wrist camera
- `action`: action sequence (same dimensionality as the state)

### Step 3 — Edit the training config

Training configs live in [`src/openpi/training/config.py`](src/openpi/training/config.py). This repo ships the reference config **`pi05_wuji_multi_54d`**. Before training, adjust:

- `data.repo_id` / `repo_ids`: path(s) to the LeRobot dataset(s) from Step 2 (multiple datasets can be mixed via `LeRobotWujiDataConfig`).
- `checkpoint_base_dir`: output directory for training checkpoints.
- The base checkpoint path inside `weight_loader` (e.g. `pi05_base/params`).
- `model.action_dim`: set to your morphology's total DOF (see [Adapting to a new morphology](#adapting-to-a-new-morphology)).
- Hyperparameters: batch size, training steps, learning rate, etc.

Reference snippet (excerpted from `pi05_wuji_multi_54d`):

```python
TrainConfig(
    name="pi05_wuji_multi_54d",
    checkpoint_base_dir="/path/to/your/checkpoints",
    model=pi0_config.Pi0Config(pi05=True, action_dim=54, action_horizon=100, max_token_len=256),
    data=LeRobotWujiDataConfig(
        repo_ids=[
            "/path/to/lerobot_dataset_1",
            "/path/to/lerobot_dataset_2",
            # ...
        ],
        ...
    ),
    weight_loader=CheckpointWeightLoader(
        "/path/to/pi05_base/params"
    ),
)
```

### Step 4 — Compute norm stats and train

```bash
# Compute normalization statistics (only on first run or after data changes)
uv run scripts/compute_norm_stats.py --config-name pi05_wuji_multi_54d

# Launch training (add --overwrite to overwrite an existing experiment)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_wuji_multi_54d \
  --exp-name=my_wuji_run
```

Training logs go to the console and to Weights & Biases; checkpoints are written under `checkpoint_base_dir`.

### Step 5 — On-robot inference and deployment

Deployment is a **policy server + ROS2 client** that exchanges observations/actions with wuji-hand-teleop.

```bash
# Policy server (ML side)
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=/path/to/checkpoint
```

```bash
# Robot control client (ROS2 side) — run from project root, in a ROS2 env
# whose Python has openpi-client and rclpy available
source /opt/ros/humble/setup.bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

The client subscribes to the joint-state and image topics published by wuji-hand-teleop, packages them into an observation, sends it to the policy server over WebSocket, receives an action chunk, and publishes the chunk back as ROS2 joint commands.

> For full deployment details (YAML config, broker modes, topic lists, troubleshooting), see [`examples/wuji/README.md`](examples/wuji/README.md).

## Adapting to a new morphology

The reference config is 54-dim, but **nothing in the pipeline is hard-wired to that number**. To run SFT and deployment on a different dual-arm + Wuji Hand setup:

1. **Deploy YAML** — set the dimension fields in [`examples/wuji/config/deploy.yaml`](examples/wuji/config/deploy.yaml):
   ```yaml
   arm_mode: "dual"   # or "single_left" / "single_right"
   arm_dof: 7         # DOF per arm
   hand_dof: 20       # DOF per Wuji Hand
   ```
   The client builds the observation/action vectors from these, so the deployed dimensionality follows the YAML.
2. **Training config** — set `model.action_dim` in [`src/openpi/training/config.py`](src/openpi/training/config.py) to the matching total DOF (e.g. `arm_mode=dual` → `2 * (arm_dof + hand_dof)`), and make sure your LeRobot dataset's `observation.state` / `action` widths agree.
3. **Migrate weights** — `PartialCheckpointWeightLoader` silently skips shape-mismatched layers (typically `action_proj`), so a base checkpoint trained at one dimensionality can seed SFT at another without manual surgery.

This is the only thing that changes between morphologies — data conversion, training, and the ROS2 deployment path are all identical.

## Architecture

This repo keeps all upstream openpi capabilities (pi0, pi0-FAST, pi0.5) and layers Wuji-specific modules on top: a dimension-agnostic action pipeline, multi-dataset training, real-time trajectory smoothing, and a full ROS2 deployment package.

<details>
<summary>Deep dive — what this fork adds on top of upstream openpi</summary>

### Dual-arm + dual dexterous-hand training and deployment

- **Config-driven action space**: the reference `pi05_wuji_multi_54d` covers 7+7 (arms) + 20+20 (hands) = 54 DOF, but the dimensionality is set by config — see [Adapting to a new morphology](#adapting-to-a-new-morphology).
- **Data config**: `LeRobotWujiDataConfig` — native arm + dexterous-hand pipeline.
- **Full ROS2 deployment package** at [`examples/wuji/`](examples/wuji/): topic subscription/publishing, timestamp synchronization, broker integration — drop-in ready to talk to wuji-hand-teleop.

### Multi-dataset training

- `ConcatLeRobotDataset` and `MultiLeRobotDataset` (with weighted sampling) for mixing multiple recording sessions.
- New `DataConfig` fields:
  - `lerobot_datasets`: list of datasets.
  - `multi_dataset_mode`: switch between concatenation and weighted sampling.
- **Per-dataset prompt transform**: every sub-dataset can carry its own language instruction.

### RTG (Real-Time Trajectory Generation)

Smooths the boundary between consecutive action chunks:

- `RTGActionBroker` and a QP-based variant.
- Smoothing utilities: `qp_smooth_prefix`, `cubic_smooth_prefix`, `build_time_window_old_reference`.

Follows the paper `arXiv:2507.17141`: when the client asynchronously receives a new chunk, its prefix is smoothed against the currently executing trajectory before being stitched in.

### General tooling improvements

- **`PartialCheckpointWeightLoader`**: when loading a base checkpoint, layers whose shape doesn't match (typically `action_proj`) are silently skipped, making it trivial to migrate between different action dimensionalities.
- **[`examples/open_loop_eval.py`](examples/open_loop_eval.py)**: a general open-loop evaluator with built-in RTG comparison, useful for replaying trained checkpoints and benchmarking smoothing strategies.

> All other features (LIBERO / ALOHA / DROID examples, PyTorch backend, etc.) are unchanged from upstream — see the original openpi docs and the corresponding subdirectories under `examples/`.

</details>

## Development

Dependencies and the full upstream model docs live under [`docs/`](docs/) and the official [openpi](https://github.com/Physical-Intelligence/openpi) repository — this README focuses only on Wuji-related usage.

## Related Projects

- [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) — teleoperation capture (ROS2 mcap) and the on-robot ROS2 bridge
- [wujihandpy](https://github.com/wuji-technology/wujihandpy) — Wuji Hand SDK (C++ core with Python bindings)
- [wujihandros2](https://github.com/wuji-technology/wujihandros2) — ROS 2 driver for Wuji Hand
- [docs.wuji.tech](https://docs.wuji.tech) — Official Wuji documentation portal

## Acknowledgements

This project builds on the following open-source projects:

- [openpi](https://github.com/Physical-Intelligence/openpi) — upstream VLA training/inference framework (pi0 / pi0-FAST / pi0.5)
- [LeRobot](https://github.com/huggingface/lerobot) — dataset format and tooling
- [JAX](https://github.com/jax-ml/jax) — the training/inference backend

## Contributors

- [Han Duo](https://github.com/HanDuo-223)

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{wuji2026openpi,
  title={Wuji-OpenPI: SFT and Deployment of VLA Policies on Dual-Arm + Wuji Hand Robots},
  author={{Wuji Technology}},
  year={2026},
  url={https://github.com/wuji-technology/wuji-openpi}
}
```

## License

Apache-2.0 (inherited from upstream openpi).
