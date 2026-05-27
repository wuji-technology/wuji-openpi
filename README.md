# wuji-openpi

**English** | [中文](README_zh.md)

This repository is a fork of [openpi](https://github.com/Physical-Intelligence/openpi), customized for **SFT (supervised fine-tuning) and deployment on the wuji dual-arm robot**. On top of upstream openpi, it adds wuji-specific data processing, training configs, and a complete ROS2 deployment pipeline.

> For upstream model details, see [`docs/`](docs/) and the official openpi repository. This README focuses only on wuji-related usage.

## 1. Repository Scope

- **Purpose**: Run SFT and on-robot deployment of VLA models (pi0 / pi0.5) on the wuji dual-arm robot.
- **Data source**: Teleoperation data is collected via the [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) repository in ROS2 mcap format.
- **Training framework**: Built on the openpi JAX training pipeline, with wuji-specific data policies, configs, and norm-stats scripts.
- **Deployment**: Communicates with wuji-hand-teleop over ROS2, exchanging observation/action messages for on-robot inference.

## 2. End-to-End Workflow

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
                                                            │  wuji dual-arm robot (real)  │
                                                            │  ROS2 bridge: wuji-hand-...  │
                                                            └──────────────────────────────┘
```

### Step 1: Collect teleoperation data

Use [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) to teleoperate the robot, recording synchronized trajectories for both arms + both dexterous hands together with the head camera and the left/right wrist cameras. Data is saved as **ROS2 mcap** bags.

### Step 2: Convert to a LeRobot v2.1 dataset

Convert the ROS2 mcap recordings to LeRobot v2.1 datasets and place them in your training data directory. Dataset fields must line up with `WujiInputs` (see [`src/openpi/policies/wuji_policy.py`](src/openpi/policies/wuji_policy.py)).

Expected fields after conversion:

- `observation.state`: 54-dim (14 from both arms + 40 from both dexterous hands)
- `observation.images.cam_high`: head camera
- `observation.images.cam_left_wrist`: left wrist camera
- `observation.images.cam_right_wrist`: right wrist camera
- `action`: 54-dim action sequence

### Step 3: Edit the training config

Training configs live in [`src/openpi/training/config.py`](src/openpi/training/config.py). This repo ships the example **`pi05_wuji_multi_54d`**. Before training, adjust these fields to match your setup:

- `data.repo_id`: path to the LeRobot dataset(s) produced in Step 2 (multiple datasets can be mixed via `LeRobotDataConfig`).
- `checkpoint_base_dir`: output directory for training checkpoints.
- The base checkpoint path inside `weight_loader` (e.g. `pi05_base/params`).
- Hyperparameters: batch size, training steps, learning rate, etc.

Reference snippet (excerpted from `pi05_wuji_multi_54d`):

```python
TrainConfig(
    name="pi05_wuji_multi_54d",
    checkpoint_base_dir="/path/to/your/checkpoints",
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

### Step 4: Compute norm stats and train

```bash
# Compute normalization statistics (only on first run or after data changes)
uv run scripts/compute_norm_stats.py --config-name pi05_wuji_multi_54d

# Launch training (add --overwrite to overwrite an existing experiment)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_wuji_multi_54d \
  --exp-name=my_wuji_run
```

Training logs go to the console and to Weights & Biases; checkpoints are written under `checkpoint_base_dir`.

### Step 5: On-robot inference and deployment

The repo provides the full wuji dual-arm deployment pipeline: a **policy server + ROS2 client** that exchanges observations/actions with wuji-hand-teleop.

Start the policy server (ML side):

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=/path/to/checkpoint
```

Start the robot control client (ROS2 side, talks to wuji-hand-teleop). Run from the project root, in a ROS2 environment whose Python has `openpi-client` and `rclpy` available:

```bash
source /opt/ros/humble/setup.bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

The client subscribes to the joint-state and image topics published by wuji-hand-teleop, packages them into an observation, sends it to the policy server over WebSocket, receives an action chunk, and publishes the chunk back as ROS2 joint commands.

> For full deployment details (YAML config, broker modes, topic lists, troubleshooting), see [`examples/wuji/README.md`](examples/wuji/README.md).

## 3. Directory Layout (wuji-related)

```
wuji-openpi/
├── examples/wuji/                    # wuji dual-arm deployment example
│   ├── config/deploy.yaml           # deployment config (ROS2 topics, broker, control rate)
│   ├── core/                        # ROS2 interface, timestamp sync, utils
│   ├── deploy/                      # deployment entry (main.py / ros_env.py)
│   └── README.md                    # detailed deployment doc
├── src/openpi/
│   ├── policies/wuji_policy.py      # WujiInputs / WujiOutputs data mapping
│   └── training/config.py           # pi05_wuji_multi_54d and other training configs
├── scripts/
│   ├── compute_norm_stats.py        # norm-stats computation
│   ├── train.py                     # training entry
│   └── serve_policy.py              # policy server entry
└── docs/                            # upstream openpi docs
```

## 4. Environment Setup

Dependencies match upstream openpi and are managed with [uv](https://docs.astral.sh/uv/):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The deployment side additionally needs:

- ROS2 Humble (Python 3.10)
- A ROS2 topic namespace that matches wuji-hand-teleop

## 5. What This Fork Adds on Top of Upstream openpi

The fork keeps all original openpi capabilities (pi0, pi0-FAST, pi0.5) and layers in five additional modules targeting the wuji dual-arm robot.

### 5.1 Wuji dual-arm + dual dexterous-hand training and deployment

- **54-dim action space**: 7+7 (arms) + 20+20 (hands), covering every arm joint and every dexterous-hand DOF.
- **Data config**: `LeRobotWujiDataConfig` — native 54-dim pipeline.
- **Training entry** (in [`src/openpi/training/config.py`](src/openpi/training/config.py)): `pi05_wuji_multi_54d` outputs 54-dim actions directly.
- **Full ROS2 deployment package** at [`examples/wuji/`](examples/wuji/): topic subscription/publishing, timestamp synchronization, broker integration — drop-in ready to talk to wuji-hand-teleop.

### 5.2 Multi-dataset training

- New `ConcatLeRobotDataset` and `MultiLeRobotDataset` (with weighted sampling) for mixing multiple recording sessions.
- New `DataConfig` fields:
  - `lerobot_datasets`: list of datasets.
  - `multi_dataset_mode`: switch between concatenation and weighted sampling.
- **Per-dataset prompt transform**: every sub-dataset can carry its own language instruction.

### 5.3 RTG (Real-Time Trajectory Generation)

Smooths the boundary between consecutive action chunks:

- `RTGActionBroker` and a QP-based variant.
- Smoothing utilities:
  - `qp_smooth_prefix`
  - `cubic_smooth_prefix`
  - `build_time_window_old_reference`

Follows the paper `arXiv:2507.17141`: when the client asynchronously receives a new chunk, its prefix is smoothed against the currently executing trajectory before being stitched in.

### 5.4 General tooling improvements

- **`PartialCheckpointWeightLoader`**: when loading a base checkpoint, layers whose shape doesn't match (typically `action_proj`) are silently skipped, making it easy to migrate between different action dimensionalities.
- **[`examples/open_loop_eval.py`](examples/open_loop_eval.py)**: a general open-loop evaluator with built-in RTG comparison, useful for replaying trained checkpoints and benchmarking smoothing strategies.

> All other features (LIBERO / ALOHA / DROID examples, PyTorch backend, etc.) are unchanged from upstream — see the original openpi docs and the corresponding subdirectories under `examples/`.

## 6. References

- Data collection repo: <https://github.com/wuji-technology/wuji-hand-teleop.git>
- Upstream openpi: <https://github.com/Physical-Intelligence/openpi>
- Wuji deployment guide: [`examples/wuji/README.md`](examples/wuji/README.md)
- Training config entry point: [`src/openpi/training/config.py`](src/openpi/training/config.py)

## License

Apache-2.0 (inherited from upstream openpi).
