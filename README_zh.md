# wuji-openpi

[English](README.md) | **中文**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-openpi)](https://github.com/wuji-technology/wuji-openpi/releases)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.5%2B-9cf?logo=google&logoColor=white)](https://github.com/jax-ml/jax)
[![CUDA](https://img.shields.io/badge/CUDA-12-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-22314E?logo=ros&logoColor=white)](https://docs.ros.org/en/humble/)
[![Stars](https://img.shields.io/github/stars/wuji-technology/wuji-openpi?style=social)](https://github.com/wuji-technology/wuji-openpi/stargazers)

> 基于 [openpi](https://github.com/Physical-Intelligence/openpi) 的 fork，用于在**任意双臂 + Wuji Hand 构型**上完成 pi0 / pi0.5 等 VLA 策略的 SFT（监督微调）与实机部署。在上游 openpi 之上，新增了 Wuji 专用的数据处理、训练配置、维度无关的 action 管线，以及完整的 ROS2 部署链路。

<p align="center">
  <img src="docs/assets/demo.gif" width="80%" alt="wuji-openpi 双臂 + Wuji Hand 演示" />
</p>

<p align="center">
  <sub>⏩ 上方为完整演示（2 倍速）。▶️ 原速 + 音频：<a href="docs/assets/demo.mp4">docs/assets/demo.mp4</a></sub>
</p>

## 构型支持

整条链路是**与构型无关**的：任意双臂 + Wuji Hand 组合均可支持，action 维度完全由配置驱动。下表列出本仓库自带的参考配置。

| 本体 | 参考配置 | Action 维度 | 部署示例 |
|---|---|---|---|
| 双臂 + 双 Wuji Hand | `pi05_wuji_multi_54d` | 54（双臂 7+7，双手 20+20） | [`examples/wuji/`](examples/wuji/) |

> **适配其他构型。** 并不需要这套固定的 54 维布局。在 [`examples/wuji/config/deploy.yaml`](examples/wuji/config/deploy.yaml) 中设置 `arm_mode`（`single_left` / `single_right` / `dual`）、`arm_dof`、`hand_dof`，并在训练配置中设置对应的 `action_dim`，同一套数据、训练与部署流程即可在**任意维度**下完成 SFT。详见下文 [适配新构型](#适配新构型)。

## 目录结构

```text
wuji-openpi/
├── examples/wuji/                    # 双臂 + Wuji Hand 部署示例
│   ├── config/deploy.yaml            # 部署配置（手臂/手部 DOF、ROS2 topic、broker、控制频率）
│   ├── core/                         # ROS2 接口、时间戳同步、工具函数
│   ├── deploy/                       # 部署入口（main.py / ros_env.py）
│   └── README.md                     # 部署详细说明
├── src/openpi/
│   ├── policies/wuji_policy.py       # WujiInputs / WujiOutputs 数据映射
│   └── training/config.py            # pi05_wuji_multi_54d 等训练配置
├── scripts/
│   ├── compute_norm_stats.py         # norm stats 计算
│   ├── train.py                      # 训练入口
│   └── serve_policy.py               # Policy server 入口
└── docs/                             # openpi 上游文档
```

## 环境要求

- Linux x86_64
- NVIDIA GPU，CUDA 12
- Python 3.11+，使用 [uv](https://docs.astral.sh/uv/) 安装
- 部署端：ROS2 Humble（Python 3.10），且 ROS2 topic 命名空间与 [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) 一致

## 安装

依赖与上游 openpi 保持一致，使用 [uv](https://docs.astral.sh/uv/) 管理：

```bash
# 1. 克隆
git clone https://github.com/wuji-technology/wuji-openpi
cd wuji-openpi

# 2. 解析环境（跳过 LFS smudge，避免提前拉取大权重）
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

## 工作流

从遥操作采集到闭环实机推理的端到端链路：

```text
┌────────────────────┐    ROS2 mcap     ┌──────────────────┐    LeRobot v2.1    ┌──────────────┐
│  wuji-hand-teleop  │ ───────────────▶ │  数据转换脚本     │ ─────────────────▶ │   SFT 训练   │
│  （遥操作采集）     │                  │  (mcap → lerobot)│                    │  (本仓库)    │
└────────────────────┘                  └──────────────────┘                    └──────┬───────┘
                                                                                       │
                                                                                       ▼
                                                            ┌──────────────────────────────┐
                                                            │  Policy Server (本仓库)       │
                                                            │  WebSocket / ROS2 推理服务    │
                                                            └──────────────┬───────────────┘
                                                                           │ ROS2 obs/action
                                                                           ▼
                                                            ┌──────────────────────────────┐
                                                            │ 双臂 + Wuji Hand (实机执行)   │
                                                            │ wuji-hand-teleop 提供 ROS2 桥 │
                                                            └──────────────────────────────┘
```

### Step 1 —— 采集遥操作数据

使用 [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) 执行遥操作，采集双臂 + 双灵巧手的同步轨迹，以及头部相机、左右腕部相机视频，保存为 **ROS2 mcap** bag。

### Step 2 —— 转换为 LeRobot v2.1 数据集

将 ROS2 mcap 数据转换为 LeRobot v2.1 格式，输出到训练数据目录。数据集字段需与 `WujiInputs` 对齐（详见 [`src/openpi/policies/wuji_policy.py`](src/openpi/policies/wuji_policy.py)）。

转换后期望的字段（以参考的 54 维布局为例）：

- `observation.state`：状态向量（54 维 = 双臂 14 + 双灵巧手 40）
- `observation.images.cam_high`：头部相机
- `observation.images.cam_left_wrist`：左腕相机
- `observation.images.cam_right_wrist`：右腕相机
- `action`：动作序列（维度与 state 一致）

### Step 3 —— 修改训练配置

训练配置位于 [`src/openpi/training/config.py`](src/openpi/training/config.py)，本仓库提供参考配置 **`pi05_wuji_multi_54d`**。训练前需调整：

- `data.repo_id` / `repo_ids`：指向 Step 2 生成的 LeRobot 数据集路径（可在 `LeRobotWujiDataConfig` 中配置多个数据集混合训练）。
- `checkpoint_base_dir`：训练 checkpoint 输出目录。
- `weight_loader` 中的 base checkpoint 路径（如 `pi05_base/params`）。
- `model.action_dim`：设为目标构型的总 DOF（详见 [适配新构型](#适配新构型)）。
- batch size、训练步数、学习率等超参数。

参考片段（节选自 `pi05_wuji_multi_54d`）：

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

### Step 4 —— 计算 norm stats 并训练

```bash
# 计算归一化统计量（仅首次或数据更新时需要）
uv run scripts/compute_norm_stats.py --config-name pi05_wuji_multi_54d

# 启动训练（如需覆盖已有实验，加上 --overwrite）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_wuji_multi_54d \
  --exp-name=my_wuji_run
```

训练日志会写入控制台并同步到 Weights & Biases，checkpoint 默认保存到 `checkpoint_base_dir` 下。

### Step 5 —— 实机推理与部署

部署采用 **policy server + ROS2 客户端** 架构，与 wuji-hand-teleop 交换 observation/action。

```bash
# Policy server（机器学习端）
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_wuji_multi_54d \
  --policy.dir=/path/to/checkpoint
```

```bash
# 机器人控制客户端（ROS2 端）—— 在项目根目录下、
# 并且已 source 含有 openpi-client 与 rclpy 的 ROS2 环境中执行
source /opt/ros/humble/setup.bash
python3 -m wuji.deploy.main -c examples/wuji/config/deploy.yaml
```

客户端会订阅 wuji-hand-teleop 发布的 ROS2 关节状态和图像 topic，将拼接后的 observation 通过 WebSocket 发送给 policy server，获取 action chunk 后再以 ROS2 命令形式发布回机器人执行。

> 更完整的部署细节（YAML 配置、broker 模式、topic 列表、故障排查）请参阅 [`examples/wuji/README.md`](examples/wuji/README.md)。

## 适配新构型

参考配置是 54 维，但**链路中没有任何地方把这个数字写死**。要在不同的双臂 + Wuji Hand 构型上完成 SFT 与部署：

1. **部署 YAML** —— 在 [`examples/wuji/config/deploy.yaml`](examples/wuji/config/deploy.yaml) 中设置维度字段：
   ```yaml
   arm_mode: "dual"   # 或 "single_left" / "single_right"
   arm_dof: 7         # 单臂 DOF
   hand_dof: 20       # 单只 Wuji Hand DOF
   ```
   客户端会根据这些字段构造 observation/action 向量，部署维度随 YAML 变化。
2. **训练配置** —— 在 [`src/openpi/training/config.py`](src/openpi/training/config.py) 中将 `model.action_dim` 设为对应的总 DOF（如 `arm_mode=dual` → `2 * (arm_dof + hand_dof)`），并确保 LeRobot 数据集的 `observation.state` / `action` 宽度一致。
3. **迁移权重** —— `PartialCheckpointWeightLoader` 会自动跳过形状不匹配的层（典型如 `action_proj`），因此在某一维度训练得到的 base checkpoint 可以无需手工改动直接用于另一维度的 SFT。

不同构型之间需要改动的只有这些 —— 数据转换、训练与 ROS2 部署链路完全一致。

## 架构

本仓库保留了上游 openpi 的全部能力（pi0、pi0-FAST、pi0.5），并在其上叠加 Wuji 专用模块：维度无关的 action 管线、多数据集训练、实时轨迹平滑，以及完整的 ROS2 部署包。

<details>
<summary>深入了解 —— 本仓库相对上游 openpi 新增的内容</summary>

### 双臂 + 双灵巧手训练与部署

- **配置驱动的 action 空间**：参考配置 `pi05_wuji_multi_54d` 覆盖双臂 7+7 + 双手 20+20 = 54 DOF，但维度由配置决定 —— 详见 [适配新构型](#适配新构型)。
- **数据配置**：`LeRobotWujiDataConfig` —— 原生手臂 + 灵巧手数据管线。
- **完整 ROS2 部署包** [`examples/wuji/`](examples/wuji/)：topic 订阅/发布、时间戳同步、broker 接入，开箱即用对接 wuji-hand-teleop。

### 多数据集训练

- 新增 `ConcatLeRobotDataset` 与 `MultiLeRobotDataset`（带加权采样），便于将多次采集的数据合并训练。
- `DataConfig` 新增字段：
  - `lerobot_datasets`：多数据集列表。
  - `multi_dataset_mode`：拼接 / 加权采样切换。
- 支持 **per-dataset prompt transform**：每个子数据集可绑定独立的语言指令。

### RTG（Real-Time Trajectory Generation）

针对 chunk 切换时的边界跳变，提供轨迹拼接平滑能力：

- `RTGActionBroker` 及其 QP 变体。
- 平滑工具：`qp_smooth_prefix`、`cubic_smooth_prefix`、`build_time_window_old_reference`。

参考论文 `arXiv:2507.17141`：客户端异步收到新 chunk 后，对其前缀与当前执行中的轨迹做平滑再拼接执行。

### 通用工具改进

- **`PartialCheckpointWeightLoader`**：加载 base checkpoint 时，遇到形状不匹配的层（典型如 `action_proj`）自动跳过，方便在不同 action 维度之间迁移。
- **[`examples/open_loop_eval.py`](examples/open_loop_eval.py)**：通用 open-loop 评估器，内置 RTG 对比，方便对训练好的 checkpoint 做轨迹复现 & 平滑算法对照。

> 其余通用功能（如 LIBERO/ALOHA/DROID 示例、PyTorch 后端等）保持与上游一致，请参考原 openpi 文档与 `examples/` 下对应子目录。

</details>

## 开发

依赖说明与完整的上游模型文档位于 [`docs/`](docs/) 以及 [openpi](https://github.com/Physical-Intelligence/openpi) 官方仓库 —— 本 README 仅聚焦于 Wuji 相关的使用方法。

## 相关项目

- [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop.git) —— 遥操作采集（ROS2 mcap）与实机 ROS2 桥
- [wujihandpy](https://github.com/wuji-technology/wujihandpy) —— Wuji Hand SDK（C++ 内核 + Python 绑定）
- [wujihandros2](https://github.com/wuji-technology/wujihandros2) —— Wuji Hand 的 ROS 2 驱动
- [docs.wuji.tech](https://docs.wuji.tech) —— Wuji 官方文档门户

## 致谢

本项目构建于以下开源项目之上：

- [openpi](https://github.com/Physical-Intelligence/openpi) —— 上游 VLA 训练/推理框架（pi0 / pi0-FAST / pi0.5）
- [LeRobot](https://github.com/huggingface/lerobot) —— 数据集格式与工具链
- [JAX](https://github.com/jax-ml/jax) —— 训练/推理后端

## 贡献者

- [Han Duo](https://github.com/HanDuo-223)

## 引用

如果本项目对你有帮助，欢迎引用：

```bibtex
@software{wuji2026openpi,
  title={Wuji-OpenPI: SFT and Deployment of VLA Policies on Dual-Arm + Wuji Hand Robots},
  author={{Wuji Technology}},
  year={2026},
  url={https://github.com/wuji-technology/wuji-openpi}
}
```

## License

Apache-2.0（继承自上游 openpi）。
