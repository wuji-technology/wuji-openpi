# Changelog

本文件记录 `wuji-technology/wuji-openpi` **相对于上游 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) `main` 分支的偏离**。仅收录本仓库引入或修改的内容；上游本身的变更请参考其原始仓库历史。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

## [Unreleased]

## [0.1.0] - 2026-05-13

汇总自 fork 起在 `upstream/main` 之上累积的全部偏离。基线：
- upstream HEAD `c23745b`（"Remove redundant tree dependency (#937)"）
- 共同祖先 `981483d`（2025-12-27）
- 本仓库领先 ~157 commit，净 +6540 / −5488 行（`uv.lock` 不再跟踪贡献了 −5407 行）

### Added

**Wuji 双臂双灵巧手机器人栈（内部）**
- `examples/wuji/`：ROS2 部署包（约 2790 行，13 文件），含 `core/`（`DeployConfig`、ROS2 接口、时间戳对齐、四元数/欧拉角/FK 工具）、`deploy/`（`main.py` CLI 入口 + `ros_env.py` Environment 适配）、`config/deploy.yaml`、`package.xml`、`README.md`。
- `src/openpi/policies/wuji_policy.py`：54 维双臂双手 transforms（替换上游 27 维单臂版本）。
- `src/openpi/training/config.py`：
  - `LeRobotWujiDataConfig` 数据配置类；
  - 训练注册项 `pi05_wuji_multi_54d`。

**客户端异步 broker**（`packages/openpi-client/src/openpi_client/`）
- `rtg_action_broker.py`（+596）：Real-Time Trajectory Generation broker，含 QP 变体 `qp_smooth_prefix` / `cubic_smooth_prefix` / `build_time_window_old_reference`。

**多数据集训练**
- `src/openpi/training/data_loader.py`：`ConcatLeRobotDataset`（1:1 拼接）与 `MultiLeRobotDataset`（加权采样，校验权重和为 1.0），均支持 per-dataset prompt transform。
- `src/openpi/training/config.py`：`LeRobotDataset` 数据类、`DataConfig.lerobot_datasets`、`DataConfig.multi_dataset_mode: Literal["concat", "weighted"]`。

**通用工具**
- `src/openpi/training/weight_loaders.py`：`PartialCheckpointWeightLoader` —— 形状不匹配时按正则跳过 `action_in_proj` / `action_out_proj` / `state_proj`，便于跨 action_dim 复用预训练权重。
- `examples/open_loop_eval.py`（+942）：通用 open-loop 策略评估器，支持 JAX / PyTorch 与本地 / `gs://` checkpoint。

**项目自动化**
- `CHANGELOG.md`（本文件）。
- `.github/workflows/auto-release-on-pr.yml`：监听 `auto-release` PR 合并，从 `release/v*` 分支名抽版本号并打 tag。
- `.github/workflows/build.yml`：tag push 触发，解析 CHANGELOG 抽取版本 release notes，创建私有 Release；非 RC 版本同时通过 App token 在公开仓库创建 Release 并推飞书通知。
- `.github/workflows/sync-docs.yml`：`main` 上 `CHANGELOG.md` / `README.md` 变更时同步至公开仓库 PR；RC 版本跳过。
- `.gitignore`：忽略 `open_loop_results*/`。

[Unreleased]: https://github.com/wuji-technology/wuji-openpi/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wuji-technology/wuji-openpi/releases/tag/v0.1.0
