# NoMaD NavArena Server Design

## Overview

在 `visualnav-transformer` 仓库中新增 NavArena 评测服务端，将 NoMaD 模型封装为 WebSocket 服务，供 `navarena-bench` 进行 ImageNav 任务评测。

## 目标

- 支持 ImageNav（目标图像导航）评测任务
- 基于 `navarena-server` 的 `NavigationModelServer` 基类
- 支持 batch 推理（多 episode 并行评测）
- 使用 PD 控制器将 NoMaD 路点输出转换为 navarena 位移动作格式
- 默认使用 `nomad.yaml` + `nomad.pth` 配置，支持通过 CLI 参数覆盖

## 项目结构

**注意**：包名为 `nomad_server`（而非 `navarena_server`），以避免与 pip 安装的 `navarena_server` 包产生命名冲突。

```
visualnav-transformer/
├── nomad_server/
│   ├── __init__.py            # 包导出
│   ├── nomad_server.py        # NoMaDServer（继承 NavigationModelServer）
│   ├── nomad_inference.py     # NoMaDInference（模型加载、推理封装）
│   ├── image_utils.py         # 图像预处理（从 deployment/src/utils.py 提取）
│   ├── pd_controller.py       # PDController（路点 → navarena 位移动作）
│   ├── __main__.py            # CLI 入口
│   └── tests/                 # 单元测试
├── deployment/                # 已有 ROS 部署（不修改）
├── train/                     # 已有训练代码（不修改）
└── diffusion_policy/          # 已有扩散子包（不修改）
```

## 组件设计

### 1. NoMaDInference（nomad_inference.py）

封装 NoMaD 的完整推理流程。

#### 职责

- 加载模型配置（YAML）和 checkpoint
- 初始化 DDPM scheduler
- 管理每个 slot 的 context queue（历史观测帧）
- 执行推理：图像预处理 → vision encoder → DDPM 去噪 → get_action

#### 接口

```python
class NoMaDInference:
    def __init__(
        self,
        config_path: str,          # nomad.yaml 路径
        checkpoint_path: str,      # 模型权重路径
        device: str = "cuda:0",
        num_samples: int = 8,      # 扩散采样数
        waypoint_index: int = 2,   # 从轨迹中选取的路点索引
    ):
        ...

    def reset(self, slot_id: str) -> None:
        """重置指定 slot 的 context queue（episode 开始时调用）"""

    def infer(
        self,
        obs_image: np.ndarray,     # 当前帧 RGB (H, W, 3) uint8
        goal_image: np.ndarray,    # 目标图像 RGB (H, W, 3) uint8
        slot_id: str,
    ) -> tuple[np.ndarray | None, float | None]:
        """
        单个 slot 的推理。
        返回 (waypoint, distance)，如果 context 帧不足返回 (None, None)。
        waypoint 形状为 (2,)，表示机体坐标系下的目标位移 (dx, dy)。
        """

    def batch_infer(
        self,
        obs_images: dict[str, np.ndarray],
        goal_images: dict[str, np.ndarray],
    ) -> dict[str, tuple[np.ndarray | None, float | None]]:
        """
        批量推理：将多个 slot 的观测拼 batch 一次前向。
        返回 slot_id → (waypoint, distance) 的映射。
        """
```

#### 推理流程细节

1. **Context Queue 管理**：
   - 每个 slot 维护独立的 FIFO 队列，长度 `context_size + 1`
   - 每步将当前观测帧（转为 PIL）加入队列
   - 队列未满时返回 `(None, None)`

2. **图像预处理**：
   - 复用 `deployment/src/utils.py:transform_images()`
   - Resize 到 `image_size`（默认 `[96, 96]`），ImageNet 归一化
   - Context 帧在 channel 维拼接
   - Goal 图像单独处理

3. **Vision Encoder**：
   - `model('vision_encoder', obs_img=..., goal_img=..., input_goal_mask=0)`
   - `input_goal_mask=0` 表示使用目标图像（ImageNav 模式）
   - 输出 `obsgoal_cond` 条件向量

4. **距离预测**：
   - `model('dist_pred_net', obsgoal_cond=obsgoal_cond)`
   - 输出标量距离值

5. **DDPM 去噪**：
   - 初始化 `(num_samples, len_traj_pred, 2)` 的高斯噪声
   - DDPMScheduler 的 `num_diffusion_iters` 步反向去噪
   - 每步调用 `model('noise_pred_net', sample=..., timestep=..., global_cond=obs_cond)`

6. **动作提取**：
   - 使用 `navarena_bench.utils.action_utils.get_action()` 反归一化 + cumsum
   - 取第一条样本的 `waypoint_index` 处路点
   - 返回 `(waypoint, distance)`

### 2. PDController（pd_controller.py）

从 `deployment/src/pd_controller.py` 移植核心逻辑，去掉 ROS 依赖。

#### 接口

```python
class PDController:
    def __init__(
        self,
        max_v: float = 0.5,   # 最大线速度 (m/s)
        max_w: float = 1.0,   # 最大角速度 (rad/s)
        dt: float = 1.0,      # 时间步长 (s)
    ):
        ...

    def control(self, waypoint: np.ndarray) -> dict[str, float]:
        """
        将 2D 路点 (dx, dy) 转换为 navarena 动作格式。

        PD 控制逻辑：
        - v = clip(dx / dt, 0, max_v)     # 前向速度
        - w = clip(arctan(dy/dx) / dt, -max_w, max_w)  # 角速度

        返回：
        - {"x": v*dt, "y": 0.0, "yaw": w*dt}
        - 差分驱动假设：无横向位移，横向偏差通过旋转补偿
        """
```

#### 参数来源

- `max_v`, `max_w`: 可通过 CLI 参数或 `robot.yaml` 配置
- `dt`: navarena 环境的仿真步长

### 3. NoMaDServer（nomad_server.py）

组合 NoMaDInference 和 PDController，实现 `NavigationModelServer` 接口。

#### 接口

```python
class NoMaDServer(NavigationModelServer):
    def __init__(
        self,
        inference: NoMaDInference,
        controller: PDController,
    ):
        ...

    async def predict(self, observation: dict, ctx: SessionContext) -> dict:
        """
        处理单个观测，返回动作。

        1. 从 observation["rgb"] 提取当前帧（取第一个相机）
        2. 从 observation["goal_image"] 提取目标图像
        3. 调用 inference.infer() 获取路点
        4. 调用 controller.control() 转换为动作
        """

    async def on_episode_start(self, task_info: dict, ctx: SessionContext) -> None:
        """重置对应 slot 的 context queue。"""

    async def on_episode_end(self, result: dict, ctx: SessionContext) -> None:
        """清理对应 slot 的状态。"""

    async def batch_predict(
        self,
        observations: dict[str, dict],
        contexts: dict[str, SessionContext],
    ) -> dict[str, dict]:
        """
        批量推理：
        1. 收集所有 slot 的观测和目标图像
        2. 调用 inference.batch_infer() 一次前向
        3. 逐 slot 调用 controller.control() 转换动作
        """
```

### 4. CLI 入口（__main__.py）

```bash
python -m navarena_server \
    --config train/config/nomad.yaml \
    --checkpoint model_weights/nomad.pth \
    --host 0.0.0.0 \
    --port 8765 \
    --device cuda:0 \
    --waypoint-index 2 \
    --num-samples 8 \
    --max-v 0.5 \
    --max-w 1.0 \
    --dt 1.0
```

使用 `argparse` 解析参数，构造 `NoMaDInference` + `PDController` + `NoMaDServer`，调用 `navarena_server.serve()` 启动 WebSocket 服务。

## 数据流

```
navarena-bench                   NoMaD Server
    │                                │
    │── EPISODE_START ──────────────►│ on_episode_start(): reset context queue
    │                                │
    │── OBSERVATION ────────────────►│ predict():
    │   {rgb, goal_image, pose}      │   1. 提取 RGB + goal_image
    │                                │   2. 更新 context queue
    │                                │   3. transform_images()
    │                                │   4. vision_encoder(obs, goal, mask=0)
    │                                │   5. DDPM denoise → waypoints
    │                                │   6. get_action() → 选定路点
    │◄── ACTION ─────────────────────│   7. pd_control() → {x, y, yaw}
    │   {x, y, yaw}                  │
    │                                │
    │── EPISODE_END ────────────────►│ on_episode_end(): cleanup
```

## 依赖

### 运行时依赖

- `navarena-server`（NavigationModelServer, serve）
- `torch`, `torchvision`
- `diffusers`（DDPMScheduler）
- `efficientnet-pytorch`
- `vint_train`（已有，`pip install -e train/`）
- `diffusion_policy`（已有，`pip install -e diffusion_policy/`）
- `Pillow`, `numpy`, `PyYAML`

### 安装方式

```bash
# 在 nomad conda 环境中
conda activate nomad_train  # 或 vint_deployment

# 最小依赖：navarena-server（提供 NavigationModelServer 基类和 serve）
pip install -e /path/to/NavArena/navarena-server

# 可选：navarena-bench（如果需要使用其 action_utils 等工具）
pip install -e /path/to/NavArena/navarena-bench

# 已有依赖
pip install -e train/
pip install -e diffusion_policy/
```

> 注意：`get_action` 和 `unnormalize_data` 工具函数在本模块内自行实现（从
> `train/vint_train/training/train_utils.py` 移植），不强制依赖 `navarena-bench`。

## 配置

### 默认值

| 参数 | 默认值 | 说明 |
|------|--------|------|
| config | `train/config/nomad.yaml` | 模型配置 |
| checkpoint | `model_weights/nomad.pth` | 模型权重 |
| device | `cuda:0` | GPU 设备 |
| waypoint_index | 2 | 轨迹路点选取索引 |
| num_samples | 8 | DDPM 采样数 |
| max_v | 0.5 | 最大前向速度 (m/s) |
| max_w | 1.0 | 最大角速度 (rad/s) |
| dt | 1.0 | 仿真步长 (s) |
| host | 0.0.0.0 | 监听地址 |
| port | 8765 | 监听端口 |

### 评测配置

使用已有的 `navarena-bench/configs/eval/imagenav_nomad_eval.yaml`，确保 `server.url` 指向 NoMaD 服务端。

## 错误处理

- context queue 未满时返回零动作 `{x: 0, y: 0, yaw: 0}`
- 模型推理异常捕获后记录日志并返回零动作
- GPU OOM 时记录错误并优雅降级

## 测试计划

1. 单元测试：PDController 的路点→动作转换
2. 集成测试：NoMaDInference 加载模型并推理
3. 端到端测试：启动服务端，用 navarena-bench 运行少量 episode
