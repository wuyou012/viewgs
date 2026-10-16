# 主流程代码地图

## 范围

当前目标是整理与复现，不改变重建、轨迹插值、损失函数或模型选择。DUSt3R 与 MASt3R 的切换属于后续实验，不属于本轮清理。

## 唯一主链路

```text
scene/images
  -> InstantSplat/init_geo.py
  -> scene/sparse_2/{0,1}
  -> ViewCrafter/inference_diff.py
  -> ViewCrafter/viewcrafter_sparse_without_inter_only.py
  -> diffusion/<scene>/{diffusion,gt,render0}
  -> work5/train_mip_diff_1.py
  -> reconstruction/<scene>
```

## 有效入口与职责

| 阶段 | 有效入口 | 主要职责 |
|---|---|---|
| 几何初始化 | `InstantSplat/init_geo.py` | MASt3R 点云、两端相机、测试相机初始化 |
| 插帧生成 | `ViewCrafter/inference_diff.py` | 参数解析与 ViewCrafter 调度 |
| 插帧实现 | `ViewCrafter/viewcrafter_sparse_without_inter_only.py` | DUSt3R 几何、点云渲染、扩散生成 |
| ViewGS | `work5/train_mip_diff_1.py` | 两张真实图与 23 张中间图监督下的 Gaussian 训练 |

## 必需模块

- InstantSplat：`mast3r/`、`dust3r/`、`utils/sfm_utils.py`、`utils/camera_utils.py`、`scene/colmap_loader.py`。
- ViewCrafter：`lvdm/`、`extern/dust3r/`、`configs/`、`utils/`、`render_colmap_pcd_2.py`。
- ViewGS：`arguments/`、`gaussian_arguments/`、`gaussian_renderer/`、`scene/`、`gaussian_scene/`、`gaussian_utils/`、`lpipsPyTorch/`、`run_point_cloud_densification.py` 及两个 CUDA 子模块。

## 不进入主代码

- 所有 `*.bak*`、`* copy*`、`*_old.py`。
- `output*`、`logs/`、`wandb/`、可视化和指标结果目录。
- `run_rebuttal_*`、权重扫描、消融与相机评估脚本；它们应进入独立的 `experiments/`，不能和主入口并列。
- 模型权重、数据集和编译产物。

## 目标目录

```text
viewgs-repro/
  configs/           # 单一 YAML/环境变量配置
  scripts/           # bootstrap、完整流程、仅重建
  src/adapters/      # 三个外部项目的薄适配层
  experiments/       # 非主流程消融与评估
  docs/              # 架构、问题、验证和权重清单
  tests/             # 输入验证、命令构造、端到端冒烟测试
```

## 清理顺序

1. 固定当前输出基线和版本信息。
2. 提取有效文件及其依赖闭包，不做算法修改。
3. 将硬编码参数集中到配置文件。
4. 把实验入口移入 `experiments/`，备份和产物不迁移。
5. 对提取后的代码重新运行 Barn 端到端测试。
6. 只有输出行为一致后，才进行类拆分、重命名和去重复代码。

