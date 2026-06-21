# EfficientMamba

这是一个把 EfficientPhys 前端、UBFC 预处理、训练、评估和 Mamba 后端串起来的项目。

## 快速入口

- 训练说明：`docs/Training.md`
- EfficientPhys 前端说明：`docs/EfficientPhysFront.md`
- Mamba 安装说明：`docs/MambaInstall.md`
- Mamba 兼容依赖：`requirements-mamba.txt`

## 最近改动摘要

- 默认预处理 ROI 策略改为 `bbox`，并保留 `--roi` 可选项切换为 `full`、`haar`、`mediapipe`、`ellipse` 或 `bbox`。
- `bbox` 表示一个紧凑的外接矩形裁剪（compact bbox crop），会尽量把脸部区域裁得更紧。
- 预处理现在在每个 `.pt` payload 中保存 `roi_type` 与 `roi_bbox`，便于可复现比较与后续分析。
- 在模型前端的 attention gating 中加入了基于显式 ROI 的空间偏置（乘性 bias map），将显式 ROI 与隐式空间注意力结合。
- 训练端新增轻量数据增强（`--augment`）与 L2 正则（`--weight_decay` 参数，AdamW 支持）。
- 脚本输出已由直接 `print()` 改为 `logging`，便于静默或详细级别控制。

## 项目结构

- `data/ubfc_clips/`：UBFC 预处理后的 clip 张量，供训练/评估直接读取
- `src/preprocessing/ubfc_to_tensors.py`：UBFC 视频转 clip 脚本
- `src/datasets/ubfc_dataset.py`：递归加载 `.pt` clip 文件的数据集封装

## 运行流程

### 1. 准备环境

建议先创建并激活虚拟环境：

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
```

如果你在 Windows 上跑大模型依赖，优先使用 WSL 的本地磁盘环境；项目默认走 Mamba，环境不满足时才会自动回退。

### 2. 预处理 UBFC 数据

先把 UBFC 原始数据转成 clip 张量，训练和评估都会读取这些 `.pt` 文件：

```bash
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips --clip_len 128 --stride 32 --roi bbox
```

参数说明：

- `--src`：UBFC 原始数据目录，每个子目录对应一个 subject
- `--out`：生成的 clip 文件目录
- `--clip_len`：每个 clip 的帧数，默认 `128`
- `--stride`：滑窗步长，默认 `8`
- `--size`：缩放后的帧大小，默认 `72`

### 2.1 生成 ROI 示例图

如果你想先确认 ROI 策略（特别是 `bbox` 的紧凑矩形裁剪）在样本上的实际效果，可以运行可视化脚本：

```bash
python -m src.utils.visualize_roi_case --dataset ubfc --sample_dir dataSet/UBFC-rPPG/subject9 --roi bbox --frame_index 20 --out results/roi_visualization
```

输出文件会保存在 `results/roi_visualization/`，例如：

- `subject12_bbox_roi_example.png`

### 3. 先做一次快速检查

可以先跑一个 smoke test，确认默认 backbone 已经是 Mamba：

```bash
PYTHONPATH=. python -c "from src.models.efficientphys_mamba import smoke_test; print(smoke_test())"
```

### 4. 训练

默认训练入口会直接使用 Mamba，不需要额外开关：

静默（默认，INFO 级别）：

```bash
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 16 --max_samples 400
```

详细（DEBUG 级别）：

```bash
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 16 --max_samples 400 --verbose
```

如果你想按 subject 划分（推荐用于评估泛化），可以使用 `--split_mode subject_disjoint`：

```bash
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 16 --split_mode subject_disjoint
```

你也可以限制使用的 subject 数量或对每个 subject 限制 clip 数：

```bash
# 只使用前 8 个 subject
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --max_subjects 40 --split_mode subject_disjoint

# 每个 subject 最多取 10 个 clip
python -m src.train --clips_dir data/ubfc_clips --per_subject_limit 10 --split_mode subject_disjoint
```

常用参数：

- `--clips_dir`：预处理后的 clip 目录
- `--epochs`：训练轮数
- `--batch_size`：批大小
- `--max_samples`：只跑前多少个样本，适合快速验证
- `--save_dir`：checkpoint 输出目录，默认 `checkpoints`
- `--neg_pearson_coef`：负皮尔森相关损失权重，默认 `0.1`（启用）

#### 高级：两阶段训练（Two-stage Training）

如果发现模型容易陷入局部最优（例如热力图坍缩，或心率预测方差过小），可以启用两阶段训练策略：

**第一阶段（前10-20个Epoch）**：重皮尔森，轻MAE
- 目的：只让模型对齐波形的频率和趋势
- 模型会自动把空间注意力（Attention Map）从背景转移到人脸皮肤
- 原因：只有皮肤里有心跳引起的周期性趋势

**第二阶段（后续训练）**：引入MAE约束
- 目的：精细化调整脉搏波的幅值、波峰和重搏脉峰等细节

用法：

```bash
# 启用两阶段训练：前10个epoch只用Pearson (mae_coef=0.01)，后续用完整MAE+Pearson
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --two_stage_training \
  --stage1_epochs 5 --stage1_mae_coef 0.01 --stage2_mae_coef 1.0 \
  --neg_pearson_coef 0.1 --split_mode subject_disjoint
```

参数说明：
- `--two_stage_training`：启用两阶段模式
- `--stage1_epochs`：第一阶段的 epoch 数（默认 `10`）
- `--stage1_mae_coef`：第一阶段 MAE 权重（默认 `0.01`，可设为 `0.0` 完全禁用MAE）
- `--stage2_mae_coef`：第二阶段 MAE 权重（默认 `1.0`，完整MAE）

损失函数公式：
$$L_{\text{total}} = \alpha \cdot L_{\text{MAE}} + \beta \cdot L_{\text{Pearson}}$$

其中：
- $\alpha$（MAE系数）：阶段1为 `stage1_mae_coef`，阶段2为 `stage2_mae_coef`
- $\beta$（皮尔森系数）：两阶段都是 `neg_pearson_coef`
- $L_{\text{MAE}}$ = 预测值与真值的平均绝对误差
- $L_{\text{Pearson}}$ = $1 - r$（其中 $r$ 为批次内皮尔森相关系数）

### 5. 评估

训练完成后，用保存的最佳 checkpoint 做评估：

```bash
python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt
```

详细（DEBUG 级别）：

```bash
python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt --verbose
```

评估说明：

- 当前 `src/eval.py` 是针对预处理好的 `.pt` clips 运行的（由 `src/preprocessing/ubfc_to_tensors.py` 生成）。请把 `--clips_dir` 指向包含 `.pt` 文件的目录（例如 `data/ubfc_clips`）。
- 评估结果会输出到 `eval_outputs/ubfc/`，包括预测 CSV（`predictions_ubfc.csv`）、每个 subject 的汇总图和 `subject_summary_ubfc.csv`。同一个 `subject` 的 clips 会自动合并，额外生成按 subject 汇总的 PPG 和 HR 预测图。

- 如果你想直接评估原始视频（不是 `.pt` clips），请先使用 `src/preprocessing/ubfc_to_tensors.py` 将视频转换为 clips，再运行 `src/eval.py`。

### 6. 只看数据预览

如果只是想先确认数据是否能读到，可以直接运行：

```bash
python main.py
```
