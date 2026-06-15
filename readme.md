# EfficientMamba

这是一个把 EfficientPhys 前端、UBFC / BH-rPPG 预处理、训练、评估和 Mamba 后端串起来的项目。

## 快速入口

- 训练说明：`docs/Training.md`
- EfficientPhys 前端说明：`docs/EfficientPhysFront.md`
- Mamba 安装说明：`docs/MambaInstall.md`
- Mamba 兼容依赖：`requirements-mamba.txt`

## 最近改动摘要

- 默认预处理 ROI 策略改为 `mediapipe`（若未安装会回退），并保留 `--roi` 可选项切换为 `full` 或 `haar`。
- 预处理现在在每个 `.pt` payload 中保存 `roi_type` 与 `roi_bbox`，便于可复现比较与后续分析。
- 在模型前端的 attention gating 中加入了基于显式 ROI 的空间偏置（乘性 bias map），将显式 ROI 与隐式空间注意力结合。
- 训练端新增轻量数据增强（`--augment`）与 L2 正则（`--weight_decay` 参数，AdamW 支持）。
- 脚本输出已由直接 `print()` 改为 `logging`，便于静默或详细级别控制。

## 项目结构

- `data/ubfc_clips/`：UBFC 预处理后的 clip 张量，供训练/评估直接读取
- `data/bh_rppg_clips/`：BH-rPPG 预处理后的 clip 张量，结构与 UBFC 一致
- `src/preprocessing/ubfc_to_tensors.py`：UBFC 视频转 clip 脚本
- `src/preprocessing/bh_rppg_to_tensors.py`：BH-rPPG 二级图片目录转 clip 脚本
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
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips
```

参数说明：

- `--src`：UBFC 原始数据目录，每个子目录对应一个 subject
- `--out`：生成的 clip 文件目录
- `--clip_len`：每个 clip 的帧数，默认 `16`
- `--stride`：滑窗步长，默认 `8`
- `--size`：缩放后的帧大小，默认 `72`

### 2.1 预处理 BH-rPPG 图片序列

如果你要把 `dataSet/BH-rPPG` 里的二级目录图片转成训练可读的 `.pt`，可以直接运行：

```bash
python -m src.preprocessing.bh_rppg_to_tensors --src dataSet/BH-rPPG --out data/bh_rppg_clips
```

这个脚本会做三件事：

- 扫描每个一层目录，例如 `dataSet/BH-rPPG/0_0`
- 读取其二层目录里的 `Frame_XXXXX.png` 图片
- 按 clip 生成 `[T, 6, H, W]` 张量并保存为 `.pt`

输出目录会按样本名分文件夹组织，例如：

```text
data/bh_rppg_clips/
	0_0/
		0_0_clip_0000.pt
		0_0_clip_0001.pt
```

参数说明：

- `--src`：BH-rPPG 原始目录
- `--out`：生成的 clip 文件目录
- `--clip_len`：每个 clip 的帧数，默认 `16`
- `--stride`：滑窗步长，默认 `8`
- `--size`：缩放后的帧大小，默认 `72`
- `--max_subjects`：只处理前多少个样本目录，便于快速测试
- `--max_clips_per_subject`：每个样本最多保存多少个 clip

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

如果你先把 BH-rPPG 转成了 `data/bh_rppg_clips/`，也可以直接替换数据目录：

```bash
python -m src.train --clips_dir data/bh_rppg_clips --epochs 10 --batch_size 16 --max_samples 400
```

如果你想按 subject 划分（推荐用于评估泛化），可以使用 `--split_mode subject_disjoint`：

```bash
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 16 --split_mode subject_disjoint
```

你也可以限制使用的 subject 数量或对每个 subject 限制 clip 数：

```bash
# 只使用前 8 个 subject
python -m src.train --clips_dir data/ubfc_clips --epochs 16 --max_subjects 41 --split_mode subject_disjoint

# 每个 subject 最多取 10 个 clip
python -m src.train --clips_dir data/ubfc_clips --per_subject_limit 10 --split_mode subject_disjoint
```

常用参数：

- `--clips_dir`：预处理后的 clip 目录
- `--epochs`：训练轮数
- `--batch_size`：批大小
- `--max_samples`：只跑前多少个样本，适合快速验证
- `--save_dir`：checkpoint 输出目录，默认 `checkpoints`

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

- 当前 `src/eval.py` 是针对预处理好的 `.pt` clips 运行的（由 `src/preprocessing/*.py` 生成）。请把 `--clips_dir` 指向包含 `.pt` 文件的目录（例如 `data/ubfc_clips` 或 `data/bh_rppg_clips`）。
- 新增 `--dataset` 参数用于标注评估数据集类型，输出会写入 `eval_outputs/<dataset>/`（例如 `eval_outputs/bh_rppg/`）。示例：

```bash
python -m src.eval --clips_dir data/bh_rppg_clips --checkpoint checkpoints/best.pt --dataset bh_rppg
```

- 评估结果会输出到 `eval_outputs/<dataset>/`，包括预测 CSV（`predictions_<dataset>.csv`）、每个 subject 的汇总图和 `subject_summary_<dataset>.csv`。同一个 `subject` 的 clips 会自动合并，额外生成按 subject 汇总的 PPG 和 HR 预测图。

- 如果你想直接评估原始视频（不是 `.pt` clips），请先使用 `src/preprocessing/ubfc_to_tensors.py` 或 `src/preprocessing/bh_rppg_to_tensors.py` 将视频/图片序列转换为 clips，再运行 `src/eval.py`。

### 6. 只看数据预览

如果只是想先确认数据是否能读到，可以直接运行：

```bash
python main.py
```
