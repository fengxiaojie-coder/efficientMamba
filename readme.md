# EfficientMamba

这是一个面向 UBFC-rPPG 数据集的训练与评估项目，包含数据预处理、模型训练、结果评估和可视化流程。本文档面向 Windows 本机环境，帮助你从零开始创建环境、准备数据、运行训练和评估。

## 1. 先看一下你需要什么

这个项目的典型流程是：

1. 创建 Python 环境
2. 安装依赖
3. 准备 UBFC 数据
4. 把视频转成 `.pt` clip 数据
5. 训练模型
6. 评估模型

如果你只是想先确认项目能否正常启动，可以先运行：

```powershell
python main.py
```

它会做一个简要的数据预览检查，并验证项目的基本导入流程。

---

## 2. 环境要求

建议配置：

- Windows 10/11
- Python 3.12 x64
- Git
- 至少 8GB 内存，推荐 16GB+
- 可选：NVIDIA GPU，训练速度会明显更快

推荐使用 PowerShell 或 CMD 进行命令行操作。

---

## 3. 创建 Python 环境

### 方式 A：使用 venv（推荐，最简单）

在项目根目录执行：

```powershell
cd E:\UclHomework\research project\efficientMamba
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 拒绝执行脚本，可以先运行：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

然后再次激活虚拟环境。

### 方式 B：使用 conda

```powershell
conda create -n efficientmamba python=3.12
conda activate efficientmamba
```

---

## 4. 安装依赖

在激活虚拟环境后，安装项目依赖：

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

这个项目默认使用标准依赖文件：

- `requirements.txt`：通用训练/评估依赖
- `requirements-mamba.txt`：可选的 Mamba 后端依赖

如果你想尝试 Mamba 后端，可以继续查看：

- `docs/MambaInstall.md`

> 如果你是在 Windows 本机上运行，Mamba 后端的安装可能比标准依赖更容易出问题。对于先跑通主流程，优先安装标准依赖即可。

---

## 5. 准备数据

### 5.1 你需要的数据目录

项目默认假设数据目录在：

```text
dataSet/UBFC-rPPG/
```

也就是说，仓库中应该有类似下面的结构：

```text
dataSet/
  UBFC-rPPG/
    subject1/
    subject2/
    ...
```

每个 subject 目录下应包含对应的视频文件和标注文件。

### 5.2 如果你已经有预处理好的 clips

如果你已经生成过 `.pt` clip 文件，可以直接放到下面目录：

```text
data/ubfc_clips/
```

这种情况下可以跳过预处理步骤，直接进行训练。

### 5.3 如果你只有原始视频

需要先把视频转为项目可读取的 `.pt` clip 文件。

---

## 6. 生成预处理后的 clip 数据

如果你手头只有 UBFC 原始视频，执行下面命令：

```powershell
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips --clip_len 128 --stride 32 --roi bbox --size 72
```

这一步会把视频切出多个 clip，并保存成 `.pt` 文件，后续训练和评估都会用到这些文件。

### 常用参数说明

- `--src`：原始数据根目录
- `--out`：输出的 clip 目录
- `--clip_len`：每个 clip 的帧数，默认 128
- `--stride`：滑窗步长，建议从 32 开始试
- `--size`：裁剪后的图像尺寸，默认 72
- `--roi`：ROI 策略，可选 `full`、`haar`、`mediapipe`、`ellipse`、`bbox`

如果你遇到视频解码问题，可以尝试：

```powershell
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips --clip_len 128 --stride 32 --roi bbox --decoder opencv
```

---

## 7. 训练模型

准备好 `data/ubfc_clips` 后，就可以开始训练：

```powershell
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 8 --max_samples 400 --save_dir checkpoints
```

### 训练时常用参数

- `--clips_dir`：clip 数据目录
- `--epochs`：训练轮数
- `--batch_size`：批大小
- `--max_samples`：只用前多少样本做快速测试
- `--save_dir`：checkpoint 保存目录
- `--verbose`：输出更详细日志

如果你想让训练更接近“跨 subject 泛化”的场景，可以加上：

```powershell
python -m src.train --clips_dir data/ubfc_clips --epochs 10 --batch_size 8 --split_mode subject_disjoint --save_dir checkpoints
```

训练结束后，checkpoint 会保存在 `checkpoints/` 目录下，例如：

```text
checkpoints/best.pt
```

---

## 8. 评估模型

训练完成后，使用保存的 checkpoint 做评估：

```powershell
python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt
```

评估结果会输出到：

```text
eval_outputs/
```

其中会包含预测结果、汇总图和统计文件。

---

## 9. 项目目录说明

```text
main.py                  # 快速入口
src/train.py             # 训练脚本
src/eval.py              # 评估脚本
src/preprocessing/      # 数据预处理代码
src/models/              # 模型定义
src/datasets/            # 数据集读取逻辑
checkpoints/             # 训练输出模型
data/ubfc_clips/         # 预处理后的 clip 数据
dataSet/UBFC-rPPG/       # 原始 UBFC 数据目录
results/                 # 结果输出
```

---

## 10. 常见问题

### 10.1 `No module named torch`

说明当前虚拟环境没有安装依赖。重新执行：

```powershell
pip install -r requirements.txt
```

### 10.2 找不到数据目录

检查项目根目录下是否存在：

```text
dataSet/UBFC-rPPG/
```

如果目录不存在，需要把 UBFC 数据下载并放进去。

### 10.3 训练时显存/内存不够

可以减小批大小或只跑少量样本：

```powershell
python -m src.train --clips_dir data/ubfc_clips --epochs 5 --batch_size 2 --max_samples 100
```

### 10.4 视频预处理失败

可以尝试切换解码器：

```powershell
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips --decoder opencv
```

---

## 11. 推荐的最小跑通顺序

如果你想先把项目跑起来，建议按这个顺序：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python main.py
python -m src.preprocessing.ubfc_to_tensors --src dataSet/UBFC-rPPG --out data/ubfc_clips --clip_len 128 --stride 32 --roi bbox --size 72
python -m src.train --clips_dir data/ubfc_clips --epochs 5 --batch_size 4 --max_samples 100 --save_dir checkpoints
python -m src.eval --clips_dir data/ubfc_clips --checkpoint checkpoints/best.pt
```

如果你愿意，我也可以继续帮你把这个 README 再补成“带截图/目录结构/每一步预期输出”的版本。