# 🎯 Attention Heatmap Visualization - 实现总结

## 实现内容

已成功添加 **attention heatmap 可视化**功能到训练流程中。在训练时，模型会自动保存 attention 热力图，帮助你验证模型是否学到了有意义的 ROI（人脸区域）。

## 核心改动

### 1️⃣ 模型修改

**文件**: `src/models/efficientphys_front.py`
**改动**:
- `forward_features()` 新增 `return_attention` 参数
- 当 `return_attention=True` 时，返回格式改为:
  ```python
  output, {'g1': [B, T, 1, 70, 70], 'g2': [B, T, 1, 33, 33]}
  ```
- g1: 早期融合层的 attention gate (appearance 分支生成)
- g2: 晚期融合层的 attention gate (appearance 分支生成)

**文件**: `src/models/efficientphys_mamba.py`
**改动**:
- `forward()` 和 `forward_features()` 支持 `return_attention` 参数
- 自动传递给 EfficientPhysFront

### 2️⃣ 训练脚本修改

**文件**: `src/train.py`
**改动**:
- 每个 epoch 完成后，从验证集采样第一个 batch
- 调用 `model(clips, return_attention=True)` 获取 attention 图
- 用 matplotlib 可视化 G1 和 G2，保存为 PNG
- 输出路径: `checkpoints/attention_epoch_001.png`, `attention_epoch_002.png` 等

### 3️⃣ 演示脚本

**文件**: `demo_attention.py`
- 独立脚本，可快速测试可视化功能
- 生成 4 个样本的 attention 热力图
- 保存到 `checkpoints/demo_attention/`

## 使用方法

### 快速测试 (推荐先运行此命令)
```bash
python demo_attention.py
```
输出:
- `checkpoints/demo_attention/demo_attention_heatmaps.png` - 可视化图像
- `checkpoints/demo_attention/demo_attention_stats.txt` - 统计信息

### 正式训练 (自动保存 attention 热力图)
```bash
python -m src.train \
  --clips_dir data/ubfc_clips \
  --epochs 20 \
  --batch_size 8 \
  --save_dir checkpoints
```

训练过程中会自动生成:
```
checkpoints/
├── attention_epoch_001.png
├── attention_epoch_002.png
├── attention_epoch_003.png
├── ...
├── model_epoch_1.pt
├── best.pt
└── training_metrics.csv
```

## 热力图解读

每张 PNG 图像显示 4 列:
- **列1**: G1 attention，第 0 帧
- **列2**: G1 attention，中间帧
- **列3**: G2 attention，第 0 帧
- **列4**: G2 attention，中间帧

**颜色含义**:
- 🔴 **红色/亮色** = 高 attention (模型聚焦这些区域)
- 🟤 **棕色/暗色** = 低 attention (模型忽略这些区域)

### ✅ 好的迹象
- 热力图在中心（人脸区域）集中
- 从 epoch 到 epoch 逐渐变清晰
- 不是完全均匀（说明学到了空间结构）

### ⚠️ 需要关注的情况
- 热力图完全均匀 → 模型没学到 ROI
- 热力图噪声很多 → 训练不稳定
- 热力图全 0 或全 1 → 可能有问题

## 技术细节

### Attention Gate 工作原理
```
Appearance Branch
    ↓
  Conv (RGB)
    ↓
Attention Gate 1 (G1)  ← 70×70
    ↓
  Conv
    ↓
Attention Gate 2 (G2)  ← 33×33
    ↓
Motion Branch features × G1
    ↓
Motion Branch features × G2
    ↓
Output
```

### 为什么需要两个 Gate?
- **G1** (早期): 捕捉粗粒度人脸位置
- **G2** (晚期): 捕捉细粒度人脸细节

## 依赖

确保已安装:
```bash
pip install torch matplotlib numpy
```

## 文档参考

详细使用指南见: [docs/AttentionVisualization.md](docs/AttentionVisualization.md)

## 故障排查

### Q: 训练时没有保存 attention 图像
**A**:
- 检查是否安装了 matplotlib: `pip install matplotlib`
- 检查 `checkpoints/` 目录是否存在且可写
- 查看训练日志是否有错误

### Q: Attention 图像全是一个颜色
**A**:
- 这表示模型没学到空间结构
- 尝试降低 weight_decay: `--weight_decay 1e-5`
- 或禁用增强: `--no_augment`

### Q: 想只在某些 epoch 保存热力图
**A**:
- 修改 `train.py` 中的可视化代码
- 加入条件: `if epoch % 5 == 0:` (每 5 个 epoch 保存一次)

## 下一步

运行训练后，定期检查 attention 热力图:
1. 早期 epoch：热力图应该逐渐变清晰
2. 中期 epoch：热力图应该集中在人脸区域
3. 后期 epoch：热力图应该保持稳定，值在合理范围

这能帮你快速判断模型是否在正确学习！
