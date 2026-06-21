"""Quick test to generate valid PNG and verify visualization."""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 生成简单的热力图
print("生成测试热力图...")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# 创建示例 attention 数据
np.random.seed(42)
for i, ax in enumerate(axes.flat):
    # 生成高斯分布的热力图（中心亮，边缘暗）
    y = np.linspace(-3, 3, 70)
    x = np.linspace(-3, 3, 70)
    X, Y = np.meshgrid(x, y)
    attention = np.exp(-(X**2 + Y**2) / 2)
    # 加一点噪声
    attention = attention + 0.1 * np.random.randn(70, 70)
    attention = np.clip(attention, 0, 1)

    im = ax.imshow(attention, cmap='hot')
    ax.set_title(f'Attention Gate - Sample {i//2} Frame {i%2}', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle('Attention Heatmap Visualization Test\n(Bright=High Attention, Dark=Low Attention)',
             fontsize=13, fontweight='bold')
plt.tight_layout()

output_path = Path('checkpoints/demo_attention/test_heatmap.png')
plt.savefig(output_path, dpi=120, bbox_inches='tight')
plt.close()

# 验证
file_size = output_path.stat().st_size
with open(output_path, 'rb') as f:
    is_valid_png = f.read(8).startswith(b'\x89PNG')

print(f"✓ 文件已保存: {output_path}")
print(f"  文件大小: {file_size} bytes")
print(f"  PNG 格式有效: {is_valid_png}")
print(f"\n✓ 图像已生成！请用图像查看器打开: {output_path}")
