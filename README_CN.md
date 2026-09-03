# SAGE-MIL

论文题目：**Pathology-Related Semantic Guidance with Normal-Tissue References for Weakly Supervised Whole-Slide Image Classification**

本仓库提供 SAGE-MIL 的训练、评估和分析代码。方法使用 Morphology、Texture 和 Microenvironment 三类病理语义锚点，通过训练侧正常组织参考字典对语义响应进行校正，再使用稀疏语义路由和空间 MIL 完成切片分类。训练阶段还包含基于 Sinkhorn OT 的连续特征替换约束；该路径在训练后用于预测敏感性分析，常规推理只保留分类路径。

## 主要配置

- 20× WSI，256×256 非重叠 patch
- CONCH / PLIP 冻结特征
- 3 个固定语义锚点
- 正常组织参考字典 `M=100`
- 语义 Top-k 比例 `0.30`
- 参考分配温度 `T_e=0.10`
- 校正掩码 `[1,1,0]`
- 训练特征高斯扰动 `σ=0.005`
- instance Top-k：训练 10→1，推理 6
- 训练特征替换 `K_train=5`
- 回顾性分析主设置 `K=200`
- Sinkhorn `ε=0.05`，30 次迭代
- `λ_verify=0.10`，warm-up 20 epochs
- `λ_orth=0.10`
- 分类概率使用 bag/instance 0.5/0.5 融合
- ECE 使用 15 个等宽置信度区间

训练示例：

```bash
PYTHONPATH=src python train/train.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --seed 2022
```

回顾性特征替换分析：

```bash
PYTHONPATH=src:. python audit/replacement_protocol.py \
  --config configs/experiments/camelyon16_conch.yaml \
  --checkpoint /path/to/best.ckpt \
  --budget 200 \
  --output outputs/audit/replacement_curves.csv
```

当前仓库包含源码、实验配置、manifest 工具、评估脚本、机制分析和定量图生成脚本；原始 WSI、第三方预训练权重、实验 checkpoint、预测结果和本地日志不在仓库中发布。
