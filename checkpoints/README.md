# checkpoints/

训练好的模型权重，**未纳入 Git**（4 个 .pt，共约 15 GB）。

仓库含 **两个模型变体**，除 `imp_upper` 外配置完全相同：

```
checkpoints/
├── vhe_net_with_weight/
│   ├── best_model.pt          ← 3.6 GB
│   └── last_model.pt          ← 3.6 GB
└── vhe_net_without_weight/
    ├── best_model.pt          ← 3.6 GB
    └── last_model.pt          ← 3.6 GB
```

## 两个变体的区别

| 字段 | `with_weight` | `without_weight` |
|---|---|---|
| `data.imp_upper` | **20.0** | **1.0** |
| 正样本权重 | 20.00 | 1.00 |
| 负样本权重 | 4.49 | 1.00 |
| 正/负权重比 | **4.45×** | **1.00×** |

## 获取 / 重建

见仓库根目录 `README.md`。也可用 `python train.py --config configs/...` 自行训练（约 25 分钟/模型）。
