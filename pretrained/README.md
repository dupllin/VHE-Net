# pretrained/

本目录用于放置 **LucaVirus** 蛋白语言模型权重，**未纳入 Git**（3.6 GB）。

## 需要的文件

```
pretrained/lucaVirus/
├── config.json
├── configuration_lucavirus.py
├── modeling_lucavirus.py
├── tokenization_lucavirus.py
├── model.safetensors          ← 3.6 GB，需单独下载
├── tokenizer_config.json
└── vocab.json
```

## 模型规格（实测）

| 项 | 值 |
|---|---|
| 参数量 | **950,889,767（0.95 B）** |
| 层数 | **12** |
| hidden_size | 2560 |
| num_attention_heads | 20 |
| ffn_dim | 10240 |
| vocab_size | 39 |
| 精度 | float32（约 3.54 GiB）|

> ⚠️ **注意**：该权重为 **0.95B** 参数，**不是 3B**。
> 核对方式：`sum(p.numel() for p in model.parameters())`

## 获取方式

见仓库根目录 `README.md` 的「模型权重与缓存」一节。
