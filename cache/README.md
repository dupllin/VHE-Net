# cache/

预计算的中间特征缓存，**未纳入 Git**（约 1 GB）。

## 需要的文件

```
cache/
├── ids_km_cache_934/              ← 934 个病毒的 token id + k-mer 谱缓存（936 文件，654 MB）
└── cls_windows_cache_934.pt       ← 窗口 CLS 特征（934 病毒 × ~59 窗口 × 2560 维，360 MB）
```

两种格式，注意区别：

| 路径 | 格式 | 谁读它 |
|---|---|---|
| `cache/ids_km_cache_934/` | **目录**：每病毒一个 `.pt` + `index.json` | `train.py` / `predict.py`（经 `FullWindowIdsHistPosData` → `load_ids_cache`） |
| `cache/cls_windows_cache_934.pt` | **单个 `.pt`**：`dict{virus: Tensor[Nwin, 2560]}` | `train.py:211`、`predict.py:84` 直接 `torch.load` |

## 为什么需要缓存

LucaVirus 是 0.95 B 模型，若每个 epoch 都重新编码 934 条病毒序列会非常慢。
预计算后，**每个 epoch 仅需约 5 秒**（编码器不进计算图）。
`train.py` 从不调用编码器前向 —— 它只读这两个缓存。

## 如何重建

两个缓存各有一个命令行入口，`--help` 可看全部参数。

```bash
# ① token id + k-mer 谱缓存（纯 CPU，输出目录格式）
python -m vhenet.encode_ids \
    --fasta data/virus_sequences.fasta \
    --cache-dir cache/ids_km_cache_934 \
    --kmer 6

# ② 窗口 CLS 特征（需要 GPU，输出目录格式 + 打包单文件）
python -m vhenet.encode \
    --fasta data/virus_sequences.fasta \
    --model-path pretrained/lucaVirus \
    --cache-dir cache/cls_windows_cache_934 \
    --packed-path cache/cls_windows_cache_934.pt \
    --window 1022 --stride 512
```

> 重建耗时取决于 GPU，约 10~30 分钟。

### 也可以作为函数调用

```python
from vhenet.encode_ids import encode_ids_to_cache
from vhenet.encode import encode_fasta_to_cache

# ① id 缓存（不需要 tokenizer，也不需要 GPU）
encode_ids_to_cache("data/virus_sequences.fasta",
                    "cache/ids_km_cache_934", kmer=6)

# ② 窗口 CLS 特征
from transformers import AutoModel
import torch
device = torch.device("cuda:0")
model = AutoModel.from_pretrained("pretrained/lucaVirus",
                                  trust_remote_code=True).to(device).eval()
encode_fasta_to_cache("data/virus_sequences.fasta",
                      "cache/cls_windows_cache_934",
                      model=model, device=device,
                      packed_path="cache/cls_windows_cache_934.pt")
```

### 校验已重建的 id 缓存

```python
from vhenet.encode import verify_gene_encoding
verify_gene_encoding("cache/ids_km_cache_934", "data/virus_sequences.fasta")
# -> gene 编码校验: 6/6 个 (病毒, 窗口) 完全一致
```

## ⚠️ 重建时的两个坑

### 1. 不要用 `tokenizer(..., seq_type="gene")` 生成 id

`AutoTokenizer.from_pretrained("pretrained/lucaVirus")` 读不到 `vocab_type`
（`tokenizer_config.json` 里没有该字段），构造函数回退到默认 `"gene_prot"`，
于是 `seq_type` 参数被**静默忽略**，DNA 字符被当作蛋白字母 tokenize：

```
正确（gene 词表）: 'GAAT' -> gene_seq_replace '4112' -> token id [8, 5, 5, 6]
错误（gene_prot）: 'GAAT' -> 原样查表           -> token id [12, 11, 11, 17]
```

**不会抛异常**，但特征全错。`vhenet.encode.tokenize_gene_ids()` 显式完成
gene 映射，不经过 tokenizer；两个缓存构建函数都已改用它。

### 2. `cls_windows_cache_934.pt` 里是**已白化**的特征

官方构建脚本先做白化再落盘：

```python
wh = (cls - wh_mean) @ wh_W        # wh_mean / wh_W 来自 data/whiten_stats.pt
```

而 `encode_fasta_to_cache(packed_path=...)` 写出的是**未白化**的原始 CLS。
若用它替换官方缓存，需要自行套用 `data/whiten_stats.pt` 的白化矩阵。
