# Obtaining weights and caches

The repository tracks code and primary data only. Four large artefacts are required to
run inference or reproduce the reported numbers:

| Artefact | Size | Git | Needed for |
|---|---|---|---|
| `pretrained/lucaVirus/` | 3.6 GB | no | both train and predict |
| `cache/ids_km_cache_934/` | 654 MB | no | training |
| `cache/cls_windows_cache_934.pt` | 360 MB | no | training and predict |
| `checkpoints/vhe_net_*/{best,last}_model.pt` | 4 x 3.6 GB | no | predict only |

**These artefacts are not hosted in this repository or in a GitHub Release.**
GitHub caps release assets at 2 GB per file, and `pretrained/lucaVirus/model.safetensors`
alone is 3.54 GiB, so no single-file upload can carry it. Request access from the
corresponding author, or rebuild the artefacts with **Option B** below.

---

## Expected layout

Place the artefacts so the tree looks like this — `verify_reproduce.py` checks exactly
these paths:

```
VHE-Net/
├── pretrained/
│   └── lucaVirus/                    <- 3.6 GB, from B1
├── cache/
│   ├── ids_km_cache_934/             <- 654 MB, from B2
│   └── cls_windows_cache_934.pt      <- 360 MB, from B3
└── checkpoints/
    ├── vhe_net_with_weight/
    │   └── best_model.pt
    └── vhe_net_without_weight/
        └── best_model.pt
```

Check it with:

```bash
python verify_reproduce.py
```

---

## Option A - request the prepared artefacts

Contact the corresponding author for `lucaVirus.zip`, `cache.zip` and
`checkpoints.zip`. They cannot be distributed through GitHub Releases; any transfer
must use a channel without a 2 GB per-file cap (institutional storage, Zenodo,
Hugging Face, or a chunked upload).

---

## Option B - rebuild everything from scratch

### B1. LucaVirus backbone

`pretrained/lucaVirus/` must contain:

```
config.json
configuration_lucavirus.py
modeling_lucavirus.py
tokenization_lucavirus.py
model.safetensors            <- 3.6 GB
tokenizer_config.json
vocab.json
```

> WARNING: this checkpoint holds **950,889,767 parameters (0.95 B)**, 12 layers with
> `hidden_size = 2560`. It is often described as "3 B" - that is wrong. Verify with:
>
> ```python
> from transformers import AutoModel
> m = AutoModel.from_pretrained("pretrained/lucaVirus", trust_remote_code=True)
> print(sum(p.numel() for p in m.parameters()))   # 950,889,767
> ```
>
> The measured value is **950,889,767**. An earlier version of this file also quoted
> "944,227,840 (encoder only)"; that number is not reproducible and has been removed.

### B2. k-mer id cache

```bash
python -m vhenet.encode_ids --fasta data/virus_sequences.fasta \
    --cache-dir cache/ids_km_cache_934 --kmer 6
```

Or programmatically — note the real function name is `encode_ids_to_cache`
(`vhenet/encode_ids.py`), not `build_ids_cache`:

```python
from vhenet.encode_ids import encode_ids_to_cache
encode_ids_to_cache("data/virus_sequences.fasta", "cache/ids_km_cache_934",
                    kmer=6)
```

Produces ~936 files (~654 MB). Each file holds
`{'ids': [Nwin, 1024] int16, 'mask': ..., 'hist': [Nwin, 4096] float32, 'starts': ..., 'seq_len': int}`.

### B3. Window CLS cache

The real function is `encode_fasta_to_cache` (`vhenet/encode.py`), not
`build_window_cls_cache`. **Pass `whiten_stats`**: `train.py` and `predict.py`
construct the model with `whiten_stats=None` and therefore expect the cache to
already hold whitened vectors.

```bash
python -m vhenet.encode --fasta data/virus_sequences.fasta \
    --cache-dir cache/cls_windows_cache_934 \
    --packed-path cache/cls_windows_cache_934.pt \
    --whiten-stats data/whiten_stats.pt
```

```python
import torch
from vhenet.encode import encode_fasta_to_cache
stats = torch.load("data/whiten_stats.pt", map_location="cpu")
encode_fasta_to_cache("data/virus_sequences.fasta", "cache/cls_windows_cache_934",
                      model=model, device=dev, whiten_stats=stats,
                      packed_path="cache/cls_windows_cache_934.pt")
```

934 viruses, **35,160 windows total**, 2560 dimensions (~360 MB).

Confirm what you built with:

```python
from vhenet.encode import verify_cache_whitening
verify_cache_whitening("cache/cls_windows_cache_934.pt", stats, n_virus=60)
# => 已白化: as-is cosine ~0.00-0.09, 反白化后升高到 ~0.70
```

### B4. Whitening statistics

`data/whiten_stats.pt` **is** tracked by Git (25 MB) and contains
`{'mean': [2560], 'W': [2560, 2560], 'n': int, 'singular': ..., 'eps': ...}`.

⚠️ `train.py` does **not** compute or regenerate this file — it reads
`whiten_stats: data/whiten_stats.pt` from the config but discards it, because
whitening is applied once, when the CLS cache is built (see B3). To rebuild the
statistics you must run the whitening fit on the uncached CLS distribution
explicitly; no script in this repository currently does that, so treat
`data/whiten_stats.pt` as a required input rather than a generated artefact.

### B5. Train the checkpoints yourself

Cheaper than downloading 15 GB if you have a GPU:

```bash
python train.py --config configs/vhe_net_with_weight.yaml
python train.py --config configs/vhe_net_without_weight.yaml
```

Roughly 25 minutes per model (~150 epochs at 5 s each), peak 4.8 GB VRAM.

---

## Sanity checks

```bash
# parameter count of the backbone
python -c "
from transformers import AutoModel
m = AutoModel.from_pretrained('pretrained/lucaVirus', trust_remote_code=True)
n = sum(p.numel() for p in m.parameters())
print(f'{n:,} = {n/1e9:.4f} B'); assert n < 1.1e9, 'unexpected size'"

# cache integrity
python - <<'PY'
import torch
cw = torch.load("cache/cls_windows_cache_934.pt", map_location="cpu")
print(len(cw), "viruses;", next(iter(cw.values())).shape)   # 934 ; (Nwin, 2560)
PY
```
