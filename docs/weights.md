# Obtaining weights and caches

The repository tracks code and primary data only. Four large artefacts are required to
run inference or reproduce the reported numbers:

| Artefact | Size | Git | Needed for |
|---|---|---|---|
| `pretrained/lucaVirus/` | 3.6 GB | no | both train and predict |
| `cache/ids_km_cache_934/` | 654 MB | no | training |
| `cache/cls_windows_cache_934.pt` | 360 MB | no | training and predict |
| `checkpoints/vhe_net_*/{best,last}_model.pt` | 4 x 3.6 GB | no | predict only |

---

## Option A - download the prepared artefacts

```bash
python - <<'PY'
import urllib.request, zipfile, io, pathlib
BASE = "https://github.com/dupllin/VHE-Net/releases/download/v1.0"
for name in ["lucaVirus.zip", "cache.zip", "checkpoints.zip"]:
    print("downloading", name)
    data = urllib.request.urlopen(f"{BASE}/{name}").read()
    zipfile.ZipFile(io.BytesIO(data)).extractall(".")
print("done")
PY
```

Verify afterwards:

```bash
python verify_reproduce.py
```

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
> print(sum(p.numel() for p in m.parameters()))   # 944,227,840 (encoder only)
> ```

### B2. k-mer id cache

```bash
python - <<'PY'
import yaml
from vhenet.encode_ids import build_ids_cache
cfg = yaml.safe_load(open("configs/vhe_net_with_weight.yaml"))
build_ids_cache(cfg["data"]["fasta_path"], "cache/ids_km_cache_934", k=6)
PY
```

Produces 936 files (~654 MB). Each file holds
`{'ids': [Nwin, 1024] int16, 'mask': ..., 'hist': [Nwin, 4096] float32, 'starts': ..., 'seq_len': int}`.

### B3. Window CLS cache

```bash
python - <<'PY'
from vhenet.encode import build_window_cls_cache
build_window_cls_cache(
    fasta="data/virus_sequences.fasta",
    model_path="pretrained/lucaVirus",
    cache_dir="cache/cls_windows_cache_934.pt",
    window=1022, stride=512)
PY
```

934 viruses x ~59 windows x 2560 dimensions (~360 MB). Alternatively export it directly
from a training run - `train.py` writes the cache whenever it is missing.

### B4. Whitening statistics

`data/whiten_stats.pt` **is** tracked by Git (25 MB) and contains
`{'mean': [2560], 'W': [2560, 2560], 'n': int, 'singular': ..., 'eps': ...}`.

To regenerate it, run `train.py` once with `whiten_stats: null`; the script computes and
saves the statistics from the training CLS distribution.

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
