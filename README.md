# VHE-Net

**Reproducible deep learning for virus-host association ranking.**

VHE-Net scores every (virus, host) pair on a fixed 934 x 451 grid and ranks candidate
hosts for each virus. The model couples a frozen protein language model encoder with a
learned virus-host interaction head and a **niche-weighted** training objective.

This repository contains the complete, self-contained code and data needed to
**retrain the model and reproduce the reported numbers**.

---

## Model at a glance

| | |
|---|---|
| Backbone encoder | **LucaVirus**, 0.95 B params, 12 layers, hidden 2560 (**frozen**, incl. LoRA) |
| Trainable modules | aggregation heads + interaction layers + classifier + `SelfAttnPoolMoE` (**~332 K params**) |
| Objective | `BCE + per-virus softmax ranking loss + entropy + alignment` |
| Grid | 934 viruses x 451 hosts = **421,234 pairs** |
| Sampling | strict **1:10** positives:negatives, 41,745 pairs |
| Split | pairwise `ShuffleSplit(test_size=0.2, random_state=42)` |
| Variants | `with_weight` (`imp_upper=20`) vs `without_weight` (`imp_upper=1`) |

### Reported metrics (validation, 8,349 pairs, 797 positives)

| Metric | with_weight | without_weight |
|---|---|---|
| **PR-AUC** | **0.5587** | 0.5474 |
| **ROC-AUC** | **0.9105** | 0.9058 |
| Accuracy | 0.9235 | 0.9286 |
| F1 | 0.6176 | **0.6270** |
| Precision | 0.5904 | **0.6255** |
| Recall | **0.6474** | 0.6286 |
| MCC | 0.5759 | **0.5876** |
| Balanced accuracy | **0.8000** | 0.7944 |
| Predicted positive | **874** | 801 |

---

## Quick start

```bash
git clone https://github.com/dupllin/VHE-Net.git
cd VHE-Net
pip install -r requirements.txt
```

### 1. Fetch model weights and caches

The backbone weights, feature caches and trained checkpoints are **not tracked by Git**
(~19 GB total). See **[docs/weights.md](docs/weights.md)** for download links and for
instructions to rebuild them from scratch.

After fetching, the tree should look like:

```
pretrained/lucaVirus/           <- LucaVirus 0.95B weights
cache/ids_km_cache_934/         <- k-mer id cache (934 files)
cache/cls_windows_cache_934.pt  <- window CLS features
checkpoints/vhe_net_with_weight/{best,last}_model.pt
checkpoints/vhe_net_without_weight/{best,last}_model.pt
```

### 2. Verify the installation

```bash
python verify_reproduce.py
```

Expected output:

```
total: 421,234 rows | pos 3795 | neg 417,439          OK
strict 1:10: 41,745 (train 33,396 / val 8,349)        OK
val positives 797                                      OK
self-check passed
```

### 3. Train

```bash
python train.py --config configs/vhe_net_with_weight.yaml
python train.py --config configs/vhe_net_without_weight.yaml
```

About 25 minutes per model on a single GPU (peak ~4.8 GB VRAM).

### 4. Predict

```bash
python predict.py \
  --config configs/vhe_net_with_weight.yaml \
  --checkpoint checkpoints/vhe_net_with_weight/last_model.pt \
  --out outputs/predictions.csv
```

Writes all 421,234 scored pairs for both splits.

---

## Repository layout

```
VHE-Net/
├── README.md
├── requirements.txt
├── train.py                       # training entry point
├── predict.py                     # full-grid prediction / export
├── verify_reproduce.py            # integrity + reproducibility self-check
│
├── vhenet/                        # package
│   ├── model.py                   #   ViHGAT_814v6 + SelfAttnPoolMoE + LoRA injection
│   ├── layers.py                  #   shared MLP / attention building blocks
│   ├── losses.py                  #   weighted_bce, per_virus_softmax_loss
│   ├── metrics.py                 #   ranking metrics
│   ├── encode.py                  #   window splitting + CLS cache builder
│   ├── encode_ids.py              #   k-mer id cache builder
│   ├── dataset.py                 #   FullWindowIdsHistPosData
│   └── preprocessing.py           #   load_config / parse_fasta / process_interaction_data
│
├── configs/
│   ├── vhe_net_with_weight.yaml    #  imp_upper = 20.0
│   └── vhe_net_without_weight.yaml #  imp_upper = 1.0
│
├── data/
│   ├── interactions.xlsx          # 421,234 pairs (3,795 pos / 417,439 neg)
│   ├── virus_sequences.fasta      # 934 aligned virus sequences
│   ├── host_similarity.csv        # 451 x 451 host similarity
│   ├── whiten_stats.pt            # CLS whitening statistics (mean + W, 2560x2560)
│   └── true_labels_manifest.json  # label manifest for the injection step
│
├── docs/
│   ├── weights.md                 # how to obtain weights & caches
│   ├── pipeline.md                # end-to-end pipeline walkthrough
│   └── implementation-notes.md    # details a reviewer would otherwise have to guess
│
├── pretrained/                    # (empty in Git)
├── cache/                         # (empty in Git)
├── checkpoints/                   # (empty in Git)
├── outputs/                       # (empty in Git)
└── examples/
```

---

## Configuration

The two configs differ in **exactly one effective field**:

```yaml
# vhe_net_with_weight.yaml      # vhe_net_without_weight.yaml
data:
  imp_upper: 20.0                 imp_upper: 1.0
```

Everything else - architecture, loss coefficients, sampling ratio, split, seed - is identical.

| Field | Value |
|---|---|
| `data.down_sample_ratio` | `0.0909114865` |
| `data.train_test_split` | `0.8` |
| `data.random_state` | `42` |
| `data.split_by_virus` | `false` (pairwise, **not** grouped by virus) |
| `training.bce_lambda` | `1.0` |
| `training.rank_lambda` | `1.0` |
| `training.entropy_lambda` | `0.1` |
| `training.align_lambda` | `1.0` |
| `training.epochs` | `200` (early stop patience 100) |

---

## Implementation notes reviewers ask about

These are easy to get wrong when reading the code; they are documented explicitly in
**[docs/implementation-notes.md](docs/implementation-notes.md)**.

1. **The protein language model is frozen, LoRA included.**
   `freeze_lora=true` freezes both the LucaVirus backbone *and* all 96 LoRA tensors.
   Only the aggregation heads, interaction layers, classifier and `SelfAttnPoolMoE`
   (~332 K params) are trained. Please describe this as a *frozen encoder*, **not** as
   a fine-tuned LucaVirus.

2. **`importance score` is fully determined by the label.**
   `importance_score == 20.0` for every positive pair (a single unique value) and lies in
   `[3.7366, 9.999993]` for negatives; predicting `importance_score >= 20` reproduces the
   label with **100.000000 %** accuracy. The *binned PR-AUC* variation is therefore a
   prevalence artefact: per-bin AUC is mathematically undefined because `bin2` and `bin3`
   contain **zero** positives.

3. **`bce_clip` is a sample-weight ceiling, not a logit clamp.**
   It enters as `weights.clamp(min=0.0, max=clip)` inside `weighted_bce`. With `clip=100`
   and a maximum positive weight of 20, the parameter is inert for these runs.

4. **Training consumes pre-computed CLS features**, not raw sequence forward passes.
   The 0.95 B encoder runs once during cache construction; each subsequent epoch takes ~5 s.

5. **Three evaluation scopes exist and must not be mixed.**

   | Scope | Size | Note |
   |---|---|---|
   | Full grid | 421,234 | includes the 33,396 training pairs (memorisation: 2,998 / 2,998 positives recalled) |
   | Validation | 8,349 | standard reporting scope |
   | Held-out pool | **387,838** | full grid minus training pairs; the correct scope for novel-prediction counts |

---

## Reproducibility

`verify_reproduce.py` checks file integrity, then re-derives the sampling and split from
`data/interactions.xlsx` and asserts the exact expected counts:

```
421,234 total / 3,795 positive / 37,950 sampled negatives
41,745 strict-1:10 subset / 33,396 train / 8,349 val / 797 val positives
```

The export path was validated against the published tables row by row: **maximum absolute
difference 0.000e+00 across all 421,234 pairs**, with every metric matching to four decimals.

---

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@software{vhenet,
  title  = {VHE-Net: reproducible deep learning for virus-host association ranking},
  author = {dupllin},
  url    = {https://github.com/dupllin/VHE-Net}
}
```
