# Pipeline

End-to-end path from raw data to reported numbers.

```
data/interactions.xlsx        421,234 (virus, host) pairs, 3,795 labelled positive
data/virus_sequences.fasta    934 aligned virus genomes
data/host_similarity.csv      451 x 451 host similarity matrix
pretrained/lucaVirus/         frozen 0.95 B encoder
        |
        v
(1) load + downsample          vhenet/preprocessing.py :: process_interaction_data
        |
        |   down_sample_ratio = 0.0909114865
        |   -> 37,950 negatives sampled from 417,439
        v
    strict 1:10 subset = 41,745 pairs (3,795 pos + 37,950 neg)
        |
        v
(2) split                      ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        |   pairwise (split_by_virus = false): all 934 viruses appear in both folds
        v
    train 33,396 (2,998 pos)  |  val 8,349 (797 pos)
        |
        v
(3) sequence encoding          vhenet/encode.py :: window_starts(len, 1022, 512)
        |   ~59 windows per virus, each 1022 nt
        |   LucaVirus -> CLS feature (2560-d)
        |   whitening:  wf = (cls - mean) @ W        (data/whiten_stats.pt)
        v
    cache/cls_windows_cache_934.pt
        |
        v
(4) training                   train.py, vhenet/model.py
        |   frozen: llm.* and all 96 LoRA tensors
        |   trained: aggregation + interaction + classifier + SelfAttnPoolMoE (332 K)
        |
        |   L = bce_lambda   * weighted_bce(logits, y, w, clip)
        |     + rank_lambda  * per_virus_softmax_loss(logits, y, w)
        |     + entropy_lambda * entropy
        |     + align_lambda   * alignment
        |
        |   w = importance_score.clamp(imp_lower, imp_upper)
        |       then positives scaled by pos_weight (= 1.0 for both reported models)
        v
    checkpoints/vhe_net_{with,without}_weight/{best,last}_model.pt
        |
        v
(5) prediction                 predict.py
        |   scores all 421,234 pairs for both splits
        v
    outputs/predictions.csv
        |
        v
(6) evaluation - THREE SCOPES, do not mix them
        |
        +-- full grid        421,234   includes 33,396 training pairs (memorisation)
        +-- validation         8,349   standard reporting scope
        +-- held-out pool    387,838   full grid minus training pairs
```

---

## Step detail

| Step | Input | Code | Output |
|---|---|---|---|
| 1 | `data/interactions.xlsx`, `data/host_similarity.csv` | `vhenet/preprocessing.py` | DataFrame + `host_to_id` + similarity matrix |
| 1 | downsampling | config `down_sample_ratio` | strict 1:10 = **41,745** |
| 2 | 41,745 pairs | `ShuffleSplit` | train **33,396** / val **8,349** |
| 3 | `data/virus_sequences.fasta` + `pretrained/lucaVirus` | `vhenet/encode.py`, `vhenet/encode_ids.py` | `cache/` |
| 4 | caches + labels + weights | `train.py` | `checkpoints/` |
| 5 | checkpoint + grid | `predict.py` | `outputs/predictions.csv` |
| 6 | predictions | - | metrics table |

---

## Why the caches exist

LucaVirus is a 0.95 B model. Re-encoding 934 virus genomes every epoch would dominate
runtime. After pre-computation:

- cache build: one pass, ~10-30 min on GPU
- each training epoch: **~5 s** (the large encoder never enters the autograd graph)

This is also why `train.py` does not accept raw sequences as input.

---

## Key numbers

| Quantity | Value |
|---|---|
| Full grid | **421,234** |
| Positives | **3,795** |
| Negatives | **417,439** |
| Sampled negatives | **37,950** = `int(417439 * 0.0909114865)` |
| Strict 1:10 subset | **41,745** |
| train / val | **33,396 / 8,349** |
| train positives / val positives | **2,998 / 797** |
| Viruses / hosts | **934 / 451** |
| Held-out pool | **387,838** |
