# Implementation notes

Details that are easy to misread from the code, and that a reviewer or a reuser will
otherwise have to guess. Each is stated with the evidence used to establish it.

---

## 1. The encoder is frozen - including LoRA

`train.py` (freeze block):

```python
for p in model.parameters():
    p.requires_grad = False                    # freeze everything
for name, p in model.named_parameters():
    if not name.startswith("llm."):
        p.requires_grad = True                 # unfreeze non-LLM parts
    elif not freeze_lora and ("lora_A" in name or "lora_B" in name):
        p.requires_grad = True                 # NOT executed when freeze_lora=True
```

Both configs set `freeze_lora: true`. Startup log confirms:

```
LoRA 注入: 48 个线性层 (r=8, alpha=16)
freeze_lora=true: 冻结 96 个 LoRA 参数张量，只训聚合头/分类器
```

**Consequence.** The trainable set is `host_rep_for`, `virus_interact`, `host_interact`,
`classifier` and `SelfAttnPoolMoE`. Measured on the shipped checkpoint:

| Component | Parameters |
|---|---|
| `SelfAttnPoolMoE` (`ckpt["moe"]`, 19 tensors) | 331,776 |
| `ckpt["model"]` excluding the frozen `llm.*` and the `sim_matrix` buffer | 2,218,241 |
| **total handed to AdamW** | **2,550,017** |
| encoder `llm.*` (304 tensors), frozen | 946,194,688 |
| of which LoRA (96 tensors), injected but frozen | 1,966,080 |
| `sim_matrix` buffer, not a parameter | 203,401 |

So the trainable fraction is **2,550,017 / 950,889,767 = 0.27 %**. An earlier version of
this file said "about 332 K", which is the `SelfAttnPoolMoE` module alone and understates
the trainable set by 7.7x. Note also that `llm_proj` (656,128 parameters) sits in the
checkpoint but is unreachable on the live path, because `cls_no_proj=true` makes
`encode_llm_windows` return before it is called.

**Wording.** Describe the model as using a *frozen* LucaVirus encoder. Do not claim that
LucaVirus was fine-tuned, and do not describe the LoRA adapters as trained - they are
injected but frozen. Stronger still: during training and inference the encoder is never
executed at all, because both read pre-computed CLS features from the cache.

---

## 2. `importance score` is label-derived, not an independent prior

Empirical check over all 421,234 pairs:

| | `importance score` |
|---|---|
| `Label == 1` | exactly `20.000000` - **one single unique value** |
| `Label == 0` | range `3.736625` - `9.999993`, 292,046 unique values |

A threshold rule `importance_score >= 20` predicts `Label` with **100.000000 %** accuracy.

**Consequences.**

- You cannot use `importance score` as an evaluation-stratification variable: it is a
  deterministic function of the label.
- The reported "binned PR-AUC" increase (e.g. 0.5587 -> 0.7636) is a **prevalence
  artefact**. The `clean` set (`bin1 u bin2`) raises the positive rate from 9.55 % to
  38.48 %, and PR-AUC is prevalence-sensitive.
- **Per-bin AUC is mathematically undefined here.** `bin1` contains all 797 validation
  positives and zero negatives; `bin2` and `bin3` contain zero positives. A single-class
  set has no ROC or PR curve.

---

## 3. `bce_clip` is a sample-weight ceiling

```python
def weighted_bce(logits, labels, weights, clip=5.0):
    w = weights.clamp(min=0.0, max=clip)       # operates on the SAMPLE WEIGHT
    loss = (bce * w).sum() / w.sum().clamp_min(1e-6)
```

It does **not** clamp logits. With `bce_clip = 100.0` and a maximum positive weight of 20,
the ceiling is never reached, so the parameter is inert for the reported runs.

---

## 4. Weight arithmetic for the two variants

```
w = importance_score.clamp(min=imp_lower, max=imp_upper)
if pos_weight > 1.0:
    w = where(label > 0.5, w * pos_weight, w)
```

| Variant | `imp_upper` | mean pos weight | mean neg weight | pos/neg ratio | positive share of loss |
|---|---|---|---|---|---|
| `with_weight` | 20.0 | 20.00 | 4.49 | **4.45x** | 30.5 % |
| `without_weight` | 1.0 | 1.00 | 1.00 | **1.00x** | 9.0 % |

For `without_weight`, `clamp(min=1.0, max=1.0)` collapses every weight to 1 - the model is
a plain unweighted BCE model. This is the correct control for the weighting ablation.

---

## 5. Training uses cached features, not raw sequences

`train.py` loads `cache/cls_windows_cache_934.pt` and never calls the encoder forward pass.
The 0.95 B model is used exactly once, during cache construction.

Practical consequences:

- epoch time ~5 s, so ~150 epochs in ~15-20 min
- the checkpoint still contains the full encoder weights (3.6 GB), because the model object
  holds them even though they are frozen
- deleting the cache without rebuilding it makes training impossible; see `docs/weights.md` B3

---

## 6. Split is pairwise, not grouped by virus

`split_by_virus: false`, so the same virus appears in both train and validation folds.
This is a **transductive** setting: the model has seen each virus's sequence during
training, and validation measures the ability to rank hosts for a virus whose
sequence is known but whose specific pair labels are held out.

If a reviewer asks about cold-start generalisation to unseen viruses, that requires a
different split (`split_by_virus: true`) and has not been evaluated here.

---

## 7. Three evaluation scopes

| Scope | Size | Definition | Use |
|---|---|---|---|
| Full grid | 421,234 | all pairs | distribution plots |
| Validation | 8,349 | the 20 % held-out fold | **metric reporting** |
| Held-out pool | **387,838** | full grid minus the 33,396 training pairs | **novel-prediction counts** |

The full grid includes training pairs, where the model recalls 2,998 of 2,998 positives.
Reporting precision on the full grid therefore inflates it by roughly 5.8x
(0.1667 vs 0.0286 for `with_weight`).

Novel-prediction counts should always be computed on the held-out pool:

```
with_weight:    predicted positive 18,072 | true positive 516 | novel 17,556
without_weight: predicted positive 15,659 | true positive 501 | novel 15,158
```

---

## 8. Backbone size

| | |
|---|---|
| parameters | **950,889,767 (0.95 B)** |
| layers | 12 (verified: indices 0-11 present in `model.safetensors`) |
| hidden size | 2560 |
| attention heads | 20 |
| ffn dim | 10240 |
| vocab | 39 |
| dtype | float32, 3.54 GiB on disk |

Single-layer cost `4H^2 + 2HF = 78,653,440`; twelve layers give 943,841,280, matching the
measured encoder total of 944,123,648 plus embeddings.

Reaching 3 B would require roughly 40 layers. **The checkpoint is a 0.95 B model**;
describing it as 3 B is incorrect and trivially falsifiable.

---

## 9. Encoding DNA for LucaVirus: the tokenizer's batch path drops `seq_type`

If you extend the pipeline to new sequences, **do not** pass a list of windows to the
tokenizer. It produces a valid-looking integer tensor and raises no error, but the ids
are wrong.

`tokenization_lucavirus.py:294`:

```python
def batch_encode_plus(self, *args, **kwargs):
    kwargs.pop("seq_type", None)          # discarded
    kwargs.pop("text_pair", None)
```

So `seq_type` survives only on the single-string path:

```python
tokenizer("GAAT",  seq_type="gene")    -> encode_plus()       -> gene mapping runs
tokenizer(["GAAT"], seq_type="gene")   -> batch_encode_plus() -> parameter dropped
```

With the parameter gone, the parent class looks the characters up in the `gene_prot`
vocabulary, where DNA letters are valid protein symbols, so they pass through as
amino-acid tokens. Measured on a real 1022 bp window:

```
tokenizer(seq,  seq_type="gene")  -> [2, 5, 6, 8, 6, 5, 6, 8, 6, 8, 5, 8]   correct
tokenizer([seq], seq_type="gene") -> [2, 11, 17, 12, 17, 11, 17, 12, ...]   wrong
tokenize_gene_ids(seq)            -> [2, 5, 6, 8, 6, 5, 6, 8, 6, 8, 5, 8]   correct
```

**Fix.** `vhenet/encode.py` does the gene mapping explicitly and never calls the
tokenizer:

```python
from vhenet.encode import tokenize_gene_ids
ids, mask = tokenize_gene_ids(seq_window, max_len=1024)
```

Both cache builders (`encode_ids_to_cache`, `encode_fasta_to_cache`) use it. Verify a
cache you already have with:

```python
from vhenet.encode import verify_gene_encoding
verify_gene_encoding("cache/ids_km_cache_934", "data/virus_sequences.fasta")
```

It re-derives tokens for a few (virus, window) pairs and compares them element by
element; the shipped caches return `6/6 完全一致`.

**Scope.** The reported runs are unaffected. `train.py` and `predict.py` read the
pre-computed `ids`/`cls` caches and never invoke the tokenizer, and the original
`encode_ids_to_cache` tokenized one string at a time — which is the correct path —
so the caches it produced are valid. This matters when encoding new sequences through
a code path that batches windows into a list.

---

## 10. Raw CLS is nearly collinear; whitening is what separates viruses

When quoting a "cosine similarity" figure for these features, say **which stage** you
mean — the two differ by more than an order of magnitude:

| Stage | Cosine between window-0 CLS of different viruses |
|---|---|
| **before** whitening | mean **0.69** (934 training set) / **0.83** (209 external set) |
| **after** whitening | mean **0.02** (934) / **0.09** (209) |

Windows of the *same* virus are also collinear before whitening (mean **0.82**),
dropping to **0.02** after.

So a statement like "LucaVirus CLS features are almost identical across viruses"
is true of the raw encoder output, and false of the cached features that the model
actually consumes (`cls_windows_cache_934.pt` stores the whitened values).

---

## 11. Pooling choice determines whether sequence *order* is visible at all

`mean` pooling averages token embeddings:

```
mean_pool(seq) = (1/L) * sum_i h(token_i)
```

Summation is permutation-invariant, so **shuffling any region leaves the mean-pooled
vector bit-identical** — regardless of how many windows you feed in. This is an
identity, not a property of any particular model. Measured on the 209 bat CoVs
(shuffle the spike CDS, 204 viruses):

| Feature | windows | max abs dP | top-1 host changed |
|---|---|---|---|
| mean-pool, first window | 1 | **0.000e+00** | **0.0 %** |
| mean-pool, all K windows | K | 6.3e-02 | **0.0 %** |
| CLS, first window | 1 | **0.000e+00** | 0.0 % (spike out of reach) |
| CLS, all K windows | K | 7.3e-02 | 32.8 % |
| VHE-Net (CLS + MoE attention) | K | **9.9e-01** | **70.4 %** |

Only `CLS` carries order information, and only an aggregation that can *weight windows*
(the MoE attention) turns that into region attribution. A fixed statistic over windows
(mean / max / std) stays permutation-invariant.


---

## 12. The 70.4 % spike-shuffle figure is checkpoint-independent

Section 11 reports that shuffling the spike CDS flips the top-1 host for 70.4 % of the
209 bat CoVs. That number was measured on a training run whose checkpoint is no longer
available, so it was re-measured on the shipped
`checkpoints/v34jbce_pairwise_ratio10_allneg_8020_spearman/best_model.pt` using the
same 71 viruses, the same spike coordinates, and the same shuffle seed:

| Metric | Original run | Shipped `best_model.pt` |
|---|---|---|
| **spike shuffle → top-1 changed** | **70.4 %** (50/71) | **71.8 %** (51/71) |
| ORF1ab shuffle → top-1 changed | 98.6 % | 95.8 % |
| Baseline top-1 agreement | — | **1/71 (1.4 %)** |
| Per-virus changed/unchanged agreement | — | 40/71 (56.3 %) |

A different checkpoint that agrees on the baseline top-1 for **one virus out of 71**
still reproduces the effect to within **1.4 percentage points**. The phenomenon is a
property of the architecture, not of one training run.

Two caveats worth stating in a manuscript:

- **Per-virus attributions do not transfer.** Baseline top-1 agreement is 1.4 %, and
  per-virus changed/unchanged agreement is 56.3 % (contingency 35/16/15/5), which is
  chance level given the base rates. Cite the population-level rate, not individual
  viruses.
- **Confidence and host prior differ between runs.** On these 209 out-of-distribution
  viruses the shipped checkpoint is less confident (mean top-1 probability 0.933 vs
  0.993) and much more *Homo sapiens*-leaning (`P_Homo` mean 0.739 vs 0.411;
  `P_Homo >= 0.5` for 62/71 vs 29/71 viruses; top-1 diversity 30 vs 39 hosts). This is
  domain shift, not a bug — the 209 bat CoVs are disjoint from the 934 training viruses.

Reproduction: `2025_9_21_web_VHE_rankBCE/` serves the shipped checkpoint with the full
sliding-window + whitening pipeline; running the same shuffle through it takes ~35 min
for all 71 viruses. Per-virus detail is in
`results/deliverable_20260904/30_spike_shuffle_复现_当前权重/`.

---

## 13. Whitening happens once, in the cache builder — not inside the model

This is the single easiest thing to get wrong in this repository, because the code,
the config and the documentation each suggested a different answer.

**What the reported models actually do.** The whitening transform

```
wf = (cls - mean) @ W
```

is applied **once, when the window CLS cache is built**. `cache/cls_windows_cache_934.pt`
therefore stores *already-whitened* 2560-d vectors, and `train.py` / `predict.py`
correctly pass `whiten_stats=None` — whitening a second time would destroy the features.

**Measured proof** (delivered 209-virus cache, `verify_cache_whitening`,
60 viruses / 1,562 windows):

| Quantity | Value | Expected for |
|---|---|---|
| cross-virus cosine, as stored | **0.0026** | whitened (≈0) |
| cross-virus cosine after un-whitening (`X @ W⁻¹ + mean`) | **0.699** | raw |
| mean abs per-dimension value, as stored | 0.029 | whitened (≈0) |

Raw LucaVirus CLS is strongly collinear — cross-virus cosine 0.69 (934 viruses) and
0.83 (209 bat CoVs) — and whitening is what makes windows from different viruses
comparable. Un-whitening the cache restores exactly that 0.70 cosine, which is the
positive control: it shows the stored vectors are the whitened ones, not raw ones.

**Why the code looked like it disagreed.** Two features of the shipped code are
deliberately misleading if read alone:

1. `train.py:241` and `predict.py:68` hard-code `whiten_stats=None`, discarding
   `configs/vhe_net_with_weight.yaml:43`. That is **correct** for a pre-whitened cache.
2. The whitening call sites (`vhenet/model.py:296`, `:318`) live inside
   `encode_llm_windows()`, which `train.py` / `predict.py` never call — they apply
   `model.llm_out` directly to cached CLS. Those call sites are for the alternative
   workflow where you pass raw CLS and let the model whiten, and they must stay unused
   when the cache is pre-whitened.

**The one real defect this exposed.** The repository's own cache builder used to write
*raw* CLS, so a cache rebuilt with the documented command fed raw features to a model
that assumed whitened ones. Fixed by making whitening an explicit builder step that
defaults to on:

```bash
python -m vhenet.encode --fasta data/virus_sequences.fasta \
    --cache-dir cache/cls_windows_cache_934 \
    --packed-path cache/cls_windows_cache_934.pt \
    --whiten-stats data/whiten_stats.pt
```

`encode_fasta_to_cache(..., whiten_stats=stats)` applies the transform before writing;
omitting it writes raw CLS and then requires passing the same `whiten_stats` to the
model. **Never both.** To check a cache you already have:

```python
from vhenet.encode import verify_cache_whitening
verify_cache_whitening("cache/cls_windows_cache_934.pt", stats, n_virus=60)
```

It reports `is_whitened`, the as-stored cosine, and the un-whitened cosine, so the
answer does not depend on reading this file.
