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
`classifier` and `SelfAttnPoolMoE` - about 332 K parameters out of ~949 M.

**Wording.** Describe the model as using a *frozen* LucaVirus encoder. Do not claim that
LucaVirus was fine-tuned, and do not describe the LoRA adapters as trained - they are
injected but frozen.

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
