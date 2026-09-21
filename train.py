"""v34FMoE 训练：MoE + 注意力-敏感性对齐正则（B-version multi-peak）。

背景/目标（用户最新要求）：
  v34E 是单 query sparsemax 注意力（只能聚焦 1-2 窗，易"孤注一掷"，如 Pa_GD_A909
  那株 spike 不决定）。用户希望做**多头稀疏混合专家(MoE)**：
    - 55 个窗口 = 55 个"专家"；
    - N 个"意图 token"(=MoE 的多个 router) 各自 sparsemax 路由，聚焦到不同窗口；
    - 多头 = 分布式决定，既保留 spike 定位、又提升稳定性。
  负载均衡设计原则（用户明确）：**防止 head 失效(head 总是选中和别的 head 同一个窗
  /或完全不选)与 head 重复(两个 head 总选中同一窗)**，**不得强迫生物学窗口全局均匀**。

实现：
  - SelfAttnPoolMoE：N=4 个 query(q_proj)，每个对 55 窗做 sparsemax(router)；
    ctx_h = attn_h @ v；virus_rep = out_proj(concat(4 头 ctx))；+全局 mean 残差。
  - 损失项：
      + bce_lambda*weighted_bce + rank_loss            (与 v34E 相同)
      + ent_lambda * mean_h entropy(attn_h)             (逐头稀疏, 允许聚焦)
      + bal_lambda * mean_{h1<h2} (attn_h1·attn_h2)     (跨头重叠惩罚=防重复/防失效,
                                                        不碰生物学窗口的均匀性)
  - importance 权重：fixed(正20/负4.5)，10:1，split_by_virus。

热启动：从 v34E best 载入 LLM/交互/分类器；SelfAttnPoolMoE 新增随机初始化。
用法(项目根目录):
    python train.py --config configs/vhe_net_with_weight.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vhenet.preprocessing import process_interaction_data, load_config, parse_fasta  # noqa: E402
from vhenet.dataset import FullWindowIdsHistPosData  # noqa: E402
from vhenet.losses import per_virus_softmax_loss, weighted_bce  # noqa: E402
from vhenet.metrics import ranking_metrics  # noqa: E402
from vhenet.model import ViHGAT_814v6, SelfAttnPoolMoE, sparsemax  # noqa: E402
from vhenet.encode import window_starts, clean_seq  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def resolve(p, base=PROJECT_ROOT):
    p = Path(p).expanduser()
    return p if p.is_absolute() else (base / p)


def _auc(pairs):
    if not pairs:
        return 0.5
    vals = np.array([p for p, _ in pairs])
    labs = np.array([l for _, l in pairs])
    o = np.argsort(vals)
    ranks = np.empty(len(vals), dtype=float)
    ranks[o] = np.arange(len(vals))
    pos = labs == 1
    denom = pos.sum() * (len(vals) - pos.sum())
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (denom + 1e-9)


def approx_ap_loss(logits, labels, temperature=1.0, eps=1e-8):
    """Differentiable AP surrogate: encourage each positive above each negative.
    logits/labels: [K] same virus. Smooth pairwise sigmoid ranking loss.
    """
    pos_mask = labels > 0.5
    if pos_mask.sum() == 0 or pos_mask.sum() == labels.numel():
        return torch.zeros((), device=logits.device)
    pos = logits[pos_mask]
    neg = logits[~pos_mask]
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)          # [P,N]
    s = torch.sigmoid(diff * temperature)
    return -torch.log(s.mean() + eps)


def entropy(a, eps=1e-8):
    a = a.clamp(min=eps)
    return -(a * torch.log(a))


def _pearson(x, y, eps=1e-6):
    x = x - x.mean()
    y = y - y.mean()
    return -(x @ y) / (x.norm() * y.norm() + eps)


def coverage_loss(a_sum, s, tau=0.3, topk=None, eps=1e-6):
    """Coverage loss: force attention mass onto high-sensitivity windows.

    - topk=None: soft target w = softmax(s/tau) (smooth).
    - topk=k    : hard top-k mask on the k most sensitive windows (multi-peak).
    """
    if topk and topk > 0 and topk < s.numel():
        w = torch.zeros_like(s)
        idx = torch.topk(s, k=min(int(topk), s.numel())).indices
        w[idx] = 1.0
    else:
        w = torch.softmax(s / float(tau), dim=0)
    a_norm = a_sum / (a_sum.sum() + eps)
    return -(w * a_norm).sum()


def drop_sensitivity_moe(win_proj, host_rep, model, moe):
    """Per-window drop sensitivity s_i = mean|Δp| over sampled hosts (MoE version).

    All differentiable; gradients flow through moe and model's non-LLM params.
    Returns s [K].
    """
    K = win_proj.shape[0]
    if K < 2:
        return torch.zeros(K, device=win_proj.device)
    vrep0, _ = moe(win_proj)
    rep_e0 = vrep0.unsqueeze(0).expand(host_rep.shape[0], -1)
    inter0 = model.virus_interact(rep_e0) * model.host_interact(host_rep)
    feats0 = torch.cat([rep_e0, inter0], dim=-1)
    p0 = torch.sigmoid(model.classifier(feats0).squeeze(-1)).detach()
    idx_all = torch.arange(K, device=win_proj.device)
    s = []
    for i in range(K):
        keep = idx_all != i
        wf_i = win_proj[keep]
        vrep_i, _ = moe(wf_i)
        rep_e = vrep_i.unsqueeze(0).expand(host_rep.shape[0], -1)
        inter = model.virus_interact(rep_e) * model.host_interact(host_rep)
        feats = torch.cat([rep_e, inter], dim=-1)
        p_i = torch.sigmoid(model.classifier(feats).squeeze(-1))
        s.append((p_i - p0).abs().mean())
    return torch.stack(s)


def main(config_path):
    cfg = load_config(config_path)
    device = torch.device(cfg["device"])
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed); np.random.seed(seed)
    print(f"设备: {device}")

    # ---- 数据 (importance fixed: 正20/负4.5, 只 clip 安全域) ----
    df, host_to_id, sim_matrix = process_interaction_data(
        str(resolve(cfg["data"]["interaction_csv"])),
        str(resolve(cfg["data"]["similarity_csv"])),
    )
    df["importance score"] = df["importance score"].clip(
        lower=cfg["data"].get("imp_lower", 1.0),
        upper=cfg["data"].get("imp_upper", 20.0)).astype(float)
    pos_mask = df["Label"] == 1
    print(f"样本 {len(df)}, 正 {pos_mask.sum()}, 病毒 {df['Virus'].nunique()}, "
          f"pos_imp={df.loc[pos_mask,'importance score'].mean():.2f} "
          f"neg_imp={df.loc[~pos_mask,'importance score'].mean():.2f}")

    dsr = cfg["data"].get("down_sample_ratio")
    if dsr and float(dsr) < 1.0:
        n_neg_keep = int((df["Label"] == 0).sum() * float(dsr))
        neg_df = df[df["Label"] == 0].sample(n=n_neg_keep, random_state=seed)
        df = pd.concat([df[df["Label"] == 1], neg_df]).reset_index(drop=True)

    _split_by_virus = cfg["data"].get("split_by_virus", True)
    _val_ratio = float(cfg["data"].get("val_ratio", 0.0))
    _test_ratio = float(cfg["data"].get("test_ratio", 0.0))
    if _test_ratio > 0:
        _train_ratio = 1.0 - _val_ratio - _test_ratio
        _rest_ratio = _val_ratio + _test_ratio
        if _split_by_virus:
            _sp1 = GroupShuffleSplit(n_splits=1, test_size=_rest_ratio, random_state=seed)
            train_idx, _rest_idx = next(_sp1.split(df, groups=df["Virus"]))
            _sp2 = GroupShuffleSplit(n_splits=1, test_size=_test_ratio/_rest_ratio, random_state=seed+1)
            _val_rel, _test_rel = next(_sp2.split(df.iloc[_rest_idx], groups=df.iloc[_rest_idx]["Virus"]))
        else:
            _sp1 = ShuffleSplit(n_splits=1, test_size=_rest_ratio, random_state=seed)
            train_idx, _rest_idx = next(_sp1.split(df))
            _sp2 = ShuffleSplit(n_splits=1, test_size=_test_ratio/_rest_ratio, random_state=seed+1)
            _val_rel, _test_rel = next(_sp2.split(df.iloc[_rest_idx]))
        _rest_arr = np.asarray(_rest_idx)
        val_idx = _rest_arr[_val_rel]
        test_idx = _rest_arr[_test_rel]
        print(f"三路划分: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
              f"(pairwise={not _split_by_virus})", flush=True)
    elif _split_by_virus:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=1 - cfg["data"]["train_test_split"],
            random_state=seed)
        train_idx, val_idx = next(splitter.split(df, groups=df["Virus"]))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=1 - cfg["data"]["train_test_split"],
            random_state=seed)
        train_idx, val_idx = next(splitter.split(df))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_data = FullWindowIdsHistPosData(
        train_df, host_to_id, str(resolve(cfg["data"]["ids_cache_dir"])))
    val_data = FullWindowIdsHistPosData(
        val_df, host_to_id, str(resolve(cfg["data"]["ids_cache_dir"])))
    cls_windows = torch.load(
        resolve("cache/cls_windows_cache_934.pt"), map_location="cpu")
    print(f"窗口 CLS 缓存(934): {len(cls_windows)}")

    true_dp = {}
    _tml = str(cfg["training"].get("true_label_manifest", ""))
    if _tml:
        import json as _json
        man = _json.load(open(resolve(_tml)))
        for v, p in man.items():
            try:
                d = _json.load(open(p))
                true_dp[v] = torch.tensor(np.abs(d["dp_top1"]), dtype=torch.float32)
            except Exception as e:
                print(f"  [true label skip] {v}: {e}")
        print(f"真·shuffle 标签注入: {len(true_dp)} 病毒")

    model_cfg = cfg["model"]
    model = ViHGAT_814v6(
        num_hosts=len(host_to_id), host_sim_matrix=sim_matrix,
        llm_model_path=str(resolve(model_cfg["model_path"])),
        embed_dim=model_cfg.get("embed_dim", 256),
        dropout=model_cfg.get("dropout", 0.25),
        n_pos_bins=model_cfg.get("n_pos_bins", 8),
        lora_r=model_cfg.get("lora_r", 8),
        lora_alpha=model_cfg.get("lora_alpha", 16),
        lora_dropout=model_cfg.get("lora_dropout", 0.1),
        llm_window_tokens=model_cfg.get("llm_window_tokens", 1024),
        use_host_head=False, main_head_inputs="virus_inter",
        use_km_branch=False, b_agg_bins=0, cls_only=True, freeze_lora=True,
        cls_no_proj=True, use_cross_attn=False,
        cross_heads=model_cfg.get("cross_heads", 4), whiten_stats=None,
    )
    model.to(device)
    _init_from = cfg["training"].get(
        "init_from", None)   # 冷启动：从头训练
    st = None
    if _init_from:
        init_ckpt = resolve(_init_from)
        st = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(st["model"], strict=False)
    else:
        print("冷启动：不加载任何历史 checkpoint，重新训练")
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if not name.startswith("llm."):
            p.requires_grad = True
        elif not model_cfg.get("freeze_lora", True) and ("lora_A" in name or "lora_B" in name):
            p.requires_grad = True

    d = model_cfg.get("embed_dim", 256)
    moe = SelfAttnPoolMoE(d, n_heads=model_cfg.get("moe_heads", 4),
                          dropout=model_cfg.get("dropout", 0.25),
                          attn_type=model_cfg.get("attn_type", "sparsemax"),
                          attn_temp=float(model_cfg.get("attn_temp", 1.0)))
    moe.to(device)
    if st is not None and "moe" in st:
        moe.load_state_dict(st["moe"], strict=True)
        print("加载 init_from 中的 MoE，保留历史注意力", flush=True)
    print(f"SelfAttnPoolMoE: heads={moe.n_heads} attn_type={moe.attn_type} "
          f"attn_temp={moe.attn_temp} "
          f"params={sum(p.numel() for p in moe.parameters()):,}")

    opt_cfg = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad]
        + list(moe.parameters()),
        lr=opt_cfg.get("learning_rate", 1e-3),
        weight_decay=opt_cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt_cfg.get("epochs", 40), eta_min=1e-5)

    epochs = opt_cfg.get("epochs", 40)
    accum = max(1, opt_cfg.get("grad_accum_viruses", 4))
    n_neg = cfg["data"].get("train_neg_per_virus", 44)
    bce_lambda = opt_cfg.get("bce_lambda", 0.5)
    rank_lambda = opt_cfg.get("rank_lambda", 1.0)
    ap_lambda = float(opt_cfg.get("ap_lambda", 0.0))  # AP/listwise surrogate loss
    spike_sens_lambda = float(opt_cfg.get("spike_sens_lambda", 0.0))  # Spike-shuffle confidence-drop loss
    spike_sens_cache = str(opt_cfg.get("spike_sens_cache", ""))
    rank_no_importance = opt_cfg.get("rank_no_importance", False)  # rank loss 不用 importance score
    pos_weight = float(opt_cfg.get("pos_weight", 1.0))   # 正类额外倍率
    bce_clip = float(opt_cfg.get("bce_clip", 5.0))       # weighted_bce 的权重 clip
    ent_lambda = opt_cfg.get("entropy_lambda", 0.1)
    bal_lambda = opt_cfg.get("bal_lambda", 0.5)     # 跨头重叠惩罚
    align_lambda = opt_cfg.get("align_lambda", 0.0)  # 注意力-敏感性对齐正则
    align_mode = str(opt_cfg.get("align_mode", "pearson"))  # pearson | coverage | both
    coverage_topk = int(opt_cfg.get("coverage_topk", 0))   # 0=soft coverage, >0=hard top-k
    true_label_manifest = str(opt_cfg.get("true_label_manifest", ""))  # 真·shuffle 标签注入
    region_prior_lambda = opt_cfg.get("region_prior_lambda", 0.0)
    region_prior_frac = float(opt_cfg.get("region_prior_frac", 0.85))
    viruses_per_epoch = int(opt_cfg.get("viruses_per_epoch", 200))
    early_stop_metric = str(opt_cfg.get("early_stop_metric", "spearman"))  # spearman | pr_auc | pr_auc_diversity
    pr_auc_floor = float(opt_cfg.get("pr_auc_floor", 0.10))
    max_rank_spearman = float(opt_cfg.get("max_rank_spearman", 0.30))
    min_distinct_top1_hosts = int(opt_cfg.get("min_distinct_top1_hosts", 15))
    diversity_penalty = float(opt_cfg.get("diversity_penalty", 0.20))
    save_epochs = [int(x) for x in str(opt_cfg.get("save_epochs", "")).split(",") if x.strip()]
    host_ids_all = torch.arange(len(host_to_id), device=device)
    rng = np.random.default_rng(seed)

    spike_cache = {}
    if spike_sens_cache:
        _sc_dir = resolve(spike_sens_cache)
        for _pf in _sc_dir.glob("*.pt"):
            _d = torch.load(_pf, map_location="cpu")
            spike_cache[_d["virus"]] = _d
        print(f"Spike 敏感度缓存加载: {len(spike_cache)} 病毒", flush=True)

    best_spearman = float("inf")
    best_pr_auc = -float("inf")
    best_combo_score = -float("inf")
    best_epoch, bad = -1, 0
    history = []
    result_dir = resolve(cfg["data"]["result_path"]); result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = resolve(cfg["training"]["model_save_path"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    def virus_rep_of(virus):
        wf = cls_windows[virus].to(device)
        return moe(model.llm_out(wf))[0]

    def forward_virus(virus, host_t):
        wf = cls_windows[virus].to(device)
        win_proj = model.llm_out(wf)
        vrep, attn = moe(win_proj)                       # attn [H,K]
        rep_e = vrep.unsqueeze(0).expand(len(host_t), -1)
        host_rep = model.host_rep_for(host_t)
        inter = model.virus_interact(rep_e) * model.host_interact(host_rep)
        feats = torch.cat([rep_e, inter], dim=-1)
        logits = model.classifier(feats).squeeze(-1)
        return logits, attn, win_proj, host_rep

    train_viruses = sorted(train_data.virus_rows.keys())
    region_mask = {}
    if region_prior_lambda > 0:
        fasta = parse_fasta(str(resolve(cfg["data"]["fasta_path"])))
        for v in train_viruses:
            seq = clean_seq(fasta[v])
            spans = list(window_starts(len(seq), 1022, 512))
            fracs = [((s + e) / 2.0) / max(len(seq), 1) for s, e in spans]
            mask = torch.tensor([f >= region_prior_frac for f in fracs], device=device)
            region_mask[v] = mask
        if region_mask:
            mean_cov = float(torch.stack([m.float().mean() for m in region_mask.values()]).mean())
        else:
            mean_cov = 0.0
        print(f"3' region prior: frac>={region_prior_frac}, {len(region_mask)} viruses, "
              f"avg windows in 3'={mean_cov:.2f}")
    for epoch in range(1, epochs + 1):
        t0 = time.time(); model.train(); moe.train()
        viruses = list(train_viruses)
        if viruses_per_epoch > 0 and len(viruses) > viruses_per_epoch:
            rng.shuffle(viruses); viruses = viruses[:viruses_per_epoch]
        tot, rs, bs, es, bls, als, rls, aps, sls, n_steps = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for vi, virus in enumerate(viruses):
            hosts, labels, weights = train_data.sample_train_rows(virus, n_neg, rng)
            host_t = torch.from_numpy(hosts).long().to(device)
            label_t = torch.from_numpy(labels).float().to(device)
            weight_t = torch.from_numpy(weights).float().to(device)
            logits, attn, win_proj, host_rep = forward_virus(virus, host_t)
            rank_loss_weights = torch.ones_like(weight_t) if rank_no_importance else weight_t
            rank_loss = per_virus_softmax_loss(logits, label_t, rank_loss_weights)
            if pos_weight > 1.0:
                bce_weight = torch.where(
                    label_t > 0.5, weight_t * pos_weight, weight_t)
            else:
                bce_weight = weight_t
            bce_loss = weighted_bce(logits, label_t, bce_weight, clip=bce_clip)
            ap_loss = approx_ap_loss(logits, label_t) if ap_lambda > 0 else torch.zeros((), device=logits.device)
            loss = (bce_lambda * bce_loss + rank_lambda * rank_loss + ap_lambda * ap_loss) / accum
            aps += ap_loss.item()
            if spike_sens_lambda > 0 and virus in spike_cache:
                def _full_scores(_cls):
                    _wf = model.llm_out(_cls)
                    _vr, _ = moe(_wf)
                    _re = _vr.unsqueeze(0).expand(len(host_ids_all), -1)
                    _hr = model.host_rep_for(host_ids_all)
                    _inter = model.virus_interact(_re) * model.host_interact(_hr)
                    _feats = torch.cat([_re, _inter], dim=-1)
                    return torch.sigmoid(model.classifier(_feats).squeeze(-1))
                _orig = spike_cache[virus]["orig_cls"].to(device).float()
                _shuf = spike_cache[virus]["shuf_cls"].to(device).float()
                _p0 = _full_scores(_orig)
                _top1 = torch.argmax(_p0)
                _p1 = _full_scores(_shuf)[_top1]
                _diff = _p0[_top1] - _p1
                _spike_loss = -torch.log(_diff.clamp(min=1e-4))
                loss = loss + (spike_sens_lambda * _spike_loss) / accum
                sls += _spike_loss.item()
            if ent_lambda > 0:
                loss = loss + (ent_lambda * entropy(attn).mean()) / accum
                es += entropy(attn).mean().item()
            bl = moe.balance_loss(attn)
            loss = loss + (bal_lambda * bl) / accum
            bls += bl.item()
            if align_lambda > 0 or region_prior_lambda > 0:
                # 多头注意力“总和”分布
                a_sum = attn.sum(0)
            if align_lambda > 0:
                if virus in true_dp and true_dp[virus].numel() == attn.shape[-1]:
                    s = true_dp[virus].to(device)
                else:
                    s = drop_sensitivity_moe(win_proj, host_rep, model, moe)
                if align_mode == "coverage":
                    align_loss = coverage_loss(a_sum, s, topk=coverage_topk)
                elif align_mode == "both":
                    align_loss = _pearson(a_sum, s) + coverage_loss(a_sum, s, topk=coverage_topk)
                else:
                    align_loss = _pearson(a_sum, s)
                loss = loss + (align_lambda * align_loss) / accum
                als += align_loss.item()
            if region_prior_lambda > 0:
                mask = region_mask.get(virus)
                if mask is not None and mask.any():
                    a_norm = a_sum / (a_sum.sum() + 1e-6)
                    region_loss = -(a_norm * mask.float()).sum()
                    loss = loss + (region_prior_lambda * region_loss) / accum
                    rls += region_loss.item()
            loss.backward()
            tot += loss.item() * accum
            rs += rank_loss.item(); bs += bce_loss.item(); n_steps += 1
            if (vi + 1) % accum == 0 or vi == len(viruses) - 1:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(moe.parameters()), 5.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        model.eval(); moe.eval()
        probs = {}
        auc_pairs, top1_hits, n_v = [], 0, 0
        ap_scores = []
        with torch.no_grad():
            for virus in val_data.virus_rows:
                wf = cls_windows[virus].to(device)
                vrep, attn = moe(model.llm_out(wf))
                rep_e = vrep.unsqueeze(0).expand(len(host_to_id), -1)
                host_rep = model.host_rep_for(host_ids_all)
                inter = model.virus_interact(rep_e) * model.host_interact(host_rep)
                feats = torch.cat([rep_e, inter], dim=-1)
                p = torch.sigmoid(model.classifier(feats).squeeze(-1)).cpu().numpy()
                probs[virus] = p
                ph = val_data.pos_hosts[virus]
                if ph:
                    y = np.zeros(len(host_to_id)); y[list(ph)] = 1
                    ap_scores.append(average_precision_score(y, p))
                    auc_pairs += [(float(p[h]), 1.0) for h in ph]
                    negp = [p[h] for h in range(len(host_to_id)) if h not in set(ph)]
                    if len(negp) > 1:
                        rx = np.random.default_rng(0).choice(
                            negp, size=min(500, len(negp)), replace=False)
                        auc_pairs += [(float(x), 0.0) for x in rx]
                if ph and int(np.argmax(p)) in set(ph):
                    top1_hits += 1
                n_v += 1
        val_auc = _auc(auc_pairs)
        val_pr_auc = float(np.nanmean(ap_scores)) if ap_scores else 0.0
        mtr = ranking_metrics(probs, val_data.id_to_host_list)
        vcos = _virus_rep_diversity(virus_rep_of, train_viruses[:24])
        sec = time.time() - t0
        hist = {"epoch": epoch, "val_auc": val_auc, "val_pr_auc": val_pr_auc,
                "mean_pairwise_spearman": mtr["mean_pairwise_spearman"],
                "top1_hit_rate": top1_hits / max(n_v, 1),
                "n_distinct_top1_hosts": mtr["n_distinct_top1_hosts"],
                "train_loss": tot / max(n_steps, 1),
                "rank_loss": rs / max(n_steps, 1), "bce_loss": bs / max(n_steps, 1),
                "entropy_loss": es / max(n_steps, 1),
                "balance_loss": bls / max(n_steps, 1),
                "ap_loss": aps / max(n_steps, 1),
                "spike_sens_loss": sls / max(n_steps, 1),
                "align_loss": als / max(n_steps, 1),
                "region_loss": rls / max(n_steps, 1),
                "virus_rep_cos_mean": vcos[0], "virus_rep_cos_min": vcos[1],
                "sec_per_epoch": round(sec, 1)}
        history.append(hist)
        print(f"[ep {epoch}] loss={hist['train_loss']:.3f} "
              f"rank={hist['rank_loss']:.3f} bce={hist['bce_loss']:.3f} "
              f"bal={hist['balance_loss']:.3f} align={hist['align_loss']:.3f} "
              f"region={hist['region_loss']:.3f} "
              f"val_auc={val_auc:.3f} val_pr_auc={val_pr_auc:.3f} "
              f"sp={hist['mean_pairwise_spearman']:.3f} "
              f"top1_hit={hist['top1_hit_rate']:.3f} vcos={vcos[0]:.3f} "
              f"({sec:.1f}s)", flush=True)
        with open(result_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)
        if epoch in save_epochs:
            torch.save({"model": model.state_dict(), "moe": moe.state_dict(),
                        "config": cfg, "host_to_id": host_to_id,
                        "epoch": epoch, "val_metrics": hist},
                       ckpt_dir / f"checkpoint_ep{epoch}.pt")
            print(f"  -> saved ep{epoch} checkpoint", flush=True)
        is_cand = val_auc >= cfg["training"].get("auc_floor", 0.72)
        if early_stop_metric == "pr_auc":
            is_cand = val_pr_auc >= pr_auc_floor
            if is_cand and val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_epoch = epoch; bad = 0
                torch.save({"model": model.state_dict(), "moe": moe.state_dict(),
                            "config": cfg, "host_to_id": host_to_id,
                            "epoch": epoch, "val_metrics": hist},
                           ckpt_dir / "best_model.pt")
                print(f"  -> best ep{epoch} pr_auc={best_pr_auc:.3f}")
            else:
                bad += 1
        elif early_stop_metric == "pr_auc_diversity":
            is_cand = (val_pr_auc >= pr_auc_floor
                       and hist["mean_pairwise_spearman"] <= max_rank_spearman
                       and hist["n_distinct_top1_hosts"] >= min_distinct_top1_hosts)
            if is_cand:
                combo = val_pr_auc - diversity_penalty * hist["mean_pairwise_spearman"]
                if combo > best_combo_score:
                    best_combo_score = combo
                    best_pr_auc = val_pr_auc
                    best_epoch = epoch; bad = 0
                    torch.save({"model": model.state_dict(), "moe": moe.state_dict(),
                                "config": cfg, "host_to_id": host_to_id,
                                "epoch": epoch, "val_metrics": hist},
                               ckpt_dir / "best_model.pt")
                    print(f"  -> best ep{epoch} pr_auc={val_pr_auc:.3f} "
                          f"sp={hist['mean_pairwise_spearman']:.3f} "
                          f"distinct={hist['n_distinct_top1_hosts']} "
                          f"combo={combo:.3f}")
                else:
                    bad += 1
            else:
                bad += 1
        else:
            if is_cand and hist["mean_pairwise_spearman"] < best_spearman:
                best_spearman = hist["mean_pairwise_spearman"]
                best_epoch = epoch; bad = 0
                torch.save({"model": model.state_dict(), "moe": moe.state_dict(),
                            "config": cfg, "host_to_id": host_to_id,
                            "epoch": epoch, "val_metrics": hist},
                           ckpt_dir / "best_model.pt")
                print(f"  -> best ep{epoch} sp={best_spearman:.3f}")
            else:
                bad += 1
        if bad >= cfg["training"].get("early_stopping_patience", 40):
            print(f"早停 ep{epoch} best_ep={best_epoch}"); break
    torch.save({"model": model.state_dict(), "moe": moe.state_dict(),
                "config": cfg, "host_to_id": host_to_id, "epoch": epoch},
               ckpt_dir / "last_model.pt")
    if early_stop_metric == "pr_auc":
        print(f"完成。best_epoch={best_epoch} best_val_pr_auc={best_pr_auc:.4f}")
    elif early_stop_metric == "pr_auc_diversity":
        print(f"完成。best_epoch={best_epoch} best_val_pr_auc={best_pr_auc:.4f} "
              f"best_combo_score={best_combo_score:.4f}")
    else:
        print(f"完成。best_epoch={best_epoch} best_val_spearman={best_spearman:.4f}")


def _virus_rep_diversity(rep_fn, viruses, n=24):
    reps = []
    with torch.no_grad():
        for v in viruses[:n]:
            r = rep_fn(v).float()
            reps.append(r / (r.norm() + 1e-9))
    if len(reps) < 2:
        return 1.0, 1.0
    M = torch.stack(reps); cos = M @ M.t()
    iu = torch.triu_indices(M.shape[0], M.shape[0], 1)
    pair = cos[iu[0], iu[1]]
    return float(pair.mean()), float(pair.min())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="configs/vhe_net_with_weight.yaml")
    args = parser.parse_args()
    main(args.config)
