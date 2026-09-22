"""导出 v34GMoE 在 train/val 的全部宿主概率预测（与训练同款划分）。

输出长表 CSV：Virus, Host, Host_ID, Probability, Label, Split
Label: 1=该病毒正宿主，0=其余（val/train 中非正宿主按 0 处理）。
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

os.environ.setdefault("USE_TF", "0"); os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, TEST_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from vhenet.preprocessing import process_interaction_data, load_config
from vhenet.dataset import FullWindowIdsHistPosData
from vhenet.model import ViHGAT_814v6, SelfAttnPoolMoE


def resolve(p, base=PROJECT_ROOT):
    p = Path(p).expanduser(); return p if p.is_absolute() else (base / p)


def main(config_path, ckpt_path, out_prefix, split_by_virus=None, cls_cache="cache/cls_windows_cache_934.pt"):
    cfg = load_config(config_path); device = torch.device(cfg["device"]); seed = cfg.get("seed", 42)
    if str(device).startswith("cuda") and device.index is not None and device.index >= torch.cuda.device_count():
        print(f"[警告] device={cfg['device']} 越界, 回退 cuda:0", flush=True); device = torch.device("cuda:0")
    if split_by_virus is not None:
        cfg["data"]["split_by_virus"] = split_by_virus
    df, host_to_id, sim_matrix = process_interaction_data(
        str(resolve(cfg["data"]["interaction_csv"])), str(resolve(cfg["data"]["similarity_csv"])))
    df["importance score"] = df["importance score"].clip(
        lower=cfg["data"].get("imp_lower", 1.0), upper=cfg["data"].get("imp_upper", 20.0)).astype(float)
    dsr = cfg["data"].get("down_sample_ratio")
    if dsr and float(dsr) < 1.0:
        n_neg_keep = int((df["Label"] == 0).sum() * float(dsr))
        neg_df = df[df["Label"] == 0].sample(n=n_neg_keep, random_state=seed)
        df = pd.concat([df[df["Label"] == 1], neg_df]).reset_index(drop=True)
    _by_virus = bool(cfg["data"].get("split_by_virus", False))
    if _by_virus:
        splitter = GroupShuffleSplit(n_splits=1, test_size=1 - cfg["data"]["train_test_split"], random_state=seed)
        tr_idx, va_idx = next(splitter.split(df, groups=df["Virus"]))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=1 - cfg["data"]["train_test_split"], random_state=seed)
        tr_idx, va_idx = next(splitter.split(df))
    print(f"split_by_virus={_by_virus} (pairwise={not _by_virus})")
    train_df = df.iloc[tr_idx].reset_index(drop=True)
    val_df = df.iloc[va_idx].reset_index(drop=True)
    train_data = FullWindowIdsHistPosData(train_df, host_to_id, str(resolve(cfg["data"]["ids_cache_dir"])))
    val_data = FullWindowIdsHistPosData(val_df, host_to_id, str(resolve(cfg["data"]["ids_cache_dir"])))
    print(f"train viruses={train_df['Virus'].nunique()}, val viruses={val_df['Virus'].nunique()}")

    mc = cfg["model"]
    model = ViHGAT_814v6(
        num_hosts=len(host_to_id), host_sim_matrix=sim_matrix,
        llm_model_path=str(resolve(mc["model_path"])),
        embed_dim=mc.get("embed_dim", 256), dropout=mc.get("dropout", 0.25),
        n_pos_bins=mc.get("n_pos_bins", 8), lora_r=mc.get("lora_r", 8),
        lora_alpha=mc.get("lora_alpha", 16), lora_dropout=mc.get("lora_dropout", 0.1),
        llm_window_tokens=mc.get("llm_window_tokens", 1024),
        use_host_head=False, main_head_inputs="virus_inter", use_km_branch=False,
        b_agg_bins=0, cls_only=True, freeze_lora=True, cls_no_proj=True,
        use_cross_attn=False, cross_heads=mc.get("cross_heads", 4), whiten_stats=None)
    st = torch.load(resolve(ckpt_path), map_location="cpu")
    model.load_state_dict(st["model"], strict=False)
    if "moe" in st:
        h = st["moe"]["q_tokens"].shape[0]
        moe = SelfAttnPoolMoE(mc.get("embed_dim", 256), n_heads=h, dropout=mc.get("dropout", 0.25),
                              attn_type=mc.get("attn_type", "sparsemax"),
                              attn_temp=float(mc.get("attn_temp", 1.0)))
        moe.load_state_dict(st["moe"], strict=True)
    else:
        moe = SelfAttnPoolMoE(mc.get("embed_dim", 256), n_heads=mc.get("moe_heads", 2),
                              dropout=mc.get("dropout", 0.25),
                              attn_type=mc.get("attn_type", "sparsemax"),
                              attn_temp=float(mc.get("attn_temp", 1.0)))
    model.to(device).eval(); moe.to(device).eval()

    cls_windows = torch.load(resolve(cls_cache), map_location="cpu")
    host_ids_all = torch.arange(len(host_to_id), device=device)
    id_to_host = {v: k for k, v in host_to_id.items()}
    all_rows = []
    with torch.no_grad():
        for split, data in [("train", train_data), ("val", val_data)]:
            for virus in data.virus_rows:
                wf = cls_windows[virus].to(device)
                win_proj = model.llm_out(wf)
                vrep, attn = moe(win_proj)
                rep_e = vrep.unsqueeze(0).expand(len(host_to_id), -1)
                host_rep = model.host_rep_for(host_ids_all)
                inter = model.virus_interact(rep_e) * model.host_interact(host_rep)
                feats = torch.cat([rep_e, inter], dim=-1)
                p = torch.sigmoid(model.classifier(feats).squeeze(-1)).detach().cpu().numpy()
                pos = set(data.pos_hosts[virus])
                for h in range(len(host_to_id)):
                    all_rows.append((virus, id_to_host[h], h, float(p[h]), int(h in pos), split))
    out = pd.DataFrame(all_rows, columns=["Virus", "Host", "Host_ID", "Probability", "Label", "Split"])
    # 该划分是 pair-level（split_by_virus=false），因此 934 个病毒在两个 fold 里都出现，
    # 上面的循环会为每个 (Virus, Host) 产出两行。全量网格是 421,234 个**唯一**配对，
    # 所以这里按键去重（保留 val 行：val 行不带训练 fold 的正例标签，语义更干净）。
    n_before = len(out)
    if out["Split"].nunique() > 1:
        out = (out.sort_values("Split", ascending=False)      # "val" > "train"
                  .drop_duplicates(subset=["Virus", "Host"], keep="first")
                  .reset_index(drop=True))
        print(f"去重: {n_before:,} -> {len(out):,} 行（pair-level 划分使每个病毒出现在两个 fold）")
    if out_prefix:
        outp = resolve(out_prefix); outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        print("saved ->", outp, len(out), "rows")
    # quick summary
    for split in ["train", "val"]:
        sub = out[out["Split"] == split]
        pos = sub[sub["Label"] == 1]
        print(f"{split}: viruses={sub['Virus'].nunique()}, rows={len(sub)}, pos={len(pos)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vhe_net_with_weight.yaml")
    ap.add_argument("--cls-cache", default="cache/cls_windows_cache_934.pt",
                    help="预计算窗口 CLS 缓存路径")
    ap.add_argument("--checkpoint", default="checkpoints/vhe_net_with_weight/last_model.pt")
    ap.add_argument("--out", default="outputs/predictions.csv")
    ap.add_argument("--split-by-virus", dest="split_by_virus", type=int, default=None,
                    help="1=按病毒分组划分, 0=pairwise 划分, 不传=沿用 config")
    a = ap.parse_args(); main(a.config, a.checkpoint, a.out, a.split_by_virus, a.cls_cache)