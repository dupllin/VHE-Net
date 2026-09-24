#!/usr/bin/env python3
"""复现性自检：确认目录完整、能加载模型、能复现关键数字"""
import os, sys, json, yaml
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["LD_LIBRARY_PATH"] = "/home/oem/miniconda3/envs/ai_work/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

print("=" * 96)
print("① 文件完整性")
print("=" * 96)
NEED = [
    ("训练脚本", "train.py"),
    ("模型定义", "vhenet/model.py"),
    ("损失函数", "vhenet/losses.py"),
    ("预处理", "vhenet/preprocessing.py"),
    ("核心数据", "data/interactions.xlsx"),
    ("病毒序列", "data/virus_sequences.fasta"),
    ("相似性", "data/host_similarity.csv"),
    ("白化统计", "data/whiten_stats.pt"),
    ("LLM权重", "pretrained/lucaVirus/model.safetensors"),
    ("kmer缓存", "cache/ids_km_cache_934"),
    ("ckpt-With", "checkpoints/vhe_net_with_weight/last_model.pt"),
    ("ckpt-Without", "checkpoints/vhe_net_without_weight/last_model.pt"),
]
ok = True
for nm, rel in NEED:
    p = ROOT / rel
    if not p.exists():
        print(f"  ❌ {nm:14} 缺失: {rel}"); ok = False; continue
    if p.is_dir():
        n = sum(1 for _ in p.rglob('*') if _.is_file())
        print(f"  ✅ {nm:14} {n} 个文件")
    else:
        print(f"  ✅ {nm:14} {p.stat().st_size/1024/1024:>9.2f} MB")
if not ok:
    print("\n❌ 文件不完整，无法复现"); sys.exit(1)

print()
print("=" * 96)
print("② 数据核验（关键数字必须与交付完全一致）")
print("=" * 96)
d = pd.read_excel(ROOT / "data/interactions.xlsx")
assert len(d) == 421234, f"全量行数 {len(d)} != 421234"
assert int(d.Label.sum()) == 3795, f"正例 {int(d.Label.sum())} != 3795"
print(f"  全量: {len(d):,} 行 | 正 {int(d.Label.sum())} | 负 {int((d.Label==0).sum()):,}  ✅")

DSR = 0.0909114865; SEED = 42
n_neg = int((d.Label == 0).sum() * DSR)
assert n_neg == 37950, f"负采样 {n_neg} != 37950"
sel = d[d.Label == 0].sample(n=n_neg, random_state=SEED)
mine = pd.concat([d[d.Label == 1], sel]).reset_index(drop=True)
assert len(mine) == 41745, f"1:10 子集 {len(mine)} != 41745"
from sklearn.model_selection import ShuffleSplit
tri, vai = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(mine))
assert len(tri) == 33396 and len(vai) == 8349, (len(tri), len(vai))
assert int(mine.iloc[vai].Label.sum()) == 797
print(f"  严格1:10: {len(mine):,} (train {len(tri):,} / val {len(vai):,}) "
      f"| val正例 {int(mine.iloc[vai].Label.sum())}  ✅")

print()
print("=" * 96)
print("③ 模型加载核验")
print("=" * 96)
import torch
cfg = yaml.safe_load(open(ROOT / "configs/vhe_net_with_weight.yaml"))
print(f"  config 关键字段:")
for k in ["down_sample_ratio", "train_test_split", "random_state", "split_by_virus",
          "val_ratio", "test_ratio", "imp_lower", "imp_upper"]:
    print(f"    data.{k:20} = {cfg['data'][k]}")
print(f"    training.bce_lambda  = {cfg['training']['bce_lambda']}")
print(f"    training.rank_lambda = {cfg['training']['rank_lambda']}")
print(f"    model.name           = {cfg['model']['name']}")

for tag, label in [("vhe_net_with_weight", "With Niche Weighting"),
                   ("vhe_net_without_weight", "Without Niche Weighting")]:
    ck = torch.load(ROOT / f"checkpoints/{tag}/last_model.pt", map_location="cpu",
                    weights_only=False)
    keys = list(ck.keys())
    n_param = sum(v.numel() for v in ck["model"].values() if hasattr(v, "numel"))
    print(f"  {tag} ({label}):")
    print(f"    keys = {keys}")
    print(f"    参数张量 {len(ck['model'])} 个, 共 {n_param:,} 个元素")
    if "moe" in ck:
        print(f"    moe q_tokens shape = {ck['moe']['q_tokens'].shape}")

print()
print("=" * 96)
print("✅ 复现性自检通过")
print("=" * 96)
print("""
Next steps:
  1. python train.py --config configs/vhe_net_with_weight.yaml
  2. python train.py --config configs/vhe_net_without_weight.yaml
  3. python predict.py --config configs/vhe_net_with_weight.yaml --checkpoint checkpoints/vhe_net_with_weight/last_model.pt --out outputs/predictions.csv
  4. compute PR-AUC / ROC-AUC / F1 / MCC on the validation fold from outputs/predictions.csv
""")
