"""排名多样性 / 序列敏感度指标 —— 判定"每个病毒预测的宿主排名是否真有区别"。

指标：
  1. pairwise_spearman: 不同病毒宿主排名之间的 Spearman 相关系数。
     旧 0602 模型 = 1.000（完全一样）；目标 << 1。
  2. top_k_overlap: 不同病毒 top-k 宿主的平均交并比，越小越分化。
  3. top1_counts: 全部病毒 top-1 宿主的分布（宿主越多说明越分化）。
  4. shuffle_sensitivity: 把序列整条打乱后重新预测，排名相关性越低
     说明序列贡献越大（模型真的在"读"序列）。
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def build_rank_matrix(probs_by_virus: Dict[str, np.ndarray],
                      host_names: List[str]) -> pd.DataFrame:
    """{virus: prob[451]} -> DataFrame [n_virus x n_host]（概率）。"""
    return pd.DataFrame(probs_by_virus, index=host_names).T


def ranking_metrics(probs_by_virus: Dict[str, np.ndarray],
                    host_names: List[str],
                    max_pairs: int = 300,
                    seed: int = 0) -> Dict:
    mat = build_rank_matrix(probs_by_virus, host_names)
    viruses = list(mat.index)
    n = len(viruses)
    rng = np.random.default_rng(seed)
    sample = viruses if n <= 60 else list(
        rng.choice(viruses, size=60, replace=False))
    sample = list(sample)
    pos_of = {v: i for i, v in enumerate(viruses)}

    # 1. pairwise spearman（采样；NaN 对不计数，设迭代上限防死循环）
    corrs = []
    pairs = min(max_pairs, len(sample) * (len(sample) - 1) // 2)
    attempts = 0
    max_attempts = pairs * 8 + 200
    while len(corrs) < pairs and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, len(sample))
        j = rng.integers(0, len(sample))
        if i == j:
            continue
        r = spearmanr(mat.iloc[pos_of[sample[i]]],
                      mat.iloc[pos_of[sample[j]]]).correlation
        if r is not None and not np.isnan(r):
            corrs.append(r)
    corrs = np.array(corrs) if corrs else np.array([np.nan])

    # 2. top-k overlap
    topk = {}
    for k in (1, 5, 10, 20):
        sets = [set(np.argsort(-mat.iloc[pos_of[v]].values)[:k]) for v in sample]
        overlaps = []
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                inter = len(sets[a] & sets[b])
                overlaps.append(inter / k)
        topk[k] = float(np.mean(overlaps))

    # 3. top-1 分布
    top1 = [host_names[int(np.argmax(mat.iloc[pos_of[v]].values))] for v in viruses]
    top1_counts = pd.Series(top1).value_counts().head(10).to_dict()

    return {
        "n_viruses": n,
        "mean_pairwise_spearman": float(corrs.mean()),
        "std_pairwise_spearman": float(corrs.std()),
        "min_pairwise_spearman": float(corrs.min()),
        "max_pairwise_spearman": float(corrs.max()),
        "topk_overlap": topk,
        "n_distinct_top1_hosts": int(pd.Series(top1).nunique()),
        "top1_host_counts": top1_counts,
        "rank_std_within_virus": float(
            mat.std(axis=1).mean()),  # 概率在宿主间的平均标准差
    }


def spearman_between(a: np.ndarray, b: np.ndarray) -> float:
    r = spearmanr(a, b).correlation
    return float(r) if r is not None and not np.isnan(r) else float("nan")


def save_metrics(metrics: Dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)
    print(json.dumps(metrics, indent=1, ensure_ascii=False))
