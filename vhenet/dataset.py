"""8_14 数据集：整条序列全滑窗特征（读预计算缓存）。

行 = (virus, host, label, weight)。同一病毒的所有宿主行共享同一组窗口特征，
由 trainer 按病毒分组前向，避免重复计算。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from vhenet.encode import load_window_cache
from vhenet.encode_ids import load_ids_cache


class _BaseInteractionData:
    """行表公共逻辑（v1 池化特征 / v2 token ids 共用）。"""

    def __init__(self, df, host_to_id):
        self.df = df.reset_index(drop=True)
        self.host_to_id = host_to_id
        self.id_to_host = {v: k for k, v in host_to_id.items()}
        self.num_hosts = len(host_to_id)
        self.id_to_host_list = [self.id_to_host[i] for i in range(self.num_hosts)]

        viruses = df["Virus"].values
        hosts = df["Host"].values
        labels = df["Label"].values.astype(float)
        if "importance score" in df.columns:
            weights = df["importance score"].values.astype(float)
        else:
            weights = np.ones(len(df), dtype=float)
        self.rows: List[Tuple[str, int, float, float]] = [
            (str(v), host_to_id.get(h, 0), float(l), float(w))
            for v, h, l, w in zip(viruses, hosts, labels, weights)
        ]
        self.virus_rows: Dict[str, List[int]] = {}
        for i, (virus, *_rest) in enumerate(self.rows):
            self.virus_rows.setdefault(virus, []).append(i)

        self.pos_hosts: Dict[str, List[int]] = {}
        self.pos_weights: Dict[str, List[float]] = {}
        for virus, idxs in self.virus_rows.items():
            ph, pw = [], []
            for i in idxs:
                _, _h, lab, w = self.rows[i]
                if lab > 0.5:
                    ph.append(_h)
                    pw.append(w)
            self.pos_hosts[virus] = ph
            self.pos_weights[virus] = pw

    def sample_train_rows(self, virus: str, n_neg: int, rng: np.random.Generator,
                          max_pos: int = 64):
        """为一个病毒抽样训练行：全部正样本 + n_neg 个负样本（去重）。

        n_neg=-1（或 None）：保留该病毒的所有负样本（全 451 宿主格局，
        宿主流行度先验在 per-virus softmax 下完全失效）。
        """
        idxs = self.virus_rows[virus]
        pos_idx = [i for i in idxs if self.rows[i][2] > 0.5]
        neg_idx = [i for i in idxs if self.rows[i][2] <= 0.5]
        if len(pos_idx) > max_pos:
            pos_idx = rng.choice(pos_idx, size=max_pos, replace=False).tolist()
        if n_neg is not None and n_neg >= 0 and len(neg_idx) > n_neg:
            neg_idx = rng.choice(neg_idx, size=n_neg, replace=False).tolist()
        sel = pos_idx + neg_idx
        hosts = np.array([self.rows[i][1] for i in sel], dtype=np.int64)
        labels = np.array([self.rows[i][2] for i in sel], dtype=np.float32)
        weights = np.array([self.rows[i][3] for i in sel], dtype=np.float32)
        return hosts, labels, weights


class FullWindowInteractionData(_BaseInteractionData):
    """v1：全滑窗池化特征（读 LucaVirus 预计算缓存）。"""

    def __init__(self, df, host_to_id, cache_dir, max_viruses_in_ram=400):
        super().__init__(df, host_to_id)
        self.get_virus, self.cache_index = load_window_cache(
            cache_dir, max_viruses_in_ram=max_viruses_in_ram)

    def virus_window_tensors(self, virus: str):
        data = self.get_virus(virus)
        feats = torch.cat([data["cls"], data["mean"], data["max"]], dim=1).float()
        mask = torch.ones(feats.shape[0], dtype=torch.float32)
        return feats, mask


class FullWindowIdsData(_BaseInteractionData):
    """v2：全滑窗 token ids（内容无损）。"""

    def __init__(self, df, host_to_id, cache_dir, max_viruses_in_ram=600):
        super().__init__(df, host_to_id)
        self.get_virus, self.cache_index = load_ids_cache(
            cache_dir, max_viruses_in_ram=max_viruses_in_ram)

    def virus_window_tensors(self, virus: str):
        data = self.get_virus(virus)
        ids = data["ids"].long()
        mask = data["mask"].bool()
        return ids, mask


class FullWindowIdsHistData(_BaseInteractionData):
    """v3：全滑窗 token ids + k-mer 谱。"""

    def __init__(self, df, host_to_id, cache_dir, max_viruses_in_ram=600):
        super().__init__(df, host_to_id)
        self.get_virus, self.cache_index = load_ids_cache(
            cache_dir, max_viruses_in_ram=max_viruses_in_ram)

    def virus_window_tensors(self, virus: str):
        data = self.get_virus(virus)
        ids = data["ids"].long()
        mask = data["mask"].bool()
        hist = torch.as_tensor(data["hist"], dtype=torch.float32)
        return ids, mask, hist


class FullWindowIdsHistPosData(_BaseInteractionData):
    """v5：全滑窗 token ids + k-mer 谱 + 窗口基因组相对位置。"""

    def __init__(self, df, host_to_id, cache_dir, max_viruses_in_ram=600):
        super().__init__(df, host_to_id)
        self.get_virus, self.cache_index = load_ids_cache(
            cache_dir, max_viruses_in_ram=max_viruses_in_ram)

    def virus_window_tensors(self, virus: str):
        data = self.get_virus(virus)
        ids = data["ids"].long()
        mask = data["mask"].bool()
        hist = torch.as_tensor(data["hist"], dtype=torch.float32)
        seq_len = float(data["seq_len"])
        spans = data["starts"]
        centers = [(float(s) + float(e)) / 2.0 / max(seq_len, 1.0)
                   for s, e in spans]
        pos_frac = torch.tensor(centers, dtype=torch.float32).clamp(0.0, 1.0)
        return ids, mask, hist, pos_frac
