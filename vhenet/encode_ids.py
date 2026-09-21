"""v2 编码：缓存每个窗口的 token ids（内容无损）。

关键诊断结论（v1 教训）：
  * LucaVirus CLS/mean/max 特征跨病毒 cosine ~0.9997 —— 几乎无内容信息；
  * mean 池化只保留碱基组成（打乱序列后均值相同）；
  * 真正的内容信号在逐 token 的核苷酸嵌入（A/C/G/T 向量 cosine 0.13-0.21）。
因此 v2 只缓存 token ids（tokenizer 输出，纯 CPU，快），下游用冻结的
LucaVirus 核苷酸嵌入表 + 可训练 token CNN 提取内容特征。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from vhenet.encode import clean_seq, window_starts

# tokenizer id -> 碱基编码 (A=0, T=1, G=2, C=3, 其他=4)
_TOKEN_ID_MAP = {5: 0, 6: 1, 8: 2, 7: 3}


def _build_kmer_hist(ids: torch.Tensor, k: int = 6, n_bins: int = 4096):
    """ids: [Nwin, L] int16（tokenizer id）-> [Nwin, n_bins] float 计数。

    用 6-mer 滚动哈希：code = Σ base_i * 4^i，每窗口直方图归一化到总和 1。
    """
    import numpy as np
    ids_np = ids.numpy()
    base = np.zeros_like(ids_np, dtype=np.int64)
    for tid, b in _TOKEN_ID_MAP.items():
        base[ids_np == tid] = b
    # 非法 token（含特殊 token）记 4
    base[~np.isin(ids_np, list(_TOKEN_ID_MAP.keys()))] = 4
    Nwin, L = base.shape
    codes = np.zeros((Nwin, max(0, L - k + 1)), dtype=np.int64)
    p4 = 1
    for i in range(k):
        codes += base[:, i:i + L - k + 1] * p4
        p4 *= 4
    hist = np.zeros((Nwin, n_bins), dtype=np.float32)
    for w in range(Nwin):
        row = codes[w]
        row = row[row < n_bins]  # 丢弃含 N 的 k-mer
        if row.size:
            counts = np.bincount(row, minlength=n_bins)
            hist[w] = counts.astype(np.float32)  # 原始计数，模型内 log1p
    return hist


def encode_ids_to_cache(fasta_path: str, cache_dir: str, tokenizer,
                        window_size: int = 1022, stride: int = 512,
                        save_every: int = 100, resume: bool = True,
                        kmer: int = 6):
    """编码 FASTA 全部窗口的 token ids + k-mer 谱到缓存。

    每个病毒一个文件：{'ids': [Nwin, L] int16, 'mask': [Nwin, L] bool,
                       'hist': [Nwin, 4^k] float32, 'starts': [[s,e],...],
                       'seq_len': int}
    """
    from vhenet.preprocessing import parse_fasta
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / "index.json"
    index = {}
    if resume and index_file.is_file():
        try:
            with open(index_file) as f:
                index = json.load(f)
            index = {k: v for k, v in index.items()
                     if (cache / v["file"]).is_file()}
        except Exception:
            index = {}

    seq_dict = parse_fasta(fasta_path)
    names = sorted(seq_dict.keys())
    L = window_size + 2  # CLS + SEQ + SEP
    print(f"共 {len(names)} 条序列待 tokenize")

    for idx, name in enumerate(names):
        if name in index:
            continue
        seq = clean_seq(seq_dict[name]) or "N"
        spans = window_starts(len(seq), window_size, stride)
        chunks = [seq[s:e] for s, e in spans]
        # LucaVirus tokenizer 的 batch 路径有 bug（核苷酸 id 映射错乱），
        # 必须逐窗口单字符串 tokenize。
        enc_list = []
        for text in chunks:
            enc = tokenizer(
                text,
                seq_type="gene",
                padding="max_length",
                truncation=True,
                max_length=L,
                return_tensors="pt",
                add_special_tokens=True,
            )
            enc_list.append(enc)
        ids = torch.cat([e["input_ids"] for e in enc_list],
                        dim=0).to(torch.int16)        # [Nwin, L]
        mask = torch.cat([e["attention_mask"] for e in enc_list],
                         dim=0).bool()                # [Nwin, L]
        hist = None
        if kmer and kmer > 0:
            hist = torch.from_numpy(
                _build_kmer_hist(ids, k=kmer, n_bins=4 ** kmer))
        fn = cache / f"{idx:04d}.pt"
        torch.save({"ids": ids, "mask": mask, "hist": hist, "starts": spans,
                    "seq_len": len(seq)}, fn)
        index[name] = {"file": fn.name, "n_windows": ids.shape[0],
                       "seq_len": len(seq)}
        if (idx + 1) % save_every == 0 or idx == len(names) - 1:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=1)
            print(f"  ... {idx + 1}/{len(names)} saved index")

    with open(index_file, "w") as f:
        json.dump(index, f, indent=1)
    print(f"tokenize 完成: {len(index)} 病毒 -> {cache}")

    # ---- 语料级 z-score 统计（log1p(原始计数) 的每 bin 均值/标准差）----
    if kmer and kmer > 0:
        stats_file = cache / "hist_stats.pt"
        if not stats_file.is_file():
            print("计算 k-mer 谱语料统计 ...")
            import torch as _t
            sums = None
            sumsq = None
            total_windows = 0
            for info in index.values():
                d = _t.load(cache / info["file"], map_location="cpu")
                h = _t.as_tensor(d["hist"], dtype=_t.float32)
                h = _t.log1p(h)
                s = h.sum(dim=0)
                q = (h * h).sum(dim=0)
                sums = s if sums is None else sums + s
                sumsq = q if sumsq is None else sumsq + q
                total_windows += h.shape[0]
            mean = sums / total_windows
            var = sumsq / total_windows - mean * mean
            std = var.clamp_min(1e-4).sqrt()
            _t.save({"mean": mean, "std": std}, stats_file)
            print(f"hist_stats 保存: {stats_file} (windows={total_windows})")
    return index


def load_ids_cache(cache_dir: str, max_viruses_in_ram: int = 500):
    cache = Path(cache_dir)
    with open(cache / "index.json") as f:
        index = json.load(f)
    ram: dict = {}
    order: list = []

    def get(name: str):
        if name in ram:
            order.remove(name)
            order.append(name)
            return ram[name]
        info = index.get(name)
        if info is None:
            raise KeyError(f"virus {name} not in cache index")
        data = torch.load(cache / info["file"], map_location="cpu")
        ram[name] = data
        order.append(name)
        if len(ram) > max_viruses_in_ram:
            old = order.pop(0)
            del ram[old]
        return data

    return get, index
