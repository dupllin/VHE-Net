"""8_14 全滑窗编码模块。

与旧 pipeline 的关键区别：
  * 旧：np.linspace 稀疏抽 5 个 510bp 窗口（丢弃 ~90% 序列信息）；
  * 新：1024bp 窗口、stride=512 从 5' 到 3' 全覆盖滑动（含尾部锚定窗口），
    每个窗口提取 LucaVirus 的 CLS / token-mean / token-max 三种向量，
    全部窗口的向量都保留，交给下游模型做 masked mean+max 池化 ——
    即"用全部序列信息"，而不是只抽几段。

缓存设计：LucaVirus 是 frozen 的，窗口特征与训练无关，只依赖序列本身，
因此一次性预计算全部 934 个训练病毒的窗口特征并落盘（fp16），
之后训练/预测只读缓存，迭代速度大幅提升。
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

GAP_CHARS = set("-. *\n\r\t")


def clean_seq(seq: str) -> str:
    """去掉 MSA gap 和非法字符，统一大写。"""
    return "".join(ch for ch in seq.upper() if ch not in GAP_CHARS)


def window_starts(seq_len: int, window_size: int, stride: int):
    """返回 (start, end) 列表，保证整个序列被窗口完全覆盖。

    - 常规步进：0, stride, 2*stride, ...
    - 尾部锚定：最后一个窗口的 end 一定等于 seq_len（可能与前一窗重叠）。
    """
    if seq_len <= window_size:
        return [(0, seq_len)]
    starts = list(range(0, seq_len - window_size + 1, stride))
    if starts[-1] + window_size < seq_len:
        starts.append(max(0, seq_len - window_size))
    return [(s, min(s + window_size, seq_len)) for s in starts]


def _encode_windows(seq: str, tokenizer, model, device,
                    window_size: int, stride: int,
                    encode_batch: int = 16):
    """对一条序列的所有窗口做 LucaVirus 前向，返回
    cls [Nwin, H], mean [Nwin, H], max [Nwin, H] (fp16) 与 starts 列表。
    """
    spans = window_starts(len(seq), window_size, stride)
    chunks = [seq[s:e] for s, e in spans]

    cls_list, mean_list, max_list = [], [], []
    n = len(chunks)
    for i in range(0, n, encode_batch):
        batch_text = chunks[i:i + encode_batch]
        # 注意：LucaVirus tokenizer 的 batch 路径有 bug（核苷酸被映射到
        # 几乎共线的 token，id 也完全不同），必须逐条单字符串 tokenize。
        batch_enc = []
        for text in batch_text:
            enc = tokenizer(
                text,
                seq_type="gene",
                padding="max_length",
                truncation=True,
                max_length=window_size + 2,  # CLS + SEQ + SEP
                return_tensors="pt",
                add_special_tokens=True,
            )
            batch_enc.append(enc)
        ids = torch.cat([e["input_ids"] for e in batch_enc], dim=0)
        mask = torch.cat([e["attention_mask"] for e in batch_enc], dim=0)
        ids = ids.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        hidden = out.last_hidden_state  # [b, L, H]
        mm = mask.to(hidden.dtype).unsqueeze(-1)
        cls = hidden[:, 0, :]
        mean = (hidden * mm).sum(dim=1) / mm.sum(dim=1).clamp_min(1.0)
        maxv = hidden.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        cls_list.append(cls)
        mean_list.append(mean)
        max_list.append(maxv)
        del out, hidden

    return (torch.cat(cls_list, 0).half(),
            torch.cat(mean_list, 0).half(),
            torch.cat(max_list, 0).half(),
            spans)


def encode_fasta_to_cache(fasta_path: str, cache_dir: str,
                          tokenizer, model, device,
                          window_size: int = 1022, stride: int = 512,
                          encode_batch: int = 16, save_every: int = 50,
                          resume: bool = True):
    """编码整个 FASTA 到缓存目录。

    每个病毒一个文件 `<safe_name>.pt`，内容：
      {'cls': [Nwin,H] fp16, 'mean': ..., 'max': ..., 'starts': [[s,e],...]}
    另有 index.json 记录 病毒名 -> 文件、窗口数。
    resume=True 时跳过上次已完成（index 中存在且文件在）的病毒。
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
            print(f"resume: {len(index)} 个病毒已完成，将跳过")
        except Exception:
            index = {}

    seq_dict = parse_fasta(fasta_path)
    names = sorted(seq_dict.keys())
    print(f"共 {len(names)} 条序列待编码")

    model.eval()
    done = 0
    for idx, name in enumerate(names):
        if name in index:
            done += 1
            continue
        seq = clean_seq(seq_dict[name])
        if not seq:
            seq = "N"
        try:
            cls, mean, maxv, spans = _encode_windows(
                seq, tokenizer, model, device, window_size, stride, encode_batch)
        except Exception as exc:  # 单条失败不阻塞整体
            print(f"  [skip] {name}: {exc}")
            continue
        fn = cache / f"{idx:04d}.pt"
        torch.save({
            "cls": cls.cpu(),
            "mean": mean.cpu(),
            "max": maxv.cpu(),
            "starts": spans,
            "seq_len": len(seq),
        }, fn)
        index[name] = {"file": fn.name, "n_windows": cls.shape[0],
                       "seq_len": len(seq)}
        if (idx + 1) % save_every == 0 or idx == len(names) - 1:
            with open(index_file, "w") as f:
                json.dump(index, f, indent=1)
            print(f"  ... {idx + 1}/{len(names)} saved index "
                  f"(newly done {idx + 1 - done})")

    with open(index_file, "w") as f:
        json.dump(index, f, indent=1)
    total = sum(v["n_windows"] for v in index.values())
    print(f"编码完成: {len(index)} 病毒, {total} 窗口 -> {cache}")
    return index


def load_window_cache(cache_dir: str, max_viruses_in_ram: int = 200):
    """懒加载缓存：按需读盘，LRU 常驻 max_viruses_in_ram 个病毒。"""
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
