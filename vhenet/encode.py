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

# --------------------------------------------------------------------------- #
# gene 模式 tokenization                                                       #
# --------------------------------------------------------------------------- #
# 背景（踩坑记录，务必先读再改）
#
# LucaVirusTokenizer 有两个词表：'gene'（核苷酸 '1'-'5' -> id 5-9）与
# 'prot'（蛋白字母 -> id 10+）。选哪个由 `seq_type` 参数决定，而它只在
# **单字符串** 路径上生效：
#
#     tokenizer("GAAT", seq_type="gene")   -> encode_plus()       -> 走 gene  ✅
#     tokenizer(["GAAT"], seq_type="gene") -> batch_encode_plus() -> 丢掉参数 ❌
#
# 见 tokenization_lucavirus.py:294：
#
#     def batch_encode_plus(self, *args, **kwargs):
#         kwargs.pop("seq_type", None)          # <-- 直接丢弃
#
# 参数被丢弃后，父类的默认路径按 protein 词表逐字符查表，于是 DNA 变成蛋白
# token，且 **不抛任何异常**：
#
#     'GAAT' 正确 -> gene_seq_replace '4112' -> id [8, 5, 5, 6]
#     'GAAT' 错误 -> 原样查 gene_prot 词表    -> id [12, 11, 11, 17]
#
# 实测（真实 1022 bp 窗口）：tok(s) 正确，tok([s]) 错误。
#
# 因此本模块不依赖 tokenizer 的 seq_type，改为显式做 gene 映射；
# 调用方也不要再把窗口攒成 list 一次性喂给 tokenizer。
_GENE_VOCAB = {"1": 5, "2": 6, "3": 7, "4": 8, "5": 9}
_DNA_TO_GENE = {"A": "1", "T": "2", "U": "2", "C": "3", "G": "4"}
_PAD_ID, _CLS_ID, _SEP_ID = 0, 2, 3


def clean_seq(seq: str) -> str:
    """去掉 MSA gap 和非法字符，统一大写。"""
    return "".join(ch for ch in seq.upper() if ch not in GAP_CHARS)


def gene_seq_replace(seq: str) -> str:
    """DNA -> gene 字母表：A->'1', T/U->'2', C->'3', G->'4'，其他->'5'。

    与 LucaVirusTokenizer.gene_seq_replace 等价，但显式调用、不依赖 tokenizer。
    """
    return "".join(_DNA_TO_GENE.get(ch.upper(), "5") for ch in seq)


def tokenize_gene_ids(seq: str, max_len: int = 1024):
    """把一段 DNA 编成 LucaVirus 的 gene 模式 token id。

    返回 (ids, mask)，二者均为 Python list，长度 == max_len。
    布局： [CLS] + gene ids + [SEP] + [PAD]*
    """
    rep = gene_seq_replace(seq)[: max_len - 2]
    ids = [_CLS_ID] + [_GENE_VOCAB[c] for c in rep] + [_SEP_ID]
    ids = ids[:max_len] + [_PAD_ID] * max(0, max_len - len(ids))
    mask = [1 if i != _PAD_ID else 0 for i in ids]
    return ids, mask


def verify_gene_encoding(ids_cache_dir, fasta_path, n_virus: int = 3,
                         n_window: int = 3) -> bool:
    """用 tokenize_gene_ids 复算若干窗口，与已落盘的 ids 缓存比对。

    用于确认缓存是由正确的 gene 模式生成的（而不是误用 tokenizer 的
    seq_default）。返回 True 表示全部一致。
    """
    from vhenet.preprocessing import parse_fasta

    cache = Path(ids_cache_dir)
    index = json.loads((cache / "index.json").read_text())
    seq_dict = parse_fasta(fasta_path)
    names = [n for n in sorted(index) if n in seq_dict][:n_virus]
    total = ok = 0
    for name in names:
        data = torch.load(cache / index[name]["file"], map_location="cpu")
        ids, starts = data["ids"].long(), data["starts"]
        seq = clean_seq(seq_dict[name]) or "N"
        for w in range(min(n_window, len(starts))):
            s, e = starts[w]
            mine, _ = tokenize_gene_ids(seq[s:e], max_len=ids.shape[1])
            total += 1
            if mine == ids[w].tolist():
                ok += 1
    print(f"gene 编码校验: {ok}/{total} 个 (病毒, 窗口) 完全一致")
    return ok == total and total > 0


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
        # 显式 gene 映射，不经 tokenizer（否则 seq_type 被忽略 -> 蛋白词表）。
        # 见本模块顶部 "gene 模式 tokenization" 注释。
        id_rows, mask_rows = [], []
        for text in batch_text:
            i_row, m_row = tokenize_gene_ids(text, max_len=window_size + 2)
            id_rows.append(i_row)
            mask_rows.append(m_row)
        ids = torch.tensor(id_rows, dtype=torch.long)
        mask = torch.tensor(mask_rows, dtype=torch.long)
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


def apply_whitening(cls: torch.Tensor, whiten_stats) -> torch.Tensor:
    """对窗口 CLS 做白化：`(cls - mean) @ W`。

    白化是本流程**唯一**去除跨病毒共线主分量的步骤，必须在**构建 CLS 缓存时**
    施加一次。实测证据（209 病毒官方缓存）：原始 CLS 的跨病毒 cosine 为
    0.69-0.83，白化后降到 0.02-0.09；把缓存反白化（`X @ W^{-1} + mean`）后
    cosine 回升到 0.765，确认缓存存的确实是白化后的向量。

    ⚠️ `train.py` / `predict.py` 用 `whiten_stats=None` 构造模型，正是因为缓存
    已白化 —— 再白化一次会破坏特征。若缓存未白化（本函数的
    `whiten_stats=None` 分支），必须在构造模型时传入同样的统计量。

    whiten_stats: `{"mean": [H], "W": [H, H]}` 或 None。
    """
    if whiten_stats is None:
        return cls
    mean = torch.as_tensor(whiten_stats["mean"], dtype=torch.float32)
    W = torch.as_tensor(whiten_stats["W"], dtype=torch.float32)
    if W.shape[0] != cls.shape[-1]:
        raise ValueError(
            f"白化矩阵维度 {tuple(W.shape)} 与 CLS 维度 {cls.shape[-1]} 不符")
    return (cls.float() - mean) @ W


def encode_fasta_to_cache(fasta_path: str, cache_dir: str,
                          tokenizer=None, model=None, device=None,
                          window_size: int = 1022, stride: int = 512,
                          encode_batch: int = 16, save_every: int = 50,
                          resume: bool = True, packed_path: str = None,
                          whiten_stats=None):
    """编码整个 FASTA 的全部窗口 CLS 特征。

    两种落盘格式：

    * 目录格式（默认）—— 每病毒一个 `<idx>.pt` + `index.json`，
      供 `load_window_cache()` / `FullWindowInteractionData` 使用。
    * 打包单文件（`packed_path='cache/cls_windows_cache_934.pt'`）——
      一个 dict `{virus: [Nwin, 2560] CLS}`，供 `train.py` / `predict.py`
      直接 `torch.load`。**这是本仓库 reported runs 使用的格式。**

    ⚠️ **白化：** 传入 `whiten_stats` 时，本函数在写入前对 CLS 施加
    `(cls - mean) @ W`，产出的缓存与 reported runs 使用的
    `cache/cls_windows_cache_934.pt` 同构，可直接被 `train.py` / `predict.py`
    （它们用 `whiten_stats=None`）消费。**不传则写入原始未白化 CLS**，此时
    必须在构造模型时把同一份统计量传给 `whiten_stats=`，否则特征分布与训练
    时不一致。两种做法都不要同时用，否则等于白化两次。

    参数 `tokenizer` 与 `device` 保留仅为兼容旧调用点；tokenization 已改由
    `tokenize_gene_ids()` 显式完成。

    用法::

        from vhenet.encode import encode_fasta_to_cache
        stats = torch.load("data/whiten_stats.pt", map_location="cpu")
        encode_fasta_to_cache("data/virus_sequences.fasta",
                              "cache/cls_windows_cache_934",
                              model=model, device=dev,
                              whiten_stats=stats,          # 与 reported runs 一致
                              packed_path="cache/cls_windows_cache_934.pt")
    """
    from vhenet.preprocessing import parse_fasta
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    print(f"白化: {'启用 (cls-mean)@W' if whiten_stats is not None else '关闭（写入原始 CLS）'}")

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

    packed = {} if packed_path else None
    model.eval()
    done = 0
    for idx, name in enumerate(names):
        if name in index:
            done += 1
            if packed is not None:
                d = torch.load(cache / index[name]["file"], map_location="cpu")
                packed[name] = d["cls"].float()
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
        cls = apply_whitening(cls, whiten_stats)
        fn = cache / f"{idx:04d}.pt"
        torch.save({
            "cls": cls.cpu(),
            "mean": mean.cpu(),
            "max": maxv.cpu(),
            "starts": spans,
            "seq_len": len(seq),
        }, fn)
        if packed is not None:
            packed[name] = cls.float()
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
    if packed is not None:
        # packed 与目录格式内容一致：传了 whiten_stats 就是已白化，否则是原始 CLS。
        # train.py / predict.py 用 whiten_stats=None 构造模型，因此它们期望的就是
        # 【已白化】缓存 —— 构建时请传 whiten_stats（见函数 docstring）。
        torch.save(packed, packed_path)
        tag = "已白化" if whiten_stats is not None else "未白化原始"
        print(f"打包单文件: {packed_path} ({len(packed)} 病毒, {tag} CLS)")
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


# --------------------------------------------------------------------------- #
# CLI：重建窗口 CLS 缓存                                                        #
# --------------------------------------------------------------------------- #
def verify_cache_whitening(cls_cache, whiten_stats, n_virus: int = 60,
                           verbose: bool = True):
    """判定一份窗口 CLS 缓存是【已白化】还是【原始未白化】。

    判据（实测标定，来自 delivered runs 的 209 病毒缓存）：

    * 已白化：跨病毒 CLS cosine ≈ 0.02-0.09，各维均值 ≈ 0
    * 未白化：跨病毒 CLS cosine ≈ 0.69-0.83（LucaVirus CLS 强共线）

    辅助判据：对缓存做反白化 `X @ W^{-1} + mean`；若缓存本已白化，cosine 会
    显著**升高**（实测 0.037 -> 0.765）；若缓存是原始的，反白化会让 cosine
    **降低**并放大范数。

    参数
    ----
    cls_cache : str | Path | dict
        打包单文件路径，或已 `torch.load` 的 `{virus: [Nwin, 2560]}`。
    whiten_stats : dict
        `{"mean": [H], "W": [H, H]}`。

    返回
    ----
    dict，含 `is_whitened`（bool）、`cosine_as_is`、`cosine_unwhitened`、
    `mean_abs` 及判据文本。
    """
    data = (torch.load(str(cls_cache), map_location="cpu", weights_only=False)
            if isinstance(cls_cache, (str, Path)) else cls_cache)
    keys = sorted(data)[:n_virus]
    X = torch.cat([data[k].float() for k in keys], dim=0)
    if X.shape[0] < 2:
        raise ValueError("样本不足，至少需要 2 个窗口向量")

    def cos(Z):
        V = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
        C = V @ V.T
        n = C.shape[0]
        return C[~torch.eye(n, dtype=bool)].mean().item()

    mean = torch.as_tensor(whiten_stats["mean"], dtype=torch.float32)
    W = torch.as_tensor(whiten_stats["W"], dtype=torch.float32)
    c_raw = cos(X)
    X_un = X @ torch.linalg.pinv(W) + mean
    c_un = cos(X_un)
    mean_abs = X.mean(dim=0).abs().mean().item()

    # 已白化 <=> 原始 cosine 低 且 反白化后 cosine 升高
    is_whitened = (c_raw < 0.35) and (c_un > c_raw)

    if verbose:
        print(f"缓存白化状态判定（{len(keys)} 病毒, {X.shape[0]} 窗口）")
        print(f"  as-is   跨病毒 cosine = {c_raw:.4f}   |mean| = {mean_abs:.4f}")
        print(f"  反白化后 跨病毒 cosine = {c_un:.4f}")
        print(f"  参考: 已白化 ~0.02-0.09（反白化后升高）| "
              f"未白化 ~0.69-0.83（反白化后降低）")
        print(f"  => {'已白化 (whitened)' if is_whitened else '原始未白化 (raw)'}")
        if is_whitened:
            print("     与 train.py / predict.py 的 whiten_stats=None 一致 ✅")
        else:
            print("     ⚠️ train.py / predict.py 默认按已白化消费，会不一致；"
                  "请构建时传 whiten_stats，或构造模型时传同一统计量")

    return {"is_whitened": is_whitened, "cosine_as_is": c_raw,
            "cosine_unwhitened": c_un, "mean_abs": mean_abs,
            "n_windows": int(X.shape[0])}


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="重建 LucaVirus 窗口 CLS 缓存（每病毒一个 .pt + index.json）。")
    ap.add_argument("--fasta", default="data/virus_sequences.fasta",
                    help="输入 FASTA")
    ap.add_argument("--cache-dir", default="cache/cls_windows_cache_934",
                    help="输出缓存目录（每病毒一个 .pt）")
    ap.add_argument("--model-path", default="pretrained/lucaVirus",
                    help="LucaVirus 模型目录")
    ap.add_argument("--window", type=int, default=1022, help="窗口长度（碱基）")
    ap.add_argument("--stride", type=int, default=512, help="滑动步长")
    ap.add_argument("--batch", type=int, default=16, help="编码批大小")
    ap.add_argument("--device", default=None,
                    help="cpu / cuda:0 …（默认自动选择）")
    ap.add_argument("--packed-path", default=None,
                    help="额外输出打包单文件 dict{virus:[Nwin,2560]}；"
                         "train.py / predict.py 读这个格式")
    ap.add_argument("--whiten-stats", default="data/whiten_stats.pt",
                    help="白化统计 .pt（含 mean/W）。reported runs 的缓存是"
                         "已白化的，因此默认施加；传 'none' 写入原始 CLS")
    a = ap.parse_args()

    from transformers import AutoModel

    if a.device:
        device = torch.device(a.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    whiten_stats = None
    if a.whiten_stats and a.whiten_stats.lower() != "none":
        ws = Path(a.whiten_stats)
        if not ws.is_file():
            raise FileNotFoundError(
                f"白化统计不存在: {ws}（传 --whiten-stats none 可写入原始 CLS）")
        st = torch.load(ws, map_location="cpu", weights_only=False)
        whiten_stats = {"mean": st["mean"].float(), "W": st["W"].float()}
        print(f"白化统计: {ws}  mean{tuple(whiten_stats['mean'].shape)} "
              f"W{tuple(whiten_stats['W'].shape)}")

    print(f"加载 LucaVirus: {a.model_path}")
    model = AutoModel.from_pretrained(a.model_path, trust_remote_code=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    # tokenizer 不再参与编码（gene 映射由 tokenize_gene_ids 显式完成），
    # 此处仅为兼容函数签名而传入 None。
    encode_fasta_to_cache(
        fasta_path=a.fasta, cache_dir=a.cache_dir, tokenizer=None, model=model,
        device=device, window_size=a.window, stride=a.stride,
        encode_batch=a.batch, packed_path=a.packed_path,
        whiten_stats=whiten_stats)


if __name__ == "__main__":
    _main()
