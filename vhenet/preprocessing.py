"""
数据预处理模块 - 处理FASTA序列、交互数据和宿主相似性矩阵
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, List
import itertools
from pathlib import Path
import sys
import warnings
# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class KmerVocab:
    """K-mer 词表管理"""

    def __init__(self, k: int = 3):
        self.k = k
        self.bases = ['A', 'C', 'G', 'T']
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """构建K-mer词表，0预留给padding"""
        kmers = []
        for item in itertools.product(self.bases, repeat=self.k):
            kmers.append("".join(item))
        vocab = {kmer: i + 1 for i, kmer in enumerate(kmers)}  # 0 for padding
        return vocab

    def __len__(self):
        return len(self.vocab) + 1  # +1 for padding token

    def __getitem__(self, kmer: str) -> int:
        return self.vocab.get(kmer, 0)

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.copy()


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """
    解析FASTA文件，返回 {virus_name: sequence} 字典

    Args:
        fasta_file: FASTA文件路径

    Returns:
        {virus_name: sequence} 字典
    """
    seq_dict = {}
    try:
        from Bio import SeqIO

        for record in SeqIO.parse(fasta_file, "fasta"):
            # 提取virus名称（通常是header的第一部分）
            if 'Accessions:' in record.description:
                # 有Accessions:时，按上述逻辑提取
                virus_name = record.description.split('Accessions:')[0].strip()
            else:
                # 无Accessions:时，沿用原有的|分割逻辑（兼容旧数据）
                virus_name = record.description.split('|')[0].strip()
            seq = str(record.seq).upper()
            seq_dict[virus_name] = seq
    except FileNotFoundError:
        print(f"Error :  FASTA file not found at {fasta_file}")
        sys.exit(1)
    except Exception:
        name = None
        parts = []
        try:
            with open(fasta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('>'):
                        if name is not None:
                            seq_dict[name] = ''.join(parts).upper()
                        desc = line[1:]
                        if 'Accessions:' in desc:
                            name = desc.split('Accessions:')[0].strip()
                        else:
                            name = desc.split('|')[0].strip()
                        parts = []
                    else:
                        parts.append(line)
                if name is not None:
                    seq_dict[name] = ''.join(parts).upper()
        except FileNotFoundError:
            print(f"Error :  FASTA file not found at {fasta_file}")
            sys.exit(1)
    return seq_dict


def seq_to_ids(seq: str, vocab: KmerVocab, max_len: int = 100) -> torch.Tensor:
    """
    将序列转换为K-mer ID序列

    Args:
        seq: DNA序列字符串
        vocab: KmerVocab实例
        max_len: 序列最大长度

    Returns:
        ID张量 shape: (max_len,)
    """
    ids = []
    for i in range(len(seq) - vocab.k + 1):
        kmer = seq[i:i + vocab.k]
        ids.append(vocab[kmer])

    # Padding 或 Truncating
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # padding with 0
    else:
        ids = ids[:max_len]

    return torch.tensor(ids, dtype=torch.long)


def seq_to_ids_sliding_window(
        seq: str,
        vocab: KmerVocab,
        max_kmers: int = 512,  # 改为K-mer数量
        window_kmers: int = 256,  # 每个窗口的K-mer数量
        stride_kmers: int = 128,  # 滑动步长（K-mer数）
        mode: str = 'train'
) -> torch.Tensor:
    """
    使用滑动窗口将长序列分割成多个片段（按K-mer数计算）

    Args:
        seq: DNA序列字符串
        vocab: KmerVocab实例
        max_kmers: 每个片段的最大K-mer数量
        window_kmers: 每个窗口的K-mer数量
        stride_kmers: 滑动步长（K-mer数）
        mode: 'train'或'val'，训练时随机采样，验证时固定采样

    Returns:
        片段ID张量 shape: (num_fragments, max_kmers)
    """
    k = vocab.k
    seq_len = len(seq)

    # 计算整个序列的K-mer数量
    total_kmers = max(0, seq_len - k + 1)

    # 分支1：短序列（生成1个片段）
    if total_kmers <= window_kmers:
        # 生成单个片段 [1, max_kmers]
        single_fragment = seq_to_ids_simple(seq, vocab, max_kmers).unsqueeze(0)
        # 关键修复：补全到10个片段（9个全0片段）
        padding = torch.zeros(
            (10 - 1, max_kmers),  # 9行2048列
            dtype=torch.long
        )
        # 拼接后形状：[10, 2048]
        full_fragments = torch.cat([single_fragment, padding], dim=0)
        return full_fragments

    # 生成所有K-mer ID
    all_kmer_ids = []
    for i in range(total_kmers):
        kmer = seq[i:i + k]
        all_kmer_ids.append(vocab[kmer])

    # 滑动窗口生成片段
    fragments = []
    for start in range(0, total_kmers - window_kmers + 1, stride_kmers):
        end = start + window_kmers
        fragment_ids = all_kmer_ids[start:end]

        # 确保每个片段长度一致（填充或截断到max_kmers）
        if len(fragment_ids) < max_kmers:
            fragment_ids = fragment_ids + [0] * (max_kmers - len(fragment_ids))
        else:
            fragment_ids = fragment_ids[:max_kmers]

        fragments.append(torch.tensor(fragment_ids, dtype=torch.long))

        # # 限制最大片段数，避免内存爆炸
        if len(fragments) >= 4:  # 最多10个片段
            break

    if not fragments:
        # 如果没生成任何片段，用全序列
        return seq_to_ids_simple(seq, vocab, max_kmers).unsqueeze(0)

    num_fragments = len(fragments)
    if num_fragments < 10:
        # 生成全0的补全张量（形状：(不足的数量, max_kmers)，dtype和原片段一致）
        padding_fragments = torch.zeros(
            (10 - num_fragments, max_kmers),
            dtype=torch.long  # 和原片段保持相同的类型（Long）
        )
        # 将补全张量添加到片段列表
        fragments = fragments + [padding_fragments[i] for i in range(padding_fragments.shape[0])]

    return torch.stack(fragments, dim=0)


def seq_to_ids_simple(seq: str, vocab: KmerVocab, max_kmers: int = 1000) -> torch.Tensor:
    """
    基础的序列转ID函数（单个片段）
    返回固定长度的K-mer ID序列

    Args:
        seq: DNA序列字符串
        vocab: KmerVocab实例
        max_kmers: 最大K-mer数量

    Returns:
        K-mer ID张量 shape: (max_kmers,)
    """
    k = vocab.k
    seq_len = len(seq)

    # 提取所有K-mer
    ids = []
    for i in range(max(0, seq_len - k + 1)):
        kmer = seq[i:i + k]
        ids.append(vocab[kmer])

    # 截断或填充到固定长度
    if len(ids) < max_kmers:
        ids += [0] * (max_kmers - len(ids))
    else:
        ids = ids[:max_kmers]

    return torch.tensor(ids, dtype=torch.long)


def process_interaction_data(
        interaction_csv: str,
        similarity_csv: str,
        host_col: str = "V-H"
) -> Tuple[pd.DataFrame, Dict[str, int], torch.Tensor]:
    """
    处理交互CSV和相似性矩阵

    Args:
        interaction_csv: V-H交互数据CSV路径
        similarity_csv: 宿主相似性矩阵CSV路径
        host_col: 包含"Virus__Host"的列名

    Returns:
        (交互数据DataFrame, 宿主ID映射, 相似性矩阵Tensor)
    """
    # 加载交互数据
    interaction_path = Path(interaction_csv)
    suffix = interaction_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(interaction_path)
    elif suffix in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(interaction_path)
        except ValueError as exc:
            # Pandas raises this when a text/CSV file has an Excel extension.
            with interaction_path.open("rb") as file_obj:
                header = file_obj.read(256)
            if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(
                    f"{interaction_path} is a Git LFS pointer, not an Excel workbook. "
                    "Run 'git lfs pull' to download the data file."
                ) from exc
            if b"\x00" in header:
                raise ValueError(
                    f"{interaction_path} is not a readable Excel workbook. "
                    "Replace the file or point data.interaction_csv to the CSV copy."
                ) from exc
            warnings.warn(
                f"{interaction_path} has an Excel extension but contains text; "
                "loading it as CSV.",
                RuntimeWarning,
                stacklevel=2,
            )
            df = pd.read_csv(interaction_path)
    else:
        raise ValueError(
            f"Unsupported interaction data format '{suffix or '<none>'}' for "
            f"{interaction_path}; expected .csv, .xlsx, or .xls."
        )

    # 解析 V-H Label 获取 Virus 和 Host 名称
    if host_col not in df.columns:
        raise KeyError(
            f"Required column '{host_col}' is missing from {interaction_path}. "
            f"Available columns: {list(df.columns)}"
        )
    df[['Virus', 'Host']] = df[host_col].str.split('__', expand=True)

    # 获取唯一宿主列表
    unique_hosts = sorted(df['Host'].unique())
    host_to_id = {h: i for i, h in enumerate(unique_hosts)}

    # 加载相似性矩阵
    sim_df = pd.read_csv(similarity_csv, index_col=0)

    # 根据host_to_id重新排序相似性矩阵
    ordered_hosts = sorted(host_to_id.keys())
    sim_matrix = sim_df.loc[ordered_hosts, ordered_hosts].values
    sim_matrix = torch.tensor(sim_matrix, dtype=torch.float32)

    return df, host_to_id, sim_matrix


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    import yaml

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        current_dir = Path("")
        print(f"Current directory: {current_dir.absolute()}")
        print(f"⚠️  Config file not found at {config_path}, using defaults")
        return get_default_config()


def get_default_config() -> Dict:
    """返回默认配置"""
    return {
        'embed_dim': 32,
        'seq_max_len': 60,
        'kmer': 3,
        'batch_size': 16,
        'epochs': 50,
        'learning_rate': 1e-3,
        'device': 'cuda',
        'train_test_split': 0.8,
    }


# ============================================
# 数据统计和验证函数
# ============================================

def validate_data(df: pd.DataFrame, seq_dict: Dict, host_to_id: Dict) -> Dict:
    """
    验证数据完整性

    Args:
        df: 交互数据
        seq_dict: 病毒序列字典
        host_to_id: 宿主ID映射

    Returns:
        验证报告字典
    """
    report = {
        'num_interactions': len(df),
        'num_viruses': df['Virus'].nunique(),
        'num_hosts': df['Host'].nunique(),
        'num_sequences': len(seq_dict),
        'missing_sequences': 0,
        'missing_hosts': 0,
        'pos_samples': (df['Label'] == 1).sum(),
        'neg_samples': (df['Label'] == 0).sum(),
    }

    # 检查缺失序列
    for virus in df['Virus'].unique():
        if virus not in seq_dict:
            report['missing_sequences'] += 1

    # 检查缺失宿主
    for host in df['Host'].unique():
        if host not in host_to_id:
            report['missing_hosts'] += 1

    return report


def print_data_stats(report: Dict):
    """打印数据统计信息"""
    print("\n" + "=" * 50)
    print("📊 数据统计信息")
    print("=" * 50)
    print(f"交互对数:        {report['num_interactions']}")
    print(f"病毒种类:       {report['num_viruses']}")
    print(f"宿主种类:       {report['num_hosts']}")
    print(f"序列数:          {report['num_sequences']}")
    print(f"缺失序列:        {report['missing_sequences']}")
    print(f"缺失宿主:       {report['missing_hosts']}")
    print(f"正样本数:       {report['pos_samples']}")
    print(f"负样本数:       {report['neg_samples']}")
    print(
        f"正负比例:       {report['pos_samples'] / report['neg_samples']:.3f}" if report['neg_samples'] > 0 else "N/A")
    print("=" * 50 + "\n")
