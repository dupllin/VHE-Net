"""v6 模型：LucaVirus-LoRA 微调分支 + k-mer 位置分箱分支 混合。

背景（回答"为什么冻结的 LucaVirus embedding 都差不多"）：
  LucaVirus 是在全病毒基因组上用 masked-LM 预训练的，模型本身没问题；
  问题在于我们此前是【冻结编码器 + 只训小头】。冻结的 CLS 是在"预测被遮住
  的碱基"这个目标下学出来的，对"预测宿主"这个下游任务未必有区分度
  （实测两个不同病毒的冻结 CLS cosine=0.9995）。标准做法（BERT 系）就是
  【微调/适配】编码器本身，LoRA 只训低秩增量、不破坏预训练权重。

本文件：给 12 层 LucaVirus 的 q/k/v/out 投影注入 LoRA(r=8, alpha=16)，
每病毒每 epoch 采样 K 个 512-token 窗口前向（控制显存与时间），
取 [CLS; mean-pool] 投影后与 k-mer 分支融合；全窗口预测时用全部窗口。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vhenet.layers import mlp


class LoRALinear(nn.Module):
    """低秩适配线性层：冻结原权重，只训 A/B 两个低秩矩阵。"""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(base.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, base.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scale


def inject_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1,
                target_names=("q_proj", "k_proj", "v_proj", "out_proj")):
    """把注意力层的 Linear 替换为 LoRALinear，并冻结所有基础参数。"""
    replaced = 0
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] in target_names and isinstance(module, nn.Linear):
            parent = model
            parts = name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r, alpha, dropout))
            replaced += 1
    for p in model.parameters():
        p.requires_grad = False
    # 重新放开 LoRA 参数
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True
    return replaced


class ViHGAT_814v6(nn.Module):
    def __init__(
        self,
        num_hosts: int,
        host_sim_matrix,
        llm_model_path: str,
        kmer_bins: int = 4096,
        hist_mean=None,
        hist_std=None,
        embed_dim: int = 256,
        dropout: float = 0.25,
        n_pos_bins: int = 8,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        llm_window_tokens: int = 512,
        use_host_head: bool = True,
        main_head_inputs: str = "virus_inter",
        use_km_branch: bool = True,
        b_agg_bins: int = 0,
        token_attn_pool: bool = False,
        use_cross_attn: bool = False,
        cross_heads: int = 4,
        whiten_stats=None,
        llm_chunk: int = 16,
        cls_only: bool = False,
        freeze_lora: bool = False,
        cls_no_proj: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_host_head = use_host_head
        self.main_head_inputs = main_head_inputs
        self.kmer_bins = kmer_bins
        self.n_pos_bins = n_pos_bins
        self.llm_window_tokens = llm_window_tokens
        self.use_km_branch = use_km_branch
        self.b_agg_bins = b_agg_bins
        self.token_attn_pool = token_attn_pool
        self.use_cross_attn = use_cross_attn
        self.llm_chunk = llm_chunk
        self.cls_only = cls_only
        self.freeze_lora = freeze_lora
        self.cls_no_proj = cls_no_proj
        if whiten_stats is not None:
            self.register_buffer("whiten_mean",
                                 torch.as_tensor(whiten_stats["mean"],
                                                 dtype=torch.float32))
            self.register_buffer("whiten_W",
                                 torch.as_tensor(whiten_stats["W"],
                                                 dtype=torch.float32))
            print(f"白化已启用: [256] x ({whiten_stats['W'].shape[0]},"
                  f"{whiten_stats['W'].shape[1]})（去除跨病毒共线主分量）")
        else:
            self.register_buffer("whiten_mean", None)
            self.register_buffer("whiten_W", None)
        if not use_km_branch:
            print("use_km_branch=false: B 单独模式（纯 LucaVirus-LoRA 分支）")
            if b_agg_bins > 0:
                print(f"  B 聚合：{b_agg_bins} 个位置分箱 mean + 全局 mean/max"
                      f"（替代全局均值池化）")

        # ---------- LucaVirus + LoRA ----------
        from transformers import AutoModel
        base = AutoModel.from_pretrained(llm_model_path, trust_remote_code=True)
        n_replaced = inject_lora(base, lora_r, lora_alpha, lora_dropout)
        print(f"LoRA 注入: {n_replaced} 个线性层 "
              f"(r={lora_r}, alpha={lora_alpha})")
        self.llm = base
        self.llm_hidden = 2560
        # 窗口特征维：cls_no_proj 时=白化 CLS 原始维（保留白化区分度，避免投影降维磨平）
        self.win_dim = self.llm_hidden if (self.cls_only and self.cls_no_proj) \
            else embed_dim
        if self.win_dim != embed_dim:
            print(f"  cls_no_proj=true: 窗口特征维 {self.win_dim}（不做 llm_proj 降维）")
        if self.token_attn_pool:
            # 窗口内 token 级注意力池化：可学习 query 挑出有区分度的 token
            # （逐层探针显示 token 特征比 CLS/mean 池化后更分化，池化抹掉了信号）
            self.tok_q = nn.Parameter(torch.randn(self.llm_hidden) * 0.02)
            self.tok_scale = float(self.llm_hidden) ** -0.5
        self.llm_proj = nn.Sequential(
            nn.Linear(self.llm_hidden * (1 if cls_only else 2), embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---------- k-mer 分支（与 v5 相同；B 单独模式不创建）----------
        if hist_mean is not None and hist_std is not None:
            self.register_buffer("hist_mean",
                                 torch.as_tensor(hist_mean, dtype=torch.float32))
            self.register_buffer("hist_std",
                                 torch.as_tensor(hist_std, dtype=torch.float32))
            self.use_zscore = True
        else:
            self.use_zscore = False
        if use_km_branch:
            self.bin_projector = nn.Sequential(
                nn.Linear(kmer_bins, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.km_repr_agg = mlp(
                [(n_pos_bins + 2) * embed_dim, 512, embed_dim], dropout)
            self.km_out = nn.Linear(embed_dim, embed_dim)

        # ---------- 融合（带 LoRA 门控，初始 ≈ 0；B 单独模式无门控，纯 LoRA 直出）----------
        # v34A: cls_no_proj 时 win_dim=2560，纯 mean 聚合 fallthrough 需要 2560->embed
        self.llm_out = nn.Linear(self.win_dim, embed_dim)
        if use_km_branch:
            # 融合门控（初始 tanh(0)=0；B 单独模式无此参数，纯 LoRA 直出）
            self.llm_gate = nn.Parameter(torch.tensor(0.0))  # tanh(0)=0（修复：-3 时 tanh≈-1）
        elif b_agg_bins > 0:
            self.b_agg = mlp(
                [(b_agg_bins + 2) * self.win_dim, 512, embed_dim], dropout)

        # ---------- 宿主分支 / 交互 / 分类器 ----------
        self.host_embedding = nn.Embedding(num_hosts, embed_dim)
        self.host_mlp = mlp([embed_dim, embed_dim * 2, embed_dim], dropout)
        self.sim_projector = mlp([num_hosts, embed_dim], dropout)
        self.host_fusion = mlp([embed_dim * 2, embed_dim], dropout)

        self.virus_interact = nn.Linear(embed_dim, embed_dim)
        self.host_interact = nn.Linear(embed_dim, embed_dim)
        if use_cross_attn:
            # 原框架 cross-attention（host query 去 attend 病毒窗口特征）
            # host_rep[..., D] 作用 query；win_proj[..., K, D] 作 key/value
            self.cross_heads = cross_heads
            self.key_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.value_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim),
                nn.GELU(), nn.Dropout(dropout))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, cross_heads, dropout=dropout, batch_first=True)
            self.attn_residual = mlp([embed_dim * 2, embed_dim], dropout)
            self.ln_cross = nn.LayerNorm(embed_dim)
        in_dim = {"full": embed_dim * 4,
                  "virus_inter": embed_dim * 2,
                  "virus_only": embed_dim}[main_head_inputs]
        self.classifier = mlp([in_dim, 256, 64, 1], dropout)

        if use_host_head:
            self.host_head = mlp([embed_dim, 128, 1], dropout)

        if not isinstance(host_sim_matrix, torch.Tensor):
            host_sim_matrix = torch.tensor(host_sim_matrix, dtype=torch.float32)
        if tuple(host_sim_matrix.shape) != (num_hosts, num_hosts):
            raise ValueError(
                f"host_sim_matrix shape {tuple(host_sim_matrix.shape)} "
                f"!= ({num_hosts}, {num_hosts})"
            )
        self.register_buffer("sim_matrix", host_sim_matrix.float())

        self._init_heads()
        if self.freeze_lora:
            # 冻结 LoRA：完全保持预训练 LucaVirus（保 CLS 0.81 的预训练区分度）
            n_f = 0
            for name, param in self.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    param.requires_grad = False
                    n_f += 1
            print(f"freeze_lora=true: 冻结 {n_f} 个 LoRA 参数张量，"
                  f"只训聚合头/分类器")

    def _init_heads(self):
        nn.init.normal_(self.host_embedding.weight, mean=0.0, std=0.02)
        # 热启动：km_out 初始为单位映射（保持 v9 基线行为），llm 门控从 0 开始
        if self.use_km_branch:
            nn.init.eye_(self.km_out.weight)
            nn.init.zeros_(self.km_out.bias)
        # ⚠️ 关键修复：跳过 self.llm 下的一切模块——预训练编码器权重绝对不能动
        # （此前版本把 LucaVirus 的 FFN fc1/fc2 和 LayerNorm 也随机重初始化了，
        #  导致 LLM 分支输出与输入无关、LoRA 永远学不动）
        km_out = getattr(self, "km_out", None)
        for name, module in self.named_modules():
            if name.startswith("llm."):
                continue
            if isinstance(module, nn.Linear) and module is not km_out:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _km_repr(self, hist, pos_frac):
        h = torch.log1p(hist)
        if self.use_zscore:
            h = (h - self.hist_mean) / self.hist_std
            h = h.clamp(-4.0, 4.0)
        feats = [self.bin_projector(h.mean(dim=0)),
                 self.bin_projector(h.amax(dim=0))]
        if pos_frac is not None and self.n_pos_bins > 0:
            bin_idx = (pos_frac * self.n_pos_bins).long().clamp(
                0, self.n_pos_bins - 1)
            one_hot = F.one_hot(bin_idx, num_classes=self.n_pos_bins).float()
            denom = one_hot.sum(dim=0).clamp_min(1.0)
            bin_mean = (one_hot.T @ h) / denom.unsqueeze(-1)
            for b in range(self.n_pos_bins):
                feats.append(self.bin_projector(bin_mean[b]))
        else:
            feats.extend([self.bin_projector(h.mean(dim=0))] * self.n_pos_bins)
        return self.km_repr_agg(torch.cat(feats, dim=0))

    def encode_llm_windows(self, win_ids, win_mask, chunk: int = 16):
        """win_ids: [K, L] -> [K, D]。

        token_attn_pool=False: CLS+mean 投影（原行为）。
        token_attn_pool=True : CLS + 可学习 token 注意力加权和（保住 token 级
                               分化信号，逐层探针：window 内 token cosine 0.82，
                               池化后 0.999）。
        """
        if win_ids.shape[1] > self.llm_window_tokens:
            win_ids = win_ids[:, :self.llm_window_tokens]
            win_mask = win_mask[:, :self.llm_window_tokens]
        outs = []
        for i in range(0, win_ids.shape[0], chunk):
            ids = win_ids[i:i + chunk]
            mk = win_mask[i:i + chunk]
            out = self.llm(input_ids=ids, attention_mask=mk)
            hidden = out.last_hidden_state
            cls = hidden[:, 0, :]
            if self.cls_only:
                # 只用 CLS（预训练 CLS 对 209 有 0.98 共线，需白化拉开）
                c = cls
                if (self.whiten_W is not None
                        and self.whiten_W.shape[0] == self.llm_hidden):
                    # 2560 维白化：作用在原始 CLS 上（投影前），拉开 209 共线
                    c = (c - self.whiten_mean) @ self.whiten_W
                if self.cls_no_proj:
                    # 不做 llm_proj 降维：白化 CLS 直接作为窗口特征
                    # （保留白化后的区分度，投影是磨平共线的主犯）
                    outs.append(c)
                else:
                    outs.append(self.llm_proj(c))
                continue
            if self.token_attn_pool:
                score = torch.einsum("blh,h->bl", hidden, self.tok_q)
                score = score * self.tok_scale
                score = score.masked_fill(~mk, float("-inf"))
                attn = F.softmax(score, dim=1)          # [b, L]
                pooled = (attn.unsqueeze(-1) * hidden).sum(dim=1)
            else:
                mm = mk.to(hidden.dtype).unsqueeze(-1)
                pooled = (hidden * mm).sum(dim=1) / mm.sum(dim=1).clamp_min(1.0)
            outs.append(self.llm_proj(torch.cat([cls, pooled], dim=-1)))
        x = torch.cat(outs, dim=0)      # [K, D]
        if (self.whiten_W is not None
                and self.whiten_W.shape[0] != self.llm_hidden):
            # 256 维白化：作用在投影后输出（旧行为）
            x = (x - self.whiten_mean) @ self.whiten_W
        return x   # [K, D]

    def encode_llm(self, win_ids, win_mask, chunk: int = 16):
        """win_ids: [K, L] -> rep_llm [D]（K 个窗口的 CLS+mean 投影后平均）。"""
        return self.encode_llm_windows(win_ids, win_mask, chunk).mean(dim=0)

    def encode_virus(self, ids, mask, hist=None, pos_frac=None,
                     llm_ids=None, llm_mask=None, llm_pos_frac=None):
        if self.use_km_branch:
            if llm_ids is not None:
                rep_llm = self.encode_llm(llm_ids, llm_mask,
                                          chunk=self.llm_chunk)
            else:
                rep_llm = self.encode_llm(ids, mask, chunk=self.llm_chunk)
            llm_part = self.llm_out(rep_llm)
            rep_km = self._km_repr(hist, pos_frac)
            return self.km_out(rep_km) + torch.tanh(self.llm_gate) * llm_part
        if self.b_agg_bins > 0:
            # B 单独 + 位置分箱聚合：保留窗口在基因组上的位置结构
            src_ids = llm_ids if llm_ids is not None else ids
            src_mask = llm_mask if llm_mask is not None else mask
            w = self.encode_llm_windows(src_ids, src_mask)   # [K, D]
            pf = llm_pos_frac if llm_pos_frac is not None else pos_frac
            feats = [w.mean(dim=0), w.amax(dim=0)]
            bin_idx = (pf * self.b_agg_bins).long().clamp(
                0, self.b_agg_bins - 1)
            one_hot = F.one_hot(bin_idx, num_classes=self.b_agg_bins).float()
            denom = one_hot.sum(dim=0).clamp_min(1.0)
            bin_mean = (one_hot.T @ w) / denom.unsqueeze(-1)
            feats.extend([bin_mean[b] for b in range(self.b_agg_bins)])
            return self.llm_out(self.b_agg(torch.cat(feats, dim=0)))
        # B 单独：绕过门控 —— gate=0 时 tanh(0)=0 会让 virus_rep=0，
        # 进而 virus_interact(0)=bias=0 -> inter=0 -> 全模型梯度死亡（死起点）
        if llm_ids is not None:
            rep_llm = self.encode_llm(llm_ids, llm_mask, chunk=self.llm_chunk)
        else:
            rep_llm = self.encode_llm(ids, mask, chunk=self.llm_chunk)
        return self.llm_out(rep_llm)

    def host_rep_for(self, host_ids):
        emb = self.host_mlp(self.host_embedding(host_ids))
        sim = self.sim_projector(self.sim_matrix[host_ids])
        return self.host_fusion(torch.cat([emb, sim], dim=-1))

    def forward_windows(self, win_features, host_ids, return_attn=False):
        """v34B cross-attn 路径。

        win_features: [K, 2560]（每窗口白化 CLS, 预计算缓存）
        host_ids:     [B]
        返回 (logits, host_logits[, attn_weights [B,K]])
        """
        win_proj = self.llm_out(win_features)        # [K, 256]
        virus_rep = win_proj.mean(dim=0)             # [256] 全局 mean
        virus_rep_e = virus_rep.unsqueeze(0).expand(host_ids.shape[0], -1)
        host_rep = self.host_rep_for(host_ids)       # [B, 256]
        inter = self.virus_interact(virus_rep_e) * self.host_interact(host_rep)
        if self.main_head_inputs == "full":
            diff = virus_rep_e - host_rep
            feats = torch.cat([virus_rep_e, host_rep, inter, diff], dim=-1)
        elif self.main_head_inputs == "virus_inter":
            feats = torch.cat([virus_rep_e, inter], dim=-1)
        else:
            feats = virus_rep_e
        logits = self.classifier(feats)
        host_logits = None
        if self.use_host_head:
            host_logits = self.host_head(host_rep)

        # cross-attention: host query attend 病毒窗口
        if self.use_cross_attn:
            # key/value: 每窗口投影
            k = self.key_proj(win_proj.unsqueeze(0))         # [1, K, 256]
            v = self.value_proj(win_proj.unsqueeze(0))       # [1, K, 256]
            q = host_rep.unsqueeze(1)                        # [B, 1, 256]
            ctx, attn = self.cross_attn(q, k.expand(host_ids.shape[0], -1, -1),
                                        v.expand(host_ids.shape[0], -1, -1))
            ctx = ctx.squeeze(1)                             # [B, 256]
            fused = self.attn_residual(torch.cat([host_rep, ctx], dim=-1))
            fused = self.ln_cross(fused)                     # [B, 256]
            # 用 cross-attn 融合结果替换/增强 inter
            prod = host_rep * fused
            ca_feats = torch.cat([fused, prod], dim=-1)      # [B, 512]
            ca_logits = self.classifier(ca_feats)
            if return_attn:
                return ca_logits, host_logits, attn.mean(dim=1)  # [B,K]
            return ca_logits, host_logits
        return logits, host_logits

    def forward(self, ids, mask, host_ids, hist=None, pos_frac=None,
                llm_ids=None, llm_mask=None, llm_pos_frac=None):
        if hist is None and self.use_km_branch:
            raise ValueError("v6 模型需要 hist 输入（use_km_branch=True）")
        virus_rep = self.encode_virus(ids, mask, hist, pos_frac, llm_ids,
                                      llm_mask, llm_pos_frac)
        virus_rep = virus_rep.unsqueeze(0).expand(host_ids.shape[0], -1)
        host_rep = self.host_rep_for(host_ids)
        inter = self.virus_interact(virus_rep) * self.host_interact(host_rep)
        if self.main_head_inputs == "full":
            diff = virus_rep - host_rep
            feats = torch.cat([virus_rep, host_rep, inter, diff], dim=-1)
        elif self.main_head_inputs == "virus_inter":
            feats = torch.cat([virus_rep, inter], dim=-1)
        else:
            feats = virus_rep
        logits = self.classifier(feats)
        host_logits = None
        if self.use_host_head:
            host_logits = self.host_head(host_rep)
        return logits, host_logits

    def score_all_hosts(self, ids, mask, all_host_ids, hist=None,
                        pos_frac=None, llm_ids=None, llm_mask=None,
                        llm_pos_frac=None):
        logits, _ = self.forward(ids, mask, all_host_ids, hist, pos_frac,
                                 llm_ids, llm_mask, llm_pos_frac)
        return logits.squeeze(-1)


class SelfAttnPoolMoE(nn.Module):
    """多头 MoE 注意力：N 个 query(router) 各自选窗口。

    attn_type:
      - "sparsemax" : 稀疏路由（默认，非 top 窗口精确 0）
      - "softmax"   : 连续/平滑路由（不加 sparsemax），温度 attn_temp 控制锐度
    """

    def __init__(self, d=256, n_heads=4, dropout=0.25,
                 attn_type="sparsemax", attn_temp=1.0):
        super().__init__()
        self.n_heads = n_heads
        self.d = d
        self.attn_type = attn_type
        self.attn_temp = attn_temp
        self.q_tokens = nn.Parameter(torch.randn(n_heads, 1, d) * 0.02)
        self.key_proj = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.value_proj = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.q_proj = nn.Sequential(
            nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.out_proj = nn.Sequential(
            nn.Linear(n_heads * d, d), nn.LayerNorm(d), nn.GELU(),
            nn.Dropout(dropout))
        self.ln = nn.LayerNorm(d)

    def forward(self, win_proj):
        # win_proj [K, d]
        k = self.key_proj(win_proj)                      # [K,d]
        v = self.value_proj(win_proj)                    # [K,d]
        q = self.q_proj(self.q_tokens).squeeze(1)        # [H,d]
        score = torch.einsum('hd,kd->hk', q, k) / (self.d ** 0.5)   # [H,K]
        if self.attn_type == "softmax":
            attn = F.softmax(score * float(self.attn_temp), dim=-1)  # 连续
        else:
            attn = sparsemax(score, dim=-1)              # 稀疏
        ctx = attn @ v                                   # [H,d]
        flat = ctx.reshape(1, self.n_heads * self.d)     # [1, H*d]
        out = self.out_proj(flat)                        # [1,d]
        out = self.ln(out + win_proj.mean(0, keepdim=True))
        return out.squeeze(0), attn                      # [d], [H,K]

    def balance_loss(self, attn):
        """跨头重叠惩罚：防止 head 重复(选同一窗)/失效，不强迫窗口均匀。
        attn: [H,K] 注意分布。返回标量。"""
        H = attn.shape[0]
        ov = 0.0; n = 0
        for h1 in range(H):
            for h2 in range(h1 + 1, H):
                ov = ov + (attn[h1] * attn[h2]).sum()
                n += 1
        return ov / max(n, 1)


def sparsemax(z, dim=-1):
    z = z - z.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    cs = torch.cumsum(z_sorted, dim=dim) - 1.0
    denom = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    denom_v = denom.view([1] * (z.dim() - 1) + [-1]).expand_as(z)
    cond = (z_sorted - cs / denom_v) > 0
    k_star = cond.sum(dim=dim, keepdim=True).clamp(min=1).to(z.dtype)
    tau_num = cs.gather(dim, (k_star - 1).long())
    tau = tau_num / k_star
    return torch.clamp(z - tau, min=0)
