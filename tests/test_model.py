"""Architecture tests - requires the package only, no weights, no GPU."""
import torch
import pytest

from vhenet.model import SelfAttnPoolMoE, sparsemax
from vhenet.losses import per_virus_softmax_loss, weighted_bce


def test_sparsemax_sums_to_one():
    z = torch.randn(4, 7)
    a = sparsemax(z, dim=-1)
    assert torch.allclose(a.sum(-1), torch.ones(4), atol=1e-5)
    assert (a >= 0).all()


def test_sparsemax_is_sparse():
    """sparsemax should assign exact zeros to low-scoring entries."""
    z = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
    a = sparsemax(z, dim=-1)
    assert a[0, 0] > 0.9
    assert int((a == 0).sum()) >= 1


def test_weighted_bce_clip_is_a_weight_ceiling():
    """`clip` clamps SAMPLE WEIGHTS, not logits."""
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    w_small = torch.tensor([1.0, 1.0])
    w_huge = torch.tensor([1e6, 1e6])
    l1 = weighted_bce(logits, labels, w_small, clip=100.0)
    l2 = weighted_bce(logits, labels, w_huge, clip=100.0)
    # after normalisation both collapse to the same value
    assert torch.allclose(l1, l2, atol=1e-5)


def test_weighted_bce_zero_weight_is_safe():
    logits = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    w = torch.zeros(2)
    out = weighted_bce(logits, labels, w, clip=100.0)
    assert torch.isfinite(out)


def test_per_virus_softmax_loss_finite():
    logits = torch.randn(10)
    labels = torch.zeros(10)
    labels[0] = 1.0
    w = torch.ones(10)
    out = per_virus_softmax_loss(logits, labels, w)
    assert torch.isfinite(out)


def test_moe_forward_shapes():
    moe = SelfAttnPoolMoE(d=32, n_heads=2, dropout=0.0, attn_type="softmax")
    win = torch.randn(5, 32)
    out, attn = moe(win)
    assert out.shape == (32,)
    assert attn.shape == (2, 5)
    assert torch.allclose(attn.sum(-1), torch.ones(2), atol=1e-5)


def test_moe_balance_loss_range():
    moe = SelfAttnPoolMoE(d=16, n_heads=2, dropout=0.0, attn_type="softmax")
    win = torch.randn(4, 16)
    _, attn = moe(win)
    b = moe.balance_loss(attn)
    assert 0.0 <= float(b) <= 1.0


def test_backbone_is_smaller_than_1_1b():
    """Guard against the '3B' mislabel: the checkpoint is a 0.95 B model."""
    import json, struct
    from pathlib import Path
    sf = Path(__file__).resolve().parent.parent / "pretrained/lucaVirus/model.safetensors"
    if not sf.exists():
        pytest.skip("backbone weights not downloaded")
    with open(sf, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode())
    total = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        numel = 1
        for s in v["shape"]:
            numel *= s
        total += numel
    assert 9.0e8 < total < 1.1e9, f"unexpected backbone size: {total}"
