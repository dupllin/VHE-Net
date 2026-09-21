"""Data integrity tests - no GPU or weights required."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data/interactions.xlsx"

DSR10 = 0.0909114865
SEED = 42


@pytest.fixture(scope="module")
def interactions():
    df = pd.read_excel(XLSX)
    df[["Virus", "Host"]] = df["V-H"].str.split("__", expand=True)
    return df


def test_total_rows(interactions):
    assert len(interactions) == 421234


def test_label_counts(interactions):
    assert int(interactions.Label.sum()) == 3795
    assert int((interactions.Label == 0).sum()) == 417439


def test_grid_dimensions(interactions):
    assert interactions.Virus.nunique() == 934
    assert interactions.Host.nunique() == 451


def test_down_sample_count(interactions):
    n_neg = int((interactions.Label == 0).sum() * DSR10)
    assert n_neg == 37950


def test_strict_1to10_subset(interactions):
    n_neg = int((interactions.Label == 0).sum() * DSR10)
    sel = interactions[interactions.Label == 0].sample(n=n_neg, random_state=SEED)
    sub = pd.concat([interactions[interactions.Label == 1], sel]).reset_index(drop=True)
    assert len(sub) == 41745
    assert int(sub.Label.sum()) == 3795


def test_split_sizes(interactions):
    from sklearn.model_selection import ShuffleSplit
    n_neg = int((interactions.Label == 0).sum() * DSR10)
    sel = interactions[interactions.Label == 0].sample(n=n_neg, random_state=SEED)
    sub = pd.concat([interactions[interactions.Label == 1], sel]).reset_index(drop=True)
    tri, vai = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(sub))
    assert len(tri) == 33396
    assert len(vai) == 8349
    assert int(sub.iloc[tri].Label.sum()) == 2998
    assert int(sub.iloc[vai].Label.sum()) == 797


def test_importance_score_is_label_derived(interactions):
    """importance score is a deterministic function of the label - documented caveat."""
    pos = interactions.loc[interactions.Label == 1, "importance score"]
    assert pos.nunique() == 1
    assert float(pos.iloc[0]) == pytest.approx(20.0)
    # threshold rule reproduces the label perfectly
    pred = (interactions["importance score"] >= 20).astype(int)
    assert (pred == interactions.Label).mean() == pytest.approx(1.0)


def test_host_cite_count_present(interactions):
    assert interactions.Host_cite_count.min() >= 1
    assert interactions.Host_cite_count.max() > 1000
