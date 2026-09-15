import pandas as pd
import pytest

from src.split.global_split import create_global_split


def _customers():
    return pd.DataFrame({
        "SK_ID_CURR": range(1000, 1200),
        "TARGET": [0] * 160 + [1] * 40,
    })


def test_global_split_is_reproducible_disjoint_and_exhaustive():
    first = create_global_split(_customers(), seed=42)
    second = create_global_split(_customers(), seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 200
    assert first.SK_ID_CURR.is_unique
    assert (first[["train_mask", "val_mask", "test_mask"]].sum(axis=1) == 1).all()
    assert first.split.value_counts().to_dict() == {
        "train": 140, "validation": 30, "test": 30
    }
    for name in ("train", "validation", "test"):
        assert first.loc[first.split == name, "TARGET"].mean() == pytest.approx(0.20)


def test_invalid_ratios_are_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        create_global_split(_customers(), 0.8, 0.15, 0.15)
