"""Chia khách hàng vào K client theo Dirichlet trên nhãn (label-skew + quantity-skew).

Với mỗi lớp c: p_c ~ Dir(alpha * 1_K), các mẫu của lớp c được chia cho client theo tỷ lệ p_c.
alpha càng nhỏ -> càng non-IID (số mẫu và tỷ lệ default giữa các client càng chênh lệch).
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def dirichlet_label_partition(labels: np.ndarray, num_clients: int, alpha: float,
                              rng: np.random.Generator) -> np.ndarray:
    client = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        p = rng.dirichlet(alpha * np.ones(num_clients))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            client[part] = k
    return client


def _satisfies(df: pd.DataFrame, num_clients: int, cons: dict) -> bool:
    """Mỗi client phải đủ lớn và có đủ mẫu dương/âm ở cả Train/Val/Test để train và đánh giá."""
    share = df["client"].value_counts(normalize=True).reindex(range(num_clients), fill_value=0)
    if (share < cons["min_client_share"]).any():
        return False
    counts = df.groupby(["client", "split", "TARGET"]).size()
    full = pd.MultiIndex.from_product([range(num_clients), ["train", "val", "test"], [0, 1]])
    return bool((counts.reindex(full, fill_value=0) >= cons["min_samples_per_class_per_split"]).all())


def partition_clients(splits: pd.DataFrame, pcfg: dict, regions: pd.Series | None = None
                      ) -> tuple[pd.DataFrame, dict]:
    """`splits` gồm SK_ID_CURR, TARGET, split (split chung). Trả về thêm cột `client`.
    Split Train/Val/Test được giữ nguyên -> hợp các Test của client = Test của nhánh Centralized.
    `regions` (cùng thứ tự với splits) bắt buộc khi method = region_territory."""
    from src.partition.region_partition import region_territory_partition

    rng = np.random.default_rng(pcfg["seed"])
    cons = pcfg["constraints"]
    out = splits[["SK_ID_CURR", "TARGET", "split"]].copy()
    for attempt in range(1, cons["max_tries"] + 1):
        info = {}
        if pcfg["method"] == "dirichlet_label":
            out["client"] = dirichlet_label_partition(out["TARGET"].values, pcfg["num_clients"],
                                                      pcfg["alpha"], rng)
        elif pcfg["method"] == "region_territory":
            out["client"], info = region_territory_partition(regions.reset_index(drop=True),
                                                             pcfg["num_clients"], pcfg["alpha"], rng)
        else:
            raise ValueError(f"Unknown partition method: {pcfg['method']}")
        if _satisfies(out, pcfg["num_clients"], cons):
            log.info("Partition accepted at attempt %d", attempt)
            return out, {"attempts": attempt, **info}
    raise RuntimeError("Không tìm được partition thỏa ràng buộc; nới constraints hoặc tăng alpha")
