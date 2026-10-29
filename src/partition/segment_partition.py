"""Chia theo cơ cấu danh mục: mọi tổ chức phục vụ mọi phân khúc khách hàng nhưng có thế mạnh riêng.

Với mỗi phân khúc s: tỷ lệ khách hàng của s giao cho các tổ chức
    p_s = (1 - floor) * Dirichlet(alpha * 1_K) + floor / K
floor > 0 bảo đảm không tổ chức nào bỏ hẳn một phân khúc. alpha nhỏ -> tổ chức chuyên môn hóa mạnh.
Không dùng TARGET khi chia.
"""
import numpy as np
import pandas as pd


def segment_mix_partition(segments: pd.Series, num_clients: int, alpha: float, floor: float,
                          rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    segments = segments.reset_index(drop=True)
    client = np.empty(len(segments), dtype=np.int64)
    shares = {}
    for s in sorted(segments.unique()):
        idx = rng.permutation(np.flatnonzero(segments.values == s))
        p = (1 - floor) * rng.dirichlet(alpha * np.ones(num_clients)) + floor / num_clients
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for k, part in enumerate(np.split(idx, cuts)):
            client[part] = k
        shares[str(s)] = p.round(4).tolist()
    return client, {"segment_to_client_share": shares}
