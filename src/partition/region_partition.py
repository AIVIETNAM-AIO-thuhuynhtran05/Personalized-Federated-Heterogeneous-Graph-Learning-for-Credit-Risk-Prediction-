"""Chia theo địa bàn: mỗi vùng thuộc đúng 1 tổ chức, quy mô tổ chức ~ Dirichlet(alpha).

Quy tắc sở hữu:
  R1  mỗi khách hàng thuộc đúng 1 client
  R2  khách hàng thuộc tổ chức phụ trách vùng nơi họ sinh sống (region_col)
  R3  mỗi vùng do đúng 1 tổ chức phụ trách (gán nguyên vùng, không cắt vùng)
Dirichlet chỉ quyết định QUY MÔ thị trường của tổ chức; lệch phân phối đặc trưng/nhãn
xuất hiện tự nhiên từ đặc điểm các vùng mà tổ chức phụ trách (không dùng TARGET khi chia).
"""
import numpy as np
import pandas as pd


def region_territory_partition(regions: pd.Series, num_clients: int, alpha: float,
                               rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    sizes = regions.value_counts()
    q = rng.dirichlet(alpha * np.ones(num_clients))       # quy mô mục tiêu của từng tổ chức
    target = q * len(regions)
    load = np.zeros(num_clients)
    region_owner = {}
    # Duyệt vùng theo thứ tự ngẫu nhiên, gán cho tổ chức còn thiếu nhiều nhất so với quy mô mục tiêu
    for r in rng.permutation(sizes.index.values):
        k = int(np.argmax(target - load))
        region_owner[r] = k
        load[k] += sizes[r]
    client = regions.map(region_owner).values.astype(np.int64)
    return client, {"target_share": q.round(4).tolist(),
                    "region_owner": {str(r): int(k) for r, k in region_owner.items()}}
