"""Seeded label-Dirichlet allocation with an explicit per-class minimum."""
import numpy as np
import pandas as pd


def partition_dirichlet(data, number_of_clients=10, alpha=0.5, seed=42, min_per_class=2):
    if number_of_clients < 2 or alpha <= 0 or min_per_class < 0:
        raise ValueError("Require N>=2, alpha>0 and min_per_class>=0")
    if data.TARGET.isna().any() or not set(data.TARGET.unique()) <= {0, 1}:
        raise ValueError("TARGET must be binary and non-null")
    if not data.SK_ID_CURR.is_unique or data.SK_ID_CURR.isna().any():
        raise ValueError("Customer IDs must be unique and non-null")
    rng = np.random.default_rng(seed)
    assignment = np.full(len(data), -1, dtype=int)
    for label in (0, 1):
        rows = np.flatnonzero(data.TARGET.to_numpy() == label)
        if len(rows) < number_of_clients * min_per_class:
            raise ValueError(f"Class {label} has too few rows for the requested minimum")
        rng.shuffle(rows)
        counts = rng.multinomial(len(rows) - number_of_clients * min_per_class,
                                 rng.dirichlet(np.full(number_of_clients, alpha))) + min_per_class
        start = 0
        for client, count in enumerate(counts):
            assignment[rows[start:start + count]] = client
            start += count
    clients = {i: data.iloc[np.flatnonzero(assignment == i)].copy() for i in range(number_of_clients)}
    assignments = pd.DataFrame({"SK_ID_CURR": data.SK_ID_CURR.to_numpy(), "client_id": assignment})
    return clients, assignments


def local_split(customers, test_size=0.2, seed=42):
    """Stratify within each client, rounding per class; singletons remain train."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    test = np.zeros(len(customers), dtype=bool)
    for label in (0, 1):
        rows = np.flatnonzero(customers.TARGET.to_numpy() == label)
        rng.shuffle(rows)
        count = min(len(rows) - 1, max(1, int(round(len(rows) * test_size)))) if len(rows) >= 2 else 0
        test[rows[:count]] = True
    return pd.DataFrame({"SK_ID_CURR": customers.SK_ID_CURR.to_numpy(),
                         "train_mask": ~test, "test_mask": test})
