# Chia Customer trước, chuyển các bảng lịch sử theo SK_ID_CURR sau; giữ schema chung giữa client.
"""Distribute globally prepared tables without selecting features per client."""
from pathlib import Path
import json
import pandas as pd

from src.partition.multidimensional_partition import partition_non_iid, DEFAULT_PARTITION_COLUMNS
from src.partition.dirichlet_partition import partition_dirichlet, local_split

TABLE_NAMES = ("application_train", "bureau", "previous_application",
               "installments_payments", "POS_CASH_balance", "credit_card_balance")


def partition_tables(input_dir: Path, output_dir: Path, number_of_clients: int = 10,
                     partition_columns=DEFAULT_PARTITION_COLUMNS, chunksize: int = 200_000,
                     strategy="dirichlet", alpha=0.5, seed=42, test_size=0.2, min_per_class=2):
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must differ")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    if not (input_dir / "missing_report.json").is_file():
        raise ValueError("Run 01_prepare_tables.py before partitioning")
    schema = {name: pd.read_csv(input_dir / f"{name}.csv", nrows=0).columns.tolist()
              for name in TABLE_NAMES}
    for name, columns in schema.items():
        if "SK_ID_CURR" not in columns:
            raise ValueError(f"{name} is missing SK_ID_CURR")
    customers = pd.read_csv(input_dir / "application_train.csv")
    if customers.SK_ID_CURR.isna().any() or not customers.SK_ID_CURR.is_unique:
        raise ValueError("Customer IDs must be unique and non-null")
    if strategy == "dirichlet":
        clients, assignments = partition_dirichlet(customers, number_of_clients, alpha, seed, min_per_class)
    elif strategy == "semantic":
        clients, assignments = partition_non_iid(customers, tuple(partition_columns), number_of_clients)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # Ánh xạ này là nguồn xác định client cho mọi hàng lịch sử.
    owner = assignments.set_index("SK_ID_CURR")["client_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"num_clients": number_of_clients, "partition_columns": list(partition_columns),
              "schema": schema, "tables": {}, "clients": [], "strategy": strategy,
              "alpha": alpha, "seed": seed, "test_size": test_size,
              "min_per_class": min_per_class, "split_counts": {}}
    for client_id, frame in clients.items():
        directory = output_dir / f"client_{client_id:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "application_train.csv", index=False)
        # Seed riêng theo client; chỉ chia Customer, lịch sử đi theo Customer tương ứng.
        split = local_split(frame, test_size, seed + client_id)
        split.to_csv(directory / "customer_split.csv", index=False)
        report["split_counts"][directory.name] = {
            subset: {str(label): int(((frame.TARGET.to_numpy() == label) & split[subset].to_numpy()).sum())
                     for label in (0, 1)} for subset in ("train_mask", "test_mask")}
        report["clients"].append(directory.name)
        for name in TABLE_NAMES[1:]:
            pd.DataFrame(columns=schema[name]).to_csv(directory / f"{name}.csv", index=False)
    report["tables"]["application_train"] = {"assigned_rows": len(customers), "unassigned_rows": 0}
    for name in TABLE_NAMES[1:]:
        assigned = unassigned = 0
        print(f"Partitioning {name}...", flush=True)
        for chunk in pd.read_csv(input_dir / f"{name}.csv", chunksize=chunksize):
            # ID không có trong population nhận NaN và được đếm unassigned, không đưa vào client.
            ids = chunk.SK_ID_CURR.map(owner)
            unassigned += int(ids.isna().sum())
            assigned += int(ids.notna().sum())
            for client_id, rows in chunk.groupby(ids):
                # Ghi nối chunk sau header đã tạo; không lọc cột riêng cho từng client.
                rows.to_csv(output_dir / f"client_{int(client_id):03d}" / f"{name}.csv",
                            mode="a", header=False, index=False)
        report["tables"][name] = {"assigned_rows": assigned, "unassigned_rows": unassigned}
    assignments.to_csv(output_dir / "assignments.csv", index=False)
    (output_dir / "partition_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
