# Đối chiếu nhãn từng ID với nguồn, kiểm tra split và phát lại Dirichlet; tổng số nhãn đúng chưa đủ.
"""Check label conservation, class coverage and seeded Dirichlet replay."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.partition.dirichlet_partition import partition_dirichlet


def check_labels(source, client_dir, output_dir):
    original = pd.read_csv(source, usecols=["SK_ID_CURR", "TARGET"])
    if original.SK_ID_CURR.isna().any() or not original.SK_ID_CURR.is_unique:
        raise ValueError("Source Customer IDs must be unique and non-null")
    if not original.TARGET.isin([0, 1]).all():
        raise ValueError("Source TARGET must be binary and non-null")
    report = json.loads((client_dir / "partition_report.json").read_text())
    names = report["clients"]
    if len(names) != len(set(names)) or len(names) != report["num_clients"]:
        raise ValueError("Invalid client list in manifest")
    reference = original.set_index("SK_ID_CURR").TARGET
    positives = int(original.TARGET.sum())
    rows, frames, errors = [], [], []
    for name in names:
        frame = pd.read_csv(client_dir / name / "application_train.csv", usecols=["SK_ID_CURR", "TARGET"])
        frame["client"] = name
        frames.append(frame)
        valid_label = frame.TARGET.isin([0, 1])
        # Đối chiếu nhãn theo ID để bắt cả trường hợp hoán đổi nhãn nhưng tổng positive vẫn đúng.
        wrong_label = frame.SK_ID_CURR.isin(reference.index) & frame.TARGET.ne(frame.SK_ID_CURR.map(reference))
        if not valid_label.all() or wrong_label.any():
            errors.append(f"{name}: {int((~valid_label).sum())} invalid labels, {int(wrong_label.sum())} labels differ from source")
        n_positive, n_negative = int(frame.TARGET.eq(1).sum()), int(frame.TARGET.eq(0).sum())
        row = {"client": name, "n_customer": len(frame), "n_positive": n_positive, "n_negative": n_negative,
               "positive_rate": n_positive / len(frame) if len(frame) else None,
               "share_of_global_positive": n_positive / positives if positives else None,
               "has_both_classes": n_positive > 0 and n_negative > 0,
               "test_n_positive": None, "test_n_negative": None, "test_auc_eligible": None}
        split_path = client_dir / name / "customer_split.csv"
        if split_path.is_file():
            split = pd.read_csv(split_path)
            valid = (set(("SK_ID_CURR", "train_mask", "test_mask")) <= set(split.columns)
                     and split.SK_ID_CURR.is_unique and not split.SK_ID_CURR.isna().any()
                     and set(split.SK_ID_CURR) == set(frame.SK_ID_CURR)
                     and (len(split) == 0 or (split.train_mask.dtype == bool and split.test_mask.dtype == bool)))
            if valid and len(split):
                valid = not (split.train_mask & split.test_mask).any() and (split.train_mask | split.test_mask).all()
            if not valid:
                errors.append(f"{name}: invalid customer_split.csv")
            else:
                test_y = split.loc[split.test_mask.astype(bool), "SK_ID_CURR"].map(reference)
                row["test_n_positive"] = int(test_y.eq(1).sum())
                row["test_n_negative"] = int(test_y.eq(0).sum())
                row["test_auc_eligible"] = row["test_n_positive"] > 0 and row["test_n_negative"] > 0
        rows.append(row)
    actual = pd.concat(frames, ignore_index=True)
    coverage = {"source_customers": len(original), "assigned_customers": len(actual),
                "source_positive": positives, "assigned_positive": int(actual.TARGET.eq(1).sum()),
                "source_negative": len(original) - positives, "assigned_negative": int(actual.TARGET.eq(0).sum()),
                "duplicate_customer_rows": int(actual.SK_ID_CURR.duplicated().sum()),
                "null_customer_rows": int(actual.SK_ID_CURR.isna().sum()),
                "missing_customers": int((~original.SK_ID_CURR.isin(actual.SK_ID_CURR)).sum()),
                "extra_customer_rows": int((~actual.SK_ID_CURR.isin(original.SK_ID_CURR)).sum())}
    if any(coverage[k] for k in ("duplicate_customer_rows", "null_customer_rows", "missing_customers", "extra_customer_rows")):
        errors.append("Customer IDs are missing, duplicated or outside the source population")
    for label in ("positive", "negative"):
        if coverage[f"source_{label}"] != coverage[f"assigned_{label}"]:
            errors.append(f"Total {label} count differs from source")
    replay = {"status": "not_checked", "reason": "Manifest does not declare Dirichlet"}
    if report.get("strategy") == "dirichlet":
        keys = ("alpha", "seed", "min_per_class", "num_clients")
        if all(key in report for key in keys):
            # Replay kiểm tra tái lập chính xác theo code/seed/thứ tự nguồn, không phải kiểm định phân phối.
            _, expected = partition_dirichlet(original, report["num_clients"], report["alpha"], report["seed"], report["min_per_class"])
            expected_names = expected.set_index("SK_ID_CURR").client_id.map(lambda value: f"client_{value:03d}")
            mismatch = int(actual.client.ne(actual.SK_ID_CURR.map(expected_names)).sum())
            replay = {"status": "match" if mismatch == 0 else "mismatch", "mismatched_customer_rows": mismatch,
                      **{key: report[key] for key in keys},
                      "note": "Exact replay depends on source row order, partition code and NumPy RNG version; not a goodness-of-fit test."}
            if mismatch:
                errors.append("Saved allocation differs from replay using declared Dirichlet settings; check source order/code/version")
        else:
            replay = {"status": "not_checked", "reason": "Dirichlet parameters missing from manifest"}
    summary = {"passed": not errors, "coverage": coverage, "errors": errors, "dirichlet_replay": replay,
               "zero_positive_clients": [r["client"] for r in rows if r["n_positive"] == 0],
               "zero_negative_clients": [r["client"] for r in rows if r["n_negative"] == 0],
               "test_auc_ineligible_clients": [r["client"] for r in rows if r["test_auc_eligible"] is False],
               "test_split_not_verified_clients": [r["client"] for r in rows if r["test_auc_eligible"] is None]}
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "label_distribution.csv", index=False)
    (output_dir / "label_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/application_train.csv")
    parser.add_argument("--client-dir", type=Path, default=ROOT / "data/processed/clients")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/label_check")
    args = parser.parse_args()
    result = check_labels(**vars(args))
    sys.exit(0 if result["passed"] else 1)
