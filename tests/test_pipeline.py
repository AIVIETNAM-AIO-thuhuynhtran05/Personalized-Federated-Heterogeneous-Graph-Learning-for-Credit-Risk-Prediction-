# Script audit artifact thực tế, không phải unit test tự sinh dữ liệu; một số mô tả schema là phiên bản cũ.
"""Kiểm tra end-to-end pipeline federated heterogeneous GNN cho Home Credit.

Chạy:
    python check_pipeline.py --root .

Script kiểm tra theo đúng flow đã thống nhất:
  1) Bảng interim đã merge bureau_balance + lọc missing >80%
  2) partition_report.json và tính nhất quán schema
  3) Không customer nào bị trùng/rơi rớt giữa các client
  4) Bảng CSV từng client khớp schema toàn cục (cột + dtype khóa)
  5) Graph output từng client: node_id liên tục, edge trong phạm vi,
     reverse edge đúng là bản đảo của forward edge, missing_previous_links = 0
  6) Nhận diện đang dùng "Cách A" (5 loại cạnh, strict hierarchy) hay
     "Cách B" (8 loại cạnh, có cạnh tắt Customer->leaf)
  7) Cảnh báo nếu chưa có cột train/test cục bộ trong từng client

Không sửa bất kỳ file dữ liệu nào — chỉ đọc và báo cáo.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ("SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV")

INTERIM_KEY_BY_TABLE = {
    "application_train": "SK_ID_CURR",
    "bureau": "SK_ID_BUREAU",
    "previous_application": "SK_ID_PREV",
}

NODE_TYPES = ("customer", "bureau", "previous_application", "installment", "pos_cash", "credit_card")

EXPECTED_SCHEMA_TABLES = {
    "application_train",
    "bureau",
    "previous_application",
    "installments_payments",
    "POS_CASH_balance",
    "credit_card_balance",
}


class Report:
    """Gom kết quả PASS / WARN / FAIL để in báo cáo cuối cùng."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warned: list[str] = []
        self.failed: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def warn(self, msg: str) -> None:
        self.warned.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def print_summary(self) -> None:
        print("\n" + "=" * 64)
        print(f"PASS: {len(self.passed)}   WARN: {len(self.warned)}   FAIL: {len(self.failed)}")
        print("=" * 64)
        if self.failed:
            print("\n[FAIL] Cần sửa:")
            for m in self.failed:
                print(f"  - {m}")
        if self.warned:
            print("\n[WARN] Nên xem lại:")
            for m in self.warned:
                print(f"  - {m}")
        if self.passed:
            print("\n[PASS]:")
            for m in self.passed:
                print(f"  - {m}")


def safe(rep: Report, name: str, fn, *args):
    """Chạy 1 bước kiểm tra, không để 1 bước lỗi làm sập toàn bộ script."""
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001 - muốn bắt mọi lỗi để tiếp tục check
        rep.fail(f"Lỗi khi chạy bước '{name}': {exc}")
        return None


# ---------------------------------------------------------------------------
# 1) Bảng interim: đã merge bureau_balance, đã lọc missing >80%
# ---------------------------------------------------------------------------
def check_interim_tables(root: Path, rep: Report) -> dict[str, pd.DataFrame] | None:
    tables_dir = root / "data/interim/tables"
    if not tables_dir.exists():
        rep.fail(f"Thiếu thư mục {tables_dir}")
        return None

    expected = [
        "application_train", "bureau", "previous_application",
        "POS_CASH_balance", "credit_card_balance", "installments_payments",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name in expected:
        path = tables_dir / f"{name}.csv"
        if not path.exists():
            rep.fail(f"Thiếu bảng interim: {name}.csv")
            continue
        df = pd.read_csv(path)
        tables[name] = df

        key = INTERIM_KEY_BY_TABLE.get(name)
        if key:
            if df[key].isna().any():
                rep.fail(f"{name}: cột khóa {key} có giá trị null")
            elif not df[key].is_unique:
                rep.fail(f"{name}: cột khóa {key} bị trùng, phải unique")
            else:
                rep.ok(f"{name}: khóa {key} hợp lệ (unique, non-null)")

        miss_ratio = df.isna().mean()
        bad_cols = miss_ratio[miss_ratio > 0.8].index.tolist()
        protected = [c for c in bad_cols if c in ID_COLS or c == "TARGET"]
        if protected:
            rep.fail(f"{name}: cột khóa/label bị lọt vào diện >80% missing: {protected}")
        real_bad = [c for c in bad_cols if c not in protected]
        if real_bad:
            rep.warn(f"{name}: vẫn còn cột >80% missing chưa lọc: {real_bad}")
        else:
            rep.ok(f"{name}: không còn cột nào >80% missing")

    if "previous_application" in tables:
        leftover = [
            c for c in ("RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED")
            if c in tables["previous_application"].columns
        ]
        if leftover:
            rep.warn(f"previous_application: 2 cột lẽ ra đã xóa vẫn còn: {leftover}")
        else:
            rep.ok("previous_application: đã xóa đúng RATE_INTEREST_PRIMARY/PRIVILEGED")

    return tables


# ---------------------------------------------------------------------------
# 2) partition_report.json: cấu trúc + schema toàn cục
# ---------------------------------------------------------------------------
def check_partition(root: Path, rep: Report) -> dict | None:
    report_path = root / "data/processed/clients/partition_report.json"
    if not report_path.exists():
        rep.fail(f"Thiếu {report_path}")
        return None

    part = json.loads(report_path.read_text(encoding="utf-8"))
    clients = part.get("clients")
    schema = part.get("schema")
    if not clients:
        rep.fail("partition_report.json thiếu hoặc rỗng 'clients'")
    if not schema:
        rep.fail("partition_report.json thiếu hoặc rỗng 'schema'")

    if schema:
        missing_tables = EXPECTED_SCHEMA_TABLES - set(schema)
        extra_tables = set(schema) - EXPECTED_SCHEMA_TABLES
        if missing_tables or extra_tables:
            rep.warn(f"schema khác kỳ vọng — thiếu: {missing_tables or '{}'}, thừa: {extra_tables or '{}'}")
        else:
            rep.ok("schema có đủ 6 bảng dùng để build graph")

    return part


# ---------------------------------------------------------------------------
# 3) Không customer bị trùng / rơi rớt giữa các client
# ---------------------------------------------------------------------------
def check_no_customer_overlap(root: Path, part: dict | None, rep: Report) -> dict[int, str] | None:
    if not part:
        return None
    seen: dict[int, str] = {}
    dup_ids: set[int] = set()

    for client in part["clients"]:
        f = root / "data/processed/clients" / client / "application_train.csv"
        if not f.exists():
            rep.fail(f"{client}: thiếu application_train.csv")
            continue
        ids = pd.read_csv(f, usecols=["SK_ID_CURR"])["SK_ID_CURR"]
        if ids.duplicated().any():
            rep.fail(f"{client}: SK_ID_CURR trùng lặp ngay trong chính client này")
        for i in ids:
            if i in seen and seen[i] != client:
                dup_ids.add(i)
            seen[i] = client

    if dup_ids:
        rep.fail(f"{len(dup_ids)} customer xuất hiện ở nhiều client (rò rỉ chéo client), ví dụ: {list(dup_ids)[:5]}")
    else:
        rep.ok(f"Không có customer trùng giữa các client (tổng {len(seen)} customer duy nhất đã gán)")

    return seen


def check_coverage(root: Path, seen: dict[int, str] | None, rep: Report) -> None:
    src = root / "data/interim/tables/application_train.csv"
    if not src.exists() or seen is None:
        return
    total = set(pd.read_csv(src, usecols=["SK_ID_CURR"])["SK_ID_CURR"])
    assigned = set(seen)
    missing_from_clients = total - assigned
    extra_in_clients = assigned - total

    if missing_from_clients:
        rep.fail(f"{len(missing_from_clients)} customer bị rớt, không được gán vào client nào")
    if extra_in_clients:
        rep.fail(f"{len(extra_in_clients)} customer trong client nhưng không có trong bảng gốc")
    if not missing_from_clients and not extra_in_clients:
        rep.ok("Mọi customer trong bảng gốc đều được phân đúng 1 lần vào client, không thừa không thiếu")


# ---------------------------------------------------------------------------
# 4) Bảng CSV từng client: khớp schema toàn cục (cột + dtype khóa)
# ---------------------------------------------------------------------------
def check_client_tables_schema(root: Path, part: dict | None, rep: Report) -> None:
    if not part:
        return
    schema = part.get("schema", {})
    any_dtype_issue = False
    for client in part["clients"]:
        cdir = root / "data/processed/clients" / client
        for table, cols in schema.items():
            f = cdir / f"{table}.csv"
            if not f.exists():
                rep.fail(f"{client}/{table}.csv không tồn tại")
                continue
            df = pd.read_csv(f, nrows=5)
            if df.columns.tolist() != cols:
                rep.fail(f"{client}/{table}: thứ tự/tên cột không khớp schema toàn cục")
            for c in ID_COLS:
                if c in df.columns and str(df[c].dtype) not in ("int64", "Int64"):
                    rep.warn(f"{client}/{table}: cột khóa {c} có dtype {df[c].dtype}, nên ép về int64")
                    any_dtype_issue = True
    if not any_dtype_issue:
        rep.ok("Schema + dtype khóa của bảng client khớp nhau ở mọi client")


# ---------------------------------------------------------------------------
# 5) Graph output từng client
# ---------------------------------------------------------------------------
def _parse_edge_name(name: str):
    if "__rev_has__" in name:
        dst, src = name.split("__rev_has__")
        return src, dst, True
    if "__has__" in name:
        src, dst = name.split("__has__")
        return src, dst, False
    return None, None, None


def check_graphs(root: Path, part: dict | None, rep: Report) -> None:
    if not part:
        return
    graph_root = root / "data/processed/graphs"
    edge_type_inventory: set[str] = set()

    for client in part["clients"]:
        gdir = graph_root / client
        rpath = gdir / "graph_report.json"
        if not rpath.exists():
            rep.fail(f"{client}: thiếu graph_report.json (chưa build graph?)")
            continue
        greport = json.loads(rpath.read_text(encoding="utf-8"))

        for k, v in greport.get("missing_previous_links", {}).items():
            if v > 0:
                rep.fail(f"{client}: {v} dòng {k} không tìm thấy previous_application cùng client")

        node_counts_actual: dict[str, int] = {}
        for node_type in NODE_TYPES:
            npath = gdir / f"{node_type}_nodes.csv"
            if not npath.exists():
                rep.fail(f"{client}: thiếu {node_type}_nodes.csv")
                continue
            ndf = pd.read_csv(npath)
            node_counts_actual[node_type] = len(ndf)
            ids = ndf["node_id"].to_numpy()
            if len(ids) and not np.array_equal(np.sort(ids), np.arange(len(ids))):
                rep.fail(f"{client}/{node_type}: node_id không liên tục 0..n-1")

        for k, v in greport.get("node_counts", {}).items():
            if node_counts_actual.get(k) != v:
                rep.fail(f"{client}: node_counts trong report ({k}={v}) khác thực tế ({node_counts_actual.get(k)})")

        epath = gdir / "edges.npz"
        if not epath.exists():
            rep.fail(f"{client}: thiếu edges.npz")
            continue
        edges = np.load(epath)
        edge_type_inventory.update(edges.files)

        for name in edges.files:
            arr = edges[name]
            if arr.shape[0] != 2:
                rep.fail(f"{client}/{name}: edge array không đúng shape (2, N)")
                continue
            src_type, dst_type, _ = _parse_edge_name(name)
            if arr.size == 0:
                continue
            if src_type in node_counts_actual and (arr[0].max() >= node_counts_actual[src_type] or arr[0].min() < 0):
                rep.fail(f"{client}/{name}: source node_id vượt phạm vi node {src_type}")
            if dst_type in node_counts_actual and (arr[1].max() >= node_counts_actual[dst_type] or arr[1].min() < 0):
                rep.fail(f"{client}/{name}: dest node_id vượt phạm vi node {dst_type}")

        for name in edges.files:
            src, dst, is_rev = _parse_edge_name(name)
            if not is_rev:
                continue
            fwd_name = f"{src}__has__{dst}"
            if fwd_name in edges.files:
                if not np.array_equal(edges[fwd_name][::-1], edges[name]):
                    rep.fail(f"{client}: {name} KHÔNG phải bản đảo đúng của {fwd_name}")
            else:
                rep.warn(f"{client}: có {name} nhưng thiếu cạnh xuôi {fwd_name} để đối chiếu")

    n_fwd = len([e for e in edge_type_inventory if "__rev_has__" not in e])
    if n_fwd:
        style = "Cách B (có cạnh tắt Customer->leaf)" if n_fwd > 5 else "Cách A (strict hierarchy)"
        rep.ok(f"Phát hiện {n_fwd} loại cạnh xuôi ({len(edge_type_inventory)} kể cả reverse) -> đang dùng {style}")


# ---------------------------------------------------------------------------
# 6) Cảnh báo train/test cục bộ (bước còn thiếu theo flow đã sửa)
# ---------------------------------------------------------------------------
def check_train_test_split(root: Path, part: dict | None, rep: Report) -> None:
    if not part or not part.get("clients"):
        return
    first_client = part["clients"][0]
    f = root / "data/processed/clients" / first_client / "application_train.csv"
    if not f.exists():
        return
    cols = pd.read_csv(f, nrows=1).columns
    marker_cols = [c for c in cols if c.lower() in ("is_test", "test_mask", "split")]
    if marker_cols:
        rep.ok(f"Đã thấy cột đánh dấu train/test cục bộ: {marker_cols}")
    else:
        rep.warn(
            "CHƯA có cột train/test cục bộ (is_test/test_mask/split) trong bảng client — "
            "theo flow đã sửa, cần chia train/test NGAY TRONG từng client, "
            "trước bước fit encoder/scaler toàn cục."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."), help="Thư mục gốc của project")
    args = parser.parse_args()

    rep = Report()

    tables = safe(rep, "interim_tables", check_interim_tables, args.root, rep)
    part = safe(rep, "partition", check_partition, args.root, rep)
    seen = safe(rep, "no_overlap", check_no_customer_overlap, args.root, part, rep)
    safe(rep, "coverage", check_coverage, args.root, seen, rep)
    safe(rep, "client_schema", check_client_tables_schema, args.root, part, rep)
    safe(rep, "graphs", check_graphs, args.root, part, rep)
    safe(rep, "train_test", check_train_test_split, args.root, part, rep)

    rep.print_summary()
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()