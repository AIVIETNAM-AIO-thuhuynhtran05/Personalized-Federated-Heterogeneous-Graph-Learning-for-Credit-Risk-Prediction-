"""Dựng local heterogeneous graph cho từng client (nhánh Federated).

Mỗi client chỉ dùng dữ liệu của khách hàng thuộc mình:
  xử lý orphan -> nối cạnh -> đặc trưng cấu trúc (cục bộ)
  -> encoder thống nhất qua 2 vòng trao đổi thống kê tổng hợp trên Train của client
  -> encode, kiểm tra, lưu graph của client.
(Bảng CSV được đọc một lần rồi cắt theo client chỉ để tiết kiệm I/O; mọi tính toán sau đó
đều thực hiện riêng trên dữ liệu của từng client.)

    python scripts/03_build_graphs.py [--scenario NAME]
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch

from src.federated.federated_encoder import FederatedTabularEncoder
from src.graph.heterograph_builder import assemble_graph, node_columns, prepare_tables
from src.graph.node_builder import load_node_tables
from src.graph.orphan_check import validate_graph
from src.graph.schema import CUSTOMER, NODE_TYPES
from src.utils.io import load_config, load_scenario, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_client_graphs")


def split_by_client(tables: dict, owner: pd.Series, clients) -> dict:
    """Cắt mọi bảng node theo client sở hữu SK_ID_CURR (mỗi khách hàng thuộc đúng 1 client)."""
    out = {k: {} for k in clients}
    for t in list(tables):
        df = tables.pop(t)
        cid = df["SK_ID_CURR"].map(owner).values
        for k in clients:
            out[k][t] = df[cid == k].reset_index(drop=True)
        del df
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    args = parser.parse_args()

    cfg, gcfg = load_config(), load_config("graph.yaml")
    name, _, paths = load_scenario(args.scenario)
    set_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    log.info("Scenario: %s", name)

    assign = pd.read_parquet(paths["assignments"])
    clients = sorted(assign["client"].unique())
    owner = assign.set_index("SK_ID_CURR")["client"]

    tables, bb_stats = load_node_tables(cfg, assign["SK_ID_CURR"].values)
    client_tables = split_by_client(tables, owner, clients)

    # 1) Cục bộ: orphan, cạnh, đặc trưng cấu trúc
    prepared, report = {}, {"bureau_balance_global": bb_stats, "clients": {}}
    for k in clients:
        parents, roots, rep = prepare_tables(client_tables[k], gcfg)
        train_ids = set(assign.loc[(assign["client"] == k) & (assign["split"] == "train"), "SK_ID_CURR"])
        train_mask = client_tables[k][CUSTOMER]["SK_ID_CURR"].isin(train_ids).values
        prepared[k] = (parents, roots, train_mask)
        report["clients"][int(k)] = rep
        log.info("client %d: %s", k, {t: len(client_tables[k][t]) for t in NODE_TYPES})

    # 2) Encoder thống nhất: 2 vòng trao đổi thống kê tổng hợp
    encoders, xs = {}, {k: {} for k in clients}
    for t in NODE_TYPES:
        cols = set().union(*(client_tables[k][t].columns for k in clients))
        for k in clients:  # cùng schema cột ở mọi client
            missing = [c for c in cols if c not in client_tables[k][t]]
            for c in missing:
                client_tables[k][t][c] = np.nan
        num, cat = node_columns(t, client_tables[clients[0]][t])
        enc = FederatedTabularEncoder(num, cat, gcfg["clip_quantiles"], gcfg["onehot_min_frequency"])

        def train_rows(k):
            _, roots, train_mask = prepared[k]
            return client_tables[k][t].iloc[np.flatnonzero(train_mask[roots[t]])]

        enc.aggregate_round1([enc.local_stats_round1(train_rows(k), gcfg["encoder_fit_max_rows"], rng)
                              for k in clients])
        enc.aggregate_round2([enc.local_stats_round2(train_rows(k)) for k in clients])
        for k in clients:
            xs[k][t] = torch.from_numpy(enc.transform(client_tables[k][t]))
        encoders[t] = enc
        log.info("encoder %-11s -> %d features (shared by all clients)", t, len(enc.feature_names_out()))

    # 3) Ráp và lưu graph của từng client
    out_dir = paths["graphs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in clients:
        data = assemble_graph(client_tables[k], prepared[k][0], xs[k])
        report["clients"][int(k)]["validation"] = validate_graph(data)
        report["clients"][int(k)]["nodes"] = {t: int(data[t].num_nodes) for t in NODE_TYPES}
        torch.save(data, out_dir / f"client_{k}.pt")
        client_tables[k], xs[k] = None, None
    report["feature_dims"] = {t: len(e.feature_names_out()) for t, e in encoders.items()}
    joblib.dump(encoders, out_dir / "federated_encoders.joblib")
    save_json(report, paths["metrics"] / "graph_report.json")

    nodes = pd.DataFrame({k: r["nodes"] for k, r in report["clients"].items()}).T
    log.info("Client graph nodes:\n%s", nodes.to_string())
    orph = pd.DataFrame({k: {f"{t}_missing_prev": r["missing_previous"][t]["rows_missing_prev"]
                             for t in ["installment", "pos", "cc"]}
                         | {"placeholders": r["missing_previous"].get("placeholder_prev_created", 0),
                            "cust_no_history": r["childless"]["customers_without_any_history"]}
                         for k, r in report["clients"].items()}).T
    log.info("Orphans per client:\n%s", orph.to_string())


if __name__ == "__main__":
    main()
