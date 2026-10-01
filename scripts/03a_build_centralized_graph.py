"""Dựng global heterogeneous graph cho nhánh Centralized + báo cáo orphan records.

    python scripts/03a_build_centralized_graph.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import torch

from src.graph.heterograph_builder import build_hetero_graph
from src.utils.io import load_config, resolve, save_json
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_graph")


def main():
    cfg, gcfg = load_config(), load_config("graph.yaml")
    set_seed(cfg["seed"])
    processed, results = resolve(cfg["paths"]["processed_dir"]), resolve(cfg["paths"]["results_dir"])

    # Dùng đúng split chung với các baseline tabular
    splits = pd.read_parquet(processed / cfg["files"]["splits"])
    train_ids = splits.loc[splits["split"] == "train", "SK_ID_CURR"]

    data, report, artifacts = build_hetero_graph(cfg, gcfg, splits["SK_ID_CURR"].values, train_ids)
    report["orphan_policy"] = gcfg["orphan_policy"]

    out = processed / gcfg["files"]["centralized_graph"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out)
    joblib.dump(artifacts, out.with_suffix(".encoders.joblib"))
    save_json(report, results / gcfg["files"]["orphan_report"])

    log.info("Graph saved -> %s", out.relative_to(resolve(".")))
    log.info("Nodes: %s", report["nodes"])
    log.info("Feature dims: %s", report["feature_dims"])
    for t, s in report["missing_previous"].items():
        log.info("missing_previous[%s]: %s", t, s)
    log.info("childless: %s", {k: v for k, v in report["childless"].items()
                               if k != "prev_payment_coverage_by_status"})
    log.info("bureau_balance: %s", report["bureau_balance"])


if __name__ == "__main__":
    main()
