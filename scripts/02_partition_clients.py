"""Chia khách hàng vào K client theo kịch bản trong configs/partition.yaml, giữ nguyên split chung.

    python scripts/02_partition_clients.py                                  # kịch bản mặc định
    python scripts/02_partition_clients.py --scenario label_dirichlet_a0.5
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.partition.dirichlet_partition import partition_clients
from src.partition.partition_diagnostics import PROFILE_COLS, partition_report
from src.utils.io import load_config, load_scenario, read_raw, resolve, save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("partition")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario")
    args = parser.parse_args()

    cfg = load_config()
    name, scfg, paths = load_scenario(args.scenario)
    splits = pd.read_parquet(resolve(cfg["paths"]["processed_dir"]) / cfg["files"]["splits"])
    extra = [scfg["segment_col"]] if "segment_col" in scfg else []
    profile = read_raw(cfg, "application_train", usecols=["SK_ID_CURR", *PROFILE_COLS, *extra])
    profile = profile.set_index("SK_ID_CURR").loc[splits["SK_ID_CURR"]].reset_index()

    attribute = None
    if scfg["method"] == "region_territory":
        attribute = profile[scfg["region_col"]]
    elif scfg["method"] == "segment_mix":
        attribute = profile[scfg["segment_col"]].map(scfg["segment_map"])
        assert attribute.notna().all(), "segment_map thiếu giá trị của " + scfg["segment_col"]
    assign, info = partition_clients(splits, scfg, attribute)
    paths["assignments"].parent.mkdir(parents=True, exist_ok=True)
    assign.to_parquet(paths["assignments"], index=False)

    report = {"scenario": name, "config": scfg, **info, **partition_report(assign, profile)}
    if attribute is not None and scfg["method"] == "segment_mix":
        mix = pd.crosstab(assign["client"].values, attribute.values, normalize="index")
        report["segment_mix_by_client"] = mix.round(4).to_dict(orient="index")
        log.info("Cơ cấu phân khúc của từng client:\n%s", (mix * 100).round(1).to_string())
    save_json(report, paths["metrics"] / "partition_report.json")

    rows = {}
    for k, c in report["clients"].items():
        rows[k] = {"n": c["n"], "share": c["share"], "default_rate": c["default_rate"],
                   "regions": c["n_regions"], "rating1": c["region_rating_share"].get(1, 0),
                   "rating3": c["region_rating_share"].get(3, 0), "med_income": c["median_income"],
                   "ext2": c["mean_ext_source_2"], "W(ext2)": c["feature_shift_wasserstein"]["EXT_SOURCE_2"],
                   "W(income)": c["feature_shift_wasserstein"]["LOG_INCOME"],
                   "JS(label)": c["label_js_divergence"],
                   **{f"n_{s}": c["split_sizes"].get(s, 0) for s in ["train", "val", "test"]}}
    log.info("Scenario %s:\n%s", name, pd.DataFrame(rows).T.to_string())
    log.info("size max/min = %s | default rate range = %s",
             report["size_max_min_ratio"], report["default_rate_range"])


if __name__ == "__main__":
    main()
