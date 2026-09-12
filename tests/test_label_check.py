# Đổi nhãn giữa ID nhưng giữ tổng số positive để kiểm tra audit phát hiện lỗi ở cấp khách hàng.
import importlib.util
import json
from pathlib import Path
import pandas as pd
from src.partition.dirichlet_partition import partition_dirichlet, local_split

spec = importlib.util.spec_from_file_location("label_check", Path(__file__).resolve().parents[1] / "scripts/09_check_label_split.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_label_swaps_are_caught_even_when_positive_total_is_preserved(tmp_path):
    source = pd.DataFrame({"SK_ID_CURR": range(12), "TARGET": [0]*6 + [1]*6})
    source.to_csv(tmp_path / "original.csv", index=False)
    clients, _ = partition_dirichlet(source, 2)
    names = []
    for i, frame in clients.items():
        name = f"client_{i:03d}"
        names.append(name)
        (tmp_path / name).mkdir()
        frame.to_csv(tmp_path / name / "application_train.csv", index=False)
        local_split(frame).to_csv(tmp_path / name / "customer_split.csv", index=False)
    (tmp_path / "partition_report.json").write_text(json.dumps({"clients": names, "num_clients": 2,
        "strategy": "dirichlet", "alpha": 0.5, "seed": 42, "min_per_class": 2}))
    result = module.check_labels(tmp_path / "original.csv", tmp_path, tmp_path / "out")
    assert result["passed"]
    assert result["dirichlet_replay"]["status"] == "match"
    path = tmp_path / names[0] / "application_train.csv"
    frame = pd.read_csv(path)
    negative = frame.index[frame.TARGET == 0][0]
    positive = frame.index[frame.TARGET == 1][0]
    frame.loc[negative, "TARGET"] = 1
    frame.loc[positive, "TARGET"] = 0
    frame.to_csv(path, index=False)
    result = module.check_labels(tmp_path / "original.csv", tmp_path, tmp_path / "out")
    assert not result["passed"]
    assert result["coverage"]["assigned_positive"] == result["coverage"]["source_positive"]
    assert any("labels differ" in error for error in result["errors"])
