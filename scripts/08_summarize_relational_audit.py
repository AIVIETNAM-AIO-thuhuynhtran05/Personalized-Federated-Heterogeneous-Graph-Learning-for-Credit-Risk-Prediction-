"""Create a portable-report input and a companion inspection notebook from the audit."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "results/relational_audit.json").read_text())
if "partition_pass" not in report:
    raise ValueError("Audit has not completed")
title = "Kiểm tra chia dữ liệu relational Home Credit"
passed = report["partition_pass"] and report["customer_consistent_previous_links"]
sections = [
    ("title", f"# {title}"),
    ("summary", "## Kết luận về khóa ngoại\n\n" + (
        "**Dữ liệu hiện tại đã được phân bổ đúng theo SK_ID_CURR.** " if passed else "**Có sai lệch cần xử lý trước khi dùng graph.** ") +
        f"Đã kiểm tra toàn bộ {report['customer']['source_rows']:,} Customer ở {report['num_clients']} client. "
        "Đối chiếu từng dòng khóa và số lần xuất hiện của từng cặp khóa với phần dữ liệu nguồn thuộc quần thể Customer train."),
    ("findings", "## Không nhầm khóa thiếu ở nguồn với lỗi partition\n\n" + "\n".join(
        f"- **{name}**: {data['client_rows']:,} dòng; {data['wrong_client_rows']:,} dòng sai client; "
        f"{data['missing_key_occurrences']:,} lần xuất hiện khóa bị thiếu và {data['extra_key_occurrences']:,} lần thừa. "
        f"{data['missing_previous_in_source']:,} dòng thiếu Previous Application ngay trong nguồn "
        f"({data['missing_previous_source_rate']:.2%} số dòng của bảng trong các client); "
        f"{data['missing_previous_only_locally']:,} dòng có Previous ở nguồn nhưng thiếu trong client; "
        f"{data['previous_customer_mismatch']:,} dòng nối Previous thuộc Customer khác."
        for name, data in report["tables"].items()) +
        "\n\nCác giao dịch thiếu Previous vẫn có thể nối trực tiếp đến Customer. "
        "Không nên xóa chúng chỉ để báo cáo không còn khóa thiếu."),
    ("scope", "## Phạm vi và quần thể được kiểm tra\n\n"
        "Nguồn đối chiếu là sáu bảng đã chuẩn bị trong data/interim/tables. "
        "Quần thể cần chia gồm Customer trong application_train; lịch sử thuộc Customer ngoài quần thể này "
        "được loại khỏi số dòng kỳ vọng, không tính là mất dữ liệu. Mỗi dòng sự kiện được tính một lần, "
        "kể cả khi nhiều sự kiện có cùng cặp khóa."),
    ("method", "## Kiểm tra toàn bộ khóa bằng đọc từng phần\n\n"
        "Xác minh Customer không null, không trùng, không thiếu/thừa và khớp assignments.csv. "
        "Sau đó đọc 100.000 dòng/lần, kiểm tra chủ sở hữu Customer của từng dòng, và dùng SQLite "
        "so sánh chính xác số lần xuất hiện của từng cặp SK_ID_CURR–SK_ID_BUREAU hoặc SK_ID_CURR–SK_ID_PREV. "
        "Với ba bảng giao dịch, kiểm tra thêm Previous tồn tại ở nguồn, tồn tại trong client, và thuộc đúng Customer."),
    ("limits", "## Những điều kết quả này chưa chứng minh\n\n"
        "Đây là kiểm tra phân bổ quan hệ, không phải so sánh mọi giá trị feature hay xác nhận không có leakage. "
        "Các sự kiện có cùng cặp khóa được so sánh theo số lượng; chưa so sánh nội dung từng sự kiện. "
        "Không đánh giá lại phép tổng hợp bureau_balance, chất lượng huấn luyện hoặc mức độ non-IID. "
        f"Manifest hiện tại ghi strategy={report['declared_strategy']}; không dùng báo cáo này để xác nhận đã chạy Dirichlet mới."),
    ("next", "## Bước tiếp theo\n\n"
        "Giữ cách phân bổ bảng liên quan theo SK_ID_CURR. Ghi nhận tỷ lệ thiếu Previous từ nguồn riêng với lỗi partition. "
        "Sau mỗi lần chia lại client, chạy lại script audit trước khi fit encoder và build graph."),
    ("questions", "## Câu hỏi còn mở\n\n"
        "Mức thiếu lịch sử Previous ảnh hưởng bao nhiêu đến AUC? Cần đánh giá mô hình hoặc ablation riêng; "
        "kiểm tra khóa hiện tại không trả lời được câu hỏi đó."),
]
artifact = {"surface": "report", "manifest": {"version": 1, "surface": "report", "title": title,
    "description": "Technical audit of customer ownership and relational partition integrity.",
    "sources": [{"id": "audit", "label": "Full relational partition audit", "path": "results/relational_audit.json"}],
    "blocks": [{"id": key, "type": "markdown", "body": text, "sourceId": "audit"}
               for key, text in sections]},
    "snapshot": {"version": 1, "generatedAt": report["checked_at_utc"], "status": "ready", "datasets": {}}}
(ROOT / "results/relational_audit_artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
cells = [
    {"cell_type": "markdown", "metadata": {}, "source": ["# Relational partition audit\n", "Read the saved full-data audit; rerun the script after changing partitions.\n", "The reusable script contains all checks and uses disk-backed counters to bound RAM."]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "from pathlib import Path\n", "import json\n", "import pandas as pd\n",
        "root = Path.cwd() if (Path.cwd() / 'scripts').is_dir() else Path.cwd().parent\n",
        "audit = json.loads((root / 'results/relational_audit.json').read_text())\n",
        "audit['customer'], audit['partition_pass'], audit['customer_consistent_previous_links']"]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "pd.DataFrame([{k: v for k, v in values.items() if k != 'clients'} | {'table': name}\n",
        "              for name, values in audit['tables'].items()]).set_index('table')"]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [
        "# Optional full rerun (reads all source/client key rows):\n",
        "# import subprocess, sys\n",
        "# subprocess.run([sys.executable, str(root / 'scripts/07_audit_relational_partition.py')], cwd=root, check=True)"]},
]
(ROOT / "notebooks/03_relational_partition_audit.ipynb").write_text(json.dumps({"nbformat": 4, "nbformat_minor": 5,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}, indent=2), encoding="utf-8")
