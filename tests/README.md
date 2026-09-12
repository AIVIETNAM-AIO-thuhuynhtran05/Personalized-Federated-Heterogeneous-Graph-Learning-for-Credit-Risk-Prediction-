# Kiểm thử

Chạy từ thư mục gốc sau khi cài dependencies. Các nhóm kiểm thử phần mềm chính:

```powershell
python -m unittest discover -s tests -p test_prepare_tables.py -v
python -m unittest discover -s tests -p test_federated_pipeline.py -v
python -m unittest discover -s tests -p test_dirichlet_training.py -v
```

| File / nhóm | Phạm vi |
| --- | --- |
| test_preprocessing.py, test_prepare_tables.py | Làm sạch, tổng hợp và chuẩn bị bảng |
| test_partition.py, test_dirichlet_training.py | Chia client, split và tính lặp lại |
| test_graph_builder.py, test_orphan_graph.py | Graph và quan hệ orphan |
| test_federated_pipeline.py | Encoder, huấn luyện và pipeline federated |
| test_auc_metrics.py | Các chỉ số AUC |
| test_relational_audit.py, test_label_check.py | Báo cáo quan hệ và nhãn |
| test_pipeline.py | Script kiểm tra các artifact dữ liệu thực tế, cần dữ liệu đã sinh |

`test_pipeline.py` còn mô tả schema graph cũ trong docstring; khi đọc kết quả cần đối chiếu schema hiện tại ở [graph](../src/graph/README.md). Không dùng kết quả synthetic test để kết luận hiệu quả mô hình trên Home Credit.

Cache .pytest_cache và __pycache__ có thể xóa; giữ các file kiểm thử.

