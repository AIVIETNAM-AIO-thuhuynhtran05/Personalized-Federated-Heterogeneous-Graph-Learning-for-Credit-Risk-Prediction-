# Chia client và bảng quan hệ

- `dirichlet_partition.py`: chia theo nhãn bằng Dirichlet có ràng buộc số mẫu tối thiểu.
- `multidimensional_partition.py`: chiến lược semantic theo các cột phân nhóm.
- `relational_partition.py`: điều phối chia khách hàng, split cục bộ và chuyển lịch sử theo chủ sở hữu.
- `iid_partition.py`, `partition_diagnostics.py`: hiện rỗng.

Entry point: [02_partition_clients.py](../../scripts/02_partition_clients.py). Mặc định 10 client, alpha=0.5, seed=42, test_size=0.2 và min_per_class=2. Có thể chọn strategy semantic.

Đầu vào là data/interim/tables; đầu ra là data/processed/clients. Khách hàng không trùng giữa client; bảng lịch sử đi theo SK_ID_CURR. Mỗi client có customer_split.csv; báo cáo tổng ghi trong partition_report.json. Khi đổi split phải fit lại shared encoder và tạo lại graph.

