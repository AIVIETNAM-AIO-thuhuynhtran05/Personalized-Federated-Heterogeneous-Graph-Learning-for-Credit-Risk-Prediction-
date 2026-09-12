# Tiền xử lý

- `aggregate_bureau.py`: tổng hợp lịch sử bureau_balance.
- `clean_application.py`: làm sạch bảng application.
- `feature_transformer.py`: tạo đặc trưng application cho tiện ích preprocessing riêng.
- `shared_encoder.py`: fit và áp dụng encoder dùng chung giữa các client.

Bước chuẩn bị bảng nằm ở [01_prepare_tables.py](../../scripts/01_prepare_tables.py). Encoder được fit sau khi chia client và train/test, chỉ từ khách hàng train và lịch sử thuộc các khách hàng đó. Khóa ID và TARGET không được dùng làm đặc trưng.

Numeric dùng imputation/scaling chung; categorical dùng vocabulary chung với giá trị missing/unknown. Encoder lưu tại data/processed/shared_encoder.json. Cờ is_orphan_prev được thêm lúc xây graph, không phải lúc fit encoder.

