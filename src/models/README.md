# Mô hình

- `hetero_gnn.py`: GNN truyền thông điệp mean theo loại quan hệ, có head phân loại Customer.
- `local_training.py`: huấn luyện và dự đoán local-only theo batch thành phần khách hàng.
- `baselines.py`, `classifier.py`: hiện rỗng.

Mặc định hidden size 32 và 2 lớp; loss BCEWithLogitsLoss không trọng số. Huấn luyện chỉ dùng train_mask. Batch giữ thành phần khách hàng cùng các node lịch sử của họ.

Chạy [04_train_local_gnn.py](../../scripts/04_train_local_gnn.py) để train từng client độc lập; đầu ra mặc định results/local_gnn. Checkpoint lấy ở epoch cuối, không chọn bằng test. Mô hình này cũng được dùng trong [FedAvg](../federated/README.md).

