# Federated learning

- `client.py`: cập nhật và dự đoán tại client.
- `server.py`: điều phối huấn luyện FedAvg và đối chứng local-only.
- `aggregation.py`, `fed_baselines.py`, `personalization.py`: hiện rỗng; không phải các thuật toán đã triển khai riêng.

FedAvg gộp cập nhật theo số khách hàng train. Adam local trong FedAvg được khởi tạo lại mỗi round; đối chứng local-only giữ optimizer qua các epoch. Hai phương pháp dùng cùng kiến trúc, encoder và split.

Chạy [04_train_federated.py](../../scripts/04_train_federated.py); đánh giá bằng [05_evaluate.py](../../scripts/05_evaluate.py). Đầu ra mặc định results/default.

Đây là mô phỏng FL tập trung với preprocessing chung trên train, chưa phải hệ thống phân tán bảo vệ riêng tư hay thuật toán personalization chuyên biệt.

