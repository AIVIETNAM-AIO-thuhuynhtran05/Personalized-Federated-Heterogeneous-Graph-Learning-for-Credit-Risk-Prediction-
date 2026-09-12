# Đánh giá

`metrics.py` triển khai các metric phân loại nhị phân. `fairness.py` và `non_iid_analysis.py` hiện rỗng.

- ROC-AUC: diện tích dưới ROC.
- PR-AUC: diện tích hình thang dưới đường precision–recall.
- Average precision: chỉ số riêng, không đồng nhất với PR-AUC hình thang.
- auc: tên tương thích cho ROC-AUC.

TARGET=1 là lớp dương. Metric không xác định cho tập rỗng hoặc chỉ có một lớp được ghi null/ô CSV trống. Pooled AUC tính trên các dự đoán ghép lại, không phải trung bình AUC từng client.

Đánh giá checkpoint qua [05_evaluate.py](../../scripts/05_evaluate.py). Các metric test dùng để báo cáo, không chọn checkpoint tốt nhất.

