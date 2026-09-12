# Script chạy pipeline

Chạy các lệnh dưới đây từ thư mục gốc, trong môi trường đã cài [requirements.txt](../requirements.txt). Các giá trị mặc định được định nghĩa trong CLI, không tự nạp YAML.

```powershell
python scripts/01_prepare_tables.py
python scripts/02_partition_clients.py --num-clients 10 --alpha 0.5 --seed 42
python scripts/02b_fit_encoder.py
python scripts/03_build_graphs.py
python scripts/04_train_federated.py --rounds 20 --local-epochs 1 --batch-size 256
python scripts/05_evaluate.py
```

| Script | Vai trò / đầu ra mặc định |
| --- | --- |
| 01_prepare_tables.py | Chuẩn bị bảng quan hệ từ raw → data/interim/tables |
| 02_partition_clients.py | Chia khách hàng và bảng liên quan → data/processed/clients |
| 02b_fit_encoder.py | Fit encoder chung trên train → data/processed/shared_encoder.json |
| 03_build_graphs.py | Tạo graph từng client → data/processed/graphs |
| 04_train_federated.py | Huấn luyện FedAvg và đối chứng local-only → results/default |
| 04_train_local_gnn.py | Chạy riêng GNN độc lập cho từng client → results/local_gnn |
| 05_evaluate.py | Đánh giá checkpoint của lần chạy federated |
| 06_sweep_clients.py | Lặp thí nghiệm theo số client/seed → results/sweep |
| 07_audit_relational_partition.py | Kiểm tra phân bổ các bảng → results/relational_audit.json |
| 08_summarize_relational_audit.py | Đọc audit, tạo bản tóm tắt và notebook; cần audit có sẵn |
| 09_check_label_split.py | Kiểm tra phân phối nhãn và split → results/label_check |
| 01_preprocess.py | Tiện ích đặc trưng application riêng, ngoài pipeline chính |

Ví dụ local-only và sweep:

```powershell
python scripts/04_train_local_gnn.py --epochs 20 --batch-size 128
python scripts/06_sweep_clients.py --client-counts 10 20 30 40 50 --seeds 42 --rounds 20
```

Các lệnh có thể ghi lại đầu ra tại đường dẫn mặc định; lưu riêng kết quả cần giữ trước khi chạy lại. Sweep giữ dữ liệu và graph cho từng cấu hình nên có thể tốn nhiều dung lượng.

`translate_eda_vi.py` và `finish_eda_translation.py` là tiện ích dịch cũ, cần notebook nguồn `notebooks/01_eda.ipynb` hiện không có. Script hoàn thiện còn cần cache dịch; phụ thuộc `nbformat` chưa nằm trong requirements hiện tại. Chúng không thuộc pipeline huấn luyện.

