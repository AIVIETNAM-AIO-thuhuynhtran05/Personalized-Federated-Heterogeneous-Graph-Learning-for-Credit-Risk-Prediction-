# Cấu hình

Các YAML ghi lại cấu hình dự kiến/mặc định. Script hiện dùng tham số CLI và **không tự động nạp YAML**; sửa YAML chưa làm thay đổi một lần chạy.

| File | Nội dung |
| --- | --- |
| config.yaml | Đường dẫn dữ liệu, seed, quy tắc preprocessing |
| partition.yaml | Dirichlet, số client, alpha, tỷ lệ test |
| model.yaml | GNN, hidden size, số lớp, loss |
| federated.yaml | FedAvg, số round, local epochs, batch size |
| graph.yaml | Hiện còn rỗng |

Xem tham số thực tế bằng `python scripts/<tên_script>.py --help` từ thư mục gốc. Hướng dẫn chạy ở [scripts](../scripts/README.md).

