# Mã nguồn

Các [script](../scripts/README.md) điều phối các module trong thư mục này.

| Module | Trách nhiệm |
| --- | --- |
| [preprocessing](preprocessing/README.md) | Chuẩn bị đặc trưng và encoder chung |
| [partition](partition/README.md) | Chia khách hàng, split và phân bổ bảng quan hệ |
| [graph](graph/README.md) | Xây dựng và kiểm tra heterogeneous graph |
| [models](models/README.md) | GNN và huấn luyện local-only |
| [federated](federated/README.md) | Huấn luyện client và FedAvg |
| [evaluation](evaluation/README.md) | Tính metric |
| [utils](utils/README.md) | Khung tiện ích, hiện chưa triển khai |

Một số file còn rỗng; tên file không đồng nghĩa chức năng đã có. Pipeline hiện là mô phỏng tập trung của FedAvg với encoder chung fit trên train, chưa triển khai thuật toán personalization chuyên biệt.

