# Heterogeneous graph

| File | Vai trò |
| --- | --- |
| schema.py | Khai báo node và quan hệ |
| node_builder.py, edge_builder.py | Xây node và cạnh |
| heterograph_builder.py | Ghép graph |
| orphan_handler.py | Xử lý quan hệ thiếu Previous và dữ liệu không hợp lệ |
| graph_dataset.py | Đọc graph đã lưu |
| validate_training_graph.py | Kiểm tra graph dùng để huấn luyện |
| central_node_identifier.py, graph_split.py | Hiện rỗng |

Schema hiện tại là orphan_fallback_v2 với sáu loại node: customer, bureau, previous_application, installment, pos_cash, credit_card.

Customer nối tới Bureau/Previous; Previous nối tới các giao dịch có cha hợp lệ. Giao dịch thiếu Previous nối về Customer qua has_orphan. Mỗi giao dịch có đúng một cạnh cha thuận; có thêm cạnh ngược. is_orphan_prev là đặc trưng nhị phân cuối của ba loại giao dịch.

Graph .npz chứa đặc trưng, mapping, cạnh, nhãn, customer_ids, train_mask/test_mask và metadata. Dữ liệu bị loại do ownership không hợp lệ được ghi vào quarantine khi phát sinh.

Chạy [03_build_graphs.py](../../scripts/03_build_graphs.py) sau partition và fit encoder. Graph, encoder và checkpoint phải tương thích schema; xem protocol đầy đủ ở [README gốc](../../README.md).

