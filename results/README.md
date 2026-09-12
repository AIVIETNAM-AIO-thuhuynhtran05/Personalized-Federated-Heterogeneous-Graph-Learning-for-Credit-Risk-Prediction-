# Kết quả thí nghiệm

Đây là đầu ra chạy chương trình, không phải mã nguồn. Giữ lại kết quả cần dùng cho báo cáo trước khi chạy lại cùng đường dẫn.

| Đường dẫn | Nội dung |
| --- | --- |
| local_gnn/ | run_config.json, summary.csv và từng client: model.pt, history.csv, test_predictions.csv |
| label_check/ | Báo cáo JSON và CSV phân phối nhãn |
| relational_audit.json | Báo cáo kiểm toán bảng quan hệ |
| relational_audit_artifact.json | Bản tóm tắt do script 08 tạo |
| relational_audit_delivery.json | Tệp phụ trợ hiện có của báo cáo audit |
| orphan_raw_vs_interim.json | Báo cáo đối chiếu orphan hiện có |
| default/ | Đầu ra mặc định của train federated/evaluate; xuất hiện khi chạy |
| sweep/ | Đầu ra theo số client và seed; xuất hiện khi chạy sweep |

Theo [protocol](../README.md), checkpoint lấy ở vòng/epoch cuối, không chọn theo test. ROC-AUC, PR-AUC hình thang và average precision là các chỉ số riêng; không mặc định coi PR-AUC là average precision. CSV cũ có thể chưa có các cột mới.

Muốn tái lập cần lưu cấu hình, split, encoder, graph và checkpoint tương ứng. Không coi toàn bộ results là cache có thể bỏ.

