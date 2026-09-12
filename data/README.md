# Dữ liệu

Luồng chính: `raw → interim/tables → processed/clients → shared_encoder.json → processed/graphs`.

| Thư mục | Nội dung | 
| --- | --- | --- |
| raw/ | CSV Home Credit gốc | 
| interim/tables/ | Bảng đã tổng hợp bureau_balance và lọc cột thiếu, cùng missing_report.json |
| processed/clients/ | Sáu bảng quan hệ/client, customer_split.csv, assignments.csv và partition_report.json | 
| processed/graphs/ | Graph .npz/client, manifest và quarantine nếu phát sinh | 
| processed/shared_encoder.json | Encoder chung fit từ các hàng thuộc khách hàng train |
| external/ | Chỗ dành cho dữ liệu bổ sung, hiện rỗng |

Raw hiện gồm application_train/test, bureau, bureau_balance, previous_application, installments_payments, POS_CASH_balance, credit_card_balance và HomeCredit_columns_description.

`TARGET` là nhãn; `SK_ID_CURR`, `SK_ID_BUREAU`, `SK_ID_PREV` là khóa quan hệ, không dùng làm đặc trưng đầu vào. Nhãn huấn luyện lấy từ application_train. Bureau balance được tổng hợp vào bureau.

Không fit encoder trên test. Khi thay partition, cần fit lại encoder và xây lại graph trước khi huấn luyện. Dữ liệu sinh ra bị Git ignore; xem [hướng dẫn chạy](../scripts/README.md). Không cần tạo README riêng trong từng client hoặc graph sinh tự động.

