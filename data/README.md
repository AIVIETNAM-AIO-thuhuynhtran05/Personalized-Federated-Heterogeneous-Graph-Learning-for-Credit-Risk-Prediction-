# Dữ liệu

Thư mục này lưu dữ liệu từ CSV Home Credit gốc đến graph dùng để huấn luyện. Các đường dẫn trong tài liệu tính từ thư mục gốc project.

Luồng chính: `raw → interim/tables → processed/clients → shared_encoder.json → processed/graphs`.

| Thư mục | Nội dung | Khi dọn project |
| --- | --- | --- |
| raw/ | CSV Home Credit gốc | Nên giữ; nguồn để tái tạo pipeline |
| interim/tables/ | Bảng đã tổng hợp bureau_balance và lọc cột thiếu, cùng missing_report.json | Có thể tạo lại bằng 01_prepare_tables.py |
| processed/clients/ | Sáu bảng quan hệ/client, customer_split.csv, assignments.csv và partition_report.json | Tạo lại bằng bước partition |
| processed/graphs/ | Graph .npz/client, manifest và quarantine nếu phát sinh | Tạo lại sau partition và encoder |
| processed/shared_encoder.json | Encoder chung fit từ các hàng thuộc khách hàng train | Giữ đồng bộ với split và graph |
| external/ | Chỗ dành cho dữ liệu bổ sung, hiện rỗng | Chưa được pipeline hiện tại sử dụng |

Raw hiện gồm application_train/test, bureau, bureau_balance, previous_application, installments_payments, POS_CASH_balance, credit_card_balance và HomeCredit_columns_description.

`TARGET` là nhãn; `SK_ID_CURR`, `SK_ID_BUREAU`, `SK_ID_PREV` là khóa quan hệ, không dùng làm đặc trưng đầu vào. Nhãn huấn luyện lấy từ application_train. Bureau balance được tổng hợp vào bureau.

## 1. Dữ liệu gốc: raw/

| File | Vai trò |
| --- | --- |
| `application_train.csv` | Khách hàng có nhãn, dùng để chia client và train/test |
| `application_test.csv` | Khách hàng không có nhãn; không phải local-test trong thí nghiệm hiện tại |
| `bureau.csv` | Hồ sơ tín dụng tại các tổ chức khác |
| `bureau_balance.csv` | Lịch sử theo tháng của hồ sơ bureau |
| `previous_application.csv` | Các hồ sơ vay trước đây tại Home Credit |
| `installments_payments.csv` | Thông tin kỳ thanh toán |
| `POS_CASH_balance.csv` | Lịch sử dư nợ POS/cash |
| `credit_card_balance.csv` | Lịch sử dư nợ thẻ tín dụng |
| `HomeCredit_columns_description.csv` | Mô tả cột để tra cứu |

Giữ nguyên raw để tái tạo pipeline. `TARGET=1` là lớp dương khi tính metric. Các khóa ID phục vụ nối bảng và xác định chủ sở hữu, không được encoder đưa vào đặc trưng.

## 2. Bảng đã chuẩn bị: interim/tables/

[01_prepare_tables.py](../scripts/01_prepare_tables.py) tổng hợp bureau_balance vào bureau rồi loại cột có tỷ lệ thiếu **lớn hơn 80%** theo mặc định. Các cột được bảo vệ gồm ba khóa ID, TARGET, REGION_RATING_CLIENT_W_CITY và OCCUPATION_TYPE nếu có.

Với application_test, script áp dụng danh sách cột bị loại từ application_train. Đầu ra gồm bảy bảng CSV: application_train/test, bureau, previous_application và ba bảng giao dịch. Không xuất bureau_balance riêng vì đã tổng hợp vào bureau.

`missing_report.json` lưu ngưỡng thiếu, số dòng, số cột trước/sau, danh sách cột giữ/loại và chính sách lọc từng bảng.

**Phạm vi thống kê:** lọc cột thiếu hiện thực hiện trên bảng nguồn trước khi chia train/test. Chỉ bước fit encoder phía sau được giới hạn trên train; không nên mô tả toàn bộ preprocessing là được học hoàn toàn từ train.

`data/interim/application_features.csv`, nếu có, là đầu ra của tiện ích `01_preprocess.py`, ngoài luồng bảng quan hệ chính.

## 3. Bảng từng client: processed/clients/

[02_partition_clients.py](../scripts/02_partition_clients.py) chia khách hàng từ application_train, sau đó phân bổ lịch sử theo SK_ID_CURR. Mỗi khách hàng chỉ thuộc một client.

| Tham số mặc định | Giá trị |
| --- | --- |
| strategy | dirichlet theo nhãn |
| num-clients | 10 |
| alpha | 0.5 |
| seed | 42 |
| min-per-class | 2 |
| test-size | 0.2 |

Đây là Dirichlet có ràng buộc số mẫu tối thiểu mỗi lớp/client. Script cũng hỗ trợ `--strategy semantic`. Train/test được chia cục bộ theo nhãn; tỷ lệ thực tế có thể khác 80/20 khi làm tròn hoặc lớp quá ít mẫu. Local-test lấy từ application_train, không phải application_test.

```text
processed/clients/
├── assignments.csv
├── partition_report.json
├── client_000/
│   ├── application_train.csv
│   ├── bureau.csv
│   ├── previous_application.csv
│   ├── installments_payments.csv
│   ├── POS_CASH_balance.csv
│   ├── credit_card_balance.csv
│   └── customer_split.csv
└── client_001/ ...
```

- `assignments.csv`: ánh xạ khách hàng sang client.
- `customer_split.csv`: SK_ID_CURR, train_mask và test_mask của khách hàng trong client.
- `partition_report.json`: cấu hình, schema, danh sách client, số mẫu theo nhãn/split, số hàng được và không được phân bổ.

Các bảng cùng loại giữ cùng schema giữa client. Hàng lịch sử không có khách hàng trong tập phân bổ được ghi nhận là `unassigned_rows`; chúng vẫn còn trong bảng interim.

## 4. Shared encoder: processed/shared_encoder.json

Shared encoder là **bộ tiền xử lý đặc trưng dùng chung giữa các client**, không phải encoder GNN được học bằng gradient. [02b_fit_encoder.py](../scripts/02b_fit_encoder.py) tạo file này; phần triển khai ở [shared_encoder.py](../src/preprocessing/shared_encoder.py).

### Mục đích

Mỗi loại node cần có cùng số chiều, thứ tự và ý nghĩa đặc trưng giữa các client. Ví dụ, một chiều one-hot biểu diễn nghề Accountant phải biểu diễn Accountant ở mọi client. Điều này giúp các mô hình có đầu vào tương thích để gộp trọng số bằng FedAvg.

### Dữ liệu dùng để fit

Encoder chỉ lấy hàng thuộc **khách hàng train của từng client**, gồm cả lịch sử của họ, rồi gộp thống kê giữa client. Sau khi fit, cùng quy tắc được áp dụng cho cả train và test; không học lại từ test.

Đây là bước tập trung trong mô phỏng FL hiện tại, chưa phải giao thức fit encoder phân tán bảo vệ riêng tư.

| Loại dữ liệu | Cách biến đổi |
| --- | --- |
| Numeric | Điền giá trị thiếu/không hữu hạn bằng mean train, rồi tính `(x - mean) / scale` |
| Numeric không có phương sai | Dùng scale=1 để tránh chia cho 0 |
| Cột hoàn toàn thiếu trên train | Giữ cột; thống kê mặc định mean=0, scale=1 |
| Categorical | One-hot theo vocabulary chung từ train |
| Categorical thiếu | Slot riêng cho missing |
| Categorical chưa gặp trên train | Slot riêng cho unknown |
| ID và TARGET | Không đưa vào đặc trưng |

JSON lưu định nghĩa cột, mean/scale hoặc vocabulary, số chiều từng bảng, số hàng fit, cấu hình partition, hash các file split và fingerprint nhận diện encoder.

Khi đổi partition/split hoặc dữ liệu đặc trưng train, cần fit lại encoder rồi xây lại graph. Cờ `is_orphan_prev` được thêm lúc xây graph, không fit/scale trong encoder. Encoder không thay thế các bảng client: xây graph cần cả bảng lẫn encoder.

## 5. Graph huấn luyện: processed/graphs/

[03_build_graphs.py](../scripts/03_build_graphs.py) đọc bảng client, split và encoder để tạo một file `client_XXX.npz` cho mỗi client, cùng `manifest.json`.

| Loại node | Bảng nguồn |
| --- | --- |
| customer | application_train |
| bureau | bureau |
| previous_application | previous_application |
| installment | installments_payments |
| pos_cash | POS_CASH_balance |
| credit_card | credit_card_balance |

Schema hiện tại là `orphan_fallback_v2`:

- Customer nối tới Bureau và Previous application.
- Previous application nối tới giao dịch có cha hợp lệ.
- Giao dịch thiếu Previous application nối về Customer bằng quan hệ has_orphan.
- Mỗi giao dịch hợp lệ có đúng một cạnh cha thuận; các quan hệ có thêm cạnh ngược.
- Ba loại giao dịch có đặc trưng nhị phân cuối is_orphan_prev, không chuẩn hóa.

### Nội dung NPZ

| Khóa | Nội dung |
| --- | --- |
| `x__<node>` | Ma trận đặc trưng node |
| `edge__<relation>` | Chỉ số cạnh theo loại quan hệ |
| `owner__<node>` | Chỉ số node Customer sở hữu node đó |
| `mapping__<node>__<key>` | Ánh xạ để đối chiếu node với dữ liệu nguồn |
| `is_orphan_prev__<node>` | Cờ orphan của ba loại giao dịch |
| `customer_ids`, `y` | ID và nhãn khách hàng |
| `train_mask`, `test_mask` | Phân tách khách hàng huấn luyện/đánh giá |
| `metadata` | Client, schema, encoder fingerprint, số node và thống kê quarantine |

Manifest ghi danh sách client, schema graph, encoder fingerprint và cấu hình partition. Graph builder kiểm tra hash split so với lúc fit encoder để phát hiện split bị đổi.

### Quarantine và orphan

`graphs/quarantine/client_XXX/` lưu giao dịch bị loại do ownership không hợp lệ, kèm dòng nguồn và lý do. Ví dụ: thiếu/không tìm thấy Customer trong client, hoặc Previous hiện có thuộc Customer khác. File có thể chỉ có header nếu không có hàng lỗi.

**Thiếu Previous đơn thuần không đồng nghĩa bị loại:** khi Customer hợp lệ, giao dịch được giữ bằng cạnh orphan. Bước xây graph không sửa các bảng nguồn.

Một graph chứa cả train và test; huấn luyện chỉ dùng thành phần thuộc khách hàng train. Chi tiết ở [module graph](../src/graph/README.md) và [protocol](../README.md).

## 6. Tạo lại dữ liệu

Chạy từ thư mục gốc sau khi cài dependencies:

```powershell
python scripts/01_prepare_tables.py
python scripts/02_partition_clients.py --num-clients 10 --alpha 0.5 --seed 42
python scripts/02b_fit_encoder.py
python scripts/03_build_graphs.py
```

Script dùng tham số CLI, không tự nạp YAML. Dùng `--help` để xem tùy chọn.

| Thay đổi | Các bước cần chạy lại |
| --- | --- |
| Raw hoặc chính sách chuẩn bị bảng | Chuẩn bị bảng → partition → encoder → graph → huấn luyện |
| Số client, chiến lược, seed hoặc split | Partition → encoder → graph → huấn luyện |
| Cách mã hóa đặc trưng | Encoder → graph → huấn luyện |
| Chỉ thay cấu trúc graph | Graph → huấn luyện; không cần refit encoder nếu bảng/split và cách mã hóa không đổi |
| Chỉ thay tham số huấn luyện | Huấn luyện, dùng lại graph tương thích |

Lưu riêng đầu ra cần giữ trước khi chạy lại cùng đường dẫn. Khi đổi số client, các thư mục/file cũ có thể còn trên đĩa; đối chiếu danh sách client trong báo cáo và manifest hiện tại.

## 7. Lưu trữ và Git

- Giữ raw làm nguồn tái tạo. Interim và processed có thể tạo lại nhưng tốn thời gian và đang được pipeline sử dụng.
- Không trộn split, encoder, graph và checkpoint từ các lần chạy khác nhau.
- Dữ liệu trong raw/interim/processed/external được [.gitignore](../.gitignore) bỏ qua khi thêm file mới; file đã được Git theo dõi cần xử lý riêng.
- Giữ README này trong Git; không cần README riêng cho từng client sinh tự động.
- Checkpoint, dự đoán và metric nằm ở [results](../results/README.md).

Hướng dẫn huấn luyện, sweep và kiểm tra dữ liệu: [scripts/README.md](../scripts/README.md).
