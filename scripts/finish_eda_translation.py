# Hoàn thiện bản dịch từ cache và nội dung cục bộ; đọc/ghi ngay khi chạy, cần notebook nguồn và cache.
"""Finish the Vietnamese Markdown translation without external services."""
from pathlib import Path
import json
import re
import nbformat

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "notebooks" / "01_eda.ipynb"
dst = ROOT / "notebooks" / "01_eda_vi.ipynb"
cache_path = ROOT / "notebooks" / ".01_eda_vi_translation_cache.json"
cache = json.loads(cache_path.read_text("utf-8"))

# Reader-facing local translations for cells not completed by the cached pass.
T = {
0: "<h1>PHÂN TÍCH DỮ LIỆU KHÁM PHÁ – RỦI RO VỠ NỢ HOME CREDIT</h1>",
2: """<h2>Bộ dữ liệu</h2>

<h3>Tổng quan dữ liệu</h3>

<b>Home Credit Group</b> cung cấp bộ dữ liệu lớn để hỗ trợ xây dựng mô hình dự đoán rủi ro vỡ nợ. Các bảng quan hệ chứa thông tin nhân khẩu học, lịch sử tín dụng tại tổ chức khác và lịch sử tại Home Credit. Đây là dữ liệu mất cân bằng vì số khách hàng vỡ nợ ít hơn nhiều số khách hàng không vỡ nợ.<br><br>
Bộ dữ liệu có thể tải từ Kaggle: <a href=https://www.kaggle.com/c/home-credit-default-risk/data>Home Credit Default Risk Dataset</a><br><br>

<h3>Thông số dữ liệu</h3>
<pre>Có tổng cộng 10 tệp CSV:
HomeCredit_columns_description.csv - 36.51 KB
POS_CASH_balance.csv - 374.51 MB
application_test.csv - 25.34 MB
application_train.csv - 158.44 MB
bureau.csv - 162.14 MB
bureau_balance.csv - 358.19 MB
credit_card_balance.csv - 404.91 MB
installments_payments.csv - 689.62 MB
previous_application.csv - 386.21 MB
sample_submission.csv - 523.63 KB</pre>""",
4: """<br><br><h4>Mô tả ngắn gọn từng bảng</h4>
<h5>application_{train|test}.csv</h5><ul><li>Bảng chính gồm tập huấn luyện có TARGET và tập kiểm tra không có TARGET; mỗi dòng là một hồ sơ vay.</li></ul>
<h5>bureau.csv</h5><ul><li>Các khoản tín dụng trước đây tại tổ chức khác được báo cáo cho Cục Thông tin Tín dụng.</li></ul>
<h5>bureau_balance.csv</h5><ul><li>Số dư hằng tháng của từng khoản tín dụng trong bureau.</li></ul>
<h5>POS_CASH_balance.csv</h5><ul><li>Số dư hằng tháng của khoản vay POS và vay tiền mặt trước đây tại Home Credit.</li></ul>
<h5>credit_card_balance.csv</h5><ul><li>Số dư hằng tháng của thẻ tín dụng trước đây tại Home Credit.</li></ul>
<h5>previous_application.csv</h5><ul><li>Các hồ sơ vay Home Credit trước đây.</li></ul>
<h5>installments_payments.csv</h5><ul><li>Lịch sử trả nợ của các khoản tín dụng trước đây.</li></ul>
<h5>HomeCredit_columns_description.csv</h5><ul><li>Mô tả các cột trong các tệp dữ liệu.</li></ul>
Nguồn: Home Credit Group (<a href=https://www.kaggle.com/c/home-credit-default-risk/data>Kaggle</a>)""",
9: "<h3>Định nghĩa các hàm tiện ích</h3>",
56: "##### Quan sát và kết luận:\n\nCODE_GENDER có 4 dòng XNA không hợp lý và có thể loại bỏ. Nữ chiếm 65,8% hồ sơ, nam chiếm 34,2%, nhưng tỷ lệ vỡ nợ của nam cao hơn (10,14% so với 7%).",
57: "<b><u>Phân phối biến phân loại FLAG_EMP_PHONE</u></b>\n\nCột này cho biết khách hàng có cung cấp số điện thoại cơ quan hay không; 1 là Có và 0 là Không.",
63: "<b><u>Phân phối biến phân loại NAME_EDUCATION_TYPE</u></b>\n\nĐặc trưng này mô tả trình độ học vấn cao nhất của khách hàng.",
65: "##### Quan sát và kết luận:\n\nKhoảng 71% khách hàng chỉ học tới trung học phổ thông/trung học chuyên biệt và 24,34% có trình độ đại học. Nhóm chỉ học tới trung học cơ sở có tỷ lệ vỡ nợ cao nhất; nhóm đại học thấp hơn, còn nhóm có học vị hàn lâm thấp nhất nhưng số lượng rất nhỏ.",
68: "##### Quan sát và kết luận:\n\nLao động phổ thông là nghề phổ biến nhất, gần 26%. Lao động tay nghề thấp có tỷ lệ vỡ nợ cao nhất (khoảng 17,5%), tiếp theo là tài xế, phục vụ, bảo vệ và đầu bếp. Các nghề có kỹ năng trung bình hoặc cao có tỷ lệ thấp hơn.",
72: "<b><u>Phân phối REG_CITY_NOT_LIVE_CITY, REG_CITY_NOT_WORK_CITY và LIVE_CITY_NOT_WORK_CITY</u></b><br><br>\nCác cột này cho biết địa chỉ thường trú có khác địa chỉ liên hệ hoặc địa chỉ làm việc ở cấp vùng/thành phố hay không. Giá trị 1 là khác và 0 là giống nhau.",
74: """##### Quan sát và kết luận:

Chỉ một bộ phận nhỏ người nộp đơn có các địa chỉ không trùng khớp: 7,52% khác địa chỉ liên hệ theo vùng, 23,05% khác địa chỉ làm việc theo vùng và 17,96% khác địa chỉ liên hệ theo thành phố. Trong cả ba trường hợp, nhóm có địa chỉ khác nhau đều có tỷ lệ vỡ nợ cao hơn nhóm có cùng địa chỉ; vì vậy, sự không trùng khớp địa chỉ có thể là một tín hiệu về rủi ro vỡ nợ.""",
75: "<b><u>Phân phối biến phân loại FLAG_DOCUMENT_3</u></b>",
76: "Cột này cho biết người nộp đơn có nộp một tài liệu bắt buộc hay không. Giá trị 0 nghĩa là đã cung cấp tài liệu và 1 nghĩa là chưa cung cấp.",
78: """##### Quan sát và kết luận:

Khoảng 71% khách hàng chưa cung cấp tài liệu này và 29% đã cung cấp. Nhóm đã cung cấp lại có tỷ lệ vỡ nợ cao hơn. Do không có mô tả chi tiết về loại tài liệu, chúng ta chưa thể giải thích chắc chắn nguyên nhân của hiện tượng này.""",
79: "#### Trực quan hóa các biến liên tục",
80: """<b><u>Phân phối biến liên tục tuổi của người nộp đơn</u></b>

Tuổi trong bộ dữ liệu được biểu diễn theo ngày. Để dễ phân tích và diễn giải, chúng ta tạo một biến mới biểu diễn tuổi theo năm.""",
82: """##### Quan sát và kết luận:

Phân phối tuổi của nhóm vỡ nợ đạt đỉnh gần 30 tuổi và nhìn chung dịch về phía trẻ hơn so với nhóm không vỡ nợ. Box plot cũng cho thấy mọi phân vị tuổi của nhóm vỡ nợ đều thấp hơn; phân vị 75% xấp xỉ 49 tuổi, so với khoảng 54 tuổi ở nhóm không vỡ nợ. Nhìn chung, khách hàng vỡ nợ thường trẻ hơn.""",
83: "<b><u>Phân phối các biến liên tục thuộc nhóm DAYS</u><b>",
84: """<b>DAYS_EMPLOYED</b><br>

Đặc trưng này biểu thị số ngày khách hàng đã làm việc tính tới ngày nộp đơn. Ta chuyển số ngày sang số năm để dễ diễn giải.""",
86: """##### Quan sát và kết luận:

Cột DAYS_EMPLOYED có các giá trị bất hợp lý bằng 365243. Sau khi xem box plot, nhóm vỡ nợ có thời gian làm việc ngắn hơn nhóm không vỡ nợ ở cả các phân vị 25%, 50% và 75%.""",
87: """<b>DAYS_ID_PUBLISH</b><br>

Cột này cho biết số ngày tính từ lần gần nhất khách hàng thay đổi giấy tờ tùy thân dùng để đăng ký khoản vay.""",
89: """##### Quan sát và kết luận:

Tương tự DAYS_REGISTRATION, nhóm vỡ nợ thường mới thay đổi giấy tờ tùy thân gần đây hơn. Tất cả các phân vị về số ngày của nhóm không vỡ nợ đều lớn hơn.""",
90: "<b><u>Phân phối các biến EXT_SOURCE</u></b>",
91: "Ba cột EXT_SOURCE chứa điểm số đã chuẩn hóa trong khoảng từ 0 đến 1, được tổng hợp từ các nguồn khác nhau.",
93: """##### Quan sát và kết luận:

Cả ba biến EXT_SOURCE đều có xu hướng tương tự: nhóm vỡ nợ có điểm thấp hơn rõ rệt, còn nhóm không vỡ nợ tập trung nhiều hơn ở các giá trị cao. Trung vị của nhóm vỡ nợ gần bằng hoặc thấp hơn phân vị 25% của nhóm không vỡ nợ. EXT_SOURCE_1 và EXT_SOURCE_3 phân tách hai nhóm tốt hơn EXT_SOURCE_2; đây là những đặc trưng có khả năng phân biệt tuyến tính tốt nhất được quan sát tới thời điểm này.""",
94: """<b><u>Phân phối FLOORSMAX_AVG và FLOORSMIN_MODE</u></b>

Các cột này là điểm chuẩn hóa tương ứng với số tầng tối đa trung bình và mode của số tầng tối thiểu trong tòa nhà nơi người nộp đơn sinh sống.""",
97: """##### Quan sát và kết luận

Nhóm vỡ nợ có trung vị FLOORSMAX_AVG thấp hơn nhóm không vỡ nợ; phân vị 25% của nhóm không vỡ nợ thậm chí gần cao hơn trung vị của nhóm vỡ nợ. Nhóm không vỡ nợ cũng có FLOORSMIN_MODE cao hơn, đặc biệt ở phân vị 75%. Hai đặc trưng này có thể hữu ích cho mô hình.""",
99: """##### Mô tả

Bảng này chứa toàn bộ lịch sử tín dụng trước đây của khách hàng tại các tổ chức tài chính khác Home Credit, được báo cáo cho Cục Thông tin Tín dụng.""",
102: """<h5>Quan sát và kết luận:</h5>

Bảng bureau.csv có gần 1,7 triệu bản ghi và 17 đặc trưng. SK_ID_BUREAU định danh khoản vay trước đây tại tổ chức khác, còn SK_ID_CURR định danh hồ sơ vay hiện tại tại Home Credit. Có khoảng 305 nghìn SK_ID_CURR duy nhất: 263 nghìn thuộc tập huấn luyện và 42,3 nghìn thuộc tập kiểm tra. Điều này cho thấy một số khách hàng hiện tại không có lịch sử tín dụng được ghi nhận tại Cục Thông tin Tín dụng.""",
105: """##### Quan sát và kết luận:

Trong 17 đặc trưng có 7 đặc trưng chứa giá trị NaN. AMT_ANNUITY có tỷ lệ thiếu cao nhất, trên 70%.""",
106: "<b>Ghép biến TARGET từ application_train vào bảng bureau.</b>",
110: """##### Quan sát và kết luận:

Heatmap Phi-K cho thấy CREDIT_TYPE có liên hệ nhất định với CREDIT_ACTIVE. Các biến phân loại nhìn chung không liên hệ mạnh với TARGET, đặc biệt là CREDIT_CURRENCY.""",
114: """##### Quan sát và kết luận:

Phần lớn các đặc trưng trong bureau có tương quan thấp. Một số cặp tương quan cao gồm DAYS_CREDIT–DAYS_CREDIT_UPDATE, DAYS_ENDDATE_FACT–DAYS_CREDIT_UPDATE, AMT_CREDIT_SUM–AMT_CREDIT_SUM_DEBT và DAYS_ENDDATE_FACT–DAYS_CREDIT. Tương quan với TARGET nhìn chung thấp, ngoại trừ DAYS_CREDIT, nên chưa thấy quan hệ tuyến tính trực tiếp rõ ràng.""",
115: "#### Trực quan hóa các biến phân loại\n\nTa trực quan hóa một số biến phân loại của bảng bureau và xem xét mối liên hệ của chúng với TARGET.",
116: "<b><u>Phân phối biến phân loại CREDIT_ACTIVE</u></b>",
117: "Cột này mô tả trạng thái khoản vay trước đây được Cục Thông tin Tín dụng báo cáo.",
119: """##### Quan sát và kết luận:

Phần lớn các khoản vay trước đây đã đóng (62,63%), tiếp theo là khoản vay đang hoạt động (36,98%); khoản vay đã bán và nợ xấu rất ít. Nhóm nợ xấu có tỷ lệ vỡ nợ cao nhất, khoảng 20%, tiếp đến là khoản vay đã bán và đang hoạt động. Khoản vay đã đóng có tỷ lệ vỡ nợ thấp nhất, phù hợp với kỳ vọng về một lịch sử tín dụng tốt.""",
121: "<u><b>Phân phối các biến liên tục thuộc nhóm DAYS</b></u>",
122: """<b>DAYS_CREDIT</b>

Cột này biểu thị số ngày từ lúc khách hàng đăng ký khoản tín dụng tại Cục Thông tin Tín dụng đến hồ sơ hiện tại. Ta chuyển số ngày sang năm để dễ diễn giải.""",
124: """##### Quan sát và kết luận:

Nhóm vỡ nợ tập trung nhiều hơn ở các khoản tín dụng được mở gần đây; phân phối của nhóm này dịch về bên trái. Box plot cũng cho thấy YEARS_CREDIT của nhóm vỡ nợ thường thấp hơn nhóm không vỡ nợ.""",
125: """<b>DAYS_CREDIT_ENDDATE</b>

Cột này cho biết thời hạn còn lại của khoản tín dụng tại thời điểm khách hàng đăng ký vay ở Home Credit.""",
127: """##### Quan sát và kết luận:

Giá trị nhỏ nhất của DAYS_CREDIT_ENDDATE lên tới 42.060 ngày, tương đương khoảng 115 năm, nên rất có thể là dữ liệu sai. Các giá trị này cần được xử lý trong bước tiền xử lý.""",
128: """<b>DAYS_ENDDATE_FACT</b>

Cột này cho biết khoản tín dụng đã kết thúc cách ngày đăng ký vay tại Home Credit bao nhiêu ngày; chỉ áp dụng cho các khoản tín dụng đã đóng.""",
130: """##### Quan sát và kết luận:

Giá trị nhỏ nhất khoảng 42.023 ngày (gần 115 năm) là bất hợp lý và cần được xử lý. Box plot cho thấy khoản tín dụng trước đây của nhóm vỡ nợ thường kết thúc gần ngày đăng ký hiện tại hơn so với nhóm không vỡ nợ.""",
131: """<b>DAYS_CREDIT_UPDATE</b>

Cột này cho biết thông tin tín dụng từ Cục Thông tin Tín dụng được cập nhật cách ngày đăng ký hiện tại bao nhiêu ngày.""",
133: """##### Quan sát và kết luận:

Giá trị ở phân vị 0% bất thường trong khi các phân vị còn lại hợp lý, vì vậy cần loại hoặc sửa giá trị này. Nhóm vỡ nợ có thông tin tín dụng được cập nhật gần đây hơn, thể hiện qua trung vị và phân vị 75% thấp hơn nhóm không vỡ nợ.""",
135: "##### Mô tả\n\nBảng này chứa số dư hằng tháng của từng khoản tín dụng trước đây tại các tổ chức tài chính khác Home Credit.",
139: """##### Quan sát và kết luận

Bảng bureau_balance.csv có khoảng 27,29 triệu dòng và 3 cột, ghi lại trạng thái hằng tháng của từng khoản vay trước đây. STATUS có 8 mã: C là đã đóng, X là không rõ, 0 là không quá hạn, 1–5 biểu thị các mức số ngày quá hạn tăng dần. Lịch sử xa nhất là 96 tháng, tương đương 8 năm.""",
143: "##### Mô tả\n\nBảng này chứa thông tin về các hồ sơ vay trước đây của khách hàng tại Home Credit.",
146: """##### Quan sát và kết luận:

Bảng previous_application.csv có khoảng 1,67 triệu bản ghi, 37 đặc trưng và 1,67 triệu SK_ID_PREV duy nhất, tương ứng mỗi dòng là một hồ sơ trước đây. Có khoảng 338 nghìn SK_ID_CURR; trong đó khoảng 291 nghìn thuộc tập huấn luyện và 47,8 nghìn thuộc tập kiểm tra.""",
149: """##### Quan sát và kết luận

Trong 37 đặc trưng có 16 cột chứa NaN. RATE_INTEREST_PRIMARY và RATE_INTEREST_PRIVILEGED thiếu gần như toàn bộ (khoảng 99,6%); một số cột ngày và AMT_DOWN_PAYMENT cũng có tỷ lệ thiếu đáng kể.""",
150: "<b>Ghép biến TARGET từ application_train vào bảng previous_application.</b>",
154: """##### Quan sát và kết luận:

Ma trận Phi-K cho thấy một số biến phân loại có liên hệ mạnh với nhau, đặc biệt NAME_CONTRACT_STATUS với CODE_REJECT_REASON và NAME_PRODUCT_TYPE với PRODUCT_COMBINATION. Mối liên hệ với TARGET nhìn chung yếu.""",
158: """##### Quan sát và kết luận:

Phần lớn biến số có tương quan thấp. Các nhóm tương quan cao chủ yếu là những đại lượng cùng bản chất, chẳng hạn AMT_APPLICATION–AMT_CREDIT, AMT_ANNUITY–AMT_CREDIT và các cột DAYS liên quan đến thời điểm đến hạn/kết thúc. Tương quan tuyến tính với TARGET không đáng kể.""",
159: "#### Trực quan hóa các biến phân loại\n\nTa trực quan hóa một số biến phân loại trong previous_application và xem xét ảnh hưởng của chúng tới TARGET.",
160: "<b><u>Phân phối biến phân loại NAME_CONTRACT_TYPE</u></b>",
161: "Cột này mô tả loại hợp đồng của khoản vay trước đây tại Home Credit.",
163: """##### Quan sát và kết luận:

Khoản vay tiền mặt và vay tiêu dùng chiếm gần như toàn bộ hồ sơ trước đây; khoản vay quay vòng ít hơn nhiều. Nhóm từng đăng ký vay tiền mặt có tỷ lệ vỡ nợ cao nhất, còn vay tiêu dùng thấp hơn.""",
164: """<b><u>Phân phối biến phân loại NAME_CONTRACT_STATUS</u></b>

Cột này mô tả trạng thái của hồ sơ vay trước đây.""",
166: """##### Quan sát và kết luận:

Phần lớn hồ sơ trước đây được chấp thuận, tiếp theo là bị hủy và bị từ chối. Nhóm có hồ sơ trước đây bị từ chối có tỷ lệ vỡ nợ cao nhất, còn nhóm được chấp thuận có tỷ lệ thấp nhất; kết quả này phù hợp với kỳ vọng về mức độ tín nhiệm.""",
167: """<b><u>Phân phối biến phân loại CODE_REJECT_REASON</u></b>

Cột này mô tả lý do hồ sơ vay trước đây tại Home Credit bị từ chối.""",
169: """##### Quan sát và kết luận:

XAP là mã phổ biến nhất (khoảng 81%), còn HC đứng thứ hai với 10,33%. Nhóm bị từ chối bởi mã SCOFR có tỷ lệ vỡ nợ cao nhất (khoảng 21%), tiếp theo là LIMIT và HC. XAP chỉ có khoảng 7,5% vỡ nợ và thấp thứ hai sau SYSTEM.""",
170: """<b><u>Phân phối biến phân loại CHANNEL_TYPE</u></b>

Cột này mô tả kênh tiếp cận khách hàng cho khoản vay trước đây.""",
172: """##### Quan sát và kết luận

Khoảng 42,47% hồ sơ đến từ văn phòng tín dụng và tiền mặt, tiếp theo là kênh toàn quốc với 29,93%. Kênh AP+ (vay tiền mặt) có tỷ lệ vỡ nợ cao nhất, khoảng 13%, còn đại lý ô tô thấp nhất, khoảng 5%.""",
173: """<b><u>Phân phối biến phân loại PRODUCT_COMBINATION</u></b>

Cột này cung cấp thông tin về tổ hợp sản phẩm của các hồ sơ trước đây.""",
175: """##### Quan sát và kết luận

Ba tổ hợp phổ biến nhất là Cash, POS household with interest và POS mobile with interest, chiếm khoảng 50% hồ sơ. Các nhóm Cash Street: mobile, Cash X-sell: high, Cash Street: high và Card Street có tỷ lệ vỡ nợ cao nhất (khoảng 11–11,5%); POS Industry without interest thấp nhất (khoảng 4,5%).""",
177: "<u><b>Phân phối các biến liên tục thuộc nhóm DAYS</b></u>",
178: "<b>DAYS_DECISION</b>\n\nCột này cho biết quyết định về hồ sơ trước đây được đưa ra cách hồ sơ hiện tại bao nhiêu ngày.",
180: "##### Quan sát và kết luận\n\nQuyết định về hồ sơ trước đây của nhóm vỡ nợ thường được đưa ra gần đây hơn so với nhóm không vỡ nợ.",
181: "<b>DAYS_FIRST_DRAWING</b>\n\nCột này cho biết lần giải ngân đầu tiên của hồ sơ trước đây cách hồ sơ hiện tại bao nhiêu ngày.",
183: """##### Quan sát và kết luận:

Nhiều giá trị DAYS_FIRST_DRAWING là bất hợp lý ngay từ phân vị 7% và cần được xử lý. Sau khi loại các điểm sai, nhóm vỡ nợ thường có lần giải ngân đầu tiên gần đây hơn; phân vị 75% cũng thấp hơn đáng kể.""",
184: """<b>DAYS_FIRST_DUE, DAYS_LAST_DUE_1ST_VERSION, DAYS_LAST_DUE và DAYS_TERMINATION</b>

Các cột này cho biết một số sự kiện của khoản vay trước đây đã xảy ra cách hồ sơ hiện tại bao nhiêu ngày.""",
186: "##### Quan sát và kết luận\n\nCác cột DAYS đều chứa một số giá trị bất hợp lý. Cần thay thế hoặc xử lý chúng để tránh ảnh hưởng tới mô hình.",
188: "##### Mô tả\n\nBảng này ghi lại lịch sử thanh toán của từng khoản vay trước đây tại Home Credit, gồm số tiền phải trả và số tiền khách hàng thực tế thanh toán cho mỗi kỳ.",
191: """##### Quan sát và kết luận

Bảng installments_payments.csv có khoảng 13,6 triệu dòng và 8 đặc trưng. Có 997 nghìn khoản vay trước đây thuộc 339 nghìn khách hàng hiện tại; 291 nghìn khách hàng thuộc tập huấn luyện và 47,9 nghìn thuộc tập kiểm tra. Phần lớn khách hàng trong hai tập từng có khoản vay tại Home Credit.""",
194: "##### Quan sát và kết luận\n\nChỉ 2 trong 8 cột chứa NaN và tỷ lệ thiếu rất nhỏ, khoảng 0,02%, nên không đáng lo ngại.",
195: "<b>Ghép biến TARGET từ application_train vào bảng installments_payments.</b>",
200: """##### Quan sát và kết luận:

Hai cặp tương quan cao là AMT_INSTALMENT–AMT_PAYMENT và DAYS_INSTALMENT–DAYS_ENTRY_PAYMENT, tương ứng với số tiền/ngày đến hạn so với số tiền/ngày thực trả. Chúng hữu ích để tạo các đặc trưng mới ít tương quan hơn. Tương quan với TARGET không rõ rệt.""",
202: "Trước tiên, nhóm dữ liệu theo SK_ID_PREV và lấy trung bình để thu được một dòng đại diện cho mỗi khoản vay trước đây.",
204: "<b><u>Phân phối biến liên tục DAYS_INSTALMENT</u></b>\n\nCột này ghi lại ngày đến hạn của kỳ thanh toán cho khoản tín dụng trước đây.",
206: "<b><u>Phân phối biến liên tục DAYS_ENTRY_PAYMENT</u></b>\n\nCột này ghi lại ngày khách hàng thực tế thanh toán kỳ trả nợ trước đây.",
208: "##### Quan sát và kết luận\n\nNhóm vỡ nợ có lần thanh toán gần nhất gần ngày đăng ký hơn ở mọi phân vị. Nhóm không vỡ nợ thường có khoảng cách từ lần thanh toán trước tới ngày đăng ký dài hơn.",
210: "##### Mô tả\n\nBảng này chứa ảnh chụp số dư hằng tháng của các khoản vay POS và vay tiền mặt trước đây, gồm trạng thái hợp đồng và số kỳ trả góp còn lại.",
213: "##### Quan sát và kết luận\n\nBảng có khoảng 10 triệu dòng và 8 cột, mỗi dòng là trạng thái theo tháng của một khoản vay POS hoặc tiền mặt trước đây. Có 936 nghìn khoản vay thuộc 337 nghìn khách hàng; 289 nghìn thuộc tập huấn luyện và 47,8 nghìn thuộc tập kiểm tra.",
216: "##### Quan sát và kết luận\n\nChỉ 2 trong 8 cột chứa NaN: số kỳ trả góp còn lại và thời hạn khoản vay. Tỷ lệ thiếu chỉ khoảng 0,26%, nên không đáng lo ngại.",
217: "<b>Ghép biến TARGET từ application_train vào bảng POS_CASH_balance.</b>",
222: "##### Quan sát và kết luận:\n\nCNT_INSTALMENT và CNT_INSTALMENT_FUTURE có tương quan vừa phải. Tương quan của các đặc trưng với TARGET rất thấp, cho thấy không có quan hệ tuyến tính rõ ràng.",
224: "Trước tiên, nhóm dữ liệu theo SK_ID_PREV và lấy trung bình để thu được một dòng đại diện cho mỗi khoản vay trước đây.",
226: "<b><u>Phân phối biến liên tục CNT_INSTALMENT_FUTURE</u></b>\n\nCột này mô tả số kỳ trả góp còn lại của khoản tín dụng trước đây.",
228: "##### Quan sát và kết luận\n\nỞ các phân vị trên 50%, CNT_INSTALMENT_FUTURE của nhóm vỡ nợ thường cao hơn nhóm không vỡ nợ; râu trên của box plot cũng cao hơn. Nhóm vỡ nợ có xu hướng còn nhiều kỳ phải trả hơn.",
230: "##### Mô tả\n\nBảng này chứa dữ liệu hằng tháng về một hoặc nhiều thẻ tín dụng trước đây của khách hàng tại Home Credit, gồm số dư, hạn mức, số tiền rút và các thông tin liên quan.",
233: """##### Quan sát và kết luận

Bảng credit_card_balance.csv có khoảng 3,84 triệu dòng và 23 đặc trưng. Có 104,3 nghìn thẻ thuộc 103,5 nghìn khách hàng, cho thấy phần lớn chỉ có một thẻ. Khoảng 86,9 nghìn khách hàng thuộc tập huấn luyện và 16,6 nghìn thuộc tập kiểm tra; chỉ 86,9 nghìn trong 307 nghìn khách hàng huấn luyện từng có thẻ tại Home Credit.""",
236: "##### Quan sát và kết luận\n\nTrong 23 đặc trưng có 9 cột chứa NaN. Bảy cột có tỷ lệ thiếu gần 20%, chủ yếu liên quan đến số tiền và số lần rút; hai cột còn lại liên quan đến thống kê trả góp.",
237: "<b>Ghép biến TARGET từ application_train vào bảng credit_card_balance.</b>",
242: """##### Quan sát và kết luận:

Các nhóm tương quan cao gồm AMT_RECEIVABLE_PRINCIPAL, AMT_RECIVABLE, AMT_TOTAL_RECEIVABLE và AMT_BALANCE; ngoài ra AMT_PAYMENT_TOTAL_CURRENT tương quan cao với AMT_PAYMENT_CURRENT. Đây là các đại lượng gần cùng bản chất. Tương quan với TARGET không đáng kể.""",
244: "Trước tiên, nhóm dữ liệu theo SK_ID_PREV và lấy trung bình để thu được một dòng đại diện cho mỗi khoản vay trước đây.",
246: "<b><u>Phân phối biến liên tục AMT_BALANCE</u></b>\n\nCột này biểu thị số dư trung bình mà khách hàng thường có trên tài khoản thẻ tín dụng trước đây.",
248: "##### Quan sát và kết luận\n\nNhóm vỡ nợ có AMT_BALANCE cao hơn ở tất cả các phân vị và cả hai râu của box plot. Điều này có thể liên quan tới hạn mức hoặc số tiền tín dụng cao hơn.",
249: "##### Quan sát và kết luận:\n\nNhóm vỡ nợ cũng có khoản trả góp tối thiểu hằng tháng cao hơn, phản ánh xu hướng chi tiêu và vay mượn nhiều hơn nhóm không vỡ nợ.",
250: "<b><u>Phân phối biến liên tục AMT_TOTAL_RECEIVABLE</u></b>\n\nCột này mô tả tổng số tiền phải thu trung bình của khoản tín dụng trước đây.",
252: "##### Quan sát và kết luận\n\nNhóm vỡ nợ thường có tổng số tiền phải thu cao hơn, có thể do từng vay số tiền lớn hơn. Hàm mật độ của nhóm không vỡ nợ đạt đỉnh cao hơn nhiều ở vùng giá trị thấp.",
253: "<b><u>Phân phối biến liên tục CNT_INSTALMENT_MATURE_CUM</u></b>\n\nCột này mô tả số kỳ trả góp trung bình đã thanh toán của các khoản tín dụng trước đây.",
255: "##### Quan sát và kết luận\n\nNhóm không vỡ nợ thường có phạm vi số kỳ đã thanh toán cao hơn. Điều này có thể phản ánh hành vi vỡ nợ: khách hàng vỡ nợ thường thanh toán ít kỳ hơn cho khoản tín dụng trước đây.",
256: "## Kết luận từ EDA",
}

HEADINGS = {
    "#### Basic Stats": "#### Thống kê cơ bản",
    "<h4>Basic Stats</h4>": "<h4>Thống kê cơ bản</h4>",
    "#### NaN Columns and Percentages": "#### Các cột NaN và tỷ lệ phần trăm",
    "<h4>NaN Columns and Percentages</h4>": "<h4>Các cột NaN và tỷ lệ phần trăm</h4>",
    "#### Phi-K Matrix": "#### Ma trận Phi-K",
    "#### Correlation Matrix of Features": "#### Ma trận tương quan giữa các đặc trưng",
    "#### Plotting Continuous Variables": "#### Trực quan hóa các biến liên tục",
    "#### Plotting Categorical Variables": "#### Trực quan hóa các biến phân loại",
    "##### Description": "##### Mô tả",
}

nb = nbformat.read(src, 4)
split_text = __import__("translate_eda_vi").split_text
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "markdown":
        continue
    if i in T:
        cell.source = T[i]
        continue
    chunks = split_text(cell.source)
    if all((not re.search(r"[A-Za-z]{3}", c)) or c in cache for c in chunks):
        cell.source = "".join(cache.get(c, c) for c in chunks)
    elif cell.source.strip() in HEADINGS:
        cell.source = HEADINGS[cell.source.strip()]

nbformat.write(nb, dst)
print(dst)
