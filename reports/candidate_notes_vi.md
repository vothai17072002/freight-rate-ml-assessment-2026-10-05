# Ghi chú để hiểu và trình bày bài assessment

Đây là bài dự đoán **giá cước cho từng load**, target là `posted_rate`. File cần
nộp ghi `predicted_rate`; không phải dự đoán tổng doanh thu hay giá trung bình
theo thành phố. Đơn vị dollar được thể hiện trong scorer và báo cáo.

## Những gì đã làm

| Hạng mục | Cách làm |
|---|---|
| Dữ liệu có nhãn | 48.000 dòng từ 01/01 đến 31/10/2025 |
| Dữ liệu phải dự đoán | 12.000 dòng từ 01/11 đến 31/12/2025; không có giá thật |
| Chọn mô hình | Train tháng 1-6, so sánh trên tháng 7-8 |
| Holdout | Train lại tháng 1-8; chấm lựa chọn đã khóa trên tháng 9-10 |
| Model nộp | Giữ cấu hình đã chọn, fit lại toàn bộ tháng 1-10 |
| December chart | Model riêng dùng đúng sáu cột feature có trong input biểu đồ |

Tại sao không chỉ chia ngẫu nhiên: bài cần dự đoán thời gian tương lai. Nếu trộn
ngẫu nhiên, mô hình có thể học từ các thời điểm sau thời điểm cần dự đoán. Cách
chia theo thời gian mô phỏng bài toán được giao tốt hơn. Tháng 9-10 không được
dùng để đổi model, feature, số iteration hoặc loss sau khi đã chấm.

## Cách hiểu mô hình

Model chính là `catboost_full_noquote_log_rpm`; model cho biểu đồ là
`catboost_reduced_log_rpm`. Cả hai đều dùng CatBoost với 700 iterations, depth 6,
learning rate 0,055 và seed 42.

Khi train, model học:

```text
y = log(posted_rate / distance)
```

Khi predict:

```text
predicted_rate = exp(model_output) × distance
```

`posted_rate / distance` là **target biến đổi trong lúc train**, không phải một
cột đầu vào lấy từ giá thật lúc predict. Phép log giúp giảm ảnh hưởng tương đối
của một số giá rất lớn; vẫn chọn model bằng RMSE tính lại theo dollar. Mô hình
tree không đảm bảo ngoại suy một xu hướng mới ngoài thời gian đã train.

Feature chính gồm khoảng cách và các phép biến đổi của nó, trọng lượng đã xử lý,
thiết bị, pickup/delivery, lane có hướng, ngày và các chu kỳ ngày. Model chính
dùng thêm tọa độ được cung cấp, chênh lệch tọa độ và `market_index`. Model giảm
feature không dùng các trường bổ sung này. `load_id` chỉ dùng ghép kết quả đúng
load, không đưa vào model.

## Chất lượng dữ liệu và target proxy

- Train có 300 dòng thiếu weight, 292 dòng weight không dương; validation lần
  lượt có 165 và 145 dòng. Chuyển weight không dương thành missing, thêm cờ lỗi,
  rồi điền median **chỉ học từ phần train**. Không lấy trị tuyệt đối weight âm.
- `market_index` thiếu 374 dòng train và 249 dòng validation. Dùng median đã fit
  trên train và cờ missing; không fit lại median theo validation.
- Không có duplicate ID hay duplicate toàn dòng trong các input load. Không
  xóa load để né yêu cầu dự đoán đủ 12.000 dòng.
- Validation có tám thành phố mới: Allentown, Charlotte, Chicago, Jackson,
  Knoxville, Laredo, Norfolk, San Diego. Có 1.447 load liên quan đến thành phố
  mới và 1.461 load có lane có hướng chưa xuất hiện trong train. CatBoost hỗ trợ
  category mới; tọa độ giúp cung cấp thêm thông tin nhưng không bảo đảm độ chính
  xác. Kiểm tra latitude/longitude trong giới hạn không chứng minh tọa độ đúng
  với thành phố thật.
- `quote_signal` có thể là tín hiệu quote có trước giá, hoặc proxy được tính từ
  target; đề không cho định nghĩa hay timestamp để kết luận. Quan sát tháng 1-8
  đã cho thấy tương quan với rate per mile đổi dấu theo tháng và gần bằng zero
  ở tháng 8. Loại trường này khỏi hai model nộp vì tính ổn định chưa đủ rõ.
  **Không tuyên bố đã chứng minh data leakage.** Các model có quote chỉ lưu để
  tham khảo, không dùng holdout tháng 9-10 để chọn lại chính sách này.
- Model chính vẫn dùng `market_index` vì input validation có trường này. Đây
  là giả định nó có trước lúc biết `posted_rate`; nếu triển khai thật phải xác
  nhận cách tính và thời điểm có dữ liệu. Việc cột xuất hiện trong CSV tự nó
  không chứng minh prediction-time availability.
- Giữ giá thật rất cao, gồm maximum $25.533,00. Giá cao không tự chứng minh sai.
  RMSE sẽ chịu ảnh hưởng mạnh của các load này, nên cần xem thêm MAE và WAPE.

## Đọc kết quả đúng phạm vi

Holdout gồm **9.523 load tháng 9-10/2025**, được dự đoán bằng model đã fit tháng
1-8, không phải model final đã fit cả tháng 1-10:

| Model | RMSE ($) | MAE ($) | WAPE | R² |
|---|---:|---:|---:|---:|
| Baseline distance × median rate per mile theo equipment | 670,66 | 229,10 | 9,58% | 0,8069 |
| Chính | 637,66 | 122,33 | 5,11% | 0,8254 |
| Giảm feature | 637,61 | 124,13 | 5,19% | 0,8254 |

RMSE khoảng $638 **không có nghĩa mọi load đều sai $638**. Nó bình phương sai
số nên chịu ảnh hưởng lớn của lỗi cực trị. MAE khoảng $122 là mean absolute
error trên holdout. WAPE 5,11% là `sum(|prediction-actual|)/sum(actual)`, không
phải MAPE trung bình từng dòng.

Model chính giảm RMSE khoảng 4,9% và MAE khoảng 46,6% so với baseline. Hai model
gần nhau trên holdout; việc chọn model theo hai input contract đã khóa từ trước,
không đổi model chính theo chênh lệch rất nhỏ này sau khi chấm.

Holdout không có thành phố hoàn toàn mới so với lịch sử train. Vì vậy có thêm
controlled cold-city stress: chọn tám thành phố theo hash tên, không theo label;
bỏ mọi load liên quan khỏi train tháng 1-6; fit cấu hình chính đã chọn trên
22.894 dòng còn lại; chấm 2.004 load tháng 7-8 liên quan đến các thành phố đó.
Kết quả RMSE $540,54, MAE $141,82, WAPE 7,04%. Đây là thử nghiệm bổ sung sau khi
chọn cấu hình; **không phải accuracy của tám thành phố mới thật trong validation**.

`score.py` chỉ kiểm tra schema, số dòng, ID, các giá trị dương/hữu hạn và input
tháng 12, rồi vẽ chart. Nó không chấm accuracy. Không có label tháng 11-12 nên
không biết điểm chính thức của bài nộp.

## Giải thích December chart

31 input cùng Lexington → Fort Wayne, 360 miles, Dry Van, 32.000 lb; chỉ date
thay đổi. Input thiếu tọa độ, `market_index`, `quote_signal`.

Thay vì gán tín hiệu thị trường tương lai tùy ý, dùng model riêng đã train và
chấm bằng đúng sáu feature có sẵn. Calendar feature tạo biến động theo ngày.
Biểu đồ là **kịch bản dự đoán**, không phải giá thật tháng 12 hay bằng chứng đã
học đúng seasonality tháng 12. Chỉ có lịch sử tháng 1-10 của một năm, không đủ
xác nhận chu kỳ December lặp lại.

## Phần code nên nắm khi phỏng vấn

| File | Nội dung cần giải thích |
|---|---|
| `src/data_audit.py` | Đếm lỗi, coverage, kiểm tra ID/ngày, ghi hash input |
| `src/features.py` | Cùng phép biến đổi lúc train/predict; không đưa target/ID vào feature |
| `src/model.py` | Lưu median đã fit trên train; target transform và predict ngược |
| `src/train.py` | Chọn trên tháng 7-8, khóa config trước holdout, refit final |
| `src/predict.py` | Ghép theo `load_id`, giữ thứ tự template, xuất đủ hai CSV |
| `src/run_all.py` | Quy trình audit → train/refit → predict → scorer → PDF |

Chạy lại từ thư mục project bằng `python -m src.run_all` trong environment đã
cài dependencies. Dùng `--recompute-selection` để rerun các thử nghiệm chọn
model, hoặc `--skip-training` để dùng model final có sẵn. Hai tùy chọn này không
dùng cùng nhau. README có đầy đủ lệnh Windows và lệnh từng bước.

## Trước khi nộp

Đọc báo cáo và source, chạy lại các lệnh chính, rồi quay video bằng lời của mình.
File `reports/loom_script_en.md` là gợi ý nội dung 2-3 phút, chưa phải video.
Có thể tạo thêm `reports/walkthrough.mp4` bằng `python -m src.build_walkthrough`
trên Windows. Đây là bản nháp AI-assisted dùng giọng tổng hợp, không phải video
Loom đã quay hay lời nói thật của ứng viên; cần xem lại và giải thích rõ nếu dùng.
Nộp repository URL mà người chấm truy cập được, `validation_predictions.csv`,
PDF và Loom link thật. Nếu repo private cần cấp quyền cho người chấm. Không có
deadline hay ngưỡng điểm đạt được nêu trong các file đề đã cung cấp.
