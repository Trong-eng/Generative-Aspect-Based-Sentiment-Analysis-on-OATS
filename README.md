# Phân tích ý kiến theo khía cạnh trong đánh giá OATS

## Dự án này làm gì?

Khi đọc một nhận xét như **“Phòng sạch sẽ, nhưng nhân viên không nhiệt tình”**, điểm đánh giá chung không cho biết rõ khách hài lòng ở đâu và chưa hài lòng ở đâu. Dự án này xây dựng một bộ công cụ nghiên cứu để đọc nhận xét và tách từng ý kiến thành thông tin có cấu trúc:

1. **Đối tượng được nhắc đến** — ví dụ: phòng, nhân viên; đôi khi đối tượng chỉ được ngầm hiểu.
2. **Khía cạnh** — ví dụ: vệ sinh phòng, chất lượng phục vụ.
3. **Cụm từ thể hiện ý kiến** — ví dụ: “sạch sẽ”, “không nhiệt tình”.
4. **Cảm xúc** — tích cực, tiêu cực, trung tính hoặc xung đột.

Nhờ cách tách này, máy tính có thể tổng hợp nhiều nhận xét theo từng chủ đề thay vì chỉ nhìn vào một điểm sao chung. Đây là **lõi capstone phục vụ nghiên cứu và thử nghiệm**, chưa phải ứng dụng thương mại hay công cụ đã được kiểm chứng trên thực tế.

## Dữ liệu và phạm vi

Dự án dùng bộ dữ liệu nghiên cứu [OATS-ABSA](https://github.com/RiTUAL-MBZUAI/OATS-ABSA), gồm nhận xét thuộc ba lĩnh vực: thực phẩm, khóa học trực tuyến và khách sạn. Dữ liệu gốc được giữ riêng dưới dạng Git submodule để duy trì nguồn gốc và ghi nhận đóng góp của nhóm tác giả.

Sau khi lấy mã nguồn dự án, cần lấy thêm dữ liệu OATS bằng lệnh sau. Đây là hướng dẫn cho lần chạy về sau, **chưa được thực hiện trong task này**:

```bash
git submodule update --init --recursive
```

## Hiện tại dự án đã làm đến đâu?

Phần **mã nguồn cốt lõi đã được viết**:

- Đọc dữ liệu OATS và chuẩn bị tập huấn luyện, có kiểm tra để các câu của cùng một bài đánh giá không bị chia sang nhiều tập.
- Giữ lại các bài không có nhãn khi chưa có bằng chứng rõ ràng; chỉ cho phép loại bài theo danh sách có nguồn xác nhận.
- Lưu dấu vết dữ liệu đầu vào và từng tập dữ liệu để nhận biết khi nguồn hoặc nội dung thay đổi.
- Tạo hai cách tổ chức đầu ra cho mô hình: GAS theo hướng trích xuất và Paraphrase-T5.
- Viết luồng huấn luyện, dự đoán, đánh giá, phân tích lỗi và so sánh hai thí nghiệm.
- Viết cấu hình mẫu, tài liệu hướng dẫn và bộ kiểm tra tự động cho các thành phần chính.

**Phần chưa làm:** chưa chạy bộ kiểm tra; chưa xử lý dữ liệu; chưa huấn luyện, dự đoán, đánh giá hay benchmark. Vì vậy hiện chưa có kết quả độ chính xác hoặc bằng chứng rằng toàn bộ pipeline chạy thành công trên máy. Những điều này được ghi là **UNVERIFIED — requires execution**.

GAS trong dự án là một cách triển khai theo hướng trích xuất cho bài toán ASQP. Bài báo OATS không công bố đầy đủ cú pháp đầu ra GAS-T5 dùng để tạo các con số trong bảng kết quả, nên dự án **không tuyên bố tái tạo chính xác** thí nghiệm GAS của bài báo.

| Hạng mục | Trạng thái |
|---|---|
| Mã nguồn pipeline và cấu hình mẫu | Đã viết; chưa xác minh khi chạy |
| Tài liệu giải thích và hướng dẫn | Đã viết |
| Bộ kiểm tra tự động | Đã viết; chưa chạy |
| Chuẩn bị dữ liệu và kiểm tra chia tập | Chưa chạy |
| Huấn luyện và dự đoán bằng mô hình | Chưa chạy |
| Kết quả đánh giá và so sánh với bài báo | Chưa có |

## Cách bắt đầu khi sẵn sàng chạy

Các lệnh bên dưới được ghi để sử dụng về sau. Chúng **chưa được chạy** trong task này. Cần cài các gói được liệt kê trong `requirements.txt` và lấy dữ liệu theo hướng dẫn ở trên trước khi dùng.

```bash
# 1. Kiểm tra dữ liệu Hotels
python -m capstone.audit_data --domain hotels --raw-root OATS-ABSA-Dataset/data

# 2. Chuẩn bị các tập dữ liệu theo bài đánh giá
python -m capstone.prepare_data --domain hotels --split-mode clean_split --seed 42 --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-root data/processed

# 3. Kiểm tra số lượng và việc tách tập
python -m capstone.audit_data --processed data/processed/hotels

# 4. Huấn luyện và đánh giá GAS
python -m capstone.train --config configs/gas_t5_hotels.yaml
python -m capstone.evaluate --config configs/gas_t5_hotels.yaml --checkpoint artifacts/experiments/gas_t5_hotels_seed42 --split test

# 5. Huấn luyện và đánh giá Paraphrase
python -m capstone.train --config configs/paraphrase_t5_hotels.yaml
python -m capstone.evaluate --config configs/paraphrase_t5_hotels.yaml --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --split test
```

Hai cấu hình mẫu dùng Hotels để bắt đầu. Các hướng dẫn chi tiết hơn, gồm cách đọc đầu ra và phân tích lỗi, có trong [hướng dẫn chạy capstone](report/Trong_02_Capstone_Core_Execution_Guide.md).

## Cách dự án giữ kết quả đáng tin cậy hơn

- Bài đánh giá không có nhãn được xem là **chưa rõ trạng thái**, không tự động coi là dữ liệu lỗi.
- Danh sách xác nhận phải có nguồn dẫn chứng; ID không tồn tại trong dữ liệu sẽ làm bước chuẩn bị dừng lại.
- Mỗi thí nghiệm lưu lại cấu hình, dấu vân tay dữ liệu và danh sách khía cạnh đã biết, để tránh vô tình đánh giá trên một bộ dữ liệu khác.
- Khi so sánh mô hình, dự án kiểm tra danh tính các tập dữ liệu chứ không chỉ so số lượng mẫu.
- `generation.early_stopping` cần là giá trị boolean thực sự trong YAML (`true` hoặc `false`).

## Cấu trúc chính

| Thư mục / tệp | Vai trò |
|---|---|
| `capstone/data/` | Đọc, làm sạch, chia tập và kiểm tra dữ liệu |
| `capstone/models/` | Định dạng đầu ra và đọc dự đoán của GAS/Paraphrase |
| `capstone/training/` | Luồng huấn luyện |
| `capstone/evaluation/` | Chuẩn hóa và tính điểm |
| `configs/` | Cấu hình thí nghiệm mẫu |
| `tests/` | Các kiểm tra tự động đã viết |
| `docs/` | Ghi chú về bài báo và lựa chọn triển khai |
| `report/` | Báo cáo triển khai và hướng dẫn sử dụng |

## Tài liệu và nguồn tham khảo

- [Báo cáo phần cốt lõi](report/Trong_01_Capstone_Core_Implementation_Report.md)
- [Hướng dẫn chạy capstone](report/Trong_02_Capstone_Core_Execution_Guide.md)
- [Ghi chú bài báo và giả định triển khai](docs/paper_reference.md)
- [Bài báo OATS tại ACL Anthology](https://aclanthology.org/2024.lrec-main.1080/)
- [Kho dữ liệu OATS-ABSA](https://github.com/RiTUAL-MBZUAI/OATS-ABSA)

**Trạng thái kiểm chứng:** `UNVERIFIED — requires execution`. Chưa chạy code, tests, xử lý dữ liệu, huấn luyện, dự đoán, đánh giá hoặc benchmark.
