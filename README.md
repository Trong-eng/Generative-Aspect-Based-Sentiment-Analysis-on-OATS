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

- Đọc trực tiếp các split TXT `train/dev/test` của OATS hoặc dùng luồng XML/review-level riêng; luồng TXT giữ nguyên phân chia do nguồn cung cấp.
- Giữ lại các bài không có nhãn khi chưa có bằng chứng rõ ràng; chỉ cho phép loại bài theo danh sách có nguồn xác nhận.
- Lưu dấu vết dữ liệu đầu vào và từng tập dữ liệu để nhận biết khi nguồn hoặc nội dung thay đổi.
- Tạo hai cách tổ chức đầu ra cho mô hình: GAS theo hướng trích xuất và Paraphrase-T5.
- Viết luồng huấn luyện, dự đoán, đánh giá, phân tích lỗi và so sánh hai thí nghiệm.
- Viết đủ sáu cấu hình cho hai mô hình trên cả ba lĩnh vực, cùng tài liệu hướng dẫn và bộ kiểm tra tự động cho các thành phần chính.

Sáu cấu hình seed 42 hiện dùng trực tiếp các split TXT. Mỗi dòng không rỗng là một mẫu riêng, được nhận diện bằng domain, split và số dòng vật lý; pipeline không suy diễn review/sentence ID và không thể xác minh leakage ở cấp review từ các file này. Nhóm xác nhận sẽ tin cậy cách chia train/dev/test có sẵn. Kết quả chạy thật được ghi riêng trong [báo cáo Thanh_03](report/Thanh_03_Seed42_Execution_Report.md); không xem số trong paper là kết quả của project.

GAS trong dự án là một cách triển khai theo hướng trích xuất cho bài toán ASQP. Bài báo OATS không công bố đầy đủ cú pháp đầu ra GAS-T5 dùng để tạo các con số trong bảng kết quả, nên dự án **không tuyên bố tái tạo chính xác** thí nghiệm GAS của bài báo.

| Hạng mục | Trạng thái |
|---|---|
| Mã nguồn pipeline và ma trận 2 mô hình × 3 lĩnh vực | Đã chạy seed 42 theo `provided_split`; kết quả trong báo cáo Thanh_03 |
| Tài liệu giải thích và hướng dẫn | Đã viết |
| Bộ kiểm tra tự động | Đã viết; trạng thái lần chạy xem trong báo cáo Thanh_03 |
| Chuẩn bị dữ liệu và kiểm tra split TXT | Dùng các file train/dev/test có sẵn; không kiểm tra leakage cấp review |
| Huấn luyện và dự đoán bằng mô hình | Trạng thái lần chạy xem trong báo cáo Thanh_03 |
| Kết quả đánh giá và so sánh với bài báo | Chỉ báo cáo kết quả thực nghiệm của project; không thay bằng số paper |

## Lệnh chạy pipeline seed 42

Các lệnh bên dưới là luồng chạy seed 42 với split TXT được cung cấp. Cần cài các gói được liệt kê trong `requirements.txt` và lấy submodule OATS trước khi dùng.

```bash
# 1. Kiểm tra các split TXT (không đọc XML)
python -m capstone.audit_data --all-domains --source-mode provided_split --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit/raw

# 2. Chuẩn bị trực tiếp từ train/dev/test TXT; seed 42 không chia lại dữ liệu
for domain in amazon_ff coursera hotels; do
  python -m capstone.prepare_data --domain "$domain" --split-mode provided_split --seed 42 --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-root data/processed
  python -m capstone.audit_data --processed "data/processed/$domain" --output-dir "artifacts/data_audit/processed/$domain"
done

# 3. Huấn luyện và đánh giá 2 mô hình × 3 lĩnh vực = 6 thí nghiệm
for domain in amazon_ff coursera hotels; do
  for model in gas paraphrase; do
    python -m capstone.train --config "configs/${model}_t5_${domain}.yaml"
    python -m capstone.evaluate --config "configs/${model}_t5_${domain}.yaml" --checkpoint "artifacts/experiments/${model}_t5_${domain}_seed42" --split test
  done
done

# 4. So sánh hai mô hình riêng trong từng lĩnh vực
for domain in amazon_ff coursera hotels; do
  python -m capstone.compare --gas-experiment "artifacts/experiments/gas_t5_${domain}_seed42" --paraphrase-experiment "artifacts/experiments/paraphrase_t5_${domain}_seed42" --output-dir artifacts/comparisons/seed42
done
```

Repository có sáu cấu hình: GAS và Paraphrase cho `amazon_ff`, `coursera`, `hotels`. Một lượt seed 42 gồm sáu lần train và sáu lần evaluate; protocol năm seed tương ứng 30 lần train và 30 lần evaluate. Kết quả phải được báo cáo riêng theo domain và mô hình, không gộp điểm ba domain. Hướng dẫn đầy đủ, gồm phân tích lỗi, cấu trúc artifact và multi-seed, có trong [hướng dẫn chạy capstone](report/Trong_02_Capstone_Core_Execution_Guide.md).

## Cách dự án giữ kết quả đáng tin cậy hơn

- Bài đánh giá không có nhãn được xem là **chưa rõ trạng thái**, không tự động coi là dữ liệu lỗi.
- Danh sách xác nhận phải có nguồn dẫn chứng; ID không tồn tại trong dữ liệu sẽ làm bước chuẩn bị dừng lại.
- Mỗi thí nghiệm lưu lại cấu hình, dấu vân tay dữ liệu và danh sách khía cạnh đã biết, để tránh vô tình đánh giá trên một bộ dữ liệu khác.
- Khi so sánh mô hình, dự án kiểm tra danh tính các tập dữ liệu chứ không chỉ so số lượng mẫu.
- `generation.early_stopping` cần là giá trị boolean thực sự trong YAML (`true` hoặc `false`).

Trong luồng TXT, mỗi dòng là một mẫu riêng; kể cả hai dòng có cùng text nhưng nhãn khác nhau vẫn được giữ. Metadata lưu source path/fingerprint và số dòng. Vì TXT không chứa review ID, báo cáo leakage cấp review là `not_verifiable`; pipeline tin cậy phân chia sẵn theo xác nhận của nhóm. Hướng dẫn đầy đủ, gồm cấu trúc artifact, audit, phân tích lỗi và multi-seed, có trong [hướng dẫn chạy capstone](report/Trong_02_Capstone_Core_Execution_Guide.md).

## Giao diện web và service dự đoán

Ngoài pipeline dòng lệnh, repository có giao diện web (`frontend/`) và một service HTTP (`capstone/service/`) để dùng mô hình đã huấn luyện:

```bash
# 1. Service dự đoán: khai báo checkpoint cho từng (domain, model)
pip install -r requirements-serve.txt
cp configs/serving.example.yaml configs/serving.yaml   # sửa đường dẫn checkpoint cho đúng máy
ASQP_SERVING_CONFIG=configs/serving.yaml ASQP_CORS_ORIGINS=http://127.0.0.1:5173 \
  python -m uvicorn capstone.service.app:get_app --factory --port 8000

# 2. Giao diện web, gọi service qua proxy dev
cd frontend && npm install
VITE_USE_MOCK_API=false DEV_PROXY_TARGET=http://127.0.0.1:8000 npm run dev
```

Service có `POST /predict` (một câu), `POST /predict/batch` (nhiều câu, dùng cho dashboard theo khía cạnh) và `GET /health`. Cặp `(domain, model)` chưa khai báo hoặc thiếu checkpoint trả về `MODEL_NOT_READY` thay vì dựng lỗi; mô hình chỉ được nạp một lần rồi dùng lại. Chi tiết contract và giao diện xem [frontend/README.md](frontend/README.md); giải thích thiết kế xem [báo cáo kỹ thuật](report/bao-cao-ky-thuat/main.tex).

## Cấu trúc chính

| Thư mục / tệp | Vai trò |
|---|---|
| `capstone/data/` | Đọc, làm sạch, chia tập và kiểm tra dữ liệu |
| `capstone/models/` | Định dạng đầu ra và đọc dự đoán của GAS/Paraphrase |
| `capstone/training/` | Luồng huấn luyện |
| `capstone/evaluation/` | Chuẩn hóa và tính điểm |
| `capstone/service/` | Service HTTP `POST /predict`, `POST /predict/batch` |
| `frontend/` | Giao diện web: phân tích một câu và dashboard theo khía cạnh |
| `configs/` | Sáu cấu hình GAS/Paraphrase cho ba lĩnh vực; `clean_split/` và `serving.example.yaml` |
| `tests/` | Các kiểm tra tự động đã viết |
| `docs/` | Ghi chú về bài báo và lựa chọn triển khai |
| `report/` | Báo cáo triển khai, hướng dẫn sử dụng và báo cáo kỹ thuật |

## Tài liệu và nguồn tham khảo

- [Báo cáo phần cốt lõi](report/Trong_01_Capstone_Core_Implementation_Report.md)
- [Hướng dẫn chạy capstone](report/Trong_02_Capstone_Core_Execution_Guide.md)
- [Ghi chú bài báo và giả định triển khai](docs/paper_reference.md)
- [Bài báo OATS tại ACL Anthology](https://aclanthology.org/2024.lrec-main.1080/)
- [Kho dữ liệu OATS-ABSA](https://github.com/RiTUAL-MBZUAI/OATS-ABSA)

**Trạng thái kiểm chứng:** xem [báo cáo Thanh_03](report/Thanh_03_Seed42_Execution_Report.md) để biết chính xác các bước đã chạy và còn chờ; không suy diễn trạng thái từ hướng dẫn này.
