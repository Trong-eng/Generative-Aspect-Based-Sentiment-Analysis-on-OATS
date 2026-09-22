# Frontend — OATS ASQP Analyzer

Giao diện web (React + Vite + TypeScript) cho ASQP trên OATS, gồm hai tab:

1. **Single review** — nhập một câu, chọn domain và mô hình, xem các quadruple `(target, aspect, opinion, sentiment)`.
2. **Review file & dashboard** — tải lên file nhiều review. Dashboard cho biết với mỗi aspect có bao nhiêu % review tích cực / tiêu cực / trung tính / xung đột, aspect nào bị chê nhiều nhất và vì sao (các cụm opinion tiêu cực phổ biến). Mục đích: giúp doanh nghiệp tìm ra phần sản phẩm/dịch vụ đang yếu — ví dụ khách sạn thấy `rooms cleanliness` có tỷ lệ tiêu cực cao nhất với “dirty”, “stained”.

Repository **chưa có HTTP backend**. Frontend vì vậy có hai chế độ:

- **Mock mode** (mặc định): chạy hoàn toàn trong trình duyệt, không cần checkpoint hay Python. Kết quả mock **không phải dự đoán của mô hình**; giao diện luôn hiện nhãn “Mock mode”.
- **API mode**: gọi `POST /predict` và `POST /predict/batch` theo contract bên dưới. Backend thật sẽ được làm ở task khác.

## Cài đặt và chạy

Yêu cầu Node.js 20+ (đã dùng Node 24.15, npm 11.12).

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173, mock mode mặc định
```

| Lệnh | Mục đích |
|---|---|
| `npm run dev` | Dev server |
| `npm run build` | Typecheck + build production vào `dist/` |
| `npm run preview` | Xem bản build |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | ESLint |
| `npm test` | Vitest (một lần) — `npm run test:watch` để chạy liên tục |

## Biến môi trường

Sao chép `.env.example` thành `.env.local` rồi sửa. Không commit file `.env*` thật.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VITE_USE_MOCK_API` | `true` | `false` (hoặc `0`) để dùng HTTP API; mọi giá trị khác là mock |
| `VITE_API_BASE_URL` | rỗng | Origin/prefix của backend. Rỗng → gọi `/predict` tương đối trên cùng origin |
| `VITE_API_TIMEOUT_MS` | `30000` | Timeout phía client cho một request single; batch dùng 4× giá trị này |
| `VITE_MOCK_DELAY_MS` | `600` | Độ trễ giả lập của mock |
| `DEV_PROXY_TARGET` | rỗng | Chỉ dùng cho `npm run dev`: proxy `/predict` (và `/predict/batch`) tới backend để tránh CORS |

`VITE_*` được nhúng vào bundle lúc build, nên **không đặt secret** vào đó.

### Mock mode

```bash
npm run dev
```

- **Single review**: fixture khác nhau cho mỗi cặp domain × model (`src/api/fixtures.ts`), không phụ thuộc nội dung câu.
- **Batch**: luật từ khóa đơn giản (`src/api/mockLexicon.ts`) ghép từ chỉ đối tượng (room, staff, lectures…) với từ chỉ ý kiến (dirty, rude, boring…) thành category OATS, có xử lý phủ định (“not clean”). Chỉ để thử dashboard; **không phản ánh chất lượng mô hình**.
- File mẫu: `public/samples/hotels_reviews_sample.csv` (30 review khách sạn tự viết; tải ngay trong UI qua link *Download a sample file*). Với mock, aspect yếu nhất là `rooms cleanliness` (74% tiêu cực).
- Thêm `#empty` vào câu → response rỗng; `#error` → lỗi `MODEL_NOT_READY` (dùng được cho cả hai tab).

### API mode

```bash
# Backend cùng origin hoặc qua proxy dev
VITE_USE_MOCK_API=false DEV_PROXY_TARGET=http://127.0.0.1:8000 npm run dev

# Backend khác origin (backend phải bật CORS)
VITE_USE_MOCK_API=false VITE_API_BASE_URL=https://asqp-api.example.org npm run dev
```

Trên Windows PowerShell, đặt biến trong `.env.local` thay vì tiền tố dòng lệnh.

## API contract

Frontend chỉ nói chuyện với backend qua contract này; không gọi Python CLI từ trình duyệt.

### Single: `POST /predict`

```http
POST /predict
Content-Type: application/json
```

```json
{ "text": "The room was spotless.", "domain": "hotels", "model_type": "paraphrase" }
```

- `domain`: `amazon_ff` | `coursera` | `hotels`
- `model_type`: `gas` | `paraphrase`
- `text` được gửi nguyên văn. Mô hình được huấn luyện ở mức câu; frontend **không tự tách câu** và chưa hỗ trợ chính thức review nhiều câu cho tới khi backend chốt preprocessing.

**Thành công (2xx):**

```json
{
  "text": "The room was spotless.",
  "domain": "hotels",
  "model_type": "paraphrase",
  "quadruples": [
    { "target": "room", "aspect": "room cleanliness", "opinion": "spotless", "sentiment": "positive" }
  ],
  "raw_generation": "room cleanliness is great because room is spotless",
  "invalid_predictions": []
}
```

- Bắt buộc: `quadruples` là list, mỗi phần tử có đủ bốn field kiểu string. Sai → UI báo “unexpected response”, không crash.
- `raw_generation`, `invalid_predictions` là optional, chỉ hiện trong phần *Diagnostics* (thu gọn). Thiếu hoặc sai kiểu thì bỏ qua.
- `text`/`domain`/`model_type` thiếu thì UI dùng giá trị của request.
- Giá trị quadruple được hiển thị nguyên văn. `NULL` ở target/opinion hiện là “Implicit / Không nêu rõ” nhưng dữ liệu gốc vẫn giữ `NULL`.
- Shape này khớp output của `capstone.predict.predict_rows` (thêm `domain`, `model_type`), nên backend có thể bọc hàm đó.

### Batch: `POST /predict/batch`

Tab dashboard gửi file theo từng lô 20 dòng (`BATCH_CHUNK_SIZE`) để có thanh tiến trình, hủy được và **tiếp tục từ chỗ dừng** khi lỗi.

```json
{
  "domain": "hotels",
  "model_type": "paraphrase",
  "items": [
    { "id": "row-2", "text": "Great location, but the room was dirty." },
    { "id": "row-3", "text": "The staff were friendly." }
  ]
}
```

```json
{
  "results": [
    {
      "id": "row-2",
      "quadruples": [
        { "target": "location", "aspect": "location general", "opinion": "Great", "sentiment": "positive" },
        { "target": "room", "aspect": "rooms cleanliness", "opinion": "dirty", "sentiment": "negative" }
      ],
      "invalid_predictions": []
    },
    { "id": "row-3", "quadruples": [] }
  ]
}
```

- `id` do frontend đặt, backend trả lại nguyên văn; thứ tự `results` không quan trọng.
- Mỗi `id` phải xuất hiện đúng một lần; thiếu, trùng hoặc lạ → UI coi là response lỗi (không gắn nhầm kết quả sang review khác).
- Mỗi `text` là một dòng của file. Nếu cần tách câu, nên làm ở backend; frontend không tách câu.

### Lỗi (non-2xx), dùng chung cho cả hai endpoint

```json
{ "error": { "code": "MODEL_NOT_READY", "message": "Checkpoint cho domain/model này chưa sẵn sàng." } }
```

UI hiện `message` (tối đa 300 ký tự) và `code`. Body không đúng shape này (HTML, stack trace…) bị bỏ qua và thay bằng thông báo chung kèm HTTP status.

## Dashboard theo aspect

### Định dạng file

| Định dạng | Nội dung |
|---|---|
| `.csv` | Có header; cột text tên `text`, `review`, `review_text`, `content`, `comment` hoặc `sentence`; tùy chọn `review_id`/`id` |
| `.txt` | Mỗi dòng một review |
| `.jsonl` | Mỗi dòng một object có `text` (tùy chọn `review_id`/`id`) hoặc một chuỗi |
| `.json` | Một list chuỗi hoặc object như trên |

- Tối đa 10 MB và 5 000 dòng. Dòng lỗi (JSON hỏng, thiếu text…) bị bỏ qua và liệt kê trong *Skipped rows*; lỗi cả file (thiếu cột text, sai định dạng) chặn không cho gửi.
- Các dòng cùng `review_id` được gộp thành **một review** — dùng khi file đã tách câu (như dữ liệu OATS).
- **File đã có dự đoán**: nếu mọi dòng có `quadruples` (hoặc `normalized_quadruples` / `predicted_quadruples` như `predictions.jsonl` của `capstone.evaluate`), dashboard dùng thẳng, **không gọi mô hình**. Nhờ vậy có thể xem dashboard từ output CLI hiện có mà chưa cần backend:

  ```bash
  python -m capstone.predict --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 \
    --model-type paraphrase --input-file reviews.jsonl --output predictions.jsonl
  ```

  Không trộn dòng có và không có dự đoán trong cùng file. Với `predictions.jsonl` của `evaluate`, dashboard dùng dự đoán, không dùng `gold_quadruples`.

### Cách đếm (theo review)

- Mỗi review được đếm **tối đa một lần cho mỗi aspect**; % = số review theo nhãn / số review nhắc tới aspect đó (`n`).
- Nhiều lần nhắc cùng một aspect trong một review được gộp: có `conflict`, hoặc vừa tích cực vừa tiêu cực → **Conflict** (hiển thị riêng); còn lại, tích cực/tiêu cực thắng trung tính; chỉ có trung tính → **Neutral**.
- Nhãn sentiment ngoài bốn nhãn OATS bị bỏ qua và được báo số lượng.
- Cụm opinion được đếm theo số review (không phân biệt hoa thường); opinion `NULL` (ngầm hiểu) không đưa vào danh sách.

### Trên giao diện

- Tóm tắt: số review, số review có ít nhất một aspect, số aspect đang hiện.
- **Most negative aspect**: aspect có % tiêu cực cao nhất (trong số aspect đủ mẫu) kèm các lời chê phổ biến.
- Bảng: % và số review cho Positive / Negative / Neutral / Conflict, thanh phân bố; sắp xếp theo % tiêu cực (mặc định), số review tiêu cực, số review nhắc tới, hoặc tên.
- **Group by**: theo category đầy đủ (`rooms cleanliness`) hoặc theo entity (`rooms`) để nhìn tổng quan rồi đi sâu.
- **Minimum reviews** (mặc định 3): ẩn aspect quá ít mẫu để tránh kết luận “100% tiêu cực” từ 1–2 review.
- Bấm tên aspect → top opinion tiêu cực/tích cực và review mẫu, lọc theo nhãn.
- **Export CSV**: bảng đang hiện (số lượng, %, top opinion tiêu cực); ô bắt đầu bằng `= + - @` được vô hiệu hóa để Excel không chạy như công thức.

## Cấu trúc

```text
src/
  api/
    types.ts          # kiểu contract (single + batch), PredictionError
    service.ts        # interface PredictionService: predict + predictBatch
    httpService.ts    # fetch + timeout + cancel cho /predict và /predict/batch
    mockService.ts    # mock theo cấu hình
    fixtures.ts       # dữ liệu mẫu single review theo domain × model
    mockLexicon.ts    # luật từ khóa cho mock batch
    validate.ts       # kiểm tra response/error body
    index.ts          # chọn service từ biến môi trường
  batch/
    parseReviewFile.ts  # đọc CSV/TXT/JSONL/JSON, gộp theo review_id, nhận file đã có dự đoán
    aggregate.ts        # đếm theo review, xếp hạng aspect, xuất CSV
  hooks/
    usePrediction.ts     # single review: idle/loading/success/error; hủy request cũ
    useBatchAnalysis.ts  # batch: chia lô, tiến trình, hủy, tiếp tục
  components/         # PredictForm, ResultsPanel, BatchTab, AspectDashboard, AspectDetail, ...
  App.tsx             # hai tab; nhận `service` qua props
public/samples/       # file review mẫu
```

Component không biết service nào đang chạy. Khi gửi request mới hoặc unmount, request cũ bị abort và response trễ bị bỏ qua, nên không ghi đè kết quả mới.

## Accessibility

- Mọi input/select có `<label>`; lỗi validation gắn qua `aria-describedby` và `aria-invalid`.
- Focus bàn phím rõ (`:focus-visible`); Ctrl+Enter để gửi ở tab single.
- Loading/kết quả được thông báo qua vùng `role="status"` `aria-live="polite"`; lỗi dùng `role="alert"`.
- Sentiment hiển thị bằng chữ + ký hiệu (`+ − ○ ±`), màu chỉ để bổ trợ.
- Tab theo chuẩn WAI-ARIA (`role="tablist"`, phím ←/→/Home/End); hai tab giữ nguyên dữ liệu khi chuyển qua lại.
- Tiến trình batch dùng `<progress>` + `aria-live`; mở chi tiết aspect sẽ chuyển focus tới tiêu đề.
- Bảng dashboard có header hàng/cột; % luôn hiện bằng số, thanh màu chỉ để bổ trợ (`aria-hidden`).
- Layout một cột trên mobile, hai cột trên desktop; bảng cuộn ngang trong khung riêng nên trang không bị tràn ngang. Hỗ trợ light/dark.

## Kiểm thử đã chạy

Chạy ngày 2026-09-22 trên Windows 11, Node 24.15.0:

| Lệnh | Kết quả |
|---|---|
| `npm test` | 10 file, 98 test — tất cả pass |
| `npm run typecheck` | Không lỗi |
| `npm run lint` | Không lỗi |
| `npm run build` | Thành công (JS ~266 kB, gzip ~82 kB) |

**Single review:** input rỗng/whitespace không gửi request; domain/model gửi đúng; loading và chống submit lặp; một và nhiều quadruple; `quadruples: []`; `NULL` hiển thị thân thiện; timeout/network/server error và retry; response malformed và exception lạ không làm crash hay lộ chi tiết nội bộ; response cũ không ghi đè response mới; cùng component render được từ mock service và HTTP service.

**Dashboard:** đọc CSV (quote, xuống dòng trong ô, BOM), TXT, JSONL, JSON; gộp theo `review_id`; file đã có dự đoán (kể cả `predictions.jsonl` của evaluate) không gọi service; file trộn/sai bị từ chối; quy tắc gán nhãn theo review (kể cả conflict); đếm opinion theo review; group theo entity; sắp xếp; CSV export (escape + chặn công thức); `/predict/batch` (body, id thiếu/trùng/lạ, timeout dài hơn); chia lô, dừng khi lỗi rồi tiếp tục đúng phần còn lại, hủy; chuyển tab bằng chuột và bàn phím; luồng đầy đủ với file mẫu (aspect yếu nhất `rooms cleanliness`), drill-down, lọc số review tối thiểu, lỗi `MODEL_NOT_READY` + Resume.

**Kiểm tra thủ công trên Chrome** (mock mode, `npm run dev`):

- Single review: validation whitespace; loading + nút bị disable; nhiều quadruple; `NULL` → “Implicit / Không nêu rõ (NULL)”; empty state; lỗi + Try again; diagnostics; không có lỗi console.
- Dashboard: tải file mẫu → 30 review, `rooms cleanliness` là aspect yếu nhất (74% tiêu cực, “dirty” ×7); bảng, thanh phân bố và drill-down (click thật) hiển thị đúng.
- Ở độ rộng ~390px (đo trong iframe): trang không tràn ngang, bảng cuộn ngang trong khung riêng. Chưa thử trên thiết bị di động thật.

Chưa chạy với backend thật.

> Lưu ý khi dev trên Windows: nếu UI chạy như code cũ sau khi đổi nhánh hoặc sửa nhiều file, watcher của Vite có thể đã bỏ lỡ thay đổi. Dừng server và chạy lại `npx vite --force`.

## Giới hạn hiện tại

- Backend HTTP (`POST /predict`, `POST /predict/batch`) **không thuộc task này** và chưa tồn tại.
- Chưa có kết quả huấn luyện/đánh giá; UI không hiển thị F1 hay số đo chất lượng. % trên dashboard khi dùng mô hình chỉ là **ước tính** — nên đo độ chính xác của cặp (aspect, sentiment) trước khi dùng để ra quyết định.
- Mô hình học ở mức câu; mỗi dòng file được gửi nguyên văn. Review dài nhiều câu nên được tách câu (ở backend, hoặc tách sẵn trong file và dùng chung `review_id`).
- Chỉ 3 domain (thực phẩm, khóa học, khách sạn), tiếng Anh, danh sách aspect cố định của OATS. Lĩnh vực hoặc ngôn ngữ khác cần gán nhãn và huấn luyện lại.
