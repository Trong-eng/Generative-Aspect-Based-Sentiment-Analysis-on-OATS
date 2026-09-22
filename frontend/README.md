# Frontend — OATS ASQP Analyzer

Giao diện web (React + Vite + TypeScript) để nhập một câu nhận xét, chọn domain và mô hình, rồi hiển thị các quadruple `(target, aspect, opinion, sentiment)`.

Repository **chưa có HTTP backend** `POST /predict`. Frontend này vì vậy có hai chế độ:

- **Mock mode** (mặc định): dữ liệu mẫu trong trình duyệt, không cần checkpoint hay Python. Kết quả mock **không phải dự đoán của mô hình**; giao diện luôn hiện nhãn “Mock mode”.
- **API mode**: gọi `POST /predict` theo contract bên dưới. Backend thật sẽ được làm ở task khác.

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
| `VITE_API_TIMEOUT_MS` | `30000` | Timeout phía client cho một request |
| `VITE_MOCK_DELAY_MS` | `600` | Độ trễ giả lập của mock |
| `DEV_PROXY_TARGET` | rỗng | Chỉ dùng cho `npm run dev`: proxy `/predict` tới backend để tránh CORS |

`VITE_*` được nhúng vào bundle lúc build, nên **không đặt secret** vào đó.

### Mock mode

```bash
npm run dev
```

Mock trả fixture khác nhau cho mỗi cặp domain × model (xem `src/api/fixtures.ts`) để kiểm tra wiring. Để thử các trạng thái khác, thêm vào câu nhập:

- `#empty` → response thành công với `quadruples: []`.
- `#error` → lỗi `MODEL_NOT_READY`.

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

**Lỗi (non-2xx):**

```json
{ "error": { "code": "MODEL_NOT_READY", "message": "Checkpoint cho domain/model này chưa sẵn sàng." } }
```

UI hiện `message` (tối đa 300 ký tự) và `code`. Body không đúng shape này (HTML, stack trace…) bị bỏ qua và thay bằng thông báo chung kèm HTTP status.

## Cấu trúc

```text
src/
  api/
    types.ts          # kiểu contract, PredictionError
    service.ts        # interface PredictionService (mock và HTTP cùng implement)
    httpService.ts    # fetch + timeout + cancel
    mockService.ts    # mock theo cấu hình
    fixtures.ts       # dữ liệu mẫu theo domain × model
    validate.ts       # kiểm tra response/error body
    index.ts          # chọn service từ biến môi trường
  hooks/usePrediction.ts  # state idle/loading/success/error; hủy request cũ
  components/         # PredictForm, ResultsPanel, QuadrupleCard, SentimentBadge
  App.tsx             # nhận `service` qua props
```

Component không biết service nào đang chạy. Khi gửi request mới hoặc unmount, request cũ bị abort và response trễ bị bỏ qua, nên không ghi đè kết quả mới.

## Accessibility

- Mọi input/select có `<label>`; lỗi validation gắn qua `aria-describedby` và `aria-invalid`.
- Focus bàn phím rõ (`:focus-visible`); Ctrl+Enter để gửi.
- Loading/kết quả được thông báo qua vùng `role="status"` `aria-live="polite"`; lỗi dùng `role="alert"`.
- Sentiment hiển thị bằng chữ + ký hiệu (`+ − ○ ±`), màu chỉ để bổ trợ.
- Layout một cột trên mobile, hai cột từ 900px; hỗ trợ light/dark.

## Kiểm thử đã chạy

Chạy ngày 2026-09-22 trên Windows 11, Node 24.15.0:

| Lệnh | Kết quả |
|---|---|
| `npm test` | 5 file, 45 test — tất cả pass |
| `npm run typecheck` | Không lỗi |
| `npm run lint` | Không lỗi |
| `npm run build` | Thành công (JS ~234 kB, gzip ~73 kB) |

Các test bao phủ: input rỗng/whitespace không gửi request; domain/model gửi đúng; loading và chống submit lặp; một và nhiều quadruple; `quadruples: []`; `NULL` hiển thị thân thiện; timeout/network/server error và retry; response malformed và exception lạ không làm crash hay lộ chi tiết nội bộ; response cũ không ghi đè response mới; cùng component render được từ mock service và HTTP service.

Kiểm tra thủ công trên Chrome (desktop, mock mode, `npm run dev`): validation whitespace; loading + nút bị disable; nhiều quadruple (GAS-T5 · Hotels); `NULL` hiển thị “Implicit / Không nêu rõ (NULL)”; empty state (`#empty`); lỗi `MODEL_NOT_READY` + nút Try again (`#error`); diagnostics mở/đóng được; không có lỗi console.

Chưa kiểm tra thủ công layout mobile và chưa chạy với backend thật.

## Giới hạn hiện tại

- Backend HTTP `POST /predict` **không thuộc task này** và chưa tồn tại.
- Chưa có kết quả huấn luyện/đánh giá; UI không hiển thị F1 hay bất kỳ số đo chất lượng nào.
- Chưa hỗ trợ chính thức review nhiều câu (chờ backend chốt preprocessing).
