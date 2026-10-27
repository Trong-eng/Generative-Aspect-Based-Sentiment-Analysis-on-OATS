# Hướng dẫn chạy ma trận `clean_split` seed 42

Dành cho người chạy lại ma trận huấn luyện trên một máy có GPU đủ khỏe.
Viết sau khi thử trên RTX 5050 8GB và gặp một loạt vấn đề, ghi lại ở mục "Những cái bẫy đã gặp".

## Mục tiêu

Chạy ma trận huấn luyện `clean_split` seed 42 cho dự án ASQP trên bộ OATS, rồi mở PR báo cáo kết quả.

## Lấy mã nguồn

```bash
git clone https://github.com/Trong-eng/Generative-Aspect-Based-Sentiment-Analysis-on-OATS.git
cd Generative-Aspect-Based-Sentiment-Analysis-on-OATS
git submodule update --init --recursive
pip install -r requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # hoặc bản CUDA hợp với driver
pip install "transformers>=4.40,<5.0" sentencepiece
```

Kiểm tra trước khi chạy: `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e9)"`
và `python -m pytest -q --basetemp=<thư mục tạm riêng>` (phải 160 passed; `--basetemp` là bắt buộc vì thư mục pytest mặc định hay bị lỗi quyền trên Windows).

## Việc cần làm

**Bước 1 — chuẩn bị dữ liệu** (nhanh, không cần GPU), cho cả ba domain:

```bash
python -m capstone.prepare_data --domain <domain> --split-mode clean_split --seed 42 \
  --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data \
  --output-root data/processed_clean_split
python -m capstone.audit_data --processed data/processed_clean_split/<domain> \
  --output-dir artifacts/data_audit/clean_split/<domain>
```

Audit phải cho `no_review_leakage: true`. Nếu không, dừng lại và báo.

**Bước 2 — đo tốc độ trước khi chạy cả dây.** Chạy `gas_t5_hotels_clean_split.yaml` khoảng 3–5 phút rồi đọc
`artifacts/experiments/gas_t5_hotels_clean_split_seed42/logs/training_log.jsonl`: lấy `elapsed_time` chia cho số bước
để ra giây/bước. Báo lại con số này **trước khi** chạy tiếp. Trên máy RTX 5050 8GB, GAS mất ~3,8 giây/bước (tràn VRAM)
còn Paraphrase ~0,5 giây/bước; nếu máy bạn cũng ~3,8 giây/bước cho GAS thì cả ma trận sẽ mất hơn 30 giờ, lúc đó phải
hỏi lại người dùng chứ đừng tự chạy.

**Bước 3 — chạy tuần tự 6 lượt** (2 mô hình × 3 domain), mỗi lượt train rồi evaluate rồi error_analysis:

```bash
python -m capstone.train --config configs/clean_split/<model>_t5_<domain>_clean_split.yaml
python -m capstone.evaluate --config configs/clean_split/<model>_t5_<domain>_clean_split.yaml \
  --checkpoint artifacts/experiments/<model>_t5_<domain>_clean_split_seed42 --split test
python -m capstone.error_analysis \
  --predictions artifacts/experiments/<model>_t5_<domain>_clean_split_seed42/evaluation/predictions.jsonl \
  --output artifacts/experiments/<model>_t5_<domain>_clean_split_seed42/evaluation/error_analysis.jsonl
```

**Bước 4 — so sánh hai mô hình trong từng domain:**

```bash
python -m capstone.compare \
  --gas-experiment artifacts/experiments/gas_t5_<domain>_clean_split_seed42 \
  --paraphrase-experiment artifacts/experiments/paraphrase_t5_<domain>_clean_split_seed42 \
  --output-dir artifacts/comparisons/clean_split_seed42
```

## Những cái bẫy đã gặp (đọc kỹ, đừng lặp lại)

1. **Chỉ chạy MỘT lượt train tại một thời điểm.** Hai lượt cùng dùng GPU sẽ làm một lượt chết với
   `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`.
2. **`best_checkpoint.json` KHÔNG phải dấu hiệu đã chạy xong** — nó được ghi lại sau mỗi epoch tốt hơn. Muốn biết lượt
   nào đã xong, dùng marker do chính script của bạn tạo sau khi lệnh train trả về 0, hoặc kiểm tra
   `training_history.json` có đủ 20 epoch.
3. **Đừng bật `training.precision: bf16` nếu VRAM dưới ~12GB.** Trên 8GB nó chậm hơn fp32 **8 lần** vì autocast tốn
   thêm VRAM và bị tràn sang RAM. Mặc định fp32 là đúng; nếu máy bạn có VRAM dư thì có thể thử và báo lại số đo.
4. **Nếu tràn VRAM**, giảm `training.batch_size` xuống 8 và đặt `gradient_accumulation_steps: 2` (giữ batch hiệu dụng
   16), ghi rõ thay đổi này trong báo cáo.
5. Pipeline **không chạy tiếp được** từ giữa chừng: lượt nào chết là phải chạy lại từ đầu.

## Báo cáo kết quả

Checkpoint rất nặng, **đừng commit**. Chỉ commit các tệp nhỏ: với mỗi thí nghiệm, copy `metrics.json`,
`diagnostic_metrics.json`, `training_history.json` và `best_checkpoint.json` vào
`report/clean_split_seed42/<tên thí nghiệm>/`, cùng ba tệp so sánh vào `report/clean_split_seed42/comparisons/`.

Rồi viết `report/clean_split_seed42/README.md` gồm:
- bảng 6 dòng theo (domain, mô hình): best epoch, dev F1, test P/R/F1, TP/FP/FN, số generation hỏng;
- chỉ số **relaxed** lấy từ `diagnostic_metrics.json` tại `relaxed.aspect` và `relaxed.aspect_sentiment` (đây là chỉ số
  mà dashboard dùng, mới thêm gần đây);
- cấu hình máy (GPU, VRAM, driver), thời gian mỗi lượt, và mọi thay đổi so với cấu hình gốc;
- xác nhận `no_review_leakage: true` cho cả ba domain.

**Không** so sánh trực tiếp với số của bài báo OATS và **không** gộp điểm ba domain. Lượt chạy `provided_split` đã có
trong `report/Thanh_03_Seed42_Execution_Report.md` dùng giao thức khác, nên để riêng hai bảng.

Cuối cùng: tạo nhánh `run/clean-split-seed42`, commit, mở PR về `main`, và báo lại số liệu.
