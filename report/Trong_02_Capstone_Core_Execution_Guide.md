# Trọng — Capstone Core Execution Guide

> Hướng dẫn này mô tả lượt seed 42 đã được phê duyệt: dùng trực tiếp các split TXT có sẵn của ba domain OATS (`amazon_ff`, `coursera`, `hotels`) cho GAS và Paraphrase. Không đọc XML và không chia lại dữ liệu.

Các kết quả thực tế, bao gồm trạng thái preflight và pipeline, phải được ghi trong [báo cáo Thanh_03](Thanh_03_Seed42_Execution_Report.md). Tài liệu này là runbook; không được xem các ví dụ lệnh hay số trong paper là kết quả đã chạy.

## Phạm vi lượt chạy

Một lượt đầy đủ với seed 42 gồm:

- 1 raw audit bao phủ cả ba domain;
- 3 lần chuẩn bị trực tiếp từ split TXT và 3 processed audit;
- 6 lần train: 2 mô hình × 3 domain;
- 6 lần evaluate trên test split;
- 3 comparison cùng domain;
- 6 error analysis.

Không gộp dữ liệu hoặc điểm số giữa các domain. Mỗi comparison chỉ đối chiếu GAS và Paraphrase trên cùng domain, split, seed và dataset identity.

## Luồng thực hiện

```text
Kiểm tra môi trường
  → audit các split TXT có sẵn của cả 3 domain
  → prepare trực tiếp từ TXT + audit processed từng domain
  → train + evaluate 6 experiment
  → compare 3 cặp cùng domain
  → error analysis 6 experiment
  → inference tùy chọn
```

Chạy từ repository root. Nếu một bước báo metadata, fingerprint, inventory hoặc compatibility không khớp thì dừng và xử lý trước khi sang bước sau.

## Step 0 — Chuẩn bị môi trường

Lấy Git submodule và cài dependencies trong môi trường của người chạy (nếu chưa chuẩn bị):

```bash
git submodule update --init --recursive
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python --version
python -m pip show torch transformers PyYAML numpy pytest
```

Kiểm tra test suite trước khi dùng GPU:

```bash
python -m pytest
```

Trên Windows, dùng đúng Python thuộc môi trường `oats-seed42`. Khi thư mục tạm mặc định của pytest không có quyền ghi, đặt `--basetemp` tới một thư mục mới có thể ghi được. Chỉ tiếp tục pipeline khi toàn bộ preflight pass.

## Step 1 — Audit dữ liệu gốc của cả ba domain

```bash
python -m capstone.audit_data \
  --all-domains \
  --source-mode provided_split \
  --sentiment-mode paper_compatible \
  --raw-root OATS-ABSA-Dataset/data \
  --output-dir artifacts/data_audit/raw
```

Expected artifacts:

```text
artifacts/data_audit/raw/
├── amazon_ff.json
├── coursera.json
├── hotels.json
└── summary.json
```

Audit đọc từng dòng của các file `train.txt`, `dev.txt`, `test.txt`, tính SHA-256 và thống kê số dòng, quadruple, nhãn rỗng, implicit elements, phân phối aspect/sentiment và annotation issues. Không có thống kê review/XML hoặc suspicious spans trong chế độ này. Không điền số liệu vào báo cáo trước khi audit thật hoàn tất.

## Step 2 — Chuẩn bị từ split TXT có sẵn cho cả ba domain

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.prepare_data \
    --domain "$domain" \
    --split-mode provided_split \
    --seed 42 \
    --sentiment-mode paper_compatible \
    --raw-root OATS-ABSA-Dataset/data \
    --output-root data/processed
done
```

Expected artifacts cho mỗi domain:

```text
data/processed/<domain>/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── metadata.json
```

Ở chế độ này, mỗi dòng TXT không rỗng là một mẫu độc lập; dòng trống bị bỏ qua nhưng số thứ tự dòng vật lý vẫn được giữ trong `source_line_number`. Các dòng có text trùng nhưng nhãn khác và các mẫu không còn quadruple sau cleaning vẫn được giữ. Quadruple có aspect không hợp lệ tiếp tục bị loại và ghi issue theo cleaning policy hiện có. Không ghép TXT với XML, không tạo `review_id`/`sentence_id`, và không chia lại dữ liệu. `seed: 42` trong config điều khiển tính ngẫu nhiên của thí nghiệm, không dùng để phân chia dữ liệu.

`metadata.json` phải giữ domain, split mode, sentiment mode, cleaning policy, aspect inventory, source paths/fingerprints, fingerprint từng split, số mẫu và issue list. Vì các TXT không chứa review ID, metadata phải ghi `review_level_leakage_check.status: not_verifiable` cùng xác nhận của người dùng rằng có thể tin cậy split có sẵn. Không ghi rằng leakage cấp review đã được kiểm tra.

## Step 3 — Audit từng processed dataset

Mỗi domain dùng output directory riêng vì processed audit ghi các tên cố định `processed.json` và `summary.json`.

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.audit_data \
    --processed "data/processed/$domain" \
    --output-dir "artifacts/data_audit/processed/$domain"
done
```

Expected artifacts:

```text
artifacts/data_audit/processed/<domain>/
├── processed.json
└── summary.json
```

Trước khi train, xác nhận domain, `provided_split`, dataset/cleaning version, sentiment mode, số dòng và fingerprints khớp giữa raw audit, source TXT, metadata và ba JSONL. Xác nhận `review_level_leakage_check.status` là `not_verifiable`; không dùng `review_overlap` hay `no_review_leakage: true` làm điều kiện vì dữ liệu không có review IDs. Nếu fingerprints hoặc metadata không khớp, dừng trước training.

## Step 4 — Train GAS trên ba domain

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.train --config "configs/gas_t5_${domain}.yaml"
done
```

Expected experiment directories:

```text
artifacts/experiments/gas_t5_amazon_ff_seed42/
artifacts/experiments/gas_t5_coursera_seed42/
artifacts/experiments/gas_t5_hotels_seed42/
```

Mỗi experiment dự kiến có snapshot `config.yaml`, `metadata.json`, `training_history.json`, `training_summary.json`, `best_checkpoint.json`, `checkpoints/` và `logs/`. Best checkpoint được chọn bằng validation exact-match F1; khi hòa, epoch đầu tiên đạt best được giữ.

GAS trong repository là **extraction-style ASQP adaptation**. OATS paper không công bố target string GAS-T5 cụ thể, nên không tuyên bố đây là reproduction chính xác implementation trong paper.

## Step 5 — Evaluate GAS trên ba test split

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.evaluate \
    --config "configs/gas_t5_${domain}.yaml" \
    --checkpoint "artifacts/experiments/gas_t5_${domain}_seed42" \
    --split test
done
```

Mỗi experiment dự kiến có:

```text
evaluation/
├── metrics.json
├── diagnostic_metrics.json
├── predictions.jsonl
├── invalid_predictions.jsonl
├── error_cases.jsonl
└── run_metadata.json
```

## Step 6 — Train Paraphrase trên ba domain

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.train --config "configs/paraphrase_t5_${domain}.yaml"
done
```

Expected experiment directories:

```text
artifacts/experiments/paraphrase_t5_amazon_ff_seed42/
artifacts/experiments/paraphrase_t5_coursera_seed42/
artifacts/experiments/paraphrase_t5_hotels_seed42/
```

Paraphrase target có dạng `{aspect} is great/ok/bad because {target} is {opinion}`; `it` biểu diễn phần tử implicit. Từ `mixed` cho nhãn conflict là extension được ghi rõ của project.

## Step 7 — Evaluate Paraphrase trên ba test split

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.evaluate \
    --config "configs/paraphrase_t5_${domain}.yaml" \
    --checkpoint "artifacts/experiments/paraphrase_t5_${domain}_seed42" \
    --split test
done
```

Kiểm tra đúng checkpoint/model type, test sample count không rỗng, parser issues được lưu và official metric là exact quadruple P/R/F1.

## Step 8 — So sánh ba cặp cùng domain

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.compare \
    --gas-experiment "artifacts/experiments/gas_t5_${domain}_seed42" \
    --paraphrase-experiment "artifacts/experiments/paraphrase_t5_${domain}_seed42" \
    --output-dir artifacts/comparisons/seed42
done
```

Expected artifacts:

```text
artifacts/comparisons/seed42/
├── amazon_ff_gas_vs_paraphrase.json
├── coursera_gas_vs_paraphrase.json
└── hotels_gas_vs_paraphrase.json
```

Comparison code kiểm tra domain, split mode, dataset version, sentiment mode, seed, cleaning policy, sample counts, dataset identity, training, sequence và generation settings. Không so hai experiment khác domain.

## Step 9 — Phân tích lỗi cho sáu experiment

```bash
for domain in amazon_ff coursera hotels; do
  for model in gas paraphrase; do
    python -m capstone.error_analysis \
      --predictions "artifacts/experiments/${model}_t5_${domain}_seed42/evaluation/predictions.jsonl" \
      --output "artifacts/experiments/${model}_t5_${domain}_seed42/evaluation/error_analysis.jsonl"
  done
done
```

Error analysis giữ `source_line_number`, text, raw generation, TP/FP/FN, field mismatch, missing/extra quadruple, implicit failures và parser errors; không tạo review/sentence ID.

## Step 10 — Inference tùy chọn

Chỉ chạy sau khi experiment tương ứng đã có `best_checkpoint.json`. Ví dụ Hotels:

```bash
python -m capstone.predict \
  --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 \
  --model-type paraphrase \
  --text "The room was clean and comfortable." \
  --output artifacts/experiments/paraphrase_t5_hotels_seed42/custom_prediction.json
```

Inference trên câu riêng không được cộng vào test metrics.

## Cấu hình chung của sáu experiment seed 42

| Parameter | GAS | Paraphrase |
|---|---:|---:|
| Backbone/tokenizer | `t5-base` | `t5-base` |
| Epochs | 20 | 20 |
| Batch / accumulation | 16 / 1 | 16 / 1 |
| Learning rate | `3e-4` | `3e-4` |
| Optimizer / scheduler | AdamW / linear | AdamW / linear |
| Input/output length | 128 / 128 | 128 / 128 |
| Beams / max new tokens | 1 / 128 | 1 / 128 |
| Seed | 42 | 42 |
| Split | `provided_split` | `provided_split` |
| Sentiment | `paper_compatible`, giữ conflict | `paper_compatible`, giữ conflict |
| Domains | Amazon_FF, Coursera, Hotels | Amazon_FF, Coursera, Hotels |

Các giá trị trên là config của repository. OATS paper không công bố đầy đủ toàn bộ hyperparameters cho hai baseline, nên kết quả tương lai không được gọi là tái tạo chính xác paper chỉ dựa trên các config này.

## Metrics cần lưu

- Official ASQP: micro precision, recall, exact-match F1, TP, FP, FN; cả bốn trường phải khớp trong cùng sentence record.
- Diagnostics: component F1, aspect/sentiment accuracy trong alignment phù hợp, explicit/implicit target và opinion, per-aspect, per-sentiment, nhóm số quadruple và example exact match.
- Training: train/validation loss, learning rate, validation exact-match F1, epoch, step, elapsed time và best checkpoint.
- Reporting: một hàng cho mỗi `(domain, model, seed)`; mean/std chỉ tính trong cùng domain và model.

## Paper comparison và multi-seed

Paper báo cáo trung bình năm seed. Với 3 domain × 2 model × 5 seed, kế hoạch đầy đủ là **30 lần train và 30 lần evaluate**. Seed 42 trong sáu config hiện tại chỉ là lượt chạy đầu tiên.

Nếu nhóm sau này phê duyệt thêm seed cho cùng protocol TXT, tạo sáu config mới hoặc bản sinh ra từ template với:

- `experiment.seed` và `experiment.name` riêng;
- `paths.processed_data` trỏ tới root riêng, ví dụ `data/processed/seed43`;
- `paths.output_dir` riêng, có seed trong tên;
- dùng nguyên các split TXT nguồn cho cả sáu config; seed mới không chia lại dữ liệu;
- compare GAS/Paraphrase trong cùng domain và cùng seed;
- tính mean/std riêng cho từng `(domain, model)` sau khi đủ năm seed.

`provided_split` là protocol của lượt chạy này và đọc trực tiếp TXT. `clean_split` vẫn là một luồng chuẩn bị khác của codebase, nhưng không tương thích với sáu config hiện tại; không dùng nó cho lượt seed 42 này. Khi báo cáo các protocol khác nhau, không đặt F1 của chúng vào cùng một cột hoặc diễn giải chênh lệch như lỗi reproduction.

Paper reference ASQP F1:

| Method | Amazon_FF | Coursera | Hotels |
|---|---:|---:|---:|
| GAS-T5 | 19.62 | 22.23 | 26.33 |
| Paraphrase | 20.84 | 19.78 | 34.51 |

Đây là số trong paper, không phải output của repository.

## Checklist hoàn tất seed 42

```text
[ ] 3 TXT-source audit JSON + summary
[ ] 3 processed dataset, mỗi dataset có train/dev/test/metadata
[ ] 3 processed audit; fingerprints/metadata hợp lệ và review-level leakage ghi not_verifiable
[ ] 6 experiment directories có best_checkpoint.json
[ ] 6 evaluation directories có metrics và predictions
[ ] 3 comparison JSON đúng domain
[ ] 6 error_analysis.jsonl
[ ] Báo cáo 6 hàng kết quả riêng theo domain/model
```

## Trạng thái kiểm chứng

- Trạng thái test suite và dependency/runtime compatibility: ghi theo preflight thực tế trong báo cáo Thanh_03.
- Cần audit TXT, preparation và fingerprints cho cả ba domain; review-level leakage không thể kiểm tra bằng nguồn TXT này.
- Sáu training/evaluation runs, checkpoint selection và GPU memory: đã chạy seed 42 theo `provided_split`; chi tiết và giới hạn trong báo cáo Thanh_03.
- Metrics, comparisons, error analysis và mọi số liệu kết quả: lấy theo báo cáo Thanh_03, không suy diễn từ hướng dẫn này. Số liệu `implicit_opinion_failure` công bố trước ngày 2026-10-16 được tính khi chưa khử trùng lặp mismatch, cần tính lại từ `predictions.jsonl` nếu muốn trích dẫn.
- `UNVERIFIED — requires execution`: mọi lượt chạy theo `clean_split` (config trong `configs/clean_split/`) và protocol multi-seed.
