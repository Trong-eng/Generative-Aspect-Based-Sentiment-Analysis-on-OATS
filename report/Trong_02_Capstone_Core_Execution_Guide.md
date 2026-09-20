# Trọng — Capstone Core Execution Guide

> Pipeline được thiết kế cho đủ ba domain OATS: `amazon_ff`, `coursera`, `hotels`, với hai mô hình GAS và Paraphrase.

**Mọi command trong tài liệu này chỉ được viết để chạy sau trên máy có GPU. Chưa command nào được thực thi trong task implementation. `UNVERIFIED — requires execution`.**

## Phạm vi lượt chạy

Một lượt đầy đủ với seed 42 gồm:

- 1 raw audit bao phủ cả ba domain;
- 3 lần chuẩn bị dữ liệu và 3 processed audit;
- 6 lần train: 2 mô hình × 3 domain;
- 6 lần evaluate trên test split;
- 3 comparison cùng domain;
- 6 error analysis.

Không gộp dữ liệu hoặc điểm số giữa các domain. Mỗi comparison chỉ đối chiếu GAS và Paraphrase trên cùng domain, split, seed và dataset identity.

## Luồng thực hiện

```text
Kiểm tra môi trường
  → audit raw cả 3 domain
  → prepare + audit processed từng domain
  → train + evaluate 6 experiment
  → compare 3 cặp cùng domain
  → error analysis 6 experiment
  → inference tùy chọn
```

Chạy từ repository root. Nếu một bước báo metadata, fingerprint, inventory hoặc compatibility không khớp thì dừng và xử lý trước khi sang bước sau.

## Step 0 — Chuẩn bị môi trường

Lấy Git submodule và cài dependencies trong môi trường của người chạy:

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

Các lệnh cài đặt và test trên chưa được chạy trong task này.

## Step 1 — Audit dữ liệu gốc của cả ba domain

```bash
python -m capstone.audit_data \
  --all-domains \
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

Đọc review/sentence/quad counts, empty-label sentences, implicit elements, aspect/sentiment distributions, invalid annotations và suspicious spans cho từng domain. Hotels clean mode ưu tiên `Hotels_corr.xml` khi có. Không điền số liệu vào báo cáo trước khi audit thật hoàn tất.

## Step 2 — Chuẩn bị clean split cho cả ba domain

```bash
for domain in amazon_ff coursera hotels; do
  python -m capstone.prepare_data \
    --domain "$domain" \
    --split-mode clean_split \
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

Clean split chia ở review level 80/10/10 để các câu trong cùng review không sang nhiều split. Review chưa có bằng chứng là lỗi được giữ với trạng thái `unknown`. Nếu cần loại review đã xác nhận chưa annotate, tạo manifest có provenance riêng cho domain, dùng `--unannotated-manifest`, và cập nhật config cùng cleaning policy trước khi train.

`metadata.json` phải giữ domain, seed, split mode, sentiment mode, cleaning policy, aspect inventory, source/split SHA-256, review IDs, zero-label statuses và issue list.

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

Trước khi train, xác nhận `review_overlap` rỗng, `no_review_leakage: true`, domain/seed/split khớp config và cả ba JSONL có fingerprint trong metadata.

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

Error analysis giữ IDs, text, raw generation, TP/FP/FN, field mismatch, missing/extra quadruple, implicit failures và parser errors.

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
| Split | `clean_split` | `clean_split` |
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

Với mỗi seed bổ sung, tạo sáu config mới hoặc bản sinh ra từ template với:

- `experiment.seed` và `experiment.name` riêng;
- `paths.processed_data` trỏ tới root riêng, ví dụ `data/processed/seed43`;
- `paths.output_dir` riêng, có seed trong tên;
- prepare lại cả ba clean split bằng đúng seed và output root đó;
- compare GAS/Paraphrase trong cùng domain và cùng seed;
- tính mean/std riêng cho từng `(domain, model)` sau khi đủ năm seed.

`original_split` dùng TXT gốc để điều tra gần protocol paper. `clean_split` là protocol chính của project để giữ review grouping. Không đặt F1 của hai protocol vào cùng một cột hoặc diễn giải chênh lệch như lỗi reproduction.

Paper reference ASQP F1:

| Method | Amazon_FF | Coursera | Hotels |
|---|---:|---:|---:|
| GAS-T5 | 19.62 | 22.23 | 26.33 |
| Paraphrase | 20.84 | 19.78 | 34.51 |

Đây là số trong paper, không phải output của repository.

## Checklist hoàn tất seed 42

```text
[ ] 3 raw audit JSON + summary
[ ] 3 processed dataset, mỗi dataset có train/dev/test/metadata
[ ] 3 processed audit, no_review_leakage = true
[ ] 6 experiment directories có best_checkpoint.json
[ ] 6 evaluation directories có metrics và predictions
[ ] 3 comparison JSON đúng domain
[ ] 6 error_analysis.jsonl
[ ] Báo cáo 6 hàng kết quả riêng theo domain/model
```

## Trạng thái kiểm chứng

- `UNVERIFIED — requires execution`: test suite và dependency/runtime compatibility.
- `UNVERIFIED — requires execution`: raw audit, preparation, fingerprints và review leakage trên cả ba domain.
- `UNVERIFIED — requires execution`: sáu training/evaluation runs, checkpoint selection và GPU memory.
- `UNVERIFIED — requires execution`: metrics, comparisons, error analysis, inference và mọi số liệu kết quả.
