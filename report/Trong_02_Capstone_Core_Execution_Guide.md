# Trọng — Capstone Core Execution Guide

> Hiện tại mới hoàn thành phần code. Pipeline chưa được chạy, model chưa được train và chưa có kết quả benchmark thực tế.

**Mọi command trong hướng dẫn này chỉ được viết để chạy sau. Chưa command nào được thực thi trong task này.**

## Execution Flow

```text
STEP 0   Environment check
   ↓
STEP 1   Audit raw OATS data
   ↓
STEP 2   Prepare clean dataset
   ↓
STEP 3   Inspect processed dataset
   ↓
STEP 4   Train GAS-T5 extraction-style adaptation
   ↓
STEP 5   Evaluate GAS-T5 adaptation
   ↓
STEP 6   Train Paraphrase-T5
   ↓
STEP 7   Evaluate Paraphrase-T5
   ↓
STEP 8   Compare GAS vs Paraphrase
   ↓
STEP 9   Error analysis
   ↓
STEP 10  Optional inference on custom sentence
```

Chạy các command từ repository root. Nếu một bước lỗi hoặc metadata không khớp config, dừng và xử lý lỗi trước khi tiếp tục.

## Step 0 — Environment check

**Purpose:** kiểm tra Python và runtime dependencies đã có trong môi trường.

**Input:** môi trường Python hiện có. Dependencies runtime liệt kê trong `requirements.txt`; pytest được liệt kê thêm trong `requirements-dev.txt`.

```bash
python --version
python -m pip show torch transformers PyYAML numpy pytest
```

**Expected output / location:** phiên bản Python và package information trên terminal; không tạo artifact.

**Check trước Step 1:** xác nhận Python tương thích với cú pháp package, cùng PyTorch, Transformers, PyYAML và NumPy. Model `t5-base` cần có sẵn trong cache hoặc được môi trường cho phép lấy về khi người dùng chạy training/inference. Không cài dependency trong task implementation này.

## Step 1 — Audit raw OATS data

**Purpose:** tạo số liệu review/sentence/quad, label distributions, implicit elements, invalid annotations và suspicious spans.

**Input:** dữ liệu OATS ở `OATS-ABSA-Dataset/data`; Hotels clean mode chọn `Hotels_corr.xml` nếu có.

```bash
python -m capstone.audit_data --domain hotels --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit
```

Để audit ba domain:

```bash
python -m capstone.audit_data --all-domains --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit
```

**Expected output / location:** `artifacts/data_audit/hotels.json` (hoặc một JSON cho mỗi domain) và `artifacts/data_audit/summary.json`.

**Check trước Step 2:** review/sentence count, empty-label sentence, quadruple count, explicit/implicit target/opinion, invalid `POLARITY#*`, malformed quadruple, suspicious annotation. Đây là output sẽ được tạo khi chạy; hiện chưa có số audit được xác nhận.

## Step 2 — Prepare clean dataset

**Purpose:** làm sạch và chia dữ liệu ở review level 80/10/10 bằng seed cố định; không tách câu cùng review sang các split khác nhau.

**Input:** XML cho Hotels và domain; XML giữ review IDs. Review có 0 quadruple mặc định mang trạng thái `unknown` và được giữ; câu rỗng trong review còn annotation cũng được giữ. Chỉ review được xác nhận rõ trong manifest có provenance mới bị loại.

```bash
python -m capstone.prepare_data --domain hotels --split-mode clean_split --seed 42 --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-root data/processed
```

Tùy chọn, tạo manifest JSON có nguồn chứng cứ rõ ràng rồi truyền `--unannotated-manifest annotation_status/hotels.json`. Schema gồm `domain`, chuỗi `provenance` không rỗng, `confirmed_unannotated_review_ids` (dùng `[]` nếu không có) và tùy chọn `legitimate_zero_opinion_review_ids`. Không đưa ID chưa được xác nhận vào danh sách. Nếu dùng manifest, cấu hình train/evaluate phải khai báo cùng đường dẫn tại `cleaning.unannotated_reviews_manifest` và bật `cleaning.remove_unannotated_reviews: true`; mặc định hai config mẫu không dùng manifest và giữ review unknown.

**Expected output / location:**

```text
data/processed/hotels/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── metadata.json
```

`metadata.json` giữ domain, split, seed, sentiment mode, cleaning policy, ba trạng thái zero-label, manifest/provenance (nếu có), aspect inventory, source paths/SHA-256, JSONL split SHA-256, statistics, review IDs, excluded reviews và issue list. Preparation dừng nếu manifest chứa ID không có trong XML đã chọn, hoặc một review có annotation attempt/parser-cleaning issue bị gán trạng thái zero-label. Để dùng original split cho investigation/reproduction, thay `--split-mode clean_split` bằng `--split-mode original_split`; câu TXT được căn chỉnh với XML. Khi câu trùng không thể gắn review duy nhất từ text và labels, preparation sẽ dừng thay vì đoán ID.

**Check trước Step 3:** đủ ba JSONL và metadata; `metadata.json` ghi `domain=hotels`, `split_mode=clean_split`, `seed=42`, `sentiment_mode=paper_compatible`, `split_fingerprints`, `source_fingerprints`, aspect inventory và zero-label status counts. Cả `raw_4_class` và `paper_compatible` hiện giữ nguyên Conflict; không tự map sang neutral.

## Step 3 — Inspect processed dataset

**Purpose:** kiểm tra train/dev/test JSONL counts, zero-label statuses, aspect inventory, fingerprints và review leakage trước khi training.

**Input:** `data/processed/hotels/{train,dev,test}.jsonl`.

```bash
python -m capstone.audit_data --processed data/processed/hotels --output-dir artifacts/data_audit
```

**Expected output / location:** `artifacts/data_audit/processed.json` và `artifacts/data_audit/summary.json`. Trong `processed.json`, kiểm tra `review_overlap` rỗng ở cả train/dev/test và `no_review_leakage: true`.

**Check trước Step 4:** xác nhận review counts, sentence counts, empty labels, quadruple counts và không overlap. Training/evaluation cũng so metadata đã chuẩn bị với config; nếu khác seed/domain/split/sentiment/cleaning policy/dataset version, code sẽ báo lỗi.

## Step 4 — Train GAS-T5 baseline adaptation

**Purpose:** train T5 theo GAS extraction-style ASQP adaptation.

**Input:** `configs/gas_t5_hotels.yaml` và processed Hotels data từ Steps 2–3.

```bash
python -m capstone.train --config configs/gas_t5_hotels.yaml
```

**Expected output / location:** `artifacts/experiments/gas_t5_hotels_seed42/` có `config.yaml`, `metadata.json`, `training_history.json`, `training_summary.json`, `best_checkpoint.json`, `checkpoints/checkpoint-epoch-XXX/` và logs trong `logs/`.

Best checkpoint được chọn bằng `validation_exact_match_f1`; khi nhiều epoch hòa, checkpoint đầu tiên giữ vị trí best. `best_checkpoint.json` ghi epoch, score và path. Có thể truyền experiment folder trực tiếp cho evaluate/predict để resolver lấy best checkpoint.

**Tham số config GAS:** T5-base; 20 epochs; batch 16; gradient accumulation 1 (effective batch 16); learning rate `0.0003`; weight decay `0`; AdamW; linear scheduler; warmup ratio `0`; input/output length 128; greedy generation (`num_beams=1`); `max_new_tokens=128`; seed 42; `hotels/clean_split`; `paper_compatible`; target format `extraction_style_quadruple_json`.

**Check trước Step 5:** experiment folder, metadata, history, best checkpoint manifest và checkpoint folder đã được tạo. Không có giá trị loss/F1 dự kiến được điền trước.

> GAS target syntax chính xác của GAS-T5 OATS trong paper không được công bố. Target format của code là extraction-style adaptation dựa trên GAS reference, không tự nhận là reproduction chính xác.

## Step 5 — Evaluate GAS-T5

**Purpose:** generation trên test split, parsing, normalization, exact-match metrics, diagnostics và error cases.

**Input:** config GAS, best checkpoint của Step 4 và `test.jsonl`.

```bash
python -m capstone.evaluate --config configs/gas_t5_hotels.yaml --checkpoint artifacts/experiments/gas_t5_hotels_seed42 --split test
```

**Expected output / location:** `artifacts/experiments/gas_t5_hotels_seed42/evaluation/`:

```text
metrics.json
diagnostic_metrics.json
predictions.jsonl
invalid_predictions.jsonl
error_cases.jsonl
run_metadata.json
```

`metrics.json` ghi TP/FP/FN, precision, recall, exact-match F1. Một quad chỉ TP nếu cả target, aspect, opinion và sentiment khớp trong cùng sentence. `diagnostic_metrics.json` không thay official score. `predictions.jsonl` giữ IDs, text, gold, raw generation, parsed/normalized output.

**Check trước Step 6:** đúng domain/split/checkpoint, sample count không rỗng, invalid generations được ghi ra artifact, không dùng dev metrics thay cho test metrics.

## Step 6 — Train Paraphrase-T5

**Purpose:** train mô hình chính theo output paraphrase tự nhiên.

**Input:** `configs/paraphrase_t5_hotels.yaml` và cùng processed Hotels data.

```bash
python -m capstone.train --config configs/paraphrase_t5_hotels.yaml
```

**Expected output / location:** `artifacts/experiments/paraphrase_t5_hotels_seed42/` với cùng layout checkpoint, config, metadata, history, summary và logs như GAS.

**Tham số config Paraphrase:** `t5-base` tokenizer/backbone; epochs 20; batch 16; gradient accumulation 1 (effective batch 16); learning rate `0.0003`; weight decay 0; AdamW; linear scheduler; warmup ratio 0; input/output length 128; greedy (`num_beams=1`); `max_new_tokens=128`; seed 42; domain Hotels; `clean_split`; `paper_compatible`. Target có dạng `{aspect} is great/ok/bad because {target} is {opinion}`; implicit target/opinion dùng `it`. `mixed` cho conflict là extension của adapter.

**Check trước Step 7:** checkpoint và best selection thuộc experiment Paraphrase, không GAS; metadata khớp processed data.

## Step 7 — Evaluate Paraphrase-T5

**Purpose:** evaluate test set qua paraphrase parser và shared evaluator.

**Input:** Paraphrase config, best checkpoint và test split.

```bash
python -m capstone.evaluate --config configs/paraphrase_t5_hotels.yaml --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --split test
```

**Expected output / location:** `artifacts/experiments/paraphrase_t5_hotels_seed42/evaluation/` có `metrics.json`, `diagnostic_metrics.json`, `predictions.jsonl`, `invalid_predictions.jsonl`, `error_cases.jsonl`, `run_metadata.json`.

**Check trước Step 8:** output split/test count tương thích GAS; raw generation và parser issues được lưu; dùng exact quadruple F1 để báo cáo.

## Step 8 — Compare models

**Purpose:** viết comparison JSON chỉ khi experiment tương thích.

**Input:** hai experiment folders đã có metadata và evaluation artifacts. Code kiểm domain, split mode, dataset version, sentiment mode, seed, cleaning policy, evaluation split/sample count, train/dev counts và training/sequence/generation settings.

```bash
python -m capstone.compare --gas-experiment artifacts/experiments/gas_t5_hotels_seed42 --paraphrase-experiment artifacts/experiments/paraphrase_t5_hotels_seed42 --output-dir artifacts/comparisons
```

**Expected output / location:** `artifacts/comparisons/hotels_gas_vs_paraphrase.json`, gồm model, precision, recall, exact-match F1, invalid generation count, seed, config, checkpoint và sample counts.

**Check trước Step 9:** xác nhận split và count của hai model giống nhau; comparison không phải paper benchmark mới.

## Step 9 — Error analysis

**Purpose:** tái tạo error case classifications từ saved predictions JSONL và lưu riêng để không ghi đè file evaluator.

**Input:** predictions của từng model.

```bash
python -m capstone.error_analysis --predictions artifacts/experiments/gas_t5_hotels_seed42/evaluation/predictions.jsonl --output artifacts/experiments/gas_t5_hotels_seed42/evaluation/error_analysis.jsonl
python -m capstone.error_analysis --predictions artifacts/experiments/paraphrase_t5_hotels_seed42/evaluation/predictions.jsonl --output artifacts/experiments/paraphrase_t5_hotels_seed42/evaluation/error_analysis.jsonl
```

**Expected output / location:** mỗi experiment có `evaluation/error_analysis.jsonl` với TP/FP/FN, field mismatch, missing/extra quadruple, implicit target/opinion failure, parser error types.

**Check trước Step 10:** sampling lỗi theo raw generation và IDs; phân biệt malformed output với quadruple hợp lệ nhưng sai nhãn.

## Step 10 — Optional inference trên câu riêng

**Purpose:** dự đoán một text hoặc file input sau khi đã có checkpoint.

**Input:** experiment directory có `best_checkpoint.json`, model type và text.

```bash
python -m capstone.predict --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --model-type paraphrase --text "The room was clean and comfortable." --output artifacts/experiments/paraphrase_t5_hotels_seed42/custom_prediction.json
```

File TXT đọc mỗi dòng là một câu; file JSONL cần mỗi object có trường string `text`. File input xuất JSONL:

```bash
python -m capstone.predict --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --model-type paraphrase --input-file input_sentences.jsonl --output artifacts/experiments/paraphrase_t5_hotels_seed42/custom_predictions.jsonl
```

**Expected output / location:** path được truyền cho `--output`, hoặc JSON/JSONL trên terminal. Config và persisted aspect inventory được tự dò theo experiment/checkpoint; inference dừng nếu không tìm được inventory hợp lệ.

**Check:** output chứa `raw_generation`, quadruples và `invalid_predictions`; inference là custom sample, không được cộng vào test score.

## Output Directory Map

```text
data/processed/hotels/
├── train.jsonl               # Sentence-level examples giữ review_id/sentence_id
├── dev.jsonl                 # Validation examples
├── test.jsonl                # Test examples
└── metadata.json             # Split, statuses, inventory, SHA-256, IDs, counts và issues

artifacts/data_audit/
├── hotels.json               # Raw audit theo domain
├── processed.json            # Processed counts và review overlap
└── summary.json              # Tóm tắt output audit

artifacts/experiments/
├── gas_t5_hotels_seed42/
│   ├── config.yaml           # Snapshot config đã dùng
│   ├── metadata.json         # Dataset/model/seed/checkpoint metadata
│   ├── best_checkpoint.json  # Epoch được chọn theo validation exact-match F1
│   ├── training_history.json # Loss, LR, validation F1 và elapsed time theo epoch
│   ├── training_summary.json # Epoch count và best selection
│   ├── checkpoints/          # Checkpoint folders theo epoch
│   ├── logs/                 # Run/training/evaluation logs
│   └── evaluation/           # Metrics, predictions, invalid và error JSONL
└── paraphrase_t5_hotels_seed42/
    └── ...                   # Layout tương tự GAS

artifacts/comparisons/
└── hotels_gas_vs_paraphrase.json # Kết quả so sánh hai experiment tương thích
```

Các path trên là expected locations từ code; hiện chưa được tạo bởi pipeline.

## Metrics to Record

**Official ASQP:** precision, recall, exact-match F1 và TP/FP/FN. Partial matches là wrong prediction; metric cộng counts sau khi match trong từng sentence record.

**Diagnostics:** target/aspect/opinion/sentiment component F1; aspect và sentiment accuracy được tính trong alignment mà ba trường còn lại khớp và có nhãn diagnostic-only; explicit/implicit target và opinion; per-aspect; per-sentiment; nhóm 0/1/2/3+ quadruple; example exact-match accuracy.

**Training:** train loss, validation loss, learning rate, validation exact-match F1, epoch/step, elapsed time; best checkpoint, selection epoch và số epoch hoàn tất. Những giá trị chỉ xuất hiện sau khi người dùng chạy.

## Model and Training Parameters

| Parameter | GAS-T5 adaptation | Paraphrase-T5 | Nguồn |
|---|---|---|---|
| `model_name` / tokenizer | `t5-base` / `t5-base` | `t5-base` / `t5-base` | Config |
| Target format | Extraction-style quadruple tuple JSON | Natural paraphrase segments `[SSEP]` | Config/code |
| Epochs | 20 | 20 | Config; OATS-specific full setting chưa công bố |
| Batch size | 16 | 16 | Config/reference |
| Gradient accumulation | 1 | 1 | Config assumption |
| Effective batch size | 16 | 16 | Batch × accumulation |
| Learning rate | `0.0003` | `0.0003` | Config/reference |
| Weight decay | `0.0` | `0.0` | Config |
| Optimizer | AdamW | AdamW | Config |
| Scheduler / warmup | linear / `0.0` | linear / `0.0` | Config |
| Input/output length | 128 / 128 | 128 / 128 | Config assumption |
| Beams / max new tokens | 1 / 128 | 1 / 128 | Config |
| Seed | 42 | 42 | Config; một seed, không phải paper mean |
| Domain / split | Hotels / clean_split | Hotels / clean_split | Config |
| Sentiment mode | paper_compatible (preserve Conflict) | paper_compatible (preserve Conflict) | Config |

Paper OATS báo cáo T5-base, 80/10/10 split và trung bình 5 seeds, nhưng không công bố đầy đủ toàn bộ hyperparameters của experiment GAS-T5/Paraphrase-T5. Do đó settings trên là config hiện tại, không phải bằng chứng đã tái tạo paper setting.

## Checkpoint Selection

Mỗi epoch sinh dev predictions và tính exact-match F1. Best được chọn theo `validation_exact_match_f1`; epoch đầu tiên giữ selection nếu các epoch sau hòa score. Manifest nằm tại:

```text
artifacts/experiments/<experiment-name>/best_checkpoint.json
```

Checkpoint folder được ghi trong field `path`, thường là `checkpoints/checkpoint-epoch-XXX/`. Evaluate/predict chấp nhận experiment folder để resolve checkpoint này. Không có checkpoint giả trong repository.

## Checklist sau từng bước

```text
[ ] Audit JSON được tạo; không điền số liệu trước khi chạy
        ↓
[ ] Processed train/dev/test và metadata khớp config
        ↓
[ ] Review overlap rỗng, no_review_leakage = true
        ↓
[ ] GAS best_checkpoint.json và checkpoint folder đã có
        ↓
[ ] GAS test metrics/predictions/error artifacts đã có
        ↓
[ ] Paraphrase best checkpoint và metadata đã có
        ↓
[ ] Paraphrase test metrics/predictions/error artifacts đã có
        ↓
[ ] Comparison được tạo và compatibility không báo mismatch
```

## Paper comparison: original_split và clean_split

`original_split` dùng TXT split repository để phục vụ investigation gần paper. TXT không chứa review IDs, nên code cần căn chỉnh an toàn với XML; ambiguous duplicate rows sẽ báo lỗi. `clean_split` chia review ở 80/10/10, giữ zero-quadruple reviews ở trạng thái `unknown` trừ khi manifest chứng cứ xác nhận, và là experiment chính để tránh leakage do review grouping. F1 của `clean_split` không cùng protocol với paper; chênh lệch không tự động đồng nghĩa reproduction failure.

Paper reference ASQP F1:

| Method | Amazon_FF | Coursera | Hotels |
|---|---:|---:|---:|
| GAS-T5 | 19.62 | 22.23 | 26.33 |
| Paraphrase | 20.84 | 19.78 | 34.51 |

Đây là con số paper, không phải output của project.

## Multi-seed và Smoke/Full runs

Thực hiện trước một seed (`42`) để tạo artifacts riêng. Multi-seed chưa có CLI aggregate tự động. Với mỗi seed tiếp theo (ví dụ 43–46), tạo config riêng đổi `experiment.name`, `experiment.seed`, `paths.processed_data`, `paths.output_dir`; prepare lại clean split tương ứng bằng cùng seed và output-root riêng. Sau khi mỗi seed đã có test F1, mới tính mean và standard deviation. Không thay config/artifacts của seed đã chạy.

Code không có smoke mode, tiny fixture training config hoặc command sinh toy dataset. Có thể thay `epochs` trong một bản config riêng vì số epoch là config, nhưng đó vẫn là training trên processed split đã cung cấp, không phải smoke pipeline isolated. Không ghi kết quả smoke/full cho đến khi người dùng chạy.

**Test suite chưa được chạy.** `UNVERIFIED — requires execution`.

## Quick Run Order

Đây là command sequence copy-paste được từ repository root. Các command chưa được chạy trong task này.

```bash
# 1. Audit raw Hotels
python -m capstone.audit_data --domain hotels --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit

# 2. Prepare clean review-level split
python -m capstone.prepare_data --domain hotels --split-mode clean_split --seed 42 --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-root data/processed

# 3. Inspect processed split and review leakage
python -m capstone.audit_data --processed data/processed/hotels --output-dir artifacts/data_audit

# 4. Train GAS extraction-style adaptation
python -m capstone.train --config configs/gas_t5_hotels.yaml

# 5. Evaluate GAS best checkpoint on test
python -m capstone.evaluate --config configs/gas_t5_hotels.yaml --checkpoint artifacts/experiments/gas_t5_hotels_seed42 --split test

# 6. Train Paraphrase
python -m capstone.train --config configs/paraphrase_t5_hotels.yaml

# 7. Evaluate Paraphrase best checkpoint on test
python -m capstone.evaluate --config configs/paraphrase_t5_hotels.yaml --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --split test

# 8. Compare compatible experiments
python -m capstone.compare --gas-experiment artifacts/experiments/gas_t5_hotels_seed42 --paraphrase-experiment artifacts/experiments/paraphrase_t5_hotels_seed42 --output-dir artifacts/comparisons

# 9. Optional separate error analysis
python -m capstone.error_analysis --predictions artifacts/experiments/gas_t5_hotels_seed42/evaluation/predictions.jsonl --output artifacts/experiments/gas_t5_hotels_seed42/evaluation/error_analysis.jsonl

# 10. Optional inference on a custom sentence
python -m capstone.predict --checkpoint artifacts/experiments/paraphrase_t5_hotels_seed42 --model-type paraphrase --text "The room was clean and comfortable." --output artifacts/experiments/paraphrase_t5_hotels_seed42/custom_prediction.json
```

**UNVERIFIED — requires execution.** Audit, preparation, review leakage, config/model loading, training, inference, evaluation, comparison, output paths and all metrics need an actual user-authorized run to verify.
