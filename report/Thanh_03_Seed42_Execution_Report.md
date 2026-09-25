# Báo cáo chạy OATS seed 42

**Trạng thái: BLOCKED tại raw audit.** Preflight môi trường và 113 test đã qua, nhưng lệnh raw audit dừng ở `amazon_ff/quads/train.txt:12`. Theo execution plan, không tiếp tục chuẩn bị dữ liệu hoặc huấn luyện cho tới khi ambiguity này được xử lý. Không có metric hay model result nào được tạo trong lượt này.

## Phạm vi và provenance

- Phạm vi dự kiến: seed 42; `amazon_ff`, `coursera`, `hotels`; GAS extraction-style ASQP adaptation và Paraphrase-T5; sáu train/evaluate runs.
- Thời gian chạy: bắt đầu `2026-09-25 09:28:01 +07:00`; dừng tại raw audit (log ghi lần cuối `2026-09-25 09:30:35 +07:00`). Múi giờ: Asia/Saigon (UTC+07:00).
- Repository: `D:\DS-AI\Web mining project\Generative-Aspect-Based-Sentiment-Analysis-on-OATS`.
- Branch: `run/full-oats-seed42`; code commit trước pipeline: `8f541e98f780f3b5481f4313adc2714236d552c8` (`fix: align tests and diagnostics with pipeline behavior`). `main` vẫn ở `d749485640a880cf95ae46589522c9fcc83f49d8`.
- Sau khi pipeline đã dừng, final review tìm thấy paired aspect mismatch bị gắn nhầm implicit-target diagnostics. Chỉ diagnostics/test được sửa ở commit `35df91766fe5470214d3138bf6a11cd201d79883`; sửa đổi này không nằm trong pipeline attempt và không đổi official TP/FP/FN, split hay config.
- OATS submodule: `2089631462f06fc0d0ac07186fa65236e19784ce`.
- Runtime: Python 3.11.16; PyTorch 2.14.0+cu132; Transformers 4.57.6; PyYAML 6.0.3; NumPy 2.4.6; pytest 9.1.1; SentencePiece 0.2.2.
- GPU: NVIDIA GeForce RTX 4060 Ti, 16,380 MiB; driver 595.79; CUDA runtime 13.2.
- Sáu config seed 42 được load và khớp ma trận domain/model; sáu CLI `--help` đều exit 0. Không thay đổi config, API, CLI hoặc methodology.

## Các bước đã chạy

| Bước | Kết quả | Bằng chứng |
|---|---|---|
| `python -m pip check` | PASS — không có broken requirements | `artifacts/setup/preflight-execution.log` |
| CUDA matrix smoke | PASS trên RTX 4060 Ti | `artifacts/setup/gpu-smoke-seed42.json` |
| T5-base load/generation smoke | PASS trên `cuda:0` | `artifacts/setup/t5-cuda-smoke-seed42.json` |
| Sáu config và sáu CLI help | PASS | `artifacts/setup/preflight-execution.log`, `artifacts/setup/cli-*-help.log` |
| Full test suite trước pipeline | PASS — 113/113, 3.50 giây; dùng `--basetemp` riêng | `artifacts/setup/pytest-full-seed42.log` |
| Raw audit cả ba domain | BLOCKED — exit code 1 ở Amazon_FF trước khi ghi JSON | `artifacts/setup/raw-audit-seed42.log` |

Sau post-stop diagnostics fix, full suite được xác minh lại trên branch cuối: 114/114 pass trong 4.25 giây; test metrics liên quan pass 13/13. Log lần lượt là `artifacts/setup/pytest-final-seed42.log` và `artifacts/setup/pytest-review-fix-green.log`.

### Commands và exit status

- `D:\Anaconda\envs\oats-seed42\python.exe --version` — exit 0; Python 3.11.16.
- `D:\Anaconda\envs\oats-seed42\python.exe -m pip check` — exit 0.
- `D:\Anaconda\envs\oats-seed42\python.exe -m pip show torch transformers PyYAML numpy pytest sentencepiece` — exit 0.
- `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` — exit 0.
- Python smoke (`torch.cuda` matrix operation, load `t5-base`, generate tối đa 4 token trên `cuda:0`) — exit 0.
- Python validation dùng `capstone.config.load_config` với sáu YAML chuẩn — exit 0. Từng lệnh `python -m capstone.audit_data --help`, `python -m capstone.prepare_data --help`, `python -m capstone.train --help`, `python -m capstone.evaluate --help`, `python -m capstone.compare --help` và `python -m capstone.error_analysis --help` — đều exit 0.
- `D:\Anaconda\envs\oats-seed42\python.exe -m pytest -ra --basetemp "D:\DS-AI\Web mining project\Generative-Aspect-Based-Sentiment-Analysis-on-OATS\artifacts\setup\pytest-basetemp-seed42-20260925-092843-432"` — exit 0, 113/113 trước pipeline.
- `D:\Anaconda\envs\oats-seed42\python.exe -m pytest -ra --basetemp "D:\DS-AI\Web mining project\Generative-Aspect-Based-Sentiment-Analysis-on-OATS\artifacts\setup\pytest-final-seed42-20260925-094559-280"` — exit 0, 114/114 trên branch sau review fix.
- Raw audit command ở trên — exit 1; traceback đầy đủ trong `artifacts/setup/raw-audit-seed42.log`.

Lệnh raw audit đã thử:

```powershell
python -m capstone.audit_data --all-domains --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit/raw
```

## Blocker tại raw audit

Traceback kết thúc với:

```text
capstone.data.parser.DataFormatError: OATS-ABSA-Dataset\data\amazon_ff\quads\train.txt:12: sentence text matches multiple unassigned XML sentences and labels do not identify one uniquely
```

Kiểm tra chỉ đọc xác nhận dòng 12 có text `Delicious !` và quadruple `NULL / food quality / delicious / positive`. Có ba XML sentence chưa được gán cùng khớp text và cùng quadruple:

| Review ID | Sentence ID | Quadruple |
|---|---|---|
| `1636` | `1636: 0` | `NULL / food quality / delicious / positive` |
| `785` | `785: 5` | `NULL / food quality / delicious / positive` |
| `244` | `244: 1` | `NULL / food quality / delicious / positive` |

Split TXT không cung cấp review/sentence ID để phân biệt ba ứng viên. Parser dừng thay vì tự gán một review không có bằng chứng. Raw audit tính kết quả cho các domain trước khi ghi artifacts, nên lần chạy này không tạo `artifacts/data_audit/raw/*.json` hay `summary.json`.

## Data gates

| Domain | Raw audit | Processed dataset / audit | Leakage | Source/split fingerprints |
|---|---|---|---|---|
| `amazon_ff` | BLOCKED — ambiguity tại `train.txt:12` | Chưa chạy | Chưa kiểm tra | Chưa có metadata |
| `coursera` | Chưa chạy vì lệnh all-domain đã dừng | Chưa chạy | Chưa kiểm tra | Chưa có metadata |
| `hotels` | Chưa chạy vì lệnh all-domain đã dừng | Chưa chạy | Chưa kiểm tra | Chưa có metadata |

Không có `data/processed/`, processed audit, split fingerprint hay leakage result nào được tạo. Không thể kết luận về chất lượng/chia tách dữ liệu từ lượt chạy này.

## Sáu experiment dự kiến

`—` nghĩa là chưa có kết quả, không phải 0. Không có experiment directory/checkpoint nào được tạo.

| Domain | Model | Status | Best epoch | Best validation F1 | Test P | Test R | Exact-match F1 | TP | FP | FN | Invalid generations | Experiment path |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `amazon_ff` | GAS extraction-style ASQP adaptation | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/gas_t5_amazon_ff_seed42/` |
| `coursera` | GAS extraction-style ASQP adaptation | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/gas_t5_coursera_seed42/` |
| `hotels` | GAS extraction-style ASQP adaptation | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/gas_t5_hotels_seed42/` |
| `amazon_ff` | Paraphrase-T5 | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/paraphrase_t5_amazon_ff_seed42/` |
| `coursera` | Paraphrase-T5 | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/paraphrase_t5_coursera_seed42/` |
| `hotels` | Paraphrase-T5 | NOT STARTED — raw audit blocked | — | — | — | — | — | — | — | — | — | `artifacts/experiments/paraphrase_t5_hotels_seed42/` |

## Comparisons và error analysis

Chưa có evaluation nên không tạo ba same-domain comparison JSON hoặc sáu `error_analysis.jsonl`. Các metric chính thức, diagnostic metrics, parser failure rates và implicit target/opinion errors đều chưa có kết quả trong lượt này. Không dùng số trong paper thay cho output của project.

## Artifacts và giới hạn

- Artifact root trên máy này: `D:\DS-AI\Web mining project\Generative-Aspect-Based-Sentiment-Analysis-on-OATS\artifacts`.
- Dung lượng `artifacts/` sau final verification: 408,718 bytes (khoảng 0.39 MiB; 534 file, chủ yếu là setup/test evidence). `data/processed/` chưa tồn tại.
- Artifacts chỉ lưu cục bộ trên ổ D:, không upload và không đưa checkpoint/dữ liệu lớn lên Git.
- Nếu chạy thành công về sau, GAS phải được mô tả là extraction-style ASQP adaptation; không tuyên bố tái tạo chính xác target syntax GAS-T5 của paper.
- Không có deviation về config/methodology. Pipeline dừng đúng tại raw audit theo stop condition; không chạy prepare, train, evaluate, compare, error analysis hay inference.

## Điều kiện để tiếp tục

Cần quyết định cách xử lý những dòng split TXT có text và quadruple trùng nhau trong nhiều XML sentence. Một hướng có thể xem xét là tách raw XML audit khỏi bước căn chỉnh split TXT; hướng khác cần nguồn mapping có review/sentence ID đáng tin cậy. Chưa áp dụng workaround hay thay đổi code trong lượt này. Sau khi có hướng được duyệt, cần thêm kiểm thử cho ambiguity và chạy lại các gate bị ảnh hưởng trước khi tiếp tục.
