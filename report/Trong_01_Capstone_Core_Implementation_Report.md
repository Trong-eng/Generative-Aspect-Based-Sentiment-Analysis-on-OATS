# Trọng — Capstone Core Implementation Report

## 1. Mục tiêu Capstone

ABSA (Aspect-Based Sentiment Analysis) xác định ý kiến theo từng khía cạnh trong văn bản. ASQP (Aspect Sentiment Quad Prediction) dự đoán đồng thời bốn trường của mỗi ý kiến: **target**, **aspect category**, **opinion phrase**, **sentiment**. OATS cung cấp annotation sentence-level cho các quadruple này ở ba domain Amazon_FF, Coursera và Hotels.

Phạm vi code là pipeline dữ liệu → mô hình sinh → đánh giá. Không xây crawler, dashboard, backend, deploy hay hệ thống cloud.

## 2. Repository ban đầu

Repository ban đầu có README, dữ liệu OATS trong `OATS-ABSA-Dataset/` và converter cũ `OATS-ABSA-Dataset/data/xml_to_aste_v2.py`; chưa có capstone package, cấu hình thí nghiệm, training/evaluation pipeline hay test suite cho ASQP. Converter ASTE không giữ đủ bốn trường ASQP và không giữ review grouping cần cho split, nên pipeline mới đọc trực tiếp XML/TXT. Dữ liệu gốc và converter cũ không bị sửa.

Đã đọc cấu trúc XML, các split TXT, README dữ liệu và paper được cung cấp `2024.lrec-main.1080.pdf`. Paper được đối chiếu thêm với [bản ACL Anthology](https://aclanthology.org/2024.lrec-main.1080/). Các tài liệu này được dùng làm nguồn định dạng và assumptions; không dùng như hướng dẫn để chạy lệnh.

## 3. Dataset OATS

Parser hỗ trợ các domain `amazon_ff`, `coursera`, `hotels`. Mặc định xử lý Hotels ưu tiên `Hotels_corr.xml` nếu file tồn tại; chế độ `original_split` dùng `Hotels.xml` để căn chỉnh với các split TXT gốc. XML giữ `Review → sentence → Opinion`, bao gồm `rid`, `sid`, text, `target`, `category`, `opinion_phrase` và `polarity`. Text split có thứ tự label `[target, aspect, sentiment, opinion]`, được đổi về shared representation `[target, aspect, opinion, sentiment]`.

Target và opinion có thể là `NULL`; câu có thể có 0, 1 hoặc nhiều quadruple. `SentenceRecord` luôn giữ `review_id` và `sentence_id` để chuẩn bị dữ liệu và phân tích lỗi không mất quan hệ review.

## 4. Vấn đề split cần lưu ý

Task brief và cấu trúc dữ liệu cho thấy cần kiểm tra các câu có `[]` và review XML chưa được annotate. Không được xóa từng câu rỗng một cách máy móc: câu rỗng trong một review đã có annotation vẫn là negative example hợp lệ. Review không có quadruple được phân thành `confirmed_unannotated`, `legitimate_zero_opinion` hoặc `unknown`; không có manifest chứng cứ thì mặc định là `unknown` và được giữ. Chỉ ID ghi rõ trong manifest có provenance mới có thể bị loại. Audit ghi các trạng thái này và leakage. **Chưa chạy audit nên không báo số liệu Amazon_FF, Coursera hoặc Hotels, và chưa kết luận nguyên nhân split từ thực nghiệm.**

## 5. Data Audit

`capstone/audit_data.py` dùng `capstone/data/audit.py` và hỗ trợ một domain, tất cả domain, cùng thư mục processed. Audit trả số review/sentence, câu có quadruple, empty-label sentence, quadruple, explicit/implicit target và opinion, aspect/sentiment distributions, malformed annotation, invalid category và suspicious offset. Raw audit dự kiến ghi `artifacts/data_audit/{domain}.json` và `summary.json`; processed audit ghi `processed.json` và `summary.json`, có các giao nhau review ID và `no_review_leakage`.

## 6. XML/TXT Parser và data model

`capstone/types.py` định nghĩa `Quadruple`, `SentenceRecord`, `ReviewRecord` có `to_dict/from_dict`. `capstone/data/parser.py` đọc XML không xóa câu rỗng, kiểm tra span offsets và ghi issue cho annotation thiếu trường, category sai hoặc offset đáng ngờ. Resolver trong `capstone/data/paths.py` ánh xạ đủ ba domain tới XML/TXT tương ứng và ưu tiên bản Hotels đã sửa khi phù hợp.

Parser split TXT dùng `ast.literal_eval`, kiểm tra đúng bốn trường và gắn row với sentence XML để giữ review ID. Khi text trùng và label không phân biệt được candidate duy nhất, parser dừng với lỗi ambiguity thay vì tự gán review.

## 7. Data Cleaning và sentiment

`capstone/data/cleaning.py` bỏ quadruple dùng namespace category sai `POLARITY#*`, bỏ sentiment ngoài `positive/negative/neutral/conflict` và ghi `DataIssue` gồm review, sentence, quadruple, reason. Các câu rỗng bên trong review có annotation được giữ lại. Review không có quadruple sau cleaning mặc định được giữ với trạng thái `unknown`; chỉ manifest có provenance mới xác nhận review chưa annotate và cho phép loại review. Metadata lưu ID, bằng chứng/provenance, path và hash của manifest; log từng review bị loại giữ trạng thái `confirmed_unannotated`.

Hai mode `raw_4_class` và `paper_compatible` được lưu trong metadata. OATS paper liệt kê cả Conflict; nguồn ASQP/GAS đã kiểm tra không xác nhận mapping `conflict → neutral`. Vì vậy hiện tại cả hai mode giữ nguyên nhãn conflict, và configs ghi `conflict_policy: preserve`. `NULL/null/None` trong target/opinion được chuẩn hóa thành `NULL`; category và text fields được whitespace-normalize và case-fold trong shared normalizer.

## 8. Clean Split và serialization

`capstone/data/splitting.py` shuffle **review records** với seed và chia mặc định 80/10/10. Tất cả sentence trong một review đi cùng split; duplicate review IDs hoặc leakage làm raise error. `capstone/prepare_data.py` xâu chuỗi parse → validate/clean → áp manifest nếu được cung cấp → tạo aspect inventory từ toàn bộ review được giữ → split → JSONL/metadata. Manifest ID không thuộc XML đang chọn làm preparation dừng; review có raw annotation attempt hoặc parser/cleaning issue không thể bị phân loại zero-label. Metadata có SHA-256 cho XML/TXT/manifest đầu vào được sử dụng và từng JSONL đầu ra, kèm source path bắt buộc cho mỗi fingerprint. Raw files không bị sửa. Output mặc định là `data/processed/{domain}/train.jsonl`, `dev.jsonl`, `test.jsonl`, `metadata.json`.

`original_split` được hỗ trợ riêng để đối chiếu/reproduction; do split TXT không chứa review ID, code căn chỉnh với XML và báo lỗi khi không thể nhận diện an toàn. Trước training/evaluation, `validate_prepared_metadata` kiểm tra domain, split mode, sentiment mode, seed (với clean split), dataset version và cleaning policy khớp với config.

## 9. GAS-T5 Baseline

`capstone/models/gas.py` hiện thực GAS **extraction-style ASQP adaptation**. Reference GAS dùng sequence generation để dự đoán sentiment element tuples; adapter mã hóa các quadruple thành ordered tuple sequence trong JSON array để ranh giới trường rõ ràng. Đây là định dạng dự án dựa trên extraction-style GAS, không phải target syntax chính thức được tái tạo: paper OATS không công bố target string GAS-T5 cụ thể của bảng ASQP. Vì vậy kết quả tương lai từ cấu hình này phải được ghi là **GAS extraction-style ASQP adaptation**, không khẳng định reproduction chính xác GAS-T5 của paper.

`capstone/models/base.py` dùng loader sinh seq2seq dùng chung và resolve experiment directory tới best checkpoint. `gas.py` chứa serializer/parser tolerant, tiếp tục giữ quadruple hợp lệ khi generation có phần lỗi. Adapter giữ duplicate để evaluator tính duplicate prediction là FP.

## 10. Paraphrase-T5

`capstone/models/paraphrase.py` serialize mỗi quadruple theo dạng paper: `{aspect} is {great|ok|bad} because {target} is {opinion}`, nhiều quad ngăn bằng `[SSEP]`. Implicit target/opinion được viết `it` và parser đổi lại `NULL`. Nhãn conflict dùng từ `mixed` như extension tường minh của OATS, không coi là setting đã xác nhận từ Paraphrase paper. Parser giữ valid segments khi một segment malformed, kiểm tra aspect theo inventory nếu có và giữ duplicate để tính false positive.

## 11. Shared ASQP Representation

Cả hai adapter chuẩn hóa về `Quadruple[]` theo thứ tự `target/aspect/opinion/sentiment`, dùng `capstone/evaluation/normalization.py` và cùng evaluator. Shared normalization chuẩn hóa NULL, whitespace, chữ hoa/thường của text/category/sentiment, aspect `#` sang khoảng trắng nhưng giữ `_`, cùng sentiment aliases. Domain aspect inventory được tạo độc lập với split từ toàn bộ review được giữ và được persist cho train/evaluation/inference/comparison; category lạ bị coi là invalid. Thiếu hoặc lệch inventory làm evaluation/comparison dừng; inference cũng dừng nếu không tìm được ontology. Invalid generation không làm dừng cả evaluation.

## 12. Official Evaluation

`capstone/evaluation/metrics.py` tính exact-match micro P/R/F1. Một prediction chỉ là TP khi target, aspect, opinion và sentiment đều khớp; partial match tạo FP và FN. Match được thực hiện **theo từng sentence record trước khi cộng counts**, vì cùng quadruple ở câu khác không thể tạo TP. Counter matching giữ duplicate: dư prediction là FP. `metrics.json` có TP, FP, FN, precision, recall, F1.

## 13. Diagnostic Metrics

`diagnostic_metrics.json` tách khỏi official ASQP F1 và gồm component F1 cho target/aspect/opinion/sentiment; accuracy bổ trợ cho aspect/sentiment (chỉ tính trong alignment mà ba trường còn lại khớp); explicit/implicit target và opinion; per-aspect, per-sentiment; nhóm 0/1/2/3+ quadruple; example exact-match accuracy và counts.

## 14. Error Analysis

Evaluator và `capstone/error_analysis.py` ghi TP/FP/FN, wrong-field mismatch (target/aspect/opinion/sentiment), missing/extra quadruple, explicit/implicit target/opinion failures và malformed/invalid generation. Output giữ review/sentence IDs, text, raw generation, parser issues và quadruple chi tiết.

## 15. Config System

Repository có sáu cấu hình tạo thành ma trận `gas|paraphrase × amazon_ff|coursera|hotels`. Mỗi cặp cùng domain dùng chung seed 42, split, preprocessing, hyperparameters và generation settings; tên experiment và output directory là duy nhất. Config có domain/split/seed, model/tokenizer, optimizer/scheduler, epochs/batch/gradient accumulation, learning rate/weight decay/warmup, input/output lengths, generation, sentiment/cleaning policy, audit/evaluation settings và paths. Paths tương đối được resolve theo repository (hoặc thư mục gốc của config trong `configs/`). Training lưu snapshot config và metadata. Raw audit yêu cầu `--domain` hoặc `--all-domains`; preparation yêu cầu `--domain` hoặc một config có `experiment.domain`, tránh vô tình chỉ chạy Hotels.

Giá trị T5-base, batch 16, learning rate `3e-4`, 20 epochs lấy từ paper/reference code trong phạm vi tài liệu có thể đối chiếu; nhiều hyperparameter OATS không công bố đầy đủ. Sequence length 128, accumulation 1, greedy decoding và seed 42 là config của pipeline này, không tự nhận là cài đặt chính thức OATS. Checkpoint được chọn theo validation exact-match F1; khi tie giữ checkpoint đầu tiên đạt best.

## 16. CLI đã được viết

| Lệnh | Chức năng |
|---|---|
| `python -m capstone.audit_data` | Audit raw domain / all domains / processed split |
| `python -m capstone.prepare_data` | Chuẩn bị original hoặc clean split |
| `python -m capstone.train` | Train GAS adaptation hoặc Paraphrase |
| `python -m capstone.predict` | Inference một text hoặc file TXT/JSONL |
| `python -m capstone.evaluate` | Sinh prediction, exact metrics, diagnostics, errors |
| `python -m capstone.compare` | So sánh hai experiment tương thích |
| `python -m capstone.error_analysis` | Dựng lại error cases từ predictions JSONL |

Các command trong README/runbook chỉ được ghi để chạy về sau; chúng chưa được chạy trong giai đoạn implementation.

## 17. Tests

Tests được viết cho types, XML/TXT parsing, clean/filter policies, evidence manifest, review-level split/leakage, audit/JSONL/source/split metadata, domain ontology, GAS và Paraphrase serialization/parsing, NULL, normalization, exact match, duplicate prediction, malformed generation, error analysis, evaluation/comparison provenance, config/early-stopping type, mocked training artifact metadata/logging và seeding. Các test đặc tả mới kiểm tra đủ sáu cấu hình, resolver cho cả ba domain, CLI không ngầm chọn Hotels và tên comparison theo domain. `requirements-dev.txt` ghi pytest. Mocked training test không tải model hay gọi network, nhưng toàn bộ test vẫn chưa được chạy.

**Tests have been written but have NOT been executed.**

## 18. Files Added / Modified

| File | Mục đích | Trạng thái |
|---|---|---|
| `README.md` | Overview và quick start | Modified |
| `requirements.txt`, `requirements-dev.txt` | Runtime và test dependencies | New |
| `capstone/__init__.py`, `capstone/types.py`, `capstone/config.py`, `capstone/logging_utils.py` | Package, ASQP records, config, logging | New |
| `capstone/data/__init__.py`, `data/parser.py`, `data/paths.py`, `data/cleaning.py`, `data/manifest.py`, `data/inventory.py`, `data/splitting.py`, `data/serialization.py`, `data/audit.py` | XML/TXT input, evidence manifest, validation, inventory, split, audit, metadata/JSONL | New |
| `capstone/audit_data.py`, `capstone/prepare_data.py` | Audit và preparation CLIs | New |
| `capstone/models/__init__.py`, `models/base.py`, `models/gas.py`, `models/paraphrase.py` | Shared seq2seq loader, adapters, parser/serializer | New |
| `capstone/training/__init__.py`, `training/seed.py`, `training/trainer.py`, `capstone/train.py` | Seed, training loop, CLI | New |
| `capstone/evaluation/__init__.py`, `evaluation/normalization.py`, `evaluation/metrics.py` | Shared normalizer, official/diagnostic scores, errors | New |
| `capstone/evaluate.py`, `capstone/predict.py`, `capstone/compare.py`, `capstone/error_analysis.py` | Evaluation, inference, comparison, errors CLIs | New |
| `configs/gas_t5_{amazon_ff,coursera,hotels}.yaml`, `configs/paraphrase_t5_{amazon_ff,coursera,hotels}.yaml` | Sáu experiment settings cho ma trận 2 mô hình × 3 domain | New |
| `docs/paper_reference.md` | Paper scores, citations, implementation assumptions | New |
| `tests/data/test_types.py`, `test_parser.py`, `test_cleaning.py`, `test_manifest.py`, `test_inventory.py`, `test_prepare_manifest.py`, `test_splitting.py`, `test_audit_serialization.py`, `test_processed_audit.py`, `tests/test_audit_data.py` | Data/type tests | New |
| `tests/models/test_gas.py`, `tests/models/test_paraphrase.py` | Adapter tests | New |
| `tests/evaluation/test_normalization.py`, `test_metrics.py` | Metric and normalizer tests | New |
| `tests/test_config.py`, `tests/test_predict.py`, `tests/test_evaluate.py`, `tests/test_compare.py`, `tests/test_error_analysis_cli.py`, `tests/training/test_seed.py`, `tests/training/test_trainer_metadata.py` | Config, CLI, provenance and mocked orchestration tests | New |
| `report/Trong_01_Capstone_Core_Implementation_Report.md`, `report/Trong_02_Capstone_Core_Execution_Guide.md` | Hai báo cáo tiếng Việt theo yêu cầu | New |

## 19. Trạng thái hiện tại

**IMPLEMENTATION STATUS: CODE WRITTEN; STATIC REVIEW COMPLETED**

**EXECUTION STATUS: NOT EXECUTED**

**TRAINING STATUS: NOT EXECUTED**

**EVALUATION STATUS: NOT EXECUTED**

## 20. Những gì chưa được xác minh

- `UNVERIFIED — requires execution`: test suite, audit, XML/TXT alignment trên toàn bộ corpus, prepare output và review leakage artifacts.
- `UNVERIFIED — requires execution`: tương thích runtime của PyTorch/Transformers, tải T5, memory/sequence truncation, convergence, training artifacts/checkpoint selection.
- `UNVERIFIED — requires execution`: generation quality, output validity, metrics, diagnostics, compare và error-analysis artifacts.
- `UNVERIFIED — requires execution`: đủ ba processed dataset, sáu experiment seed 42 và ba comparison cùng domain.
- Chưa cài dependency nào; chưa tạo processed dataset, checkpoint, prediction, metric hoặc benchmark result.

## Tham khảo paper

OATS paper báo cáo ASQP F1 GAS-T5 = 19.62/22.23/26.33 và Paraphrase = 20.84/19.78/34.51 cho Amazon_FF/Coursera/Hotels; kết quả dùng trung bình 5 seeds và split 80/10/10. Các con số này chỉ là paper reference, không phải kết quả của repository. OATS paper cũng nói quadruple chỉ đúng khi mọi trường khớp. Xem [OATS paper](https://aclanthology.org/2024.lrec-main.1080/), [GAS paper](https://aclanthology.org/2021.acl-short.64/) và [Paraphrase paper](https://aclanthology.org/2021.emnlp-main.726/).
