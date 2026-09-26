# Thanh_03 — Báo cáo thực thi OATS seed 42 từ split TXT

**Trạng thái: COMPLETE.** Đã chạy đủ sáu train/evaluate, ba comparison cùng domain và sáu error analysis. Mọi command kết thúc với exit code 0; mỗi model train đủ 20 epoch. Kết quả dưới đây lấy từ artifacts của lần chạy này, không lấy số liệu từ paper.

## 1. Phạm vi và quyết định phương pháp

Phạm vi: seed 42, ba domain amazon_ff, coursera, hotels; GAS-T5 và Paraphrase-T5. Mọi run dùng cùng code commit 8a6e20cb880ad8749370a48a6d8b53a19a75aa2b và OATS submodule commit 2089631462f06fc0d0ac07186fa65236e19784ce.

Theo quyết định rõ ràng của người dùng, pipeline dùng trực tiếp các file train/dev/test TXT có sẵn làm nguồn phân chia chuẩn. Đây là deviation có chủ ý so với mission ban đầu: không chạy XML audit, không ghép dòng TXT với XML, không dùng clean_split và không chia lại dữ liệu. Mỗi source line là một mẫu riêng; danh tính mẫu được lưu bằng domain, split và source_line_number. Các dòng trùng text vẫn được giữ độc lập, kể cả khi nhãn khác nhau. Seed 42 dùng cho quá trình huấn luyện; seed không tạo lại split.

Một lần thử raw audit dựa trên XML trước đó đã dừng ở dòng “Delicious” bị trùng; lỗi đó thuộc protocol cũ và được thay thế bởi quyết định dùng split TXT có sẵn. Lượt chạy được báo cáo ở đây không đọc XML và hoàn thành raw audit theo provided_split.

Giữ nguyên model backbone, hyperparameter, generation settings, cách tính metrics và quy tắc chọn checkpoint trong sáu config. Làm sạch theo cấu hình đã khai báo: invalid_aspect_policy=drop_and_log, conflict_policy=preserve, remove_unannotated_reviews=false. Vì thế mọi dòng/mẫu được giữ; chỉ 12/35/37 annotation quadruple có aspect category không hợp lệ lần lượt bị bỏ khỏi amazon_ff/coursera/hotels và được ghi vào metadata.

## 2. Provenance và preflight

- Thời gian thực thi: raw audit bắt đầu 2026-09-25 11:58:45; pipeline outputs hoàn tất 2026-09-25 21:36:10; snapshot dung lượng lấy lúc 2026-09-25 21:41:52, múi giờ Asia/Saigon (UTC+07:00).
- Repository: D:/DS-AI/Web mining project/Generative-Aspect-Based-Sentiment-Analysis-on-OATS.
- Branch: run/full-oats-seed42. Code commit dùng cho cả sáu run: 8a6e20cb880ad8749370a48a6d8b53a19a75aa2b. main giữ nguyên ở d749485640a880cf95ae46589522c9fcc83f49d8.
- OATS submodule: 2089631462f06fc0d0ac07186fa65236e19784ce.
- Các run được tạo trong checkout trên D:. Sau khi hoàn thành, người dùng yêu cầu chuyển artifact tree sang E:; run metadata vẫn giữ các absolute path D: ghi tại thời điểm thực thi.
- Runtime: Python 3.11.16; PyTorch 2.14.0+cu132; Transformers 4.57.6; PyYAML 6.0.3; NumPy 2.4.6; pytest 9.1.1; SentencePiece 0.2.2.
- GPU: NVIDIA GeForce RTX 4060 Ti, 16,380 MiB VRAM; driver 595.79; CUDA 13.2.
- pip check, CUDA/T5 smoke, sáu config validation và các CLI help: PASS theo artifacts/setup.
- Full test suite: PASS, 117/117 trong 4.83 giây; dùng thư mục pytest basetemp riêng để tránh lỗi quyền của thư mục tạm mặc định. Lệnh: D:/Anaconda/envs/oats-seed42/python.exe -m pytest -ra --basetemp "artifacts/setup/pytest-basetemp-txt-final-20260925-2135".
- Deviation: chỉ thay nguồn và cách đọc split theo yêu cầu người dùng; không đổi thuật toán mô hình, metric, config huấn luyện hay seed.

## 3. Data audit và giới hạn leakage

Lệnh raw audit chạy với source_mode=provided_split trên TXT cho cả ba domain và exit 0. Mỗi bộ số bên dưới theo thứ tự train/dev/test. “Raw quadruples” là trước policy loại aspect category không hợp lệ; “prepared quadruples/empty labels” là số sau làm sạch. Mẫu rỗng nhãn vẫn nằm trong split và được đưa vào train/evaluation như dữ liệu nguồn cung cấp.

| Domain | Số dòng TXT train/dev/test | Raw quadruples tổng | Empty labels raw train/dev/test | Invalid aspect annotations | Quadruples prepared train/dev/test | Empty labels prepared train/dev/test | Malformed câu/quadruple |
|---|---:|---:|---:|---:|---:|---:|---:|
| amazon_ff | 7360 / 920 / 920 | 8260 | 2160 / 216 / 237 | 12 | 6304 / 978 / 966 | 2168 / 216 / 237 | 0 / 0 |
| coursera | 7188 / 898 / 899 | 7875 | 1884 / 215 / 726 | 35 | 6676 / 928 / 236 | 1901 / 215 / 726 | 0 / 0 |
| hotels | 7834 / 979 / 980 | 11474 | 1923 / 386 / 850 | 37 | 10146 / 1049 / 242 | 1936 / 387 / 850 | 0 / 0 |

Raw audit: artifacts/data_audit/raw/{amazon_ff,coursera,hotels,summary}.json. Processed datasets: data/processed/<domain>/{train,dev,test}.jsonl và metadata.json. Processed audit: artifacts/data_audit/processed/<domain>/{processed,summary}.json. Sau relocation, mọi đường dẫn artifacts/... trong report này nằm dưới E:/models/OATS_seed42; processed datasets vẫn ở D:. Đối chiếu trước train xác nhận đúng domain, provided_split, line counts và fingerprints giữa TXT nguồn, metadata, JSONL và processed audit.

| Domain | Review-level leakage | Ghi chú |
|---|---|---|
| amazon_ff | not_verifiable | TXT không có review ID; người dùng xác nhận split có sẵn là đúng |
| coursera | not_verifiable | TXT không có review ID; người dùng xác nhận split có sẵn là đúng |
| hotels | not_verifiable | TXT không có review ID; người dùng xác nhận split có sẵn là đúng |

Do nguồn TXT không có review_id/sentence_id, pipeline không thể xác minh overlap/leakage ở cấp review. Đây là giới hạn kiểm chứng, không phải kết quả “đã xác nhận không leakage”. Pipeline tin cậy assignment train/dev/test theo xác nhận của người dùng. Không đọc XML trong lượt chạy này.

Fingerprint SHA-256 lưu trong metadata; mỗi bộ ba theo thứ tự train/dev/test:

| Domain | Source TXT fingerprints: train / dev / test | Prepared split fingerprints: train / dev / test |
|---|---|---|
| amazon_ff | 14c7d0cd107fea03e8b7eda2ce4b23a9eb3369435f246910f8c7fd73b4b2fb63 / 595759302db051501af6e3f945c667f18326bcc4baec2b68f061c39adc22d010 / f4ca175815f5c8590390caa996528ea0be7a728ffe4b147a28360b4abfa39f25 | 8c9d133bddfb6afa25ea95761562b4d64476fab17b4b8885440546aefbed79f1 / 9ebed451c719b853201e12644e20546a56ea4b00f6eec549933455afbd54e8ac / 2396a96da7c060ebd8207a254fe1c3a2e9873c48b41082769468dad5128449ec |
| coursera | 5bebecc64985adcdb743891f9841c9df2096cdcdeb7c60aa0f7c35f6eb707af7 / 16ab131f52a18a4fed4f417b91766db1cc6b4488111b318aa457865976acc5e9 / a89a910da6815cb6ac4aeb0ff1857141da2d043f2148c13f4f6b916be6883141 | 083eec8a3f38712a15f4f5ccd735aa6f456184d787f0999b5102d4e0ceb33f8f / 44958cee0ac1ccdb1c909161de9bc6f5b7a62d400ceff956a0d0fe7bd6dec5ea / 69b0a4b11e0e82d7031f0215f465d310096ad3f31b681e83351f7824af3f89a6 |
| hotels | df0cc3166c3374df312bfb8aa304b983897bccdef534d248541bf84577827283 / ff6d3b96fd75bf0ac4271e953add62f8b467976d4a0393ccc5278d258333a21c / 5171f4ec1835c89a29742b9150e71d89ea90d0aa954ce4f1718cc14553fa4e87 | d46051865d815a71906a2ddfd8a8018edae9ab214d011ea3162d0ad414c0b7b8 / d20be3564d9178582228da5947848e56ca88c6799ffbf8494dee9c1d71bddec3 / fb6c15022437b75d6b866c894c40ba4e6c1384d54e32960caaa6b32f5b9cf52d |

Lưu ý khi diễn giải test: Coursera có 726/899 mẫu và Hotels có 850/980 mẫu test không có quadruple sau cleaning. Các mẫu này được giữ nguyên theo split TXT đã được người dùng phê duyệt; không lọc hay chia lại để cải thiện phân bố. Vì vậy precision/F1 chịu ảnh hưởng từ các mẫu test không nhãn và kết quả cần được hiểu trong đúng protocol này.

## 4. Sáu kết quả train/evaluate

Tất cả sáu lệnh train và sáu lệnh evaluate exit 0. Mỗi train hoàn tất 20/20 epoch; best epoch lấy từ best_checkpoint.json, validation F1 từ training_summary.json, test metrics và invalid generation count từ artifacts/evaluation. Predictions có đúng số dòng bằng test sample count. Tất cả artifact bắt buộc đều tồn tại.

| Domain | Model | Status | Best epoch | Dev F1 | Test P | Test R | Test exact-match F1 | TP | FP | FN | Invalid generations | Experiment |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| amazon_ff | GAS-T5 extraction-style ASQP adaptation | PASS | 14 | 0.297149 | 0.247246 | 0.209110 | 0.226584 | 202 | 615 | 764 | 14 | artifacts/experiments/gas_t5_amazon_ff_seed42 |
| coursera | GAS-T5 extraction-style ASQP adaptation | PASS | 5 | 0.294936 | 0.071006 | 0.254237 | 0.111008 | 60 | 785 | 176 | 5 | artifacts/experiments/gas_t5_coursera_seed42 |
| hotels | GAS-T5 extraction-style ASQP adaptation | PASS | 19 | 0.441860 | 0.299401 | 0.413223 | 0.347222 | 100 | 234 | 142 | 3 | artifacts/experiments/gas_t5_hotels_seed42 |
| amazon_ff | Paraphrase-T5 | PASS | 5 | 0.294056 | 0.215452 | 0.204969 | 0.210080 | 198 | 721 | 768 | 2 | artifacts/experiments/paraphrase_t5_amazon_ff_seed42 |
| coursera | Paraphrase-T5 | PASS | 3 | 0.307514 | 0.094560 | 0.309322 | 0.144841 | 73 | 699 | 163 | 1 | artifacts/experiments/paraphrase_t5_coursera_seed42 |
| hotels | Paraphrase-T5 | PASS | 16 | 0.454288 | 0.337386 | 0.458678 | 0.388792 | 111 | 218 | 131 | 0 | artifacts/experiments/paraphrase_t5_hotels_seed42 |

## 5. Same-domain comparisons

Ba comparison đều xác nhận cùng domain, seed 42, provided_split, test split và split fingerprint giữa hai model. Delta là Paraphrase F1 trừ GAS F1; không gộp hay lấy trung bình qua domain.

| Domain | Test n | GAS F1 | Paraphrase F1 | Delta | Comparison artifact |
|---|---:|---:|---:|---:|---|
| amazon_ff | 920 | 0.226584 | 0.210080 | -0.016505 | artifacts/comparisons/seed42/amazon_ff_gas_vs_paraphrase.json |
| coursera | 899 | 0.111008 | 0.144841 | +0.033833 | artifacts/comparisons/seed42/coursera_gas_vs_paraphrase.json |
| hotels | 980 | 0.347222 | 0.388792 | +0.041569 | artifacts/comparisons/seed42/hotels_gas_vs_paraphrase.json |

Trong lần chạy và split này, Paraphrase có F1 thấp hơn GAS trên amazon_ff, cao hơn trên coursera và hotels. Đây là mô tả seed 42 trên split được cung cấp, không phải kết luận thống kê hay một điểm chung cho ba domain.

## 6. Error analysis

Đã tạo error_analysis.jsonl cho cả sáu run trong từng thư mục evaluation. Tổng hợp trực tiếp từ artifacts: 25 invalid generations (19 malformed_generation, 4 invalid_quadruple_shape, 2 missing_field). Trong error-analysis records, các cờ lỗi phổ biến là extra_quadruple (2706), missing_quadruple (1553), wrong_field_mismatch (1265), implicit_target_failure (1184) và implicit_opinion_failure (790). Trong các trường của mismatch, opinion (1030) xuất hiện nhiều nhất, tiếp theo aspect (477), target (469), sentiment (249).

Các số đếm error-type là số lần gắn cờ qua artifacts của sáu run; một mẫu có thể có nhiều cờ lỗi nên các nhóm không loại trừ nhau. 2,997 error-analysis record có ít nhất một loại lỗi. Không sửa tay metrics hay predictions.

Artifacts:
- artifacts/experiments/gas_t5_amazon_ff_seed42/evaluation/error_analysis.jsonl
- artifacts/experiments/gas_t5_coursera_seed42/evaluation/error_analysis.jsonl
- artifacts/experiments/gas_t5_hotels_seed42/evaluation/error_analysis.jsonl
- artifacts/experiments/paraphrase_t5_amazon_ff_seed42/evaluation/error_analysis.jsonl
- artifacts/experiments/paraphrase_t5_coursera_seed42/evaluation/error_analysis.jsonl
- artifacts/experiments/paraphrase_t5_hotels_seed42/evaluation/error_analysis.jsonl

## 7. Lệnh và kết quả thực thi

- Raw audit: python -m capstone.audit_data --all-domains --source-mode provided_split --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-dir artifacts/data_audit/raw — exit 0.
- Prepare cho từng domain: python -m capstone.prepare_data --domain <domain> --split-mode provided_split --seed 42 --sentiment-mode paper_compatible --raw-root OATS-ABSA-Dataset/data --output-root data/processed — exit 0 cho cả ba.
- Processed audit cho từng domain: python -m capstone.audit_data --processed data/processed/<domain> --output-dir artifacts/data_audit/processed/<domain> — exit 0 cho cả ba.
- Train: python -m capstone.train --config configs/gas_t5_<domain>.yaml và python -m capstone.train --config configs/paraphrase_t5_<domain>.yaml — exit 0 cho sáu run.
- Evaluate: python -m capstone.evaluate --config configs/<model>_t5_<domain>.yaml --checkpoint artifacts/experiments/<model>_t5_<domain>_seed42 --split test — exit 0 cho sáu run.
- Compare: python -m capstone.compare cho từng cặp experiment cùng domain, output artifacts/comparisons/seed42 — exit 0 cho ba domain.
- Error analysis: python -m capstone.error_analysis cho predictions từng run, output vào evaluation/error_analysis.jsonl — exit 0 cho sáu run.

## 8. Artifacts, giới hạn và kết luận

- Artifact root sau relocation: E:/models/OATS_seed42/artifacts — 23,286,697,743 bytes (1,683 files; khoảng 23.29 GB), kiểm tra lúc 2026-09-26 15:09 +07:00. Toàn bộ file count và byte count khớp nguồn trước khi xóa; không thiếu/thừa file hay lệch size. SHA-256 của model.safetensors trong best checkpoint của cả sáu experiment khớp giữa nguồn và đích.
- Nguồn D:/DS-AI/Web mining project/Generative-Aspect-Based-Sentiment-Analysis-on-OATS/artifacts đã được gỡ sau xác minh; các đường dẫn tuyệt đối D: bên trong metadata là provenance của thời điểm chạy, không phải vị trí hiện tại. Các đường dẫn artifacts/... trong các bảng phía trên được resolve tương đối từ E:/models/OATS_seed42.
- Processed data root: D:/DS-AI/Web mining project/Generative-Aspect-Based-Sentiment-Analysis-on-OATS/data/processed — 6,663,017 bytes (12 files; khoảng 6.66 MB).
- Cộng hai root trên: 23,293,360,760 bytes. E: còn 731,058,827,264 bytes trống sau khi chuyển. Artifacts và checkpoints nằm cục bộ trên ổ E:, processed data trên D:, không upload dữ liệu lớn lên GitHub. Chỉ code và report được đưa lên branch.
- Chỉ chạy seed 42. Chưa chạy protocol năm seed, inference bổ sung, hay gộp kết quả giữa domain.
- GAS ở đây là extraction-style ASQP adaptation, không phải reproduction chính xác target syntax của GAS-T5 paper (target syntax gốc chưa được xác định trong codebase).

Kết luận: pipeline trực tiếp từ các split TXT đã hoàn tất đủ checklist. Giới hạn cần giữ rõ khi báo cáo: split assignment được tin cậy theo xác nhận của người dùng nhưng review-level leakage không thể kiểm chứng từ TXT; các test domain coursera/hotels có rất nhiều dòng rỗng nhãn; và kết quả chỉ đại diện cho seed 42.
