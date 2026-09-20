# OATS ASQP paper reference

The OATS paper reports the following ASQP F1 values for its paper setup:

| Method | Amazon_FF | Coursera | Hotels |
|---|---:|---:|---:|
| GAS-T5 | 19.62 | 22.23 | 26.33 |
| Paraphrase | 20.84 | 19.78 | 34.51 |
| Template-ILO | 21.01 | 21.24 | 24.62 |
| Template-DLO | 20.39 | 21.96 | 26.59 |
| GEN-SCL-NAT | 20.36 | 20.20 | 28.58 |

These are paper reference numbers. They are not results from this repository, and they are not automatically comparable to an experiment using the project’s review-level `clean_split`. Keep paper reference, `original_split`, and `clean_split` results separate in reports.

The paper describes an 80/10/10 train/validation/test partition, exact matching of every quadruple element, five-seed averages, T5-base for the generative baselines, and the use of `it` to represent implicit targets and opinions in Paraphrase. It does not publish all run parameters or an OATS-specific implementation of the GAS-T5 target format. The original GAS paper presents annotation-style and extraction-style generation for its ABSA tasks; the OATS paper applies GAS to ASQP, but does not provide the target string syntax. This project therefore uses an extraction-style ordered quadruple tuple sequence with JSON delimiters as a source-aligned ASQP adaptation. Its target syntax is a documented project choice, so this baseline is not claimed as an exact reproduction of the OATS paper's GAS-T5 implementation.

OATS includes four sentiment labels, including `conflict`. No source inspected here establishes a conversion of OATS conflict to neutral for ASQP, so both `raw_4_class` and `paper_compatible` preserve that label. The Paraphrase adapter represents it as `mixed`, an explicit extension to the paper's three paraphrase phrases.

References:

- Chebolu et al., [OATS: A Challenge Dataset for Opinion Aspect Target Sentiment Joint Detection for ABSA](https://aclanthology.org/2024.lrec-main.1080/).
- Zhang et al., [Aspect Sentiment Quad Prediction as Paraphrase Generation](https://aclanthology.org/2021.emnlp-main.726/); official [ABSA-QUAD code and data](https://github.com/IsakZhang/ABSA-QUAD).
- Zhang et al., [Towards Generative Aspect-Based Sentiment Analysis](https://aclanthology.org/2021.acl-short.64/); official [Generative-ABSA code](https://github.com/IsakZhang/Generative-ABSA).

The configs use T5-base, batch size 16, learning rate `3e-4`, 20 epochs, greedy decoding, and sequence length 128 where the source references support those settings. Gradient accumulation 1 and sequence length 128 are explicit project defaults, not verified OATS-paper settings. OATS-specific training values are not claimed as a reproduction of the paper. The project's default seed 42 is a single-run pipeline default; the paper table averages five different seeds.
