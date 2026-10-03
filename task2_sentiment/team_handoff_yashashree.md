# DATA266 Task 2 — Yelp Polarity Partner Handoff

This document separates Yashashree's completed individual Task 2 work from the configuration recommended for her partner. The dataset and assignment requirements are shared, but the model architectures, hyperparameters, random split, and results must remain independent.

## Shared assignment requirements

- Dataset: Yelp Polarity
- Binary labels: `0 = negative`, `1 = positive`
- Analyze review-length distribution, class distribution, and class balance
- Check missing and malformed reviews
- Lowercase and normalize text
- Remove punctuation/special characters while preserving sentiment-bearing negations
- Tokenize the processed reviews
- Learn embeddings from scratch
- Train one baseline and two experimental models
- Report all required classification, calibration, robustness, hardware, and efficiency metrics
- Manually review 20 errors: 5 confident false positives, 5 confident false negatives, 5 near-threshold errors, and 5 slice-specific errors
- Do not use pretrained embeddings or pretrained language models

## Yashashree's individual Task 2 work

### Dataset and preprocessing

| Component | Yashashree's setting |
|---|---|
| SID4 | 4349 |
| Split seed | 4349 |
| Training seeds | 4349, 4350, 4351 |
| Dataset | Yelp Polarity |
| Vocabulary source | Training split only |
| Vocabulary size limit | 30,000 |
| Minimum token frequency | 2 |
| Maximum sequence length | 256 tokens |
| Batch size | 256 |
| Validation fraction | 10% of the official training split |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Gradient clipping | 1.0 |
| Training epochs | 5 |
| Device recorded | CUDA |
| GPU recorded | NVIDIA GeForce RTX 4090 |

The preprocessing lowercases the review, normalizes non-alphanumeric characters, tokenizes by whitespace, and preserves important negation words such as `not`, `no`, and `never`. The vocabulary is learned only from the training portion to avoid validation/test leakage.

### Yashashree's model lineup

| Model | Architecture | Main settings | Result recorded |
|---|---|---|---:|
| `baseline_mean_pool` | Learned embedding → masked mean pooling → MLP | Embedding 128, hidden 128, dropout 0.20 | Accuracy 93.44% |
| `experiment_cnn` | Learned embedding → parallel 1-D CNNs → max pooling → MLP | Embedding 160, 128 channels, kernels 3/5/7, dropout 0.25 | Accuracy 92.19% |
| `experiment_bigru` | Learned embedding → bidirectional GRU → MLP | Embedding 192, hidden 128 per direction, dropout 0.30 | Accuracy 94.59% |

The best recorded individual model was `experiment_bigru`. The CNN showed signs of overfitting relative to the other models.

## Recommended partner configuration

The partner should use a different seed and a different architecture family. The following lineup is recommended, but it must still be checked against the other teammate's choices before implementation so no two members accidentally use the same models.

### Partner data and training settings

| Component | Partner's recommended setting |
|---|---|
| SID4 | 4349 |
| Split seed | 4350 |
| Training seed | 4350 |
| Training seeds for stability | 4350, 4351, 4352 |
| Vocabulary size limit | 25,000 |
| Minimum token frequency | 3 |
| Maximum sequence length | 320 tokens |
| Batch size | 256 |
| Validation fraction | 10% of the official training split |
| Optimizer | AdamW |
| Learning rate | 0.0002 |
| Weight decay | 0.0005 |
| Gradient clipping | 1.0 |
| Training epochs | 8 |
| Device | Use the assigned GPU and record its exact name |

The partner must create her own stratified split using seed `4350`. She must build her own vocabulary from her training split only and must not copy Yashashree's vocabulary, split indices, checkpoints, predictions, or metric outputs.

### Partner model lineup

| Partner model | Recommended architecture | Recommended settings |
|---|---|---|
| `partner_max_pool` | Learned embedding → masked max pooling → MLP | Embedding 96, hidden 128, dropout 0.10 |
| `partner_lstm` | Learned embedding → 2-layer unidirectional LSTM → MLP | Embedding 160, hidden 128, 2 layers, dropout 0.30 |
| `partner_attention_pool` | Learned embedding → learned scalar attention pooling → MLP | Embedding 128, attention hidden 64, classifier hidden 128, dropout 0.20 |

These models do not overlap with Yashashree's mean-pooling baseline, multi-width CNN, or BiGRU experiment.

## Partner configuration cell

```python
SID4 = 4349
SEED = 4350
SLICE = 349
HP_ID = 5
CLS_A = 9
CLS_B = 2
TRAIN_SEEDS = [4350, 4351, 4352]

CONFIG = {
    "validation_fraction": 0.10,
    "vocab_size": 25_000,
    "min_frequency": 3,
    "max_length": 320,
    "batch_size": 256,
    "epochs": 8,
    "learning_rate": 2e-4,
    "weight_decay": 5e-4,
    "gradient_clip": 1.0,
    "seed": SEED,
}
```

## Partner model justifications

### `partner_max_pool`

This is a simple and interpretable baseline. It learns word embeddings from scratch and uses masked maximum pooling to retain the strongest sentiment-bearing feature across the review. It differs from Yashashree's mean-pooling baseline because max pooling emphasizes the most activated sentiment features rather than averaging all token representations.

### `partner_lstm`

The two-layer unidirectional LSTM models sequential dependencies and long-range word order. It is intentionally not bidirectional, so it does not duplicate Yashashree's BiGRU architecture. The extra layer increases sequence capacity while dropout limits overfitting.

### `partner_attention_pool`

This model learns a scalar attention weight for each token and computes a weighted review representation. It can focus on sentiment-bearing words such as `excellent`, `terrible`, or `disappointing` instead of treating every token equally. It is a distinct architecture from both the CNN and recurrent models used by Yashashree.

## Metrics the partner must report independently

For each of the three partner models, report:

- Accuracy
- Precision, recall, and F1: macro, micro, and weighted
- Confusion matrix
- ROC-AUC
- PR-AUC
- Matthews correlation coefficient
- Brier score
- Expected calibration error
- 95% bootstrap confidence intervals for accuracy, macro-F1, and MCC
- Paired McNemar test against the partner baseline
- Macro-F1 and error rate per data slice
- Parameter count
- Training time
- Examples per second
- Peak GPU memory

The partner must also complete the 20-error manual review using her own best model and her own generated error table.

## Do not overlap

- Do not use split seed `4349`; use `4350`.
- Do not reuse Yashashree's vocabulary or split indices.
- Do not use `baseline_mean_pool`, `experiment_cnn`, or `experiment_bigru` as model names or architectures.
- Do not copy Yashashree's checkpoints, predictions, metric values, or error-review examples.
- Do not report the recorded `94.59%` BiGRU accuracy as the partner's result.
- The team comparison should be added only after both individual notebooks are complete.

## Important coordination note

The assignment requires every member's three-model set to differ from every other member's models. Before training, the partner should tell the team which three model families she selected and confirm that no other member is using the same lineup.
