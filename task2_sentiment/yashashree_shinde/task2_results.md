# DATA266 Task 2: Yelp Polarity Sentiment Classification

## Individual Work

Three models were trained independently using learned embeddings initialized from scratch.
No pretrained embeddings or pretrained language models were used.

## Model Comparison

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Training time | Peak memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_mean_pool | 0.9348 | 0.9348 | 0.9822 | 0.9824 | 0.8696 | 0.0482 | 0.4364 | 3,856,770 | 255.83s | 0.16 GB |
| experiment_cnn | 0.9285 | 0.9285 | 0.9818 | 0.9823 | 0.8579 | 0.0533 | 0.4525 | 5,181,890 | 292.50s | 0.45 GB |
| experiment_bigru | 0.9469 | 0.9469 | 0.9869 | 0.9871 | 0.8938 | 0.0412 | 0.4629 | 6,057,026 | 290.14s | 1.80 GB |

## Required Analysis Notes

- The baseline uses masked mean pooling over learned word embeddings.
- Experiment A uses multiple convolution widths to capture local phrases.
- Experiment B uses a bidirectional GRU to model word order and longer context.
- The 20 manually reviewed errors are stored in `task2_error_review.csv`.
- Slice metrics are stored in `task2_slice_metrics.json`.
- McNemar tests are stored in `task2_mcnemar_results.json`.
- Best model by test Macro-F1: experiment_bigru.
- Best model test accuracy: 0.9469.
- The CNN and BiGRU should be interpreted using their best validation checkpoints.
- The 20 error rows require manual completion before submission.
- Team-level comparison must be added after the teammate supplies her metrics.