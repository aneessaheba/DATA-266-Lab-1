
# DATA266 Task 1: GPT From Scratch

## Student and Reproducibility Information

- SID4: 4349
- Seed: 4349
- Slice: 349
- HP_ID: 5
- CLS_A: 9
- CLS_B: 2
- Device: cuda
- GPU: NVIDIA GeForce RTX 4090
- Python version: 3.13.3
- PyTorch version: 2.14.0+cu132

## Dataset and Preprocessing

- Dataset: TinyStories
- Tokenization: Character-level
- Training sequences: 100,000
- Validation sequences: 10,000
- Sequence input length: 256
- Vocabulary size: 101
- Training and validation split: Independently shuffled using seed 4349
- Character dictionaries: `char_to_idx` and `idx_to_char`

## Model Architecture

- Transformer blocks: 6
- Embedding dimension: 192
- Attention heads: 3
- Dimension per attention head: 64
- Feed-forward dimension: 768
- Activation: gelu
- Normalization: pre_layer_norm
- Dropout: 0.15
- Positional embeddings: Learnable
- Total parameters: 2,757,605
- Attention implementation: Manual scaled dot-product attention
- Causal masking: Implemented and numerically verified
- Prebuilt Transformer or attention modules: Not used

## Training Configuration

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.01
- Warm-up steps: 1000
- Scheduler: Linear warm-up followed by cosine decay
- Gradient clipping: 1.0
- Batch size: 64
- Epochs: 10

## Final Evaluation Results

| Metric | Value |
|---|---:|
| Training cross-entropy loss | 0.708605 |
| Validation cross-entropy loss | 0.707674 |
| Training perplexity | 2.031157 |
| Validation perplexity | 2.029266 |
| Training bits per character | 1.022302 |
| Validation bits per character | 1.020958 |
| Generalization gap | -0.000931 |
| Training top-1 accuracy | 0.775641 |
| Validation top-1 accuracy | 0.775887 |
| Average gradient norm | 0.607040 |
| Maximum gradient norm | 5.447399 |
| Total NaN count | 0 |
| Total loss spike count | 0 |
| Parameter count | 2,757,605 |
| Training tokens/second | 610617.51 |
| Generation tokens/second | 351.29 |
| Peak GPU memory GB | 1.9707 |
| Total training time seconds | 435.48 |

## Generation Diversity Results

| Metric | Average Value |
|---|---:|
| Distinct-1 | 0.097446 |
| Distinct-2 | 0.422754 |
| Distinct-3 | 0.666328 |
| Repeated 4-gram rate | 0.230967 |

Generation used temperature sampling with temperature 0.8.

## Observations

The model successfully learned common TinyStories character and word patterns.
The validation loss decreased throughout training, and the final validation
perplexity was approximately 2.03.
The model generated recognizable story structures, but some samples contained
grammar errors, repeated phrases, abrupt topic changes, and semantically
inconsistent descriptions.

The complete failure analysis is stored in `task1_failure_analysis.md`.

## Individual Task 1 Artifacts

- `metrics_report.csv`
- `task1_model_configuration.json`
- `task1_training_history.json`
- `task1_generated_samples.txt`
- `task1_loss_curves.svg`
- `task1_failure_analysis.md`
- `task1_checkpoints/task1_best_model.pt`
- `task1_checkpoints/task1_epoch_01.pt` through `task1_epoch_10.pt`
