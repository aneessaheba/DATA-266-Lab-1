# DATA266 Task 1 — Partner Handoff

This document explains the individual configuration already used by Yashashree and the separate configuration that the partner should use for Questions 1 and 2.

The purpose is to keep both implementations valid but non-overlapping. The dataset and assignment requirements remain the same, but the random split, context length, and GPT architecture must be independently chosen.

## Requirements shared by both members

- Dataset: TinyStories
- Tokenization: character-level
- Input length: sequence length excluding the final target character
- Target: next-character sequence shifted by one position
- Training sequences: 100,000
- Validation sequences: 10,000
- Own `char_to_idx` and `idx_to_char` dictionaries
- Own deterministic train/validation shuffle using the member's seed
- No pretrained Transformer or attention module
- Causal masking must prevent access to future characters
- Model must include token embeddings, positional embeddings, self-attention, feed-forward layers, layer normalization, residual connections, and a language-modeling head

## Yashashree's completed individual configuration

Yashashree used the following configuration:

| Component | Yashashree's value |
|---|---:|
| SID4 | 4349 |
| Split seed | 4349 |
| Training seeds | 4349, 4350, 4351 |
| Context length | 256 characters |
| Transformer layers | 6 |
| Model dimension | 192 |
| Attention heads | 3 |
| Head dimension | 64 |
| Feed-forward dimension | 768 |
| Dropout | 0.15 |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Warm-up steps | 1,000 |
| Gradient clipping | 1.0 |
| Generation temperature | 0.8 |
| Generation length | 300 characters |

Do not copy this architecture or use the same split seed in the partner experiment.

## Partner's recommended individual configuration

Use this configuration for a distinct but valid GPT-style model:

| Component | Partner's value |
|---|---:|
| SID4 | 4349 |
| Split seed | 4350 |
| Training seed | 4350 |
| Context length | 128 characters |
| Transformer layers | 4 |
| Model dimension | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| Feed-forward dimension | 1,024 |
| Dropout | 0.10 |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Batch size | 64 |
| Epochs | 10 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Warm-up steps | 1,000 |
| Gradient clipping | 1.0 |
| Generation temperature | 0.8 |
| Generation length | 300 characters |

This model is different because it uses:

- 4 layers instead of 6
- 256 hidden dimensions instead of 192
- 4 attention heads instead of 3
- 1,024 feed-forward dimensions instead of 768
- 128-character context instead of 256
- 0.10 dropout instead of 0.15
- Split seed 4350 instead of 4349

The attention head dimension remains 64 because `256 / 4 = 64`.

## Required configuration cell for the partner

```python
SID4 = 4349
SEED = 4350
SLICE = 349
HP_ID = 5
CLS_A = 9
CLS_B = 2

TRAIN_SEQUENCES = 100_000
VALIDATION_SEQUENCES = 10_000

BLOCK_SIZE = 128
N_LAYERS = 4
D_MODEL = 256
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
D_FF = 1024
DROPOUT = 0.10
ACTIVATION = "gelu"
NORMALIZATION = "pre_layer_norm"

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 1000
GRADIENT_CLIP = 1.0

TEMPERATURE = 0.8
GENERATION_LENGTH = 300

assert HEAD_DIM == 64
print("Partner head dimension:", HEAD_DIM)
```

## Data preprocessing differences

The partner should create the same number of sequences but shuffle and select them independently:

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
```

The partner must use `SEED = 4350` when creating the individual split. The resulting examples will not be identical to Yashashree's examples.

The input-target construction remains:

```python
input_ids = encoded_sequence[:-1]
target_ids = encoded_sequence[1:]
```

For the partner's model, each example contains 128 input characters and 128 next-character targets.

## Model implementation checklist

The partner must implement these manually:

1. Token embedding layer
2. Positional embedding layer
3. Manual query, key, and value projections
4. Multi-head self-attention
5. Lower-triangular causal mask
6. Attention dropout
7. Output projection
8. Pre-LayerNorm
9. Feed-forward network with GELU
10. Residual connections
11. Final layer normalization
12. Language-modeling head projecting to vocabulary size

The partner should verify that the attention weights assigned to future positions are exactly zero.

## Do not overlap these items

- Do not use split seed 4349 for the partner's split.
- Do not use the 6-layer, 192-dimensional, 3-head architecture.
- Do not use context length 256.
- Do not copy Yashashree's generated samples or metric outputs.
- Do not report Yashashree's loss, accuracy, perplexity, or generation metrics as the partner's results.
- Train and evaluate the partner's model independently.

The final team report can compare both members after both independent runs are complete.
