# Task 1 GPT Style LLM From Scratch

**Member:** anees_saheba
**Dataset:** TinyStories
**Notebook:** `src/char_gpt_tinystories.ipynb`
**Checkpoint:** `checkpoints/best_model.pt`
**Seed:** 5330

## 1. What I built

A decoder only Transformer that predicts the next character of a story. It has
4 Transformer blocks at a model width of 256, with 4 attention heads per block
and a context window of 128 characters, giving 3,238,912 trainable parameters
over a vocabulary of 91 characters.

Every part of the Transformer is written with plain tensor operations. The
multihead self attention computes its own query, key and value projections,
scales the dot product by the square root of the head dimension, applies a
lower triangular mask before the softmax and projects the concatenated heads
back to the model width. No `nn.Transformer`, no `nn.MultiheadAttention` and no
`F.scaled_dot_product_attention` appear anywhere, as the brief requires.

The model reached a validation cross entropy of 0.7538 after 10 epochs, which
is 1.0875 bits per character and 76.07 percent top 1 next character accuracy.

## 2. Architecture

| Component | Choice | Why I chose it |
|---|---|---|
| Transformer blocks | 4 | The team handoff fixed a lineup distinct from my teammate's 6 blocks. 4 blocks at width 256 gives me a wider but shallower model than her narrower deeper one, so the comparison isolates depth against width rather than varying everything at once. |
| Model width `d_model` | 256 | Wider than my teammate's 192. Width increases the amount of information each position can carry, which matters at the character level because a single character is a weak signal and the model has to accumulate meaning across many positions. |
| Attention heads | 4 | 256 divided by 4 gives a head dimension of 64, the same head size used in Attention Is All You Need. One attention operation can only express one notion of relevance, so 4 heads let the model attend to several relationships at once at no extra parameter cost compared to a single head of width 256. |
| Head dimension | 64 | Follows from 256 / 4. Kept at 64 deliberately: the scaling factor in attention is the square root of this number, and 64 is the size the original scaling was designed around. |
| Feedforward width | 1024 | 4 times the model width, the ratio used in the original Transformer. Attention mixes information across positions but is linear in the values, so the feedforward block is where per position nonlinear transformation happens. |
| Context window | 128 characters | Half my teammate's 256. Attention cost grows with the square of the context, so 256 would have cost 4 times as much attention compute per layer. 128 characters is roughly two sentences of TinyStories, enough to carry a clause but not a whole paragraph, and section 6 of my failure analysis shows this is the binding limit on coherence. |
| Normalisation | Pre LayerNorm | Normalise before each sublayer and add the raw residual. This keeps an unnormalised identity path from input to output so gradients reach the early blocks directly. Post LayerNorm places a normalisation on that path and is measurably harder to train at depth. |
| Activation | GELU | Smooth everywhere, unlike ReLU which has a kink at zero. Standard in GPT style models. |
| Positional embeddings | Learned, 128 by 256 | Attention is permutation invariant on its own, so without a position signal the model could not tell `abc` from `cba`. Learned rather than sinusoidal because the context is short and fixed, so there is no need to extrapolate to unseen positions. |
| Weight initialisation | Normal, standard deviation 0.02 | As in GPT 2. PyTorch's default scales with fan in and produces larger initial logits, which makes the first few hundred steps noisier than necessary. |
| Language modelling head | Linear 256 to 91, no bias | Projects the final hidden state to one score per vocabulary character. No bias because the final LayerNorm already supplies a per feature shift. |

**Parameter breakdown**, which reconciles exactly to the recorded total:

| Part | Parameters |
|---|---|
| Token embedding (91 by 256) | 23,296 |
| Position embedding (128 by 256) | 32,768 |
| Attention per block | 263,168 |
| Feedforward per block | 525,568 |
| LayerNorms per block | 1,024 |
| One block, total | 789,760 |
| 4 blocks | 3,159,040 |
| Final LayerNorm | 512 |
| LM head (256 by 91) | 23,296 |
| **Total** | **3,238,912** |

Almost 98 percent of the parameters sit in the 4 blocks, and two thirds of each
block is the feedforward network rather than attention.

## 3. Hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Training sequences | 100,000 | Set by the brief. |
| Validation sequences | 10,000 | Set by the brief. Cut as non overlapping windows and shuffled with my seed, so no character appears in both splits. Overlapping windows would leak validation text into training and make the validation loss look better than it is. |
| Sequence length | 128 input, 128 target | One window of 129 characters gives both: input is the window without its last character, target is the window without its first. Position i of the input predicts position i of the target, so a single window supplies 128 training signals rather than one. |
| Batch size | 64 | Gives 1,562 steps per epoch and 15,620 in total, enough steps for the warmup and cosine schedule to be meaningful. Larger batches would have reduced the step count and left the schedule with too little to work with. |
| Epochs | 10 | The brief sets a minimum of 10. Validation loss was still improving at epoch 10, so this is a floor rather than a converged model. |
| Optimiser | AdamW, betas 0.9 and 0.95 | The lower second beta is the GPT convention; it makes the variance estimate adapt faster than Adam's default 0.999, which suits a short run. |
| Learning rate | 3e-4 peak | Standard for a model of this size. Reached after warmup and then decayed. |
| Warmup | 1,000 steps, linear | At step 0 the weights are random so the first gradients are close to arbitrary, and Adam's second moment estimate starts near zero so early updates are divided by a very small number and become large. Warmup lets those estimates stabilise before any large step is taken. 1,000 steps is 6.4 percent of training. |
| Schedule after warmup | Cosine decay to 10 percent of peak | Anneals smoothly so late training refines rather than bounces. A floor of 10 percent rather than zero keeps the last epochs doing useful work. |
| Weight decay | 0.01, on matmul weights only | Biases, LayerNorm parameters and embeddings are excluded. Shrinking a LayerNorm gain toward zero fights the normalisation rather than regularising the function the model computes. |
| Gradient clipping | 1.0 | A safety net against a single bad batch. In practice the mean gradient norm was 0.668, below the threshold, so clipping was rarely active and the run was stable on its own rather than being forced into stability. |
| Dropout | 0.10 | Lower than my teammate's 0.15, because my model never overfitted. The generalization gap was negative at every epoch, so stronger regularisation would have cost capacity for no benefit. |
| Generation temperature | 0.8 reported, plus 0.5 and 1.2 compared | 0.8 is the value named in the handoff. The other two are included to show the repetition against incoherence tradeoff in the failure analysis. |

## 4. How my metrics were computed

- **Notebook:** `src/char_gpt_tinystories.ipynb`, run top to bottom. The committed copy carries its outputs.
- **Config:** `src/configs/char_gpt_config.yaml`, which mirrors the configuration cell.
- **Checkpoint:** `checkpoints/best_model.pt`, saved at the epoch with the lowest validation loss, which was epoch 10.
- **Split:** all losses, perplexity, bits per character and accuracy come from the 10,000 sequence validation split, never from training data.
- **Perplexity:** the exponential of the validation cross entropy.
- **Bits per character:** validation cross entropy divided by the natural log of 2.
- **Generalization gap:** validation loss minus training loss, both at epoch 10.
- **Top 1 accuracy:** fraction of the 1,280,000 validation target positions where the highest scoring character was correct.
- **Distinct 1, 2 and 3 and repeated 4gram rate:** computed over the concatenated temperature 0.8 samples, which is the reported decoding setting.
- **Gradient norms:** returned by `clip_grad_norm_` before clipping, recorded every step.
- **Throughput, memory and time:** measured in the training loop and written to the raw log every epoch.

## 5. Results

| Metric | Value |
|---|---|
| Training cross entropy loss | 0.7841 |
| Validation cross entropy loss | 0.7538 |
| Perplexity | 2.125 |
| Bits per character | 1.0875 |
| Generalization gap | 0.0303 negative |
| Top 1 next character accuracy | 76.07 percent |
| Distinct 1 | 0.0467 |
| Distinct 2 | 0.2703 |
| Distinct 3 | 0.5412 |
| Repeated 4gram rate | 0.2943 |
| Gradient norm, mean | 0.6679 |
| Gradient norm, max | 6.4868 |
| Loss spikes | 0 |
| NaN or infinite steps | 0 |
| Parameter count | 3,238,912 |
| Vocabulary size | 91 |
| Training throughput | 37,528 tokens per second |
| Generation throughput | 233 characters per second |
| Peak memory | 1.104 GB |
| Total training time | 3,557 seconds, 59.3 minutes |

Full values in `metrics_report.csv`. Loss curves in `outputs/loss_curves.png`.

**Reading these numbers.** A model guessing uniformly over 91 characters would
score a cross entropy of 4.51 and 6.51 bits per character. At 1.0875 bits per
character this model is about 6 times better than chance. Perplexity 2.125
means that at each position it is effectively choosing between roughly 2
characters rather than 91.

**The generalization gap is negative at every epoch**, moving from 0.777
negative at epoch 1 to 0.030 negative at epoch 10. This is not an error.
Dropout is active while the training loss is measured and disabled during
validation, so the training figure is taken on a handicapped model. The gap
shrinking toward zero means that handicap matters less as the model becomes
confident. It never turns positive, so this model does not overfit, and with
3.2 million parameters against 12.8 million training characters that is the
expected outcome. It also means I had capacity headroom: a larger model, a
longer context or more epochs would all have been reasonable next steps.

**Stability.** No loss spikes and no non finite steps across 15,620 optimiser
steps. Mean gradient norm 0.668 against a clipping threshold of 1.0, so
clipping was rarely active.

**Seed sensitivity.** I ran this configuration twice, once at seed 4350 and
once at seed 5330, which changes both the initialisation and the split. Bits
per character came out 1.0823 and 1.0875, and top 1 accuracy 76.15 and 76.07
percent, a difference of under 0.5 percent. The result follows from the
architecture rather than from a favourable seed. The superseded log is kept in
`reproducibility/raw_logs/`.

## 6. How my model differs from my teammate's

| | Mine (anees_saheba) | Yashashree |
|---|---|---|
| Blocks | 4 | 6 |
| Model width | 256 | 192 |
| Attention heads | 4 | 3 |
| Head dimension | 64 | 64 |
| Feedforward width | 1024 | 768 |
| Context window | 128 | 256 |
| Dropout | 0.10 | 0.15 |
| Split seed | 5330 | 4349 |

The two models sit on opposite sides of the same budget. Mine is wider and
shallower with a shorter context; hers is narrower and deeper with a context
twice as long. Because the head dimension is 64 in both, the difference in head
count follows directly from the difference in width rather than being a
separate choice.

The comparison I expect to matter most is context length against depth. My
failure analysis shows coherence breaking down over spans longer than about two
sentences, which 128 characters cannot cover. If her 256 character context
produces measurably better long range consistency at a similar bits per
character, that would be evidence the context window rather than depth is the
binding constraint at this scale.

### Measured comparison

Her run is now in the repository, so this is against her numbers rather than an
expectation.

| | Mine | Yashashree | Better |
|---|---|---|---|
| Validation cross entropy | 0.7538 | 0.7077 | hers |
| Validation perplexity | 2.125 | 2.029 | hers |
| Validation bits per character | 1.0875 | 1.0210 | hers by 6 percent |
| Validation top 1 accuracy | 76.07 percent | 77.59 percent | hers |
| Generalisation gap | -0.0303 | -0.0009 | hers |
| Parameters | 3,238,912 | 2,757,605 | mine is 17 percent larger |
| Vocabulary | 91 | 101 | see below |
| Distinct 3 at temperature 0.8 | 0.763 | 0.666 | mine |
| Repeated 4gram at temperature 0.8 | 0.132 | 0.231 | mine |

**She wins on prediction and this supports what my failure analysis predicted.**
Her bits per character is 1.0210 against my 1.0875, with 15 percent fewer
parameters. The two obvious candidates are her deeper stack and her longer
context. My failure analysis argued that coherence broke down over spans longer
than roughly two sentences, which 128 characters cannot hold, and named context
length as the first thing I would change. Her 256 character context reaching a
better bits per character on a smaller model is consistent with that, though
depth and context changed together so this is evidence rather than proof.

**I win on generation diversity, at the same temperature.** My repeated 4gram
rate is 0.132 against her 0.231, and my distinct 3 is 0.763 against her 0.666.
So the model with the better validation loss produces the more repetitive text.
That is worth stating plainly because it is easy to assume lower perplexity
means better generation. Perplexity measures how well a model predicts the next
character given a true prefix. Repetition is a property of what happens when the
model is fed its own output instead, which perplexity never observes.

**The vocabularies are not the same set.** Mine is 91 characters after I found
and repaired mojibake in the source text, where UTF-8 had been decoded as
cp1252. Hers is 101, which is the size mine was before the repair, so her run
spends part of its embedding and output layers on corrupted character sequences
such as the three byte artifacts that should be a single quotation mark. Bits
per character is still comparable between us, because it is normalised per
character either way, but it is not quite like for like: a few of the characters
she is predicting are artifacts rather than real text. If anything this makes
her result stronger, since her model carries that overhead and still predicts
better.

**The hardware is not comparable.** She trained on an RTX 4090 in 435 seconds;
I trained on Apple M4 through MPS in 3,557 seconds. The ratio reflects the
hardware and not the models, so no timing conclusion should be drawn from it.

**What I would take from this.** If we had one more run between us, the
experiment worth doing is my architecture at a 256 character context, with
everything else held fixed. That isolates context length from depth, which this
comparison cannot do because both changed at once.

## 7. Hardware disclosure

| Field | Value |
|---|---|
| Machine | Apple Mac16,12 |
| CPU | Apple M4, 10 cores, 4 performance and 6 efficiency |
| GPU | Apple M4 integrated, 10 cores, used through PyTorch MPS |
| RAM | 32 GB unified |
| Operating system | macOS 26.6.2, build 25G83 |
| Python | 3.11.8 |
| PyTorch | 2.14.0 |
| NumPy | 2.4.6 |
| Compute device reported by the run | `mps` |
| Peak memory during training | 1.104 GB |
| Total training time | 3,557 seconds |

Training throughput fell from 49,066 tokens per second in epoch 1 to about
37,000 later in the run. This is thermal throttling on a passively managed
laptop, not a change in the workload. The reported figure of 37,528 tokens per
second is the mean across all 10 epochs.

## 8. Evidence trail

| Item | Path |
|---|---|
| Notebook with outputs | `src/char_gpt_tinystories.ipynb` |
| Config | `src/configs/char_gpt_config.yaml` |
| Checkpoint | `checkpoints/best_model.pt` |
| Metrics | `metrics_report.csv`, `outputs/metrics.json` |
| Loss curves | `outputs/loss_curves.png` |
| Generated samples | `outputs/generated_samples.txt`, `outputs/generated_samples.json` |
| Per epoch history | `outputs/training_history.json` |
| Failure analysis | `failure_analysis.md` |
| Raw log, reported run | `reproducibility/raw_logs/task1_gpt_20261004_212529.log` |
| Raw log, superseded seed 4350 run | `reproducibility/raw_logs/task1_gpt_20261002_212332_seed4350_superseded.log` |
| Raw log, partial run stopped externally | `reproducibility/raw_logs/task1_gpt_20261002_205302_PARTIAL_killed_at_step10600.log` |
| Manifest | `reproducibility/manifests/anees_saheba_task1_llm.yaml` |
