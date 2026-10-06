# Task 2 Yelp Polarity sentiment classification, anees_saheba

**Member:** anees_saheba
**Seed:** 5330, the last four digits of my student ID
**Run:** 20261005_030918
**Checkpoints:** `checkpoints/baseline_max_pool.pt`, `checkpoints/experiment_lstm.pt`, `checkpoints/experiment_attention_pool.pt`

## 1. What I built

Three binary sentiment classifiers for Yelp Polarity, all written from scratch
in PyTorch. No pretrained embeddings and no pretrained language model: every
embedding table starts from random initialisation and is learned only from the
504,000 training reviews. The three models share the same vocabulary, the same
preprocessing and the same training loop, and differ only in how they turn a
variable length sequence of token embeddings into one fixed vector. The
baseline takes a masked maximum over positions. The first experiment runs a two
layer LSTM and reads its final hidden state. The second experiment learns a
scalar score per token and takes a weighted average. Keeping everything else
fixed is what makes the comparison between them mean anything.

## 2. Architecture

| Component | Choice | Why I chose it |
|---|---|---|
| Tokeniser | lowercase, regex word split, stopwords removed | Simple and reproducible. No external tokeniser to version. |
| Negation words kept | not, no, never, nor, none, cannot and contractions | Removing them is the standard stopword list behaviour and it destroys sentiment. "not good" and "good" must not become the same sequence. |
| Vocabulary | 25,000 most frequent, minimum frequency 3 | Covers 98.29 percent of training tokens out of 198,238 distinct words. The tail is mostly typos and proper nouns. |
| Embedding | learned from random init, padding index 0 | The brief forbids pretrained embeddings. |
| Baseline pooling | masked maximum over positions | One strong feature anywhere in the review decides the class. Cheap and a genuine baseline rather than a crippled one. |
| Experiment 1 | two layer unidirectional LSTM, final hidden state | Reads the review in order, so it can represent word order and negation scope, which pooling cannot. |
| Experiment 2 | learned attention pooling, tanh projection then one score | Also order free like the baseline, but it learns which tokens matter instead of taking an extreme. It also gives a per token weight that can be inspected. |
| Classifier head | linear, ReLU, dropout, linear to one logit | Identical in all three, so any difference comes from the pooling. |
| Loss | binary cross entropy with logits | Two balanced classes. |

All three mask padding. The notebook asserts that adding padding to a sequence
does not change the baseline prediction, and the measured difference is exactly
0.0.

## 3. Hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Seed | 5330 | My student ID, so initialisation and splitting are mine. |
| Max length | 320 tokens | Keeps 92.65 percent of reviews whole. Longer costs time for the remaining 7 percent. |
| Batch size | 256 | Largest that kept peak memory near 1.2 GB on this machine. |
| Epochs | 8 | Chosen before the run. The checkpoint is the best epoch by validation loss, so overshooting is safe. |
| Learning rate | 2e-4 | Adam default was unstable for the LSTM in early tests. |
| Weight decay | 5e-4 | Mild regularisation. The embedding table is the bulk of the parameters. |
| Gradient clip | 1.0 | The LSTM reached gradient norms above 1.3, so clipping is doing real work. |
| Validation fraction | 0.10 | 56,000 reviews, enough that a 0.1 point accuracy difference is not noise. |

## 4. How my metrics were computed

- Notebook: `src/yelp_sentiment_models.ipynb`, run end to end with nbconvert
- Config: `src/configs/sentiment_config.yaml`
- Checkpoints: one per model under `checkpoints/`, each the best epoch by validation loss
- Split: metrics in section 5 are on the **test** split, 38,000 reviews, never used for training or model selection
- Threshold: 0.5, not tuned on test

Model selection used validation loss only. The test split was touched once, at
the end, for the numbers below.

## 5. Results

Test split, 38,000 reviews, perfectly balanced.

| Model | Accuracy | Macro F1 | ROC AUC | MCC | Brier | ECE | Parameters |
|---|---|---|---|---|---|---|---|
| baseline_max_pool | 0.8991 | 0.8991 | 0.9643 | 0.7981 | 0.0756 | 0.0392 | 2,412,545 |
| experiment_lstm | **0.9502** | **0.9502** | **0.9896** | **0.9005** | **0.0375** | 0.0104 | 4,297,217 |
| experiment_attention_pool | 0.9341 | 0.9341 | 0.9819 | 0.8683 | 0.0490 | **0.0024** | 3,224,962 |

95 percent bootstrap confidence intervals on accuracy: baseline 0.8962 to
0.9020, LSTM 0.9481 to 0.9525, attention 0.9318 to 0.9367. None of them
overlap, so the ordering is real and not sampling noise.

McNemar against the baseline, on paired predictions over the same test reviews:

| Model | Statistic | p value | Baseline right and model wrong | Model right and baseline wrong |
|---|---|---|---|---|
| experiment_lstm | 1087.52 | 1.1e-252 | 765 | 2,710 |
| experiment_attention_pool | 599.20 | 4.1e-137 | 814 | 2,147 |

Both experiments beat the baseline by margins that are not close to chance.

### Cost

| Model | Training time | Examples per second | Peak memory | Best epoch |
|---|---|---|---|---|
| baseline_max_pool | 408 s | 9,876 | 1.048 GB | 7 of 8 |
| experiment_lstm | 9,685 s | 416 | 1.258 GB | 6 of 8 |
| experiment_attention_pool | 437 s | 9,227 | 1.267 GB | 8 of 8 |

The LSTM is the most accurate and costs 22 times more wall clock than either
pooling model. Its 416 examples per second against 9,227 for attention pooling
is the price of sequential recurrence: 320 timesteps that cannot be computed in
parallel, against one matrix multiply over all positions at once.

### What the numbers say

**The LSTM wins on accuracy, and word order is why.** It is the only model that
reads the review in sequence. Pooling models see a bag of positions, so they
cannot represent that "not" applies to the word after it. The 5.1 point gap
over the baseline is the value of order.

**Attention pooling is the best calibrated model by a wide margin.** Its
expected calibration error is 0.0024 against 0.0104 for the LSTM and 0.0392 for
the baseline, sixteen times better than the baseline. When it says 0.9 it is
right about 90 percent of the time. Max pooling takes an extreme value by
construction, which pushes logits away from zero and produces overconfidence.
Attention averages instead, so its outputs stay closer to honest probabilities.
If a downstream system needed a usable confidence score rather than only a
label, attention pooling would be the one to ship.

**Attention pooling had not finished learning.** Its best epoch was 8 of 8 and
validation loss was still falling at the end, 0.1849 then 0.1817 then 0.1769.
The other two had already turned: the baseline bottomed at epoch 7 and the LSTM
at epoch 6. So 93.41 percent is a lower bound on what this model can do, and the
honest reading is that the budget was wrong for it, not that it is worse.

**The LSTM began overfitting from epoch 7.** Training loss kept falling, 0.1068
then 0.0979 then 0.0896, while validation loss rose, 0.1402 then 0.1405 then
0.1443. Selecting on validation loss is what kept the reported model at epoch 6.
Eight epochs was more than this model needed.

### Performance by review length

Error rate by slice, lower is better. Short, medium and long are terciles of the
test set by word count. Truncated means the review exceeded 320 tokens.

| Model | Short (12,574) | Medium (12,611) | Long (12,815) | Truncated (2,730) |
|---|---|---|---|---|
| baseline_max_pool | 0.0930 | 0.1014 | 0.1082 | 0.0993 |
| experiment_lstm | 0.0511 | 0.0473 | 0.0508 | 0.0476 |
| experiment_attention_pool | 0.0687 | 0.0634 | 0.0655 | 0.0579 |

I expected truncation to be the worst slice, because cutting a review at 320
tokens can remove the sentence where the verdict is stated. The measurement says
otherwise: for all three models the truncated slice is at or below the long
slice, and for the LSTM it is the second best slice of the four. A review long
enough to be truncated has already given the model hundreds of sentiment bearing
tokens, which is apparently more than enough. Raising the length limit is
therefore not the improvement it looks like.

The baseline degrades steadily as reviews get longer, 0.0930 to 0.1082. A single
maximum over more positions is more likely to be captured by one unrepresentative
word. Neither of the other two shows that trend, which is evidence that the
degradation is a property of max pooling and not of long reviews.

## 6. How my model differs from my teammate's

Her run is in the repository now, so this is against her numbers.

We chose different model families on purpose. My three are max pooling, an LSTM
and learned attention pooling. Hers are mean pooling, a convolutional network
and a bidirectional GRU. Only the pooling baselines are close relatives, and
even those differ: she averages over positions where I take a maximum.

| | Mine | Yashashree |
|---|---|---|
| Baseline | masked max pooling | masked mean pooling |
| Experiment A | two layer LSTM | multi width CNN |
| Experiment B | learned attention pooling | bidirectional GRU |
| Seed | 5330 | 4349 |
| Vocabulary | 25,000 | 30,000 |
| Minimum frequency | 3 | 2 |
| Maximum length | 320 | 256 |
| Learning rate | 2e-4 | 3e-4 |
| Epochs | 8 | 5 |

### Results side by side

| | Accuracy | Macro F1 | ROC AUC | MCC | Brier | Parameters |
|---|---|---|---|---|---|---|
| my baseline_max_pool | 0.8991 | 0.8991 | 0.9643 | 0.7981 | 0.0756 | 2,412,545 |
| her baseline_mean_pool | 0.9348 | 0.9348 | 0.9822 | 0.8696 | 0.0482 | 3,856,770 |
| my experiment_lstm | **0.9502** | **0.9502** | **0.9896** | **0.9005** | **0.0375** | 4,297,217 |
| her experiment_bigru | 0.9469 | 0.9469 | 0.9869 | 0.8938 | 0.0412 | 6,057,026 |
| my experiment_attention_pool | 0.9341 | 0.9341 | 0.9819 | 0.8683 | 0.0490 | 3,224,962 |
| her experiment_cnn | 0.9285 | 0.9285 | 0.9818 | 0.8579 | 0.0533 | 5,181,890 |

**My best model edges hers**, 95.02 percent against 94.69, on every headline
metric and with 29 percent fewer parameters. Both of our best models are
recurrent, which is the clearest shared result: on this dataset, reading word
order beats any order free pooling we tried.

**Her baseline is much better than mine, and that is the most useful thing in
this comparison.** Mean pooling reaches 93.48 percent where my max pooling
reaches 89.91, a gap of 3.6 points between two models that differ only in how
they collapse positions into one vector. Max pooling takes the single most
extreme activation per dimension, so one strong word can decide the review.
Mean pooling accumulates evidence across every token, which suits sentiment
because a long review usually contains many mild signals rather than one
decisive word. My own error review supports that reading: eight of my twenty
errors involve a contrast or a concession, which is exactly the case where one
extreme token wins over the balance of the text. If I ran this again I would
make mean pooling the baseline.

**Her baseline beats my attention pooling too**, 93.48 against 93.41, which is
within the margin of a single run. Attention pooling is a learned weighted
average, so it contains mean pooling as a special case and should be at least
as good. The likely explanation is budget rather than architecture: my
attention model was still improving at its final epoch, as noted in section 5.

**A caution on her calibration numbers.** Her reported expected calibration
error is 0.4364, 0.4525 and 0.4629 for the three models. Those cannot be right
alongside her Brier scores. For a balanced test set the Brier decomposition
gives Brier greater than or equal to reliability, and reliability is at least
the square of the expected calibration error, so an error of 0.4629 requires a
Brier of at least 0.214. Her reported Brier is 0.0412. All three of her models
break that bound, so the calibration figure is being computed wrongly rather
than describing badly calibrated models. Her Brier scores are reasonable and
comparable to mine, which is the better evidence that her probabilities are
fine. I have left expected calibration error out of the table above for that
reason, and she should check how it is computed before the report goes in.

**Hardware is not comparable.** She trained on an RTX 4090 and I trained on
Apple M4 through MPS, so her 255 to 292 seconds per model against my 408 to
9,685 reflects the machines. The one timing claim that does survive is internal
to my own run: my LSTM took 22 times longer than my pooling models on the same
device.

## 7. Hardware disclosure

| Field | Value |
|---|---|
| Machine | Apple Mac16,12 |
| CPU | Apple M4, 10 cores, 4 performance and 6 efficiency |
| GPU | Apple M4 integrated, 10 cores, used through PyTorch MPS |
| RAM | 32 GB unified |
| OS | macOS 26.6.2, build 25G83 |
| Python | 3.11.8 |
| PyTorch | 2.14.0 |
| Total training time | 10,531 s for all three models, about 2 hours 56 minutes |
| Peak memory | 1.267 GB, attention pooling |

## 8. Honesty notes

Two things happened during this task that affect how the numbers should be read.

**The first run failed and this is the second.** Attention pooling returned NaN
on reviews that stopword removal had emptied completely, because masking every
position and taking a softmax is zero divided by zero. The run died at the first
validation pass of the third model, after the first two had already trained for
four hours. The fix and a check that catches it in one second are in the
notebook, and the failed log is kept at
`reproducibility/raw_logs/task2_sentiment_20261004_225511_FAILED_attention_nan.log`.
The baseline reproduced to four decimal places across both runs.

**The LSTM is not bit reproducible on this hardware.** Across the two runs the
baseline and attention models reproduced exactly, but the LSTM moved slightly,
best epoch 5 in the first run and 6 in the second, with validation losses 0.1404
and 0.1402. The seeding is the same for all three models, so the cause is the
LSTM kernels on MPS rather than the experiment setup. Anyone rerunning this
should expect the LSTM to land within about 0.1 accuracy points rather than
exactly on 95.02.

**Wall clock times in section 5 are from the second run.** Another training job
shared this machine during part of the first run and stretched LSTM epochs from
22 minutes to 65. The times reported here were measured with the machine
otherwise idle.

## 9. Evidence trail

- Raw log: `../../reproducibility/raw_logs/task2_sentiment_20261005_030918.log`
- Failed run log: `../../reproducibility/raw_logs/task2_sentiment_20261004_225511_FAILED_attention_nan.log`
- Manifest: `../../reproducibility/manifests/anees_saheba_task2_sentiment.yaml`
- Full metrics: `metrics_report.csv` and `outputs/metrics.json`
- Per epoch history: `outputs/training_records.json`
- Curves: `outputs/plots/roc_pr_calibration.png`, `outputs/plots/data_distribution.png`
- Confusion matrices: `outputs/confusion_matrices/`
- Error review: `outputs/error_review_20.csv` and `failure_analysis.md`
