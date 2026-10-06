# Task 2 Yelp Polarity sentiment classification, yashashree_shinde

**Member:** yashashree_shinde
**Seed:** 4349, the last four digits of my student ID
**Best model:** `experiment_bigru`, test accuracy 0.9469

## 1. What I built

Three binary sentiment classifiers for Yelp Polarity, trained from scratch. No
pretrained embeddings and no pretrained language model: every embedding table
starts from random initialisation and is learned only from the training split.
The three share the same vocabulary, preprocessing and training loop, and
differ only in how they turn a variable length sequence of token embeddings
into one fixed vector. The baseline averages over positions. The first
experiment slides convolutions of several widths over the sequence. The second
runs a bidirectional GRU. Holding everything else fixed is what makes the
comparison between them mean anything.

## 2. Architecture

| Component | Choice | Why I chose it |
|---|---|---|
| Tokeniser | lowercase, word split, stopwords removed | Simple and reproducible, with no external tokeniser to version. |
| Negation words kept | not, no, never, cannot and contractions | Removing them is the default stopword behaviour and it destroys sentiment. "not good" and "good" must not become the same sequence. |
| Vocabulary | 30,000, minimum frequency 2 | Larger than my teammate's 25,000 so that food and service specific words survive. |
| Embedding | learned from random initialisation | The brief forbids pretrained embeddings. |
| Baseline pooling | masked mean pooling | Averages evidence across every token. Sentiment is usually spread across a review rather than concentrated in one word. |
| Experiment A | multi width convolutions | Convolutions of several widths capture local phrases such as "not good" as single features, which pooling cannot. |
| Experiment B | bidirectional GRU | Reads the review in both directions, so it can represent word order and carry negation scope across a span. |
| Classifier head | linear, activation, dropout, linear to one logit | Identical in all three, so any difference comes from the encoder. |
| Loss | binary cross entropy | Two balanced classes. |

## 3. Hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Seed | 4349 | My student ID, so initialisation and splitting are mine. |
| Maximum length | 256 tokens | Covers most reviews. The error review shows this is also where my long review failures sit. |
| Vocabulary | 30,000 | Deliberately larger than my teammate's 25,000. |
| Minimum frequency | 2 | Lower than his 3, so rarer words survive. |
| Learning rate | 3e-4 | Converged within five epochs without instability. |
| Epochs | 5 | Enough for all three models to stop improving on validation. |

Full configuration in `task2_model_configuration.json`.

## 4. How my metrics were computed

- Notebook: `src/code.ipynb`, run end to end
- Checkpoints: one per model under `checkpoints/`, each the best epoch by validation loss
- Split: the numbers below are on the **test** split, 38,000 reviews, never used for training or model selection
- Threshold: 0.5, not tuned on test
- Confidence intervals: bootstrap, 95 percent, on accuracy, macro F1 and MCC
- Full numbers: `metrics_report.csv`, `task2_metrics_report.json`, `task2_slice_metrics.json`, `task2_mcnemar_results.json`

## 5. Results

Test split, 38,000 reviews, balanced.

| Model | Accuracy | Macro F1 | ROC AUC | PR AUC | MCC | Brier | Parameters | Training time |
|---|---|---|---|---|---|---|---|---|
| baseline_mean_pool | 0.9348 | 0.9348 | 0.9822 | 0.9824 | 0.8696 | 0.0482 | 3,856,770 | 255.83 s |
| experiment_cnn | 0.9285 | 0.9285 | 0.9818 | 0.9823 | 0.8579 | 0.0533 | 5,181,890 | 292.50 s |
| **experiment_bigru** | **0.9469** | **0.9469** | **0.9869** | **0.9871** | **0.8938** | **0.0412** | 6,057,026 | 290.14 s |

95 percent bootstrap intervals on accuracy: baseline 0.9324 to 0.9373, CNN
0.9263 to 0.9308, BiGRU 0.9447 to 0.9492. The BiGRU interval does not overlap
either of the others.

McNemar against the baseline, on paired predictions over the same reviews:

| Model | Chi square | p value | Baseline right, model wrong | Model right, baseline wrong |
|---|---|---|---|---|
| experiment_cnn | 22.74 | 1.9e-06 | 1,376 | 1,136 |
| experiment_bigru | 109.88 | 1.0e-25 | 725 | 1,184 |

The CNN result is worth reading carefully. It is significant, but the counts
run the wrong way: the baseline is right and the CNN wrong on 1,376 reviews,
while the reverse happens on only 1,136. So the CNN is **significantly worse**
than the baseline, not better.

### Performance by slice

| Slice | Reviews | Macro F1 | Error rate |
|---|---|---|---|
| short_reviews | 12,899 | 0.9434 | 0.0549 |
| medium_reviews | 14,843 | 0.9508 | 0.0492 |
| long_reviews | 10,258 | 0.9413 | 0.0565 |
| contains_negation | 23,262 | 0.9430 | 0.0532 |
| whole test set | 38,000 | 0.9469 | 0.0531 |

### What the numbers say

**The BiGRU wins, and word order is why.** It is the only one of my three that
reads the review in sequence and in both directions. Mean pooling and
convolutions both see the review as a set of local observations. The 1.2 point
gain over the baseline is what reading order is worth on this data, and the
McNemar result rules out chance.

**The CNN is worse than the baseline.** That surprised me, since convolutions
should capture phrases that mean pooling cannot, and it costs 34 percent more
parameters to be 0.6 points worse. The likely explanation is reach: a
convolution only sees as far as its widest kernel, so a negation separated from
its target by several words is still out of range, while mean pooling at least
accumulates every token evenly. On this dataset the local phrase features did
not pay for what they lost.

**Negation is not the weakness I expected it to be.** Reviews containing
negation fail at 0.0532 against 0.0531 for the test set as a whole, so there is
no effect at all. This matters because my error review initially read the
opposite from a sample of 20 errors, 16 of which contained negation. That turned
out to be the base rate: 61.2 percent of all test reviews contain negation. The
slice metrics are the right instrument for this question and the error sample is
not.

**Long reviews are the real weak slice**, 0.0565 against 0.0492 for medium, a 15
percent relative increase. My maximum length is 256 tokens, and all five long
review failures in my error review sit exactly at that limit. Raising it to 512
is the experiment I would run first.

**My calibration numbers are wrong and should not be quoted.** I report expected
calibration error of 0.4364, 0.4525 and 0.4629. Those cannot be correct
alongside my Brier scores of 0.0482, 0.0533 and 0.0412. For a balanced test set
the Brier decomposition gives Brier at least equal to reliability, and
reliability is at least the square of the calibration error, so an error of
0.4629 would require a Brier of at least 0.214. All three of my models break
that bound. The Brier scores themselves are reasonable, which is the better
evidence that my probabilities are fine. The column stays in
`metrics_report.csv` for transparency but the calculation needs fixing before
the number means anything.

## 6. How my model differs from my teammate's

We chose different model families on purpose. Only the pooling baselines are
close relatives, and even those differ: I average over positions where he takes
a maximum.

| | Mine (yashashree_shinde) | Anees |
|---|---|---|
| Baseline | masked mean pooling | masked max pooling |
| Experiment A | multi width CNN | two layer LSTM |
| Experiment B | bidirectional GRU | learned attention pooling |
| Seed | 4349 | 5330 |
| Vocabulary | 30,000 | 25,000 |
| Minimum frequency | 2 | 3 |
| Maximum length | 256 | 320 |
| Learning rate | 3e-4 | 2e-4 |
| Epochs | 5 | 8 |

| | Accuracy | Macro F1 | ROC AUC | MCC | Brier | Parameters |
|---|---|---|---|---|---|---|
| my baseline_mean_pool | 0.9348 | 0.9348 | 0.9822 | 0.8696 | 0.0482 | 3,856,770 |
| his baseline_max_pool | 0.8991 | 0.8991 | 0.9643 | 0.7981 | 0.0756 | 2,412,545 |
| my experiment_bigru | 0.9469 | 0.9469 | 0.9869 | 0.8938 | 0.0412 | 6,057,026 |
| his experiment_lstm | **0.9502** | **0.9502** | **0.9896** | **0.9005** | **0.0375** | 4,297,217 |
| my experiment_cnn | 0.9285 | 0.9285 | 0.9818 | 0.8579 | 0.0533 | 5,181,890 |
| his experiment_attention_pool | 0.9341 | 0.9341 | 0.9819 | 0.8683 | 0.0490 | 3,224,962 |

**My baseline is much stronger than his,** 93.48 percent against 89.91, on two
models that differ only in how they collapse positions into one vector. Max
pooling takes the single most extreme activation per dimension, so one strong
word can decide a review. Mean pooling accumulates evidence across every token.
For sentiment that matters, because a long review usually carries many mild
signals rather than one decisive word. His own error review reached the same
conclusion from the other direction: eight of his twenty errors turn on a
contrast or a concession, which is exactly where one extreme token overrides the
balance of the text.

**His best model beats mine,** 95.02 against 94.69, with 29 percent fewer
parameters. Both of our best models are recurrent, which is the clearest shared
result between us: on this data, reading word order beats every order free
method either of us tried.

**His third model beats my third model** by a wider margin than the headline
suggests, 93.41 against 92.85, and his attention pooling is also order free. So
my CNN underperforming is a property of that architecture on this task rather
than of order free models generally.

**Calibration cannot be compared** until my numbers are recomputed, as set out
in section 5.

**Training times are not comparable.** I trained on an RTX 4090 and he trained
on Apple M4 through MPS.

## 7. Hardware disclosure

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| CPU | AMD64 Family 25 Model 97 Stepping 2 |
| OS | Windows 11, build 10.0.26200 |
| CUDA | 13.2 |
| Total training time | 838 s for all three models |
| Peak memory | 1.80 GB, the BiGRU |

## 8. Shortcomings of this run

**The calibration calculation is wrong** and needs recomputing, as set out in
section 5.

**No raw training log was captured.** The brief asks for logs to be kept
unedited as the evidence trail. A rerun should write one.

**No loss curves or confusion matrices were saved.** The outputs folder is
empty, so the training behaviour cannot be inspected visually the way it can
for Task 1.

**Five epochs** was enough for validation loss to stop improving, but it is
fewer than my teammate's eight, so the comparison between our runs is not
purely architectural.

**The CNN was not investigated further** after it underperformed. Given more
time the obvious check is whether widening its largest kernel closes the gap.

## 9. Evidence trail

- Metrics: `metrics_report.csv`, `task2_metrics_report.json`
- Slices: `task2_slice_metrics.json`
- Significance tests: `task2_mcnemar_results.json`
- Configuration: `task2_model_configuration.json`
- Error review: `task2_error_review.csv` and `failure_analysis.md`
- Checkpoints: `checkpoints/baseline_mean_pool.pt`, `experiment_cnn.pt`, `experiment_bigru.pt`
- Code: `src/code.ipynb`
- Manifest: `../../reproducibility/manifests/yashashree_shinde_task2_sentiment.yaml`
