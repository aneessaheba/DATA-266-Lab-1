# DATA266 Lab 1, Team 8

## Repository

### https://github.com/aneessaheba/DATA-266-Lab-1

**Members:** Anees Saheba Guddi (SID4 5330), Yashashree Shinde (SID4 4349)

**Date:** 6 October 2026

## 1. Team ownership statement

Both members independently designed, coded, trained and evaluated their own
models for all three tasks, so every number in this report exists twice, once
per member, from two separate runs. Anees built the four block character level
GPT for Task 1, the max pooling, LSTM and attention pooling classifiers for
Task 2, and the six block CycleGAN for Task 3, and also set up the repository,
the branching model, the data download script and the one command smoke test.
Yashashree built the six block character level GPT for Task 1, the mean
pooling, convolutional and bidirectional GRU classifiers for Task 2, and the
nine block CycleGAN for Task 3. Both members submitted independently to the
Kaggle competition. The comparison tables, the joint analyses and this report
were written together from both sets of results.

## 2. Task 1, character level language model

### 2.1 Comparison table

| | Anees | Yashashree |
|---|---|---|
| Transformer blocks | 4 | 6 |
| Embedding width | 256 | 192 |
| Attention heads | 4 | 3 |
| Head dimension | 64 | 64 |
| Feedforward width | 1024 | 768 |
| Context length | 128 | 256 |
| Dropout | 0.10 | 0.15 |
| Positional embedding | learned | learned |
| Normalisation | pre layer norm | pre layer norm |
| Vocabulary | 91 | 101 |
| Parameters | 3,238,912 | 2,757,605 |
| Optimiser | AdamW | AdamW |
| Learning rate | 3e-4 | 3e-4 |
| Weight decay | 0.01 | 0.01 |
| Warmup steps | 1000 | 1000 |
| Schedule | cosine decay | cosine decay |
| Batch size | 64 | 64 |
| Epochs | 10 | 10 |
| Seed | 5330 | 4349 |
| **Validation cross entropy** | 0.7538 | **0.7077** |
| **Validation perplexity** | 2.1251 | **2.0293** |
| **Validation bits per character** | 1.0875 | **1.0210** |
| **Top 1 next character accuracy** | 0.7607 | **0.7759** |
| Generalisation gap | -0.0303 | -0.0009 |
| Gradient norm mean | 0.6679 | 0.6070 |
| Gradient norm max | 6.4868 | 5.4474 |
| Loss spikes | 0 | 0 |
| NaN count | 0 | 0 |
| Distinct 3 at T = 0.8 | **0.7630** | 0.6663 |
| Repeated 4gram at T = 0.8 | **0.1320** | 0.2310 |
| Training tokens per second | 37,528 | 610,618 |
| Total training time | 3,557 s | 435 s |
| Hardware | Apple M4, MPS | RTX 4090 |

Neither member used a prebuilt transformer or attention module. Both wrote
multi head causal self attention with explicit matrix multiplications and
verified numerically that no position can attend to a future position.

### 2.2 Joint analysis

**Strengths.** Both models learned the TinyStories distribution to a similar
degree, around 2.0 to 2.1 perplexity and roughly 76 to 78 percent next
character accuracy, with no loss spikes and no non finite steps in either run.
Both generalisation gaps are negative, meaning validation loss sits at or below
training loss, so neither model overfitted in ten epochs.

**Where they differ and what it shows.** Yashashree's model reaches a better
bits per character, 1.0210 against 1.0875, with 15 percent fewer parameters.
The two candidate explanations are her greater depth and her longer context.
Anees's failure analysis, written before the comparison, predicted that the
128 character context was the binding constraint because coherence broke down
over spans longer than about two sentences. Her 256 character context reaching
a better result on a smaller model is consistent with that, but depth and
context changed together, so this is evidence rather than proof.

**The result that surprised us.** The model with the better validation loss
produces the more repetitive text. At the same temperature of 0.8, Anees's
repeated 4gram rate is 0.1320 against Yashashree's 0.2310, and his distinct 3
is 0.7630 against 0.6663. Perplexity measures how well a model predicts the
next character given a true prefix. Repetition is a property of what happens
when a model is fed its own output, which perplexity never observes. These are
different questions and the ranking does not have to agree.

**Limitations.** The two vocabularies are not the same set. Anees found that
the source text contained mojibake, UTF-8 that had been decoded as cp1252, and
repaired it, which took his vocabulary from 101 characters to 91. Yashashree's
run models the corrupted sequences as if they were real characters. Bits per
character is still comparable because it is normalised per character either
way, but a few of the characters in her run are artifacts. If anything this
makes her result stronger, since her model carries that overhead and still
predicts better.

Training times are not comparable. Apple M4 through MPS against an RTX 4090 is
a hardware difference of roughly eight times, not a model difference.

**What we would try next.** One run of Anees's architecture at a 256 character
context with everything else held fixed. That isolates context length from
depth, which this comparison cannot do because both changed at once.

## 3. Task 2, Yelp Polarity sentiment classification

### 3.1 Comparison table

Test split, 38,000 reviews, balanced. Neither member used pretrained embeddings
or a pretrained language model.

| | Anees baseline | Anees exp A | Anees exp B | Yash baseline | Yash exp A | Yash exp B |
|---|---|---|---|---|---|---|
| Encoder | masked max pooling | 2 layer LSTM | learned attention pooling | masked mean pooling | multi width CNN | bidirectional GRU |
| Vocabulary | 25,000 | 25,000 | 25,000 | 30,000 | 30,000 | 30,000 |
| Minimum frequency | 3 | 3 | 3 | 2 | 2 | 2 |
| Maximum length | 320 | 320 | 320 | 256 | 256 | 256 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 | 3e-4 | 3e-4 | 3e-4 |
| Epochs | 8 | 8 | 8 | 5 | 5 | 5 |
| Seed | 5330 | 5330 | 5330 | 4349 | 4349 | 4349 |
| Parameters | 2,412,545 | 4,297,217 | 3,224,962 | 3,856,770 | 5,181,890 | 6,057,026 |
| **Accuracy** | 0.8991 | **0.9502** | 0.9341 | 0.9348 | 0.9285 | 0.9469 |
| **Macro F1** | 0.8991 | **0.9502** | 0.9341 | 0.9348 | 0.9285 | 0.9469 |
| ROC AUC | 0.9643 | **0.9896** | 0.9819 | 0.9822 | 0.9818 | 0.9869 |
| PR AUC | 0.9657 | 0.9899 | 0.9822 | 0.9824 | 0.9823 | 0.9871 |
| MCC | 0.7981 | **0.9005** | 0.8683 | 0.8696 | 0.8579 | 0.8938 |
| Brier | 0.0756 | **0.0375** | 0.0490 | 0.0482 | 0.0533 | 0.0412 |
| Expected calibration error | 0.0392 | 0.0104 | **0.0024** | see note | see note | see note |
| Training time | 408 s | 9,685 s | 437 s | 256 s | 293 s | 290 s |
| Peak memory | 1.05 GB | 1.26 GB | 1.27 GB | 0.16 GB | 0.45 GB | 1.80 GB |
| Hardware | Apple M4, MPS | Apple M4, MPS | Apple M4, MPS | RTX 4090 | RTX 4090 | RTX 4090 |

**Note on calibration.** Yashashree's reported expected calibration error of
0.4364, 0.4525 and 0.4629 cannot be correct alongside her Brier scores of
0.0482, 0.0533 and 0.0412. For a balanced test set the Brier decomposition
gives Brier at least equal to reliability, and reliability is at least the
square of the expected calibration error, so an error of 0.4629 would require a
Brier of at least 0.214. All three of her models break that bound, so the
calculation is wrong rather than the models being badly calibrated. Her Brier
scores are sound and comparable to Anees's. The values are left in her
`metrics_report.csv` for transparency but are not quoted here.

### 3.2 Significance tests

McNemar against each member's own baseline, on paired predictions over the same
test reviews.

| Model | Statistic | p value | Baseline right, model wrong | Model right, baseline wrong |
|---|---|---|---|---|
| Anees experiment_lstm | 1087.52 | 1.1e-252 | 765 | 2,710 |
| Anees experiment_attention_pool | 599.20 | 4.1e-137 | 814 | 2,147 |
| Yash experiment_bigru | 109.88 | 1.0e-25 | 725 | 1,184 |
| Yash experiment_cnn | 22.74 | 1.9e-06 | 1,376 | 1,136 |

The convolutional model is the one to read carefully. Its result is
statistically significant but the counts run against it: the baseline is right
and the CNN wrong on 1,376 reviews, while the reverse happens on only 1,136.
It is significantly **worse** than its own baseline.

### 3.3 Joint analysis

**Strengths.** Both members' best models are recurrent, and both beat their own
baselines by margins that are not close to chance. The best model across the
team is Anees's LSTM at 95.02 percent, with Yashashree's bidirectional GRU at
94.69 percent and 29 percent more parameters. That both of us independently
arrived at a recurrent model as the strongest is the clearest shared result:
on this dataset, reading word order beats every order free method either of us
tried.

**The most useful disagreement is between the two baselines.** Yashashree's
mean pooling reaches 93.48 percent where Anees's max pooling reaches 89.91, a
gap of 3.6 points between two models that differ only in how they collapse
positions into one vector. Max pooling takes the single most extreme activation
per dimension, so one strong word can decide a review. Mean pooling accumulates
evidence across every token. For sentiment that matters, because a long review
usually carries many mild signals rather than one decisive word. Anees's own
error review supports this from the other direction: eight of his twenty errors
turn on a contrast or a concession, which is exactly where one extreme token
overrides the balance of the text. Had we known this before running, mean
pooling would have been the better baseline for both of us.

**Weaknesses.** Yashashree's convolutional model is worse than her own baseline
despite 34 percent more parameters. A convolution only reaches as far as its
widest kernel, so a negation separated from its target by several words is
still out of range, while mean pooling at least accumulates every token evenly.
Anees's attention pooling was still improving at its final epoch, validation
loss falling 0.1849 then 0.1817 then 0.1769, so its 93.41 percent is a lower
bound rather than a ceiling. His LSTM began overfitting from epoch 7, with
training loss falling while validation loss rose, and was saved by selecting on
validation loss.

**A methodological finding worth recording.** Yashashree's error review
initially concluded that negation was her main failure mode, because 16 of her
20 sampled errors contained negation. Her own slice metrics refute it: reviews
containing negation fail at 0.0532 against 0.0531 for the test set as a whole,
so there is no effect at all. The explanation is the base rate, since 61.2
percent of all test reviews contain negation, and a binomial test on 16 of 20
against that base rate gives p = 0.063. A hand inspection of 20 errors is a
good way to generate a hypothesis and a bad way to test one, because an error
sample carries no information about the base rate it was drawn from. The
corrected analysis is in her `failure_analysis.md`.

**Limitations.** Training times are not comparable across Apple MPS and an RTX
4090. Anees ran 8 epochs and Yashashree 5, so the comparison between our runs
is not purely architectural. Anees's LSTM is not bit reproducible on MPS:
across two runs its best epoch moved from 5 to 6, while his other two models
reproduced exactly, so the cause is the LSTM kernels rather than the seeding.

**What we would try next.** Mean pooling as the baseline for both members, and
a length ablation at 512 tokens, which Yashashree's slice metrics motivate:
long reviews are her worst slice at 0.0565 against 0.0492 for medium, and all
five of her sampled long review failures sit at exactly her 256 token limit.

## 4. Task 3, CycleGAN style transfer

### 4.1 Comparison table

| | Anees | Yashashree |
|---|---|---|
| Generator | resnet, 6 residual blocks | resnet, 9 residual blocks |
| Base filters | 80 | 64 |
| Discriminator | PatchGAN, 16 by 16 | PatchGAN, 16 by 16 |
| Normalisation | instance | instance |
| Adversarial loss | least squares | least squares |
| Parameters per generator | 12,239,363 | 11,378,179 |
| **Total parameters** | 30,008,200 | 28,285,832 |
| Image size | 256 | 256 |
| Batch size | 1 | 1 |
| Epochs | 50 | 50 |
| Learning rate | 2e-4 | 2e-4 |
| Lambda cycle | 10.0 | 10.0 |
| **Lambda identity** | **2.5** | **5.0** |
| Schedule | constant to 25, then linear decay | constant to 25, then linear decay |
| Seed | 5330 | 4349 |
| **FID, Monet to photo** | **178.94** | 189.22 |
| **FID, photo to Monet** | **86.59** | 92.39 |
| **KID, Monet to photo** | **0.0328** | 0.0431 |
| **KID, photo to Monet** | **0.0075** | 0.0122 |
| Density, Monet to photo | 0.9333 | 0.6222 |
| Density, photo to Monet | 0.5420 | 0.4163 |
| Coverage, Monet to photo | 0.0117 | 0.0070 |
| Coverage, photo to Monet | **0.7267** | 0.6867 |
| Cycle L1, Monet side | 0.1647 | **0.1411** |
| Cycle L1, photo side | **0.0949** | 0.0962 |
| Content cosine, Monet to photo | 0.7348 | **0.8011** |
| Content cosine, photo to Monet | **0.8658** | 0.7463 |
| LPIPS, Monet to photo | **0.3556** | 0.3620 |
| LPIPS, photo to Monet | **0.3820** | 0.4177 |
| Non finite steps | 0 | 0 |
| Training time | 32,198 s, 8.94 h | 18,051 s, 5.01 h |
| Throughput | 19.76 img/s | 35.26 img/s |
| Peak memory | 1.568 GB | 1.563 GB |
| Hardware | RTX 4090 | RTX 5090 |
| **Kaggle public score** | **-52.8390** | -54.6990 |

The competition takes a one row `submission.csv` with ID, FID and MiFID, scored
by the instructor's evaluation script, and the leaderboard shows the negative
mean of the two values. A less negative score is better. Anees submitted FID
105.2606 and MiFID 0.4175; Yashashree submitted FID 108.9811 and MiFID 0.4171.

The FID values in the table differ from the submitted ones because they are not
the same measurement. Each member's notebook compares 300 real Monets against
all generated images using its own Inception preprocessing, while the
instructor's scorer subsamples both sides to 300, sorts by filename and resizes
differently. Only the scorer's version is comparable across the class.

### 4.2 Joint analysis

**Strengths.** Both runs completed 50 epochs with zero non finite steps, no
loss spikes and no sign of either discriminator overpowering its generator.
Discriminator losses settled near 0.17 and 0.16 respectively and stayed there.
Both models translate photographs into recognisably Monet like images and both
reconstruct the round trip closely enough to confirm that cycle consistency is
holding.

**The comparison separates cleanly along one hyperparameter.** The two runs
used the same data, epochs, resolution, learning rate and schedule. Three
things differ: the seed, the generator shape, and the identity weight. Anees is
better on every FID and KID number in both directions. Yashashree is better on
cycle reconstruction for the Monet direction, 0.1411 against 0.1647, and on
content similarity, 0.8011 against 0.7348. That is exactly the trade the
identity term controls. Identity loss penalises changing an image that is
already in the target domain, which mostly means holding the colour palette
fixed. At 2.5 the generator is freer to repaint, which the distribution metrics
reward. At 5.0 the palette is held, which keeps the round trip faithful and
leaves outputs looking closer to the source photograph.

This is visible as well as measurable. The same input photograph of a
lighthouse is repainted by Anees's model and comes back from Yashashree's model
still reading as a photograph with brush texture applied over it.

**Weaknesses.** Both models paint fake artist signatures into the lower part of
the frame. Real Monets are signed, so a corner mark is evidence of authenticity
to the discriminator and the generator learns to produce one. Measured over
each member's 30 validation translations by comparing gradient energy in the
bottom strip against the rest of the image, the artifact appears in 7 of 30 of
Anees's outputs and 16 of 30 of Yashashree's. No automatic metric here
penalises it, and both members' Kaggle submissions contain it.

Both runs also show grid structured artifacts in flat regions such as open sky.
Both generators upsample with transposed convolutions using a kernel size of 3
and a stride of 2, and because 3 does not divide evenly by 2 some output pixels
receive more contributions than others at a fixed spacing.

**Limitations.** Three variables changed at once between the two runs, so
neither member can attribute the difference to the identity weight alone. The
Monet to photo direction is weak for both, FID 178.94 and 189.22, partly
because inventing photographic detail is harder than applying a style, and
partly because it is measured on only 30 generated images against 7,038 real
photographs, which depresses coverage to 0.0117 and 0.0070 and makes that
figure less reliable than the reverse direction. Both runs used 50 epochs where
the CycleGAN paper uses 200 for this task, and both members' cycle losses were
still falling slowly at the end. Training times are not comparable across an
RTX 4090 and an RTX 5090.

**What we would try next.** A single run with only the identity weight changed,
holding the generator shape and the seed fixed, which would turn the
correlation we observed into an attribution. Replacing the transposed
convolutions with nearest neighbour upsampling followed by a plain convolution,
which is the standard fix for the grid artifact. And masking signatures out of
the training paintings, to test whether the signature artifact is learned from
them as we believe.

## 5. Failure and error analysis

Full write ups with every snippet are in each member's `failure_analysis.md`.
The cases below are quoted from those files.

### 5.1 Task 1, Anees

**Repetition, greedy decoding.** "One day, Tom went to the park with his mom. He
saw a big storm on the ground. He wanted to see what was inside. He wanted to
see what was inside." The same sentence appears twice verbatim. This sample has
the worst repetition of the twelve generated, 0.316 against 0.104 for the most
diverse. Greedy decoding is deterministic, so once the model returns to a
similar hidden state the same continuation becomes most probable again and
nothing can break the loop.

**Loss of coherence.** "They saw a big boy who was very sad and angry. They
wanted to show her mommy how to be careful when they saw the boy was so happy
to have a new friend." The boy is sad and angry then happy within one sentence.
The grammar is correct throughout; what fails is state tracking, because the
model predicts from the previous 128 characters only and has no representation
of facts already asserted.

**Hallucination at T = 1.2.** "She loved playing outside and majwming running
around ... Carefully, here no hourse. She saw her mom radious." Three strings
are not English words. This is specific to character level modelling: a word
level model can only emit words from its vocabulary, while a character level
model can reach any letter sequence.

### 5.2 Task 1, Yashashree

**Broken grammar and semantic inconsistency.** "She had a big smile on her house
and a special wind." A smile belongs to a person, not a house, and the wind is
not connected to the sentence. The model learned common words and patterns
without learning relationships between objects and actions.

**Loss of coherence and abrupt topic change.** "Every day, she saw a big tree
with lots of money. The pink was a big pretty girl with a playground beautiful
pla" The sentence introduces money, a pink girl and a playground with no
transition, and ends incomplete.

**Repetition and semantic inconsistency.** "They saw a big tree with a big rock.
The tree was filled with a tall green ball on the ground." The model repeats
"big", "tree" and "ground" and describes a tree filled with a ball.

Her closing observation is the one worth carrying forward: the model reaches
77.6 percent next character accuracy while still producing these failures,
because next character accuracy can be high even when the errors that occur
damage the meaning of a longer story.

### 5.3 Task 2, both members

Both members reviewed 20 of their own errors in the four categories the brief
requires: five confident false positives, five confident false negatives, five
near threshold cases and five from a slice.

**Anees, twenty errors.** Eight turn on a reversal or a concession, four on
mixed or hedged sentiment, two on third party narrative, two on vocabulary
gaps, two on truncation and two look like wrong gold labels. His largest group
makes contrast marker weighting the first fix to try. His slice numbers
contradicted the intuitive hypothesis: the truncated slice has a **lower**
error rate than the long slice for all three of his models, so raising the
length limit is the obvious fix and the wrong one.

**Yashashree, twenty errors.** Five involve negation over a long span, five
mixed or aspect split sentiment, four truncation at 256 tokens, three sarcasm,
two temporal reversal and one too little signal. As recorded in section 3.3,
her negation hypothesis did not survive her own slice metrics, while truncation
did: all five of her long review failures sit at exactly her 256 token limit
and long reviews are her worst slice.

Taken together, the two reviews disagree about truncation and the disagreement
is explained by the limits themselves. Anees truncates at 320 tokens and sees
no truncation effect; Yashashree truncates at 256 and sees one. That is
consistent with the conclusion rather than against it: the cost of truncation
appears somewhere between 256 and 320 tokens on this dataset.

### 5.4 Task 3, both members

Both members documented three visual cases with the images attached in their
folders.

**Anees.** A hallucinated artist signature in `validation_B2A/0000.png`,
measured in 7 of 30 samples. Repeated blobs and colour banding across a flat
sky in `validation_B2A/0007.png`, from the transposed convolution upsampling.
A colour shift in cycle reconstruction, where Monet's haystacks return with the
composition intact but the palette moved from warm pink to cool blue.

**Yashashree.** Under stylisation in `validation_B2A/0000.png`, where the
output keeps the photograph's structure, lighting and colour with brush texture
laid over it. A bright blue and red checkerboard block in
`validation_B2A/0007.png`. And her cycle reconstruction of a valley with
poplars and cottages, included as the counterexample because it returns almost
exactly, which is the same identity weight that caused her first case.

## 6. Evidence trail

Every number in this report is traceable to a committed file.

| Evidence | Location |
|---|---|
| Raw training logs, unedited | `reproducibility/raw_logs/` |
| Manifests, six, one per member per task | `reproducibility/manifests/` |
| Task 1 metrics | `task1_llm/<member>/metrics_report.csv` |
| Task 1 loss curves | `task1_llm/anees_saheba/outputs/loss_curves.png`, `task1_llm/yashashree_shinde/outputs/task1_loss_curves.svg` |
| Task 1 generated samples | `task1_llm/<member>/outputs/` |
| Task 1 checkpoints | `task1_llm/anees_saheba/checkpoints/best_model.pt`, `task1_llm/yashashree_shinde/checkpoints/task1_best_model.pt` |
| Task 2 metrics | `task2_sentiment/<member>/metrics_report.csv` |
| Task 2 slice and significance | `task2_sentiment/anees_saheba/metrics_report.csv`, `task2_sentiment/yashashree_shinde/task2_slice_metrics.json` and `task2_mcnemar_results.json` |
| Task 2 error reviews | `task2_sentiment/<member>/` error review csv and `failure_analysis.md` |
| Task 2 checkpoints | `task2_sentiment/<member>/checkpoints/`, three per member |
| Task 3 metrics | `task3_gan/<member>/metrics_report.csv` and `outputs/metrics/full_metrics_report.json` |
| Task 3 training curves | `task3_gan/<member>/outputs/figures/training_curves.png` |
| Task 3 sample translations | `task3_gan/<member>/outputs/validation_A2B`, `validation_B2A`, `validation_cycle_A`, `validation_cycle_B` |
| Task 3 checkpoints | `task3_gan/<member>/checkpoints/generators_epoch_050.pt` |
| Kaggle submissions | `task3_gan/<member>/outputs/submission.csv` |
| Reproduction script | `scripts/smoke_test.sh`, one command, runs Task 1 end to end |

Checkpoints are tracked with Git LFS. Optimiser state was removed from
Yashashree's Task 1 and Task 2 checkpoints before committing, which reduced
them from 206 MB to 70 MB without affecting the weights, since Adam moment
buffers are needed only to resume training.

## 7. Known gaps

Stated rather than hidden.

**The human audit is not complete for either member.** Both
`human_audit_30_samples.csv` files are 30 row templates with empty rater
columns. The notebooks deliberately do not generate ratings. This is the only
part of the Task 3 evaluation that would catch the signature artifact, since no
automatic metric penalises it.

**Yashashree's Task 1 and Task 2 runs captured no raw training logs.** Her
notebooks did not write them and they cannot be reconstructed after the fact.
Her Task 3 log is present. Both members' manifests record this.

**Yashashree's Task 2 outputs folder is empty**, so there are no loss curves or
confusion matrices for her three classifiers.

**Yashashree's Task 2 calibration numbers are wrong**, as set out in section
3.1, and need recomputing.

**Anees's Task 3 checkpoint carries a stale config field** recording nine
residual blocks where the weights are unmistakably six at 80 filters,
12,239,363 parameters per generator. The YAML config and the write ups match
the actual weights.

## 8. References

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L. and Polosukhin, I. (2017). Attention Is All You Need. *Advances in
Neural Information Processing Systems 30*.

Eldan, R. and Li, Y. (2023). TinyStories: How Small Can Language Models Be and
Still Speak Coherent English? *arXiv:2305.07759*.

Zhu, J.-Y., Park, T., Isola, P. and Efros, A. A. (2017). Unpaired Image to
Image Translation using Cycle Consistent Adversarial Networks. *IEEE
International Conference on Computer Vision (ICCV)*.

Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y. and Yoo, J. (2020). Reliable Fidelity
and Diversity Metrics for Generative Models. *International Conference on
Machine Learning (ICML)*. Source of the density and coverage metrics used in
Task 3.
