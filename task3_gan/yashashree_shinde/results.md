# Task 3 CycleGAN style transfer, Monet and photo, yashashree_shinde

**Member:** yashashree_shinde
**Seed:** 4349, the last four digits of my student ID
**Run:** 20261002_233139
**Checkpoint:** `checkpoints/generators_epoch_050.pt`

## 1. What I built

A CycleGAN that translates between Monet paintings and photographs in both
directions, trained from scratch with no pretrained weights. The data is
unpaired, so there is no target image to compare any output against. CycleGAN
handles this with two generators that reverse each other: one paints a photo
into a Monet, the other turns a Monet back into a photo, and the training
signal is that passing an image through both should return the original. Two
PatchGAN discriminators push each output to look like a real member of its
target domain.

## 2. Architecture

| Component | Choice | Why I chose it |
|---|---|---|
| Generator | resnet, nine residual blocks on 64 base filters | The standard CycleGAN generator for 256 by 256 inputs, and the configuration the paper uses for the Monet task. I wanted the reference architecture as my starting point. |
| Downsampling | two stride 2 convolutions | Keeps the residual stack at a quarter resolution so nine blocks stay affordable. |
| Upsampling | two transposed convolutions | Standard. It is also the source of the grid artifacts discussed in the failure analysis. |
| Normalisation | instance | Per image rather than per batch, which suits style transfer where each image has its own colour statistics, and is necessary at batch size 1. |
| Padding | reflection | Avoids the dark border that zero padding leaves at image edges. |
| Discriminator | PatchGAN, 16 by 16 output grid | Judges local patches, which forces local texture to look painterly rather than only the global colour being right. |
| Adversarial loss | least squares | More stable than binary cross entropy for GANs and the standard choice for CycleGAN. |

Total parameters across two generators and two discriminators: 28,285,832.

## 3. Hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Seed | 4349 | My student ID, so initialisation and shuffling are mine. |
| Image size | 256 | The size the competition scorer uses. |
| Batch size | 1 | The CycleGAN default. Instance normalisation means a larger batch adds little. |
| Epochs | 50 | What fitted the booked GPU slot. |
| Learning rate | 2e-4 | The CycleGAN paper value. |
| Betas | 0.5 and 0.999 | Lower first moment is standard for GANs and reduces oscillation. |
| Schedule | constant to epoch 25, then linear decay to zero | Lets the model settle before the learning rate is pulled away. |
| Lambda cycle | 10.0 | The paper value. Cycle loss is the main constraint holding content in place. |
| Lambda identity | 5.0 | Half the cycle weight, which is the usual setting. Identity loss asks the generator to leave an image alone when it is already in the target domain, which holds the colour palette steady. |
| Gradient clip | 1.0 | Generator gradient norms reached 43 early in training, so clipping is doing real work. |

## 4. How my metrics were computed

- Notebook: `src/task_3_code.ipynb`, run end to end
- Checkpoint: `checkpoints/generators_epoch_050.pt`, both generators at the final epoch
- Split: a held out validation split, 30 Monet paintings and 703 photographs, never seen during training
- FID and KID use Inception v3 features. KID is averaged over random subsets rather than truncating both sets, which matters because only 300 Monet paintings exist.
- Full numbers: `metrics_report.csv` and `outputs/metrics/full_metrics_report.json`

## 5. Results

| Metric | Monet to photo (A2B) | Photo to Monet (B2A) |
|---|---|---|
| FID | 189.22 | **92.39** |
| KID | 0.0431 | **0.0122** |
| Density | 0.6222 | 0.4163 |
| Coverage | 0.0070 | 0.6867 |
| Cycle reconstruction L1 | 0.1411 | 0.0962 |
| Content cosine similarity | 0.8011 | 0.7463 |
| LPIPS | 0.3620 | 0.4177 |

### Cost

| Field | Value |
|---|---|
| Parameters | 28,285,832 |
| Training time | 18,051 s, 5.01 hours for 50 epochs |
| Throughput | 35.26 images per second |
| Peak memory | 1.563 GB |
| Non finite steps | 0 |
| GPU | NVIDIA GeForce RTX 5090 |

### Kaggle result

Public leaderboard score **-54.6990**, from a submitted FID of 108.9811 and
MiFID of 0.4171. The competition takes a one row `submission.csv` with ID, FID
and MiFID, and the leaderboard shows the negative mean of the two, so a less
negative score is better. The submitted file is `outputs/submission.csv`.

### What the numbers say

**The photo to Monet direction works much better than the reverse,** FID 92.39
against 189.22. Going towards Monet means applying a consistent style with a
limited palette and visible brush texture. Going towards photography means
inventing detail that was never in the painting, and there is no single correct
photographic appearance to converge on. Coverage says the same from the other
side: 0.6867 of real Monets have a nearby generated sample against 0.0070 of
real photographs, though that second figure is also depressed by generating
only 30 images against 7,038 real photos.

**Cycle consistency is the strongest part of this run.** L1 is 0.1411 on the
Monet side and 0.0962 on the photo side, and content cosine similarity on the
Monet to photo direction is 0.8011. Visually the round trip is close to exact,
as case 3 of the failure analysis shows. The identity weight of 5.0 is the main
reason: holding the palette in place keeps the reconstruction faithful.

**That faithfulness has a cost in style.** My FID and KID are both worse than my
teammate's, who ran the same number of epochs at the same resolution with
identity weight 2.5 instead of 5.0. Her photo to Monet FID is 86.59 against my
92.39 and her KID is 0.0075 against my 0.0122. The failure analysis shows what
this looks like: my outputs often still read as photographs with texture applied
rather than as repainted scenes.

## 6. How my model differs from my teammate's

Both runs used the same data, the same 50 epochs, the same image size, the same
learning rate and schedule. Three things differ.

| | Mine (yashashree_shinde) | Anees |
|---|---|---|
| Seed | 4349 | 5330 |
| Residual blocks | 9 | 6 |
| Base filters | 64 | 80 |
| Parameters per generator | 11,378,179 | 12,239,363 |
| Total parameters | 28,285,832 | 30,008,200 |
| Lambda identity | 5.0 | 2.5 |
| GPU | RTX 5090 | RTX 4090 |
| Training time | 5.01 h | 8.94 h |

| Metric | Mine | Anees | Better |
|---|---|---|---|
| Photo to Monet FID | 92.39 | **86.59** | his |
| Photo to Monet KID | 0.0122 | **0.0075** | his |
| Photo to Monet coverage | 0.6867 | **0.7267** | his |
| Monet to photo FID | 189.22 | **178.94** | his |
| Cycle reconstruction A | **0.1411** | 0.1647 | mine |
| Content cosine A2B | **0.8011** | 0.7348 | mine |
| Kaggle score | -54.6990 | **-52.8390** | his |

**The comparison separates cleanly along the identity weight.** His model is
better at looking like a Monet and mine is better at staying faithful to the
input. That is what the identity term controls: it penalises changing an image
that is already in the target domain, which mostly means holding colour. At 5.0
it keeps my reconstructions accurate and holds my outputs closer to the source
photograph than the distribution metrics reward.

The parameter budgets are within six percent of each other, so neither model is
simply larger. The difference is where the capacity sits, his wider and
shallower and mine deeper and narrower, and how the objective is weighted.

Because three things changed at once, neither of us can attribute the whole
difference to the identity weight alone. A clean attribution needs a run with
only that term changed.

The training times are not comparable. My RTX 5090 ran at 35.26 images per
second against his 19.76 on a 4090.

## 7. Hardware disclosure

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Peak GPU memory | 1.563 GB |
| Total training time | 18,051 s, 5.01 hours |
| Throughput | 35.26 images per second |

## 8. Shortcomings of this run

**The human audit is not filled in.**
`outputs/metrics/human_audit_30_samples.csv` is a 30 row template with empty
rater columns. It needs two people and the notebook does not generate ratings.

**Only 50 epochs.** The CycleGAN paper trains the Monet task for 200. Cycle
losses were still falling slowly at epoch 50.

**Only the final checkpoint is kept.** The milestone checkpoints and the
optimiser state are not in the repository. Only epoch 50 is needed to reproduce
the reported translations.

**The Monet to photo direction is evaluated on only 30 generated images**
against 7,038 real photographs, which depresses coverage to 0.0070 and makes
that FID less reliable than the reverse direction.

## 9. Evidence trail

- Raw log: `../../reproducibility/raw_logs/task3_cyclegan_20261002_233139.log`
- Manifest: `../../reproducibility/manifests/yashashree_shinde_task3_gan.yaml`
- Full metrics: `metrics_report.csv`, `outputs/metrics/full_metrics_report.json`
- Per epoch history: `outputs/metrics/training_history.json`
- Training curves: `outputs/figures/training_curves.png`
- Sample translations: `outputs/validation_A2B`, `validation_B2A`, `validation_cycle_A`, `validation_cycle_B`
- Which input maps to which output: `outputs/metrics/validation_translation_manifest.csv`
- Kaggle submission: `outputs/submission.csv`
