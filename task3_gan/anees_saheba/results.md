# Task 3 CycleGAN style transfer, Monet and photo, anees_saheba

**Member:** anees_saheba
**Seed:** 5330, the last four digits of my student ID
**Run:** 20261005_201541
**Checkpoint:** `checkpoints/generators_epoch_050.pt`

## 1. What I built

A CycleGAN that translates in both directions between Monet paintings and
photographs, trained from scratch with no pretrained weights anywhere in the
model. The data is unpaired: no photograph is the same scene as any painting,
so there is no target image to compare an output against. CycleGAN solves that
with two generators that undo each other. One turns a photo into a Monet, the
other turns a Monet back into a photo, and the training signal is that putting
an image through both should return the original. Two PatchGAN discriminators
push the outputs to look like real members of their target domain. The three
losses together are what make the task learnable without pairs.

## 2. Architecture

| Component | Choice | Why I chose it |
|---|---|---|
| Generator | resnet, six residual blocks on 80 base filters | My teammate uses the standard nine blocks on 64. Mine spends almost the same budget, 12.2 million parameters against her 11.4, on a wider and shallower bottleneck. A wider bottleneck carries more channels at the smallest spatial size, which is where the texture and palette of a style live. |
| Downsampling | two stride 2 convolutions | Standard. Keeps the bottleneck at a quarter resolution so the residual blocks are affordable. |
| Upsampling | two transposed convolutions | Standard, and the source of the checkerboard artifacts discussed in the failure analysis. |
| Normalisation | instance | Per image rather than per batch, which suits style transfer where each image has its own colour statistics. Also necessary at batch size 1. |
| Padding | reflection | Avoids the dark border that zero padding leaves at image edges. |
| Discriminator | PatchGAN, 16 by 16 output grid | Judges local patches rather than the whole image, which is what forces local texture to look painterly instead of only the global colour being right. |
| Adversarial loss | least squares | More stable than binary cross entropy for GANs and the standard choice for CycleGAN. |

## 3. Hyperparameters

| Hyperparameter | Value | Why |
|---|---|---|
| Seed | 5330 | My student ID, so initialisation and shuffling are mine. |
| Image size | 256 | The size Kaggle scores at, and the size my teammate used, so the comparison is fair. |
| Batch size | 1 | The CycleGAN default. With instance normalisation a larger batch gives little benefit. |
| Epochs | 50 | Matches my teammate's run so the comparison is like for like. |
| Learning rate | 2e-4 | The CycleGAN paper value. |
| Betas | 0.5 and 0.999 | Lower first moment is standard for GANs and reduces oscillation. |
| Schedule | constant to epoch 25, then linear decay to zero | Lets the model settle before the learning rate is pulled away. |
| Lambda cycle | 10.0 | The paper value. Cycle loss is the main constraint holding content in place. |
| Lambda identity | **2.5** | My teammate runs 5.0, the usual half of cycle. I halved it. Identity loss asks the generator to leave an image alone when it is already in the target domain, which mostly pins the colour palette. The Monet direction is scored on how repainted the output looks, so a strong identity term works against the thing being measured. |
| Gradient clip | 1.0 | Generator gradient norms reached 37 early, so clipping is doing real work. |

## 4. How my metrics were computed

- Notebook: `src/cyclegan_monet_photo.ipynb`, run end to end with nbconvert
- Config: `src/configs/cyclegan_config.yaml`
- Checkpoint: `checkpoints/generators_epoch_050.pt`, both generators at the final epoch
- Split: a held out validation split, 30 Monet images and 703 photographs, never seen in training
- FID and KID use Inception v3 features. KID is averaged over random subsets rather than truncating both sets, which matters here because only 300 Monet paintings exist.

## 5. Results

| Metric | Monet to photo (A2B) | Photo to Monet (B2A) |
|---|---|---|
| FID | 178.94 | **86.59** |
| KID | 0.0328 | **0.0075** |
| Density | 0.9333 | 0.5420 |
| Coverage | 0.0117 | 0.7267 |
| Cycle reconstruction L1 | 0.1647 | 0.0949 |
| Content cosine similarity | 0.7348 | 0.8658 |
| LPIPS | 0.3556 | 0.3820 |

### Cost

| Field | Value |
|---|---|
| Parameters | 30,008,200 across two generators and two discriminators |
| Training time | 32,198 s, 8.94 hours for 50 epochs |
| Throughput | 19.76 images per second |
| Peak memory | 1.568 GB |
| Non finite steps | 0 |

### What the numbers say

**The photo to Monet direction works much better than the reverse,** FID 86.59
against 178.94. That is not a defect in the model, it is a property of the two
domains. Going towards Monet means adding a consistent style: a limited palette,
visible brush texture, soft edges. Going towards photography means inventing
detail that was never in the painting, and there is no single correct
photographic appearance to converge on. The coverage numbers say the same
thing from the other side: 0.7267 of real Monets have a nearby generated
sample, against 0.0117 of real photographs, though that second number is also
depressed by generating only 30 images against 7,038 real photos.

**Density and coverage disagree in an informative way.** In the Monet direction
density is 0.5420 while coverage is 0.7267. Coverage above density means the
outputs are spread across the variety of real Monets rather than clustered on
a few safe ones, which is the opposite of mode collapse. In the photo
direction the pattern reverses, density 0.9333 against coverage 0.0117: the few
outputs produced sit close to the real manifold but represent almost none of
its variety.

**Cycle consistency holds better in the photo direction,** L1 0.0949 against
0.1647. A photograph carries more information than a painting of it, so
discarding detail and then re inventing it loses more than the reverse. The
visual evidence in the failure analysis shows what that number looks like: the
structure comes back, the colour does not.

## 6. How my model differs from my teammate's

Both runs used the same data, the same 50 epochs, the same image size, the same
learning rate and schedule. Three things differ: the seed, the generator shape
and the identity weight.

| | Mine (anees_saheba) | Yashashree |
|---|---|---|
| Seed | 5330 | 4349 |
| Residual blocks | 6 | 9 |
| Base filters | 80 | 64 |
| Parameters per generator | 12,239,363 | 11,378,179 |
| Total parameters | 30,008,200 | 28,285,832 |
| Lambda identity | 2.5 | 5.0 |
| GPU | RTX 4090 | RTX 5090 |
| Training time | 8.94 h | 5.01 h |

Results side by side:

| Metric | Mine | Hers | Better |
|---|---|---|---|
| Photo to Monet FID | **86.59** | 92.39 | mine by 6 percent |
| Photo to Monet KID | **0.0075** | 0.0122 | mine by 38 percent |
| Photo to Monet coverage | **0.7267** | 0.6867 | mine |
| Photo to Monet LPIPS | **0.3820** | 0.4177 | mine |
| Monet to photo FID | **178.94** | 189.22 | mine by 5 percent |
| Monet to photo KID | **0.0328** | 0.0431 | mine by 24 percent |
| Cycle reconstruction A | 0.1647 | **0.1411** | hers |
| Content cosine A2B | 0.7348 | **0.8011** | hers |

**My model is better at looking like Monet and worse at staying faithful to the
input, and that is exactly the trade the identity weight controls.** Lowering
it from 5.0 to 2.5 released the generator to shift colour, which improved every
distribution metric in both directions. The cost shows up in the two metrics
that measure similarity to the original: cycle reconstruction on the Monet side
is 17 percent worse than hers, and content similarity is 8 percent lower.

Because three things changed at once, I cannot attribute the whole difference
to the identity weight alone. The architecture change and the seed also
contribute. A clean attribution would need a run with only the identity weight
changed, which did not fit the GPU slot. The direction of the result is at
least consistent with what the identity term is supposed to do.

The parameter budgets are within six percent of each other on purpose, so
neither model is simply bigger than the other.

## 7. Hardware disclosure

| Field | Value |
|---|---|
| Machine | GPU lab container |
| GPU | NVIDIA GeForce RTX 4090 |
| Peak GPU memory | 1.568 GB |
| PyTorch | 2.1.2 |
| Precision | bfloat16 autocast on CUDA |
| Total training time | 32,198 s, 8.94 hours |
| Throughput | 19.76 images per second |

The 8.94 hours against my teammate's 5.01 is mostly the GPU, not the model. She
had an RTX 5090 and measured 35.26 images per second against my 19.76 on a
4090. My generator benchmarks 1.07 times her cost, so the architecture accounts
for a small part of the gap and the hardware for the rest.

### Kaggle result

Submitted 6 October 2026. **Public leaderboard score -52.8390**, against
Yashashree's -54.6990 on the same competition.

The competition does not take images. It takes a one row `submission.csv` with
columns ID, FID and MiFID, scored by the instructor's own evaluation script,
and the leaderboard shows the negative mean of those two numbers. A less
negative score is better.

| | Mine | Yashashree |
|---|---|---|
| FID submitted | 105.2606 | 108.9811 |
| MiFID submitted | 0.4175 | 0.4171 |
| **Leaderboard score** | **-52.8390** | -54.6990 |

Per direction under that scorer: photo to Monet FID 95.4884, Monet to photo FID
115.0328, each measured on 300 pairs.

Those FID values differ from the ones in the table above, 86.59 and 178.94,
because the two are not the same measurement. My notebook compares 300 real
Monets against all 703 generated images and uses its own Inception
preprocessing. The instructor's scorer subsamples both sides to 300, sorts by
filename and uses a different resize and crop. Neither is wrong; they answer
slightly different questions, and only the scorer's version is comparable
across the class.

The submitted file is `outputs/submission.csv` and the script that produces it
is `src/make_kaggle_submission.py`, which embeds the instructor's scoring logic
unchanged so the number can be reproduced.

## 8. Honesty notes

**The run itself only translated 30 of the 300 Monet paintings,** because that
was the validation split. The scorer needs 300, and FID on 30 images is badly
biased, so the remaining 270 were generated afterwards from the saved epoch 50
checkpoint before scoring. No retraining was involved.

**The human audit is not done.** `outputs/metrics/human_audit_30_samples.csv` is
a 30 row template with empty rater columns. It needs two people. The notebook
deliberately refuses to generate ratings.

**The config dictionary saved inside the checkpoint is wrong in one field.**
It records `residual_blocks: 9` where the weights are unmistakably six blocks
at 80 filters, 12,239,363 parameters per generator. The notebook wrote that
field from a stale value. Everything else, including
`src/configs/cyclegan_config.yaml` and the numbers in this document, matches
the actual weights.

**Only the final checkpoint is kept.** The milestone checkpoints at epochs 1
through 40 and the optimiser state, 850 MB in total, are not in the repository.
Only epoch 50 is needed to reproduce the reported translations.

## 9. Evidence trail

- Raw log: `../../reproducibility/raw_logs/task3_cyclegan_20261005_201541.log`
- Manifest: `../../reproducibility/manifests/anees_saheba_task3_gan.yaml`
- Full metrics: `metrics_report.csv`, `outputs/metrics/full_metrics_report.json`
- Per epoch history: `outputs/metrics/training_history.json`
- Training curves: `outputs/figures/training_curves.png`
- Sample translations: `outputs/validation_A2B`, `validation_B2A`, `validation_cycle_A`, `validation_cycle_B`
- Which input maps to which output: `outputs/metrics/validation_translation_manifest.csv`
