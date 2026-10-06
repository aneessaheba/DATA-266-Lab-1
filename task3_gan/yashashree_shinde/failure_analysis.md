# Task 3 Failure and Shortcoming Analysis, yashashree_shinde

**Model:** CycleGAN, nine residual blocks on 64 base filters, seed 4349
**Checkpoint:** `checkpoints/generators_epoch_050.pt`
**Images:** every case below is a file in `outputs/`, unedited

## Visual failure cases

### Case 1: Under stylisation, the output still reads as a photograph

**Image:** `outputs/validation_B2A/0000.png`, a lighthouse and pier translated
to Monet
**Artifact type:** insufficient style transfer

**Observation:**

The output keeps the photograph's structure, its lighting and most of its
colour. Brush texture has been laid over the surface but the scene has not been
repainted: the sky gradient, the red railing and the water all sit where the
camera put them, in the shades the camera recorded. Next to a real Monet the
difference is immediate.

The cause is my identity weight. Identity loss penalises the generator for
changing an image that is already in the target domain, and its main effect is
to hold the colour palette steady. I run it at 5.0, half the cycle weight,
which is the usual setting. The distribution metrics reward the opposite: FID
and KID measure how close my outputs sit to the real Monet distribution, and a
Monet is defined partly by a palette that departs from the photographic one.

My teammate ran the same architecture family at identity weight 2.5 and reached
FID 86.59 against my 92.39 and KID 0.0075 against my 0.0122. The visual
difference between our outputs on this same image is exactly this: his is
repainted, mine is a photograph with texture.

The fix to test is lowering the identity weight to 2.5 and measuring whether
FID improves and by how much cycle reconstruction degrades.

### Case 2: Checkerboard colour blocks in flat regions

**Image:** `outputs/validation_B2A/0007.png`, a snow covered mountain under
open sky
**Artifact type:** checkerboard from transposed convolution

**Observation:**

A bright block of saturated blue and red appears on the left of the image,
roughly 30 pixels across, arranged on a visible grid. Nothing in the input
photograph corresponds to it. The surrounding terrain translates cleanly.

This is the signature of transposed convolution. The upsampling layers use a
kernel size of 3 with a stride of 2, and because 3 does not divide evenly by 2
some output pixels receive contributions from more input pixels than others.
The imbalance repeats at a fixed spacing, which is why the artifact is a grid
rather than a smear. It appears in flat regions because there is no strong
input signal there to dominate the uneven overlap.

Replacing the transposed convolutions with nearest neighbour upsampling
followed by a plain convolution is the standard fix and is the first change I
would test.

### Case 3: Cycle reconstruction, what works rather than what fails

**Images:** input `../data/monet_jpg/bc4b364a44.jpg`, reconstruction
`outputs/validation_cycle_A/0000.png`. The pairing is recorded in
`outputs/metrics/validation_translation_manifest.csv`.
**Artifact type:** none significant, included as the counterexample

**Observation:**

A Monet of a valley with poplars, cottages and a wooden fence is translated to
a photograph and back. The reconstruction returns the composition, the poplars,
the cottages on the hillside, the fence in the foreground, and the palette:
the blue sky, the ochre hill, the green field all come back in roughly the
right hue. The visible loss is in fine detail, where brush strokes have been
softened.

This is the measured cycle reconstruction L1 of 0.1411 shown as an image, and
it is the best of the two runs on this metric. My teammate's is 0.1647.

It is included here because it is the direct consequence of the choice that
caused case 1. The identity weight that holds my outputs too close to the
source photograph is the same term that makes my round trip faithful. The two
cases are one tradeoff seen from both ends.

## Measured artifact rate

Case 2 and a related signature artifact in the lower part of the frame are
visible in several outputs, so I measured rather than counted by eye. For each
of the 30 validation translations I compared gradient energy in the bottom
strip of the image against the rest of it.

| | Mean ratio | Median | Maximum | Above 1.25 |
|---|---|---|---|---|
| Mine | 1.27 | 1.26 | 2.73 | 16 of 30 |
| Teammate | 1.02 | 0.99 | 1.55 | 7 of 30 |

More than half of my validation outputs carry elevated structure along the
bottom edge, against roughly a quarter of my teammate's. Part of this is the
painted signature that both models learned, since real Monets are signed and
the discriminator rewards a corner mark. The larger ratio on my side suggests
the deeper nine block generator reproduces that artifact more strongly.

## Training stability observations

Stable throughout. **Zero non finite steps across all 50 epochs.**

| Epoch | Generator | Discriminator | Cycle A | Cycle B | Generator gradient norm |
|---|---|---|---|---|---|
| 1 | 7.7211 | 0.4264 | 0.2168 | 0.2362 | 43.17 |
| 10 | 5.0264 | 0.1614 | 0.1024 | 0.1299 | 43.59 |
| 20 | 4.4901 | 0.1527 | 0.0872 | 0.1123 | 34.70 |
| 30 | 4.0969 | 0.1589 | 0.0808 | 0.0990 | 29.42 |
| 40 | 3.7943 | 0.1596 | 0.0808 | 0.0852 | 25.79 |
| 50 | 3.5460 | 0.1671 | 0.0812 | 0.0735 | 16.94 |

No loss spikes and no sign of the discriminator winning. Discriminator loss
settles near 0.16 and stays there, which is roughly where it should sit when
neither network dominates. Generator gradient norms fall from 43 to 17, and the
sharp drop over the final ten epochs follows the learning rate decaying to
zero.

One thing in the log is worth stating rather than hiding. **Validation
generator loss rises while training loss falls,** from 6.89 at epoch 1 to 7.66
at epoch 50, with training loss going the other way from 7.72 to 3.55. In a
supervised model that reads as overfitting. Here it means less, because the
validation generator loss is measured against a discriminator that is itself
still improving, so the generator's loss can rise simply because its opponent
got better. The numbers that track quality are the cycle losses, which fall
throughout, and FID and KID, computed from the final checkpoint against held
out images.

## Cycle consistency verification

| Direction | L1 | Images |
|---|---|---|
| Monet to photo to Monet | 0.1411 | 30 unique Monet paintings |
| Photo to Monet to photo | 0.0962 | 703 unique photographs |

Measured over unique images per domain rather than over a repeated set.

Visual evidence is case 3 above. Reconstructions for all 30 validation pairs
are in `outputs/validation_cycle_A` and `outputs/validation_cycle_B`.

## Known shortcomings of my run

**Style transfer is too conservative.** The clearest weakness, and it traces to
one hyperparameter. Case 1 shows it and the FID gap against my teammate
measures it.

**The grid artifact is in the outputs and went into the Kaggle submission.**
I did not notice it until after training finished.

**The Monet to photo direction is weak,** FID 189.22, and is measured on only
30 generated images against 7,038 real photographs, which depresses coverage to
0.0070 and makes the number less reliable than the reverse direction.

**Only 50 epochs** against the 200 the CycleGAN paper uses for this task. The
cycle losses were still falling slowly at the end.

**The human audit is not filled in.** It needs two raters, and it is the only
part of the evaluation that would catch the signature and grid artifacts, since
no automatic metric here penalises them.
