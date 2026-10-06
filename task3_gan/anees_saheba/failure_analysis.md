# Task 3 Failure and Shortcoming Analysis, anees_saheba

**Model:** CycleGAN, six residual blocks on 80 base filters, seed 5330
**Checkpoint:** `checkpoints/generators_epoch_050.pt`
**Images:** all cases below are files in `outputs/`, unedited

## Visual failure cases

### Case 1: Hallucinated signature

**Image:** `outputs/validation_B2A/0000.png`, a lighthouse photograph translated to Monet
**Artifact type:** hallucinated content, specifically a painted signature

**Observation:**

In the lower right corner the model has painted what reads as a cursive artist
signature. There is nothing resembling text in the input photograph. The model
invented it.

The cause is in the training data rather than the architecture. Monet
paintings are signed, and the signature sits in a corner, usually the lower
right. The discriminator therefore learns that a small region of dark cursive
strokes in that corner is evidence of a real Monet, and the generator learns to
produce one because it lowers the adversarial loss. Nothing in the objective
distinguishes style from provenance marks.

I checked whether this was a one off by measuring gradient energy in the lower
right of all 30 validation translations against the rest of each image. Seven
of the thirty have a ratio above 1.25, with a maximum of 1.97. So it is
systematic but not universal, which fits an artifact learned from a feature
present in most but not all of the training paintings.

This matters for the Kaggle submission. A painted signature is exactly the kind
of local texture that improves FID while being obviously wrong to a person.

### Case 2: Repeated texture blobs and colour banding in flat regions

**Image:** `outputs/validation_B2A/0007.png`, a railway track under open sky
**Artifact type:** checkerboard and texture hallucination

**Observation:**

The sky, which is close to featureless in the input, comes back filled with
repeated flower shaped blobs arranged on a rough grid, and the colour bands
through green, pink and blue rather than varying smoothly. The track and the
ground, which have real structure, translate cleanly in the same image.

Two separate causes meet here. The grid regularity is the signature of
transposed convolution: the upsampling layers have a kernel size that does not
divide evenly by the stride, so some output pixels receive contributions from
more input pixels than others, and the imbalance repeats at a fixed spacing.
The content of the blobs is the second cause. A flat region gives the generator
almost no signal to preserve, while the discriminator still demands painterly
texture, so the generator invents brushwork. Monet's skies do contain visible
strokes, so the invented texture is drawn from a plausible distribution. It is
just not anchored to anything in the input.

Replacing the transposed convolutions with nearest neighbour upsampling
followed by a plain convolution is the standard fix for the grid, and it is the
first change I would test.

### Case 3: Colour shift in cycle reconstruction

**Images:** input `../data/monet_jpg/cb9c553ded.jpg`, reconstruction
`outputs/validation_cycle_A/0000.png`. The pairing is recorded in
`outputs/metrics/validation_translation_manifest.csv`.
**Artifact type:** colour shift, with structure preserved

**Observation:**

The input is Monet's haystacks in warm pink and orange light. After going to
the photo domain and back, every shape returns in the right place, the two
haystacks, the horizon, the field texture, but the palette has moved to cool
blue and cyan. The painting is recognisable and the colour is wrong.

This is the clearest visual statement of what the cycle reconstruction number
means. L1 on this direction is 0.1647, the worse of the two, and this is what
that value looks like: structure recovered, colour not.

It is also the visible cost of my own hyperparameter choice. Identity loss is
the term that holds the palette in place, and I run it at 2.5 where my
teammate runs 5.0. Her cycle reconstruction on this direction is 0.1411
against my 0.1647. I traded this for better FID and KID in both directions, and
section 6 of results.md reports both sides of that trade.

## Training stability observations

Stable throughout. **Zero non finite steps across all 50 epochs**, which is
316,750 optimiser steps.

| Epoch | Generator | Discriminator | Cycle A | Cycle B | Generator gradient norm |
|---|---|---|---|---|---|
| 1 | 6.6975 | 0.4254 | 0.2166 | 0.2382 | 36.78 |
| 11 | 4.3762 | 0.1966 | 0.0997 | 0.1254 | 38.77 |
| 21 | 4.0013 | 0.1718 | 0.0869 | 0.1083 | 31.39 |
| 31 | 3.7625 | 0.1623 | 0.0827 | 0.0960 | 28.63 |
| 41 | 3.5178 | 0.1681 | 0.0815 | 0.0837 | 28.81 |
| 50 | 3.3199 | 0.1800 | 0.0797 | 0.0752 | 17.05 |

No loss spikes and no sign of the discriminator winning. Discriminator loss
settles near 0.17 and stays there, which is roughly where it should sit when
neither network is dominating. Generator gradient norms fall steadily from 37
to 17, and the sharp drop over the last ten epochs follows the learning rate
decaying to zero rather than anything about the model.

One thing in the log is worth stating plainly rather than hiding. **Validation
generator loss rises while training loss falls**, from 6.15 at epoch 1 to 6.80
at epoch 50, with training loss going the other way from 6.70 to 3.32. In a
supervised model that reads as overfitting. Here it does not mean much, because
the validation generator loss is measured against a discriminator that is
itself still training. A generator loss can rise simply because its opponent
improved. The numbers that actually track quality are the cycle losses, which
fall throughout on both training and validation, and FID and KID, which are
computed from the final checkpoint against held out images.

## Cycle consistency verification

Cycle consistency is the assumption the whole method rests on, so it is
measured rather than asserted.

| Direction | L1 | Images |
|---|---|---|
| Monet to photo to Monet | 0.1647 | 30 unique Monet paintings |
| Photo to Monet to photo | 0.0949 | 703 unique photographs |

Measured over unique images per domain. The naive version of this measurement
repeats the 300 Monet images to match the 7,038 photographs, which weights the
Monet side by that repetition and reports a number that is not an average over
the data.

Visual evidence is Case 3 above: `validation_cycle_A/0000.png` against its
input. The composition returns, the palette does not. Reconstructions for all
30 validation pairs are in `outputs/validation_cycle_A` and
`outputs/validation_cycle_B`.

The photo direction reconstructs better, 0.0949 against 0.1647. A photograph
contains more information than a painting of the same scene, so the round trip
that starts from a painting has to invent more and loses more.

## Known shortcomings of my run

**The signature artifact is in the output and would be in a Kaggle submission.**
Seven of thirty samples show it. I did not discover it until after training
finished, and removing it means either cropping the corner or masking
signatures out of the training paintings and retraining.

**The Monet to photo direction is weak,** FID 178.94. Partly real, since
inventing photographic detail is harder than applying a style. But it is also
measured on only 30 generated images against 7,038 real ones, which depresses
coverage to 0.0117 and makes the FID less reliable than the reverse direction.
Generating more images in that direction would give a fairer number.

**Three variables changed at once** against my teammate's run: seed, generator
shape, identity weight. The results are better on every FID and KID number, but
I cannot say how much of that is the identity weight on its own. The clean
experiment is a single run with only that term changed.

**Only 50 epochs.** The CycleGAN paper trains the Monet task for 200. The cycle
losses were still falling slowly at epoch 50, so the model had not finished
learning. 50 was chosen to match my teammate's run and to fit the GPU slot.

**The human audit is not filled in.** It needs two raters, and it is the only
part of the evaluation that would catch the signature artifact, since no
automatic metric here penalises it.
