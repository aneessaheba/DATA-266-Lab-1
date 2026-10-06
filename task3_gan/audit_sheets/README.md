# Human audit, 30 fixed samples

Brief section 3.2.6 asks for a blinded audit of 30 fixed samples on style,
content and artifacts, rated by two people, with inter rater agreement
reported. These sheets exist so the rating can be done by looking at one image
rather than opening 30 files.

## Sheets

| File | What it shows |
|---|---|
| anees_saheba_validation_B2A.png | photo to Monet, samples 0 to 29 |
| anees_saheba_validation_A2B.png | Monet to photo, samples 0 to 29 |
| yashashree_shinde_validation_B2A.png | photo to Monet, samples 0 to 29 |
| yashashree_shinde_validation_A2B.png | Monet to photo, samples 0 to 29 |

Each tile is labelled with its sample number, matching the sample_id column in
the audit file.

## Scale, 1 to 5

| Criterion | 1 means | 5 means |
|---|---|---|
| style | nothing like a Monet | indistinguishable from a real Monet |
| content | the original scene is unrecognisable | the scene is fully preserved |
| artifacts | obvious artifacts, grids, blobs, false signatures | clean, no visible artifacts |

Artifacts is scored so that higher is better, like the other two, so the three
can be averaged.

## How to do it

1. Open the two sheets for the member being rated.
2. Each rater fills their own columns in
   `task3_gan/MEMBER/outputs/metrics/human_audit_30_samples.csv`.
   Rater 1 fills rater1_style, rater1_content, rater1_artifacts. Rater 2 fills
   the rater2 columns.
3. Rate independently. Do not look at the other rater's scores first, since
   the agreement number is meaningless if the ratings are not independent.
4. Run the scorer from the repository root:

       python scripts/score_human_audit.py

   It computes per criterion means, exact agreement, agreement within one
   point, and Cohen's kappa with quadratic weights, then writes
   `human_audit_summary.json` next to the audit file and prints the two values
   to paste into metrics_report.csv.

## Things worth looking for

These came out of the automated analysis and are the cases the metrics miss.

- **False signatures.** Both models paint cursive marks in the lower part of
  the frame, because real Monets are signed. Measured in 7 of 30 of Anees's
  outputs and 16 of 30 of Yashashree's. No automatic metric penalises this, so
  the audit is the only place it is caught. It belongs under artifacts.
- **Grid blocks in flat regions.** Saturated checkerboard patches, usually in
  open sky, from the transposed convolution upsampling. Sample 7 is a clear
  example for both members.
- **Under stylisation.** An output that still reads as a photograph with brush
  texture laid over it rather than a repainted scene. This should score low on
  style and high on content, and is expected to be more common in
  Yashashree's outputs because her identity weight is higher.
