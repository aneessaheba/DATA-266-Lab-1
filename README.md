# DATA266 Lab 1, Team 8

LLM pretraining, sentiment classification, CycleGAN style transfer.

**Report:** [report/DATA266_Lab1_Report_Team_8.pdf](report/DATA266_Lab1_Report_Team_8.pdf)

## Team ownership

| Member folder | Name | SJSU ID | GitHub | Built |
|---|---|---|---|---|
| `anees_saheba` | Anees Saheba Guddi | 018205330 | [@aneessaheba](https://github.com/aneessaheba) | Task 1 four block character GPT, Task 2 max pooling, LSTM and attention pooling classifiers, Task 3 six block CycleGAN. Also the repository setup, branching model, data download script and smoke test. |
| `yashashree_shinde` | Yashashree Shinde | 019134349 | [@yashashree5](https://github.com/yashashree5) | Task 1 six block character GPT, Task 2 mean pooling, CNN and bidirectional GRU classifiers, Task 3 nine block CycleGAN. |

Each member independently designed, coded and trained their own models for all
three tasks. No two models in this repository share the same architecture and
hyperparameters, and every reported number exists twice, once per member.

## Headline results

| Task | Metric | Anees | Yashashree |
|---|---|---|---|
| 1 | validation bits per character | 1.0875 | **1.0210** |
| 1 | top 1 next character accuracy | 0.7607 | **0.7759** |
| 2 | best model test accuracy | **0.9502** (LSTM) | 0.9469 (BiGRU) |
| 2 | best model macro F1 | **0.9502** | 0.9469 |
| 3 | photo to Monet FID | **86.59** | 92.39 |
| 3 | photo to Monet KID | **0.0075** | 0.0122 |
| 3 | Kaggle public score | **-52.8390** | -54.6990 |

Full numbers are in each member's `metrics_report.csv` and in the report.

## Repository layout

```
DATA-266-Lab-1/
├── README.md
├── environment/              shared env definition and pinned versions
├── scripts/                  data download, smoke test, audit scorer
├── task1_llm/
│   ├── data/                 shared raw TinyStories, gitignored, see scripts/
│   └── <member>/
├── task2_sentiment/
│   ├── data/                 shared raw Yelp Polarity, gitignored, see scripts/
│   └── <member>/
├── task3_gan/
│   ├── data/{monet_jpg,photo_jpg}/   shared images, gitignored, see scripts/
│   ├── audit_sheets/         contact sheets for the human audit
│   └── <member>/
├── reproducibility/
│   ├── manifests/            one manifest per member per task, six in total
│   └── raw_logs/             unedited training logs, never clean these up
└── report/
    ├── DATA266_Lab1_Report_Team_8.pdf
    ├── DATA266_Lab1_Report_Team_8.docx
    ├── DATA266_Lab1_Report_Team_8.md
    └── figures/
```

Each member folder under each task:

```
<member>/
├── src/                 notebooks with their outputs, and configs/
├── data_processed/      own preprocessing output, gitignored
├── checkpoints/         trained weights, tracked with Git LFS
├── outputs/             samples, predictions, plots, confusion matrices
├── metrics_report.csv   every required metric for this task
├── failure_analysis.md  required failure or error write up
└── results.md           architecture and hyperparameter justification
```

## Branching model

`main` holds work that runs. Task work happens on a branch owned by one member:

```
<member>/<task>          e.g.  anees_saheba/task1-llm
                               yashashree_shinde/task3-gan
```

Rules we follow:

- One branch per member per task. You only touch files inside your own member
  folder, so the two of us never conflict.
- Merge to `main` through a pull request, so the other member sees the change.
- Shared files such as `README.md`, `scripts/` and `environment/` change on a
  `setup/<what-changed>` branch, since they affect everyone's runs.
- Commit messages say what changed and why, not just "update".

## Setup

```bash
git clone https://github.com/aneessaheba/DATA-266-Lab-1.git
cd DATA-266-Lab-1
git lfs install
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
```

Git LFS is required. The checkpoints are stored through it, and without LFS
they arrive as small text pointers rather than weights.

## Getting the data

```bash
bash scripts/download_data.sh          # all three datasets
bash scripts/download_data.sh task1    # or one at a time
```

Raw datasets are gitignored. The download script is the reproducible source of
truth. Task 3 expects `monet_jpg` and `photo_jpg` under `task3_gan/data/`, 300
and 7,038 images.

## One command smoke test

```bash
bash scripts/smoke_test.sh
```

That single command reproduces a shortened version of anees_saheba's Task 1 run
from a clean checkout. It fetches a small slice of TinyStories if the data is
not already present, trains the character level GPT for one epoch on 2,000
sequences, and then checks that the run produced what it should.

It takes a few minutes and does not need a GPU. It passes only if every code
cell ran without an exception, the causal masking check reported that no future
information reaches any position, training completed, and the metrics file,
loss curves and generated samples were all written.

The test runs in a temporary directory outside the repository and modifies
nothing that is tracked, so it is safe to run on a checkout that already holds
real results.

## Reproducing a full run

Every task is a notebook that runs top to bottom. There is no separate training
script. No path is hard coded and there are no secrets.

```bash
jupyter nbconvert --to notebook --execute \
  task1_llm/anees_saheba/src/char_gpt_tinystories.ipynb \
  --output run.ipynb --ExecutePreprocessor.timeout=86400
```

Substitute the notebook for the run you want:

| Task | Member | Notebook |
|---|---|---|
| 1 | anees_saheba | `task1_llm/anees_saheba/src/char_gpt_tinystories.ipynb` |
| 1 | yashashree_shinde | `task1_llm/yashashree_shinde/src/task_1.ipynb` |
| 2 | anees_saheba | `task2_sentiment/anees_saheba/src/yelp_sentiment_models.ipynb` |
| 2 | yashashree_shinde | `task2_sentiment/yashashree_shinde/src/code.ipynb` |
| 3 | anees_saheba | `task3_gan/anees_saheba/src/cyclegan_monet_photo.ipynb` |
| 3 | yashashree_shinde | `task3_gan/yashashree_shinde/src/task_3_code.ipynb` |

Settings can be changed without editing a notebook:

| Variable | Effect |
|---|---|
| `DATA266_EPOCHS` | Task 1 epoch count |
| `DATA266_TINYSTORIES` | Task 1 data path |
| `DATA266_TRAIN_SUBSET` | Task 2 training rows, 0 means all |
| `DATA266_NUM_EPOCHS` | Task 3 epoch count |
| `DATA266_IMAGE_SIZE` | Task 3 training resolution, defaults to 256 |
| `DATA266_MONET_DIR` | Task 3 Monet folder |
| `DATA266_PHOTO_DIR` | Task 3 photo folder |

Expected runtimes, measured: Task 1 about 59 minutes on Apple M4 MPS or 7
minutes on an RTX 4090. Task 2 about 3 hours on M4 MPS or 14 minutes on an RTX
4090. Task 3 about 9 hours on an RTX 4090 or 5 hours on an RTX 5090. Time one
epoch before committing to a booked GPU slot.

Every completed run records a config, an unedited raw log in
`reproducibility/raw_logs/`, and a manifest in `reproducibility/manifests/`
mapping each checkpoint to the result it produced.

## Kaggle submission

Task 3 submits a one row `submission.csv` with ID, FID and MiFID, scored by the
instructor's evaluation script. The leaderboard shows the negative mean of the
two, so a less negative score is better.

```bash
python task3_gan/anees_saheba/src/make_kaggle_submission.py
```

That regenerates the translations the scorer needs and writes the submission
file. It embeds the instructor's scoring logic unchanged so the number is
reproducible.

## Human audit

The 30 sample audit needs two raters and is not complete. Contact sheets are in
`task3_gan/audit_sheets/` and the scale is defined in the README there. Once
both raters have filled in their columns:

```bash
python scripts/score_human_audit.py
```

That computes per criterion means, exact agreement, agreement within one point,
and Cohen's kappa with quadratic weights.

## Where each result lives

| Task | Member | Metrics | Write up | Checkpoint | Raw log |
|---|---|---|---|---|---|
| 1 | anees_saheba | `task1_llm/anees_saheba/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/best_model.pt` | `task1_gpt_20261004_212529.log` |
| 1 | yashashree_shinde | `task1_llm/yashashree_shinde/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/task1_best_model.pt` | not captured |
| 2 | anees_saheba | `task2_sentiment/anees_saheba/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/`, three models | `task2_sentiment_20261005_030918.log` |
| 2 | yashashree_shinde | `task2_sentiment/yashashree_shinde/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/`, three models | not captured |
| 3 | anees_saheba | `task3_gan/anees_saheba/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/generators_epoch_050.pt` | `task3_cyclegan_20261005_201541.log` |
| 3 | yashashree_shinde | `task3_gan/yashashree_shinde/metrics_report.csv` | `results.md`, `failure_analysis.md` | `checkpoints/generators_epoch_050.pt` | `task3_cyclegan_20261002_233139.log` |

Raw logs live in `reproducibility/raw_logs/`. Two of Anees's Task 1 logs are
kept from superseded runs, one stopped by an external time limit and one at a
different seed, renamed to mark their status rather than deleted. One Task 2
log is from a run that failed on a NaN in attention pooling, kept as the
evidence for that fix.

## Known gaps

- The 30 sample human audit is not complete for either member. It needs two
  raters.
- Yashashree's Task 1 and Task 2 runs captured no raw training logs. Her
  notebooks did not write them and they cannot be reconstructed.
- Yashashree's Task 2 outputs folder holds no loss curves or confusion
  matrices.
- Yashashree's Task 2 expected calibration error is computed wrongly. The
  reported values cannot hold alongside her Brier scores, as set out in section
  II.C of the report. Her Brier scores are sound.
- Anees's Task 3 checkpoint carries a stale config field recording nine
  residual blocks where the weights are six at 80 filters. The YAML config and
  all write ups match the actual weights.

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Eldan and Li, *TinyStories: How Small Can Language Models Be and Still Speak
   Coherent English?*, arXiv:2305.07759, 2023.
3. Zhu et al., *Unpaired Image to Image Translation using Cycle Consistent
   Adversarial Networks*, ICCV 2017.
4. Naeem et al., *Reliable Fidelity and Diversity Metrics for Generative
   Models*, ICML 2020.
