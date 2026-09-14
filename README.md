# DATA266 Lab 1 — Team 8

LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer

## Team ownership

| Member folder | Name | GitHub | Built |
|---|---|---|---|
| `anees_saheba` | Anees Saheba | [@aneessaheba](https://github.com/aneessaheba) | _fill in before submission_ |
| `yashashree_shinde` | Yashashree Shinde | [@yashashree5](https://github.com/yashashree5) | _fill in before submission_ |

Per the brief, each member independently designs, codes, and trains their own
models for all three tasks. No two models in this repo share the same
architecture *and* hyperparameters.

## Repository layout

```
DATA-266-Lab-1/
├── README.md
├── environment/              # shared env definition + per-run pip freezes
├── scripts/                  # dataset download + smoke test entry points
├── task1_llm/
│   ├── data/                 # shared raw TinyStories (gitignored, see scripts/)
│   └── <member>/
├── task2_sentiment/
│   ├── data/                 # shared raw Yelp Polarity (gitignored, see scripts/)
│   └── <member>/
├── task3_gan/
│   ├── data/{monet_jpg,photo_jpg}/   # shared images (gitignored, see scripts/)
│   └── <member>/
├── reproducibility/
│   ├── manifests/            # one manifest per member per task
│   └── raw_logs/             # UNEDITED training logs — never clean these up
└── report/
    └── DATA266_Lab1_Report_Team_8.pdf
```

Each member folder under each task:

```
<member>/
├── src/                 # code (task notebooks with output + configs/)
├── data_processed/      # your own preprocessing output (gitignored)
├── checkpoints/         # your trained weights
├── outputs/             # samples, predictions, plots, confusion matrices
├── metrics_report.csv   # every required metric for this task
├── failure_analysis.md  # required failure/error write-up
└── results.md           # architecture + hyperparameter justification
```

## Branching model

`main` holds work that runs. Task work happens on a branch owned by one member:

```
<member>/<task>          e.g.  anees_saheba/task1-llm
                               yashashree_shinde/task3-gan
```

Rules we follow:

- One branch per member per task — you only ever touch files inside your own
  member folder, so the two of us never conflict.
- Merge to `main` through a pull request, so the other member sees the change.
- Shared files (`README.md`, `scripts/`, `environment/`) change on a
  `setup/<what-changed>` branch and get reviewed by both of us, since they
  affect everyone's runs.
- Commit messages say what changed and why, not just "update".

## Setup

```bash
git clone <this-repo>
cd DATA-266-Lab-1
python3 -m venv .venv && source .venv/bin/activate
pip install -r environment/requirements.txt
```

## Getting the data

```bash
bash scripts/download_data.sh          # all three datasets
bash scripts/download_data.sh task1    # or one at a time
```

Raw datasets are gitignored; the download script is the reproducible source of truth.

## One-command smoke test

> **Not yet available.** `scripts/smoke_test.sh` is added alongside the first
> trained model (Task 1). It must run end-to-end from a fresh clone in a few
> minutes on CPU, proving one member's pipeline works without a GPU.

## Reproducing a specific member's full run

All runs are config-driven — no hard-coded paths, no secrets.

```bash
python <task>/<member>/src/train.py --config <task>/<member>/src/configs/<config>.yaml
```

Every completed run records: a config, an unedited raw log in `reproducibility/raw_logs/`,
and a manifest in `reproducibility/manifests/` mapping checkpoint → reported result.

## Where each result lives

| Task | Member | Metrics | Write-up | Logs |
|---|---|---|---|---|
| | | | | |

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023)
- Zhu et al., *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks* (ICCV 2017)
