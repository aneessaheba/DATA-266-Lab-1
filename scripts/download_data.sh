#!/usr/bin/env bash
# Download the three shared raw datasets for DATA266 Lab 1.
#
# Raw data is gitignored; this script is the reproducible source of truth for
# how it was obtained. Paths are derived from the repo root, never hard-coded,
# so this works on any machine (laptop, GPU lab node, Colab).
#
# Usage:
#   bash scripts/download_data.sh          # all datasets
#   bash scripts/download_data.sh task1    # TinyStories only
#   bash scripts/download_data.sh task2    # Yelp Polarity only
#   bash scripts/download_data.sh task3    # Monet/Photo only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Interpreter used for the download helpers. Defaults to the repository virtual
# environment when it exists, because the dataset libraries are installed there
# and not in the system python. Override with PYTHON=/path/to/python
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi
TARGET="${1:-all}"

log() { printf '[download_data] %s\n' "$*"; }

# --- Task 1: TinyStories ----------------------------------------------------
# Streamed rather than fully downloaded: the complete dataset is several GB,
# and a character-level model needs only a small slice of it. TINYSTORIES_N
# controls how many stories are pulled (default 20000, roughly 20 MB), which
# is far more than the 100K/10K character split the brief asks for.
#
#   TINYSTORIES_N=50000 bash scripts/download_data.sh task1
download_task1() {
  local out_dir="$REPO_ROOT/task1_llm/data"
  local n_stories="${TINYSTORIES_N:-20000}"
  mkdir -p "$out_dir"
  if [[ -s "$out_dir/TinyStories-sample.txt" ]]; then
    log "Task 1: already present, skipping. Delete the file to re-download."
    return
  fi
  log "Task 1: streaming $n_stories TinyStories examples..."
  "${PYTHON}" - "$out_dir" "$n_stories" <<'PYSTREAM'
import sys, pathlib, itertools
from datasets import load_dataset

out_dir = pathlib.Path(sys.argv[1])
n_stories = int(sys.argv[2])

# streaming=True pulls examples one at a time over the network instead of
# downloading the entire dataset to disk first.
stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

path = out_dir / "TinyStories-sample.txt"
n_chars = 0
with path.open("w", encoding="utf-8") as fh:
    for row in itertools.islice(stream, n_stories):
        text = row["text"].strip()
        fh.write(text + "\n<|endofstory|>\n")
        n_chars += len(text)

print(f"  wrote {path}")
print(f"  {n_stories} stories, {n_chars:,} characters, {path.stat().st_size / 1e6:.1f} MB")

# The streaming reader leaves worker processes and a shared memory manager
# behind that do not always terminate, which leaves this script hanging
# indefinitely after the file is already complete. The file is closed and
# flushed by this point, so exit immediately rather than waiting on cleanup
# that may never finish.
import sys, os
sys.stdout.flush()
os._exit(0)
PYSTREAM
}

# --- Task 2: Yelp Polarity --------------------------------------------------
# Written as CSV so preprocessing is inspectable without re-hitting the network.
download_task2() {
  local out_dir="$REPO_ROOT/task2_sentiment/data"
  mkdir -p "$out_dir"
  if [[ -s "$out_dir/yelp_polarity_train.csv" ]]; then
    log "Task 2: already present, skipping."
    return
  fi
  log "Task 2: downloading Yelp Polarity (fancyzhx/yelp_polarity)..."
  "${PYTHON}" - "$out_dir" <<'PY'
import sys, pathlib
from datasets import load_dataset

out_dir = pathlib.Path(sys.argv[1])
ds = load_dataset("fancyzhx/yelp_polarity")
for split, filename in (("train", "yelp_polarity_train.csv"),
                        ("test", "yelp_polarity_test.csv")):
    path = out_dir / filename
    ds[split].to_csv(path, index=False)
    print(f"  wrote {path} ({len(ds[split])} rows)")
PY
}

# --- Task 3: Monet / Photo (Kaggle) ----------------------------------------
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json.
# Credentials live OUTSIDE the repo by design — never commit kaggle.json.
download_task3() {
  local out_dir="$REPO_ROOT/task3_gan/data"
  mkdir -p "$out_dir"
  if [[ -n "$(ls -A "$out_dir/monet_jpg" 2>/dev/null | grep -v '^.gitkeep$' || true)" ]]; then
    log "Task 3: already present, skipping."
    return
  fi
  if ! command -v kaggle >/dev/null 2>&1; then
    log "ERROR: kaggle CLI not found. Install it with: pip install kaggle"
    exit 1
  fi
  if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    log "ERROR: missing ~/.kaggle/kaggle.json"
    log "  Kaggle > Account > Create New API Token, then:"
    log "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
    exit 1
  fi
  log "Task 3: downloading gan-getting-started competition data..."
  kaggle competitions download -c gan-getting-started -p "$out_dir"
  unzip -q -o "$out_dir/gan-getting-started.zip" -d "$out_dir"
  rm -f "$out_dir/gan-getting-started.zip"
  log "  monet_jpg: $(ls "$out_dir/monet_jpg" | wc -l | tr -d ' ') images"
  log "  photo_jpg: $(ls "$out_dir/photo_jpg" | wc -l | tr -d ' ') images"
}

case "$TARGET" in
  all)   download_task1; download_task2; download_task3 ;;
  task1) download_task1 ;;
  task2) download_task2 ;;
  task3) download_task3 ;;
  *)     log "Unknown target '$TARGET'. Use: all | task1 | task2 | task3"; exit 1 ;;
esac

log "Done."
