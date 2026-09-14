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
TARGET="${1:-all}"

log() { printf '[download_data] %s\n' "$*"; }

# --- Task 1: TinyStories ----------------------------------------------------
# Pulled through the HuggingFace datasets library and written out as plain text
# so the character-level tokenizer in Task 1 reads one simple file.
download_task1() {
  local out_dir="$REPO_ROOT/task1_llm/data"
  mkdir -p "$out_dir"
  if [[ -s "$out_dir/TinyStories-train.txt" ]]; then
    log "Task 1: already present, skipping."
    return
  fi
  log "Task 1: downloading TinyStories (roneneldan/TinyStories)..."
  python3 - "$out_dir" <<'PY'
import sys, pathlib
from datasets import load_dataset

out_dir = pathlib.Path(sys.argv[1])
ds = load_dataset("roneneldan/TinyStories")
for split, filename in (("train", "TinyStories-train.txt"),
                        ("validation", "TinyStories-valid.txt")):
    if split not in ds:
        continue
    path = out_dir / filename
    with path.open("w", encoding="utf-8") as fh:
        for row in ds[split]:
            fh.write(row["text"].strip() + "\n<|endofstory|>\n")
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
PY
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
  python3 - "$out_dir" <<'PY'
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
