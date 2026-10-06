#!/usr/bin/env bash
#
# One command smoke test for this repository, required by brief section 5.
#
# Reproduces a shortened version of anees_saheba's Task 1 run from a fresh
# clone: it fetches a small slice of TinyStories, trains the character level
# GPT for one epoch on a reduced split, and checks that the run produced the
# things it is supposed to produce.
#
# Designed to finish in a few minutes on CPU, so a grader does not need a GPU.
#
# Usage:
#   bash scripts/smoke_test.sh
#
# Exit code 0 means the pipeline works end to end. Anything else is a failure
# and the reason is printed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The work directory is deliberately OUTSIDE the repository. The notebook
# locates its output folder by walking up from the working directory looking
# for a .git directory, so running it anywhere inside the repo would write into
# task1_llm/anees_saheba and overwrite the real results. From outside, that
# search finds nothing and the notebook falls back to a local folder.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/data266_smoke_XXXXXX")"
NOTEBOOK="${REPO_ROOT}/task1_llm/anees_saheba/src/char_gpt_tinystories.ipynb"

# Reduced settings. Small enough to run on CPU, large enough that every code
# path still executes: preprocessing, the model, training, generation and the
# full metrics table.
SMOKE_TRAIN_SEQUENCES=2000
SMOKE_VALIDATION_SEQUENCES=400
SMOKE_WARMUP_STEPS=10
SMOKE_EPOCHS=1
SMOKE_STORIES=3000

log()  { printf '[smoke] %s\n' "$*"; }
fail() { printf '[smoke] FAILED: %s\n' "$*" >&2; exit 1; }

log "repository root: ${REPO_ROOT}"

# ---------------------------------------------------------------------------
# 1. Python and dependencies
# ---------------------------------------------------------------------------
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
  log "using the virtual environment at .venv"
else
  PYTHON="$(command -v python3 || true)"
  [[ -n "${PYTHON}" ]] || fail "python3 not found. Create the environment first: python3 -m venv .venv && source .venv/bin/activate && pip install -r environment/requirements.txt"
  log "using system python3. For an exact match run: python3 -m venv .venv && source .venv/bin/activate && pip install -r environment/requirements.txt"
fi

"${PYTHON}" - <<'PY' || exit 1
import importlib.util, sys
missing = [m for m in ("torch", "numpy", "matplotlib", "nbformat", "nbconvert", "ipykernel", "datasets")
           if importlib.util.find_spec(m) is None]
if missing:
    print(f"[smoke] FAILED: missing packages: {', '.join(missing)}")
    print("[smoke] install them with: pip install -r environment/requirements.txt nbconvert ipykernel")
    sys.exit(1)
import torch
print(f"[smoke] python {sys.version.split()[0]}, torch {torch.__version__}")
PY

[[ -f "${NOTEBOOK}" ]] || fail "notebook not found at ${NOTEBOOK}"

# ---------------------------------------------------------------------------
# 2. Data. Only a small slice is needed for the smoke test.
# ---------------------------------------------------------------------------
DATA_FILE="${REPO_ROOT}/task1_llm/data/TinyStories-sample.txt"
if [[ ! -s "${DATA_FILE}" ]]; then
  log "TinyStories not present, downloading a ${SMOKE_STORIES} story slice"
  PYTHON="${PYTHON}" TINYSTORIES_N="${SMOKE_STORIES}" bash "${REPO_ROOT}/scripts/download_data.sh" task1 \
    || fail "dataset download failed. Check network access, or place TinyStories-sample.txt in task1_llm/data/"
else
  log "TinyStories already present: $(wc -c < "${DATA_FILE}" | tr -d ' ') bytes"
fi

# ---------------------------------------------------------------------------
# 3. Build a shortened copy of the notebook.
#
# The committed notebook is never modified. A copy is made with the sequence
# counts and warmup reduced, so the smoke test cannot overwrite the real
# results or leave the committed notebook out of step with its own outputs.
# ---------------------------------------------------------------------------
log "work directory: ${WORK_DIR}"
log "preparing a reduced copy of the notebook"

"${PYTHON}" - "${NOTEBOOK}" "${WORK_DIR}/smoke.ipynb" \
  "${SMOKE_TRAIN_SEQUENCES}" "${SMOKE_VALIDATION_SEQUENCES}" "${SMOKE_WARMUP_STEPS}" <<'PY'
import json, sys, pathlib

source, target, train_n, val_n, warmup = sys.argv[1:6]
nb = json.loads(pathlib.Path(source).read_text())

replacements = {
    "TRAIN_SEQUENCES = 100_000": f"TRAIN_SEQUENCES = {train_n}",
    "VALIDATION_SEQUENCES = 10_000": f"VALIDATION_SEQUENCES = {val_n}",
    "WARMUP_STEPS = 1000": f"WARMUP_STEPS = {warmup}",
}
applied = {k: 0 for k in replacements}
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    text = "".join(cell["source"])
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
            applied[old] += 1
    # Outputs are cleared so the copy cannot be mistaken for a real run.
    cell["outputs"] = []
    cell["execution_count"] = None
    cell["source"] = text.splitlines(keepends=True)

for old, count in applied.items():
    if count == 0:
        print(f"[smoke] FAILED: could not find '{old}' in the notebook")
        sys.exit(1)

pathlib.Path(target).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"[smoke] reduced notebook written, {train_n} train and {val_n} validation sequences")
PY

# ---------------------------------------------------------------------------
# 4. Execute it from the work directory outside the repository, so the
#    notebook writes its outputs there instead of into the member folder.
# ---------------------------------------------------------------------------
log "running the notebook, this takes a few minutes on CPU"
cd "${WORK_DIR}"
DATA266_EPOCHS="${SMOKE_EPOCHS}" \
DATA266_TINYSTORIES="${DATA_FILE}" \
"${PYTHON}" -m nbconvert --to notebook --execute smoke.ipynb \
  --output executed.ipynb \
  --ExecutePreprocessor.timeout=3600 \
  --allow-errors \
  > nbconvert.log 2>&1 || true
cd "${REPO_ROOT}"

[[ -f "${WORK_DIR}/executed.ipynb" ]] \
  || { tail -20 "${WORK_DIR}/nbconvert.log" >&2; fail "the notebook did not produce an executed copy"; }

# ---------------------------------------------------------------------------
# 5. Check what the run actually produced.
# ---------------------------------------------------------------------------
log "checking results"
"${PYTHON}" - "${WORK_DIR}/executed.ipynb" "${WORK_DIR}" <<'PY' || exit 1
import json, sys, pathlib, re

executed, work_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
nb = json.loads(executed.read_text())
cells = nb["cells"]

failures = []

errors = [(i, o["ename"], o["evalue"])
          for i, c in enumerate(cells)
          for o in c.get("outputs", []) if o.get("output_type") == "error"]
if errors:
    for i, name, value in errors:
        print(f"[smoke]   cell {i} raised {name}: {value[:200]}")
    failures.append(f"{len(errors)} cell(s) raised an exception")

code_cells = [c for c in cells if c["cell_type"] == "code"]
executed_cells = [c for c in code_cells if c.get("execution_count")]
if len(executed_cells) != len(code_cells):
    failures.append(f"only {len(executed_cells)} of {len(code_cells)} code cells ran")

text = "".join("".join(o.get("text", []) or "")
               for c in cells for o in c.get("outputs", []))

# The causal mask is the one thing that would invalidate the whole task if it
# silently broke, so the smoke test checks it explicitly.
if "Causal masking verified" not in text:
    failures.append("causal masking check did not pass")

for needle, label in (("TRAINING COMPLETE", "training did not finish"),
                      ("parameter_count=", "parameter count was not recorded")):
    if needle not in text:
        failures.append(label)

metrics = list(work_dir.rglob("metrics_report.csv"))
if not metrics:
    failures.append("metrics_report.csv was not written")

curves = list(work_dir.rglob("loss_curves.png"))
if not curves:
    failures.append("loss_curves.png was not written")

samples = list(work_dir.rglob("generated_samples.txt"))
if not samples:
    failures.append("generated_samples.txt was not written")

if failures:
    print("[smoke] FAILED:")
    for f in failures:
        print(f"[smoke]   - {f}")
    sys.exit(1)

loss = re.search(r"EPOCH 1/\d+ train_loss=([\d.]+) val_loss=([\d.]+)", text)
print(f"[smoke] all {len(code_cells)} code cells ran with no exceptions")
print("[smoke] causal masking verified, no future information reaches any position")
if loss:
    print(f"[smoke] epoch 1 train loss {loss.group(1)}, validation loss {loss.group(2)}")
print(f"[smoke] metrics, loss curves and generated samples all written")
PY

log "PASSED. The Task 1 pipeline runs end to end from a clean checkout."
log "Artifacts from this test are in ${WORK_DIR} and can be deleted."
log "Nothing in the repository was modified."
