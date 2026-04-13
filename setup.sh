#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — one-shot installation & model download for detect-track
#
# This script:
#   1. Creates a Python virtual environment (optional; skip with --no-venv)
#   2. Installs PyTorch with CUDA support
#   3. Installs the SAM2 package from source (Facebook Research GitHub)
#   4. Installs this package and all other Python dependencies
#   5. Downloads OWLv2 and SAM2 model weights from Hugging Face Hub
#
# After this script completes the system can operate FULLY OFFLINE.
# The runtime sets TRANSFORMERS_OFFLINE=1 to prevent any network calls.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh                        # default settings
#   ./setup.sh --no-venv              # skip venv creation (use current env)
#   ./setup.sh --skip-models          # skip model download (dev mode)
#   ./setup.sh --hf-token <TOKEN>     # HF token for gated models
#   ./setup.sh --owlv2-dir /fast/ssd/owlv2 --sam2-dir /fast/ssd/sam2
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CREATE_VENV=true
VENV_DIR=".venv"
SKIP_MODELS=false
HF_TOKEN=""
OWLV2_DIR="./models/owlv2"
SAM2_DIR="./models/sam2"
OWLV2_REPO="google/owlv2-base-patch16-ensemble"
SAM2_REPO="facebook/sam2-hiera-large"
CUDA_VERSION="cu121"          # Change to cu118 / cu124 / cpu as needed
TORCH_VERSION="2.5.1"

# ── Parse arguments ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv)       CREATE_VENV=false ;;
    --skip-models)   SKIP_MODELS=true ;;
    --hf-token)      HF_TOKEN="$2"; shift ;;
    --owlv2-dir)     OWLV2_DIR="$2"; shift ;;
    --sam2-dir)      SAM2_DIR="$2"; shift ;;
    --owlv2-repo)    OWLV2_REPO="$2"; shift ;;
    --sam2-repo)     SAM2_REPO="$2"; shift ;;
    --cuda)          CUDA_VERSION="$2"; shift ;;
    --torch-version) TORCH_VERSION="$2"; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
  shift
done

log() { echo "[setup] $*"; }
die() { echo "[setup] ERROR: $*" >&2; exit 1; }

# ── 1. Virtual environment ─────────────────────────────────────────────────────
if $CREATE_VENV; then
  log "Creating virtual environment in ${VENV_DIR} …"
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
  log "Virtual environment activated."
else
  log "Skipping venv creation (using current Python: $(which python3))."
fi

PYTHON="$(which python3)"
PIP="${PYTHON} -m pip"

$PIP install --upgrade pip wheel setuptools -q

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
log "Installing PyTorch ${TORCH_VERSION}+${CUDA_VERSION} …"
if [[ "${CUDA_VERSION}" == "cpu" ]]; then
  $PIP install "torch==${TORCH_VERSION}" torchvision --index-url https://download.pytorch.org/whl/cpu
else
  $PIP install "torch==${TORCH_VERSION}" torchvision \
    --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}"
fi

# ── 3. SAM2 from source ────────────────────────────────────────────────────────
log "Installing SAM2 from GitHub …"
$PIP install "git+https://github.com/facebookresearch/sam2.git" -q

# ── 4. This package + remaining deps ──────────────────────────────────────────
log "Installing detect-track and dependencies …"
$PIP install -e ".[dev]" -q

# ── 5. Model downloads ────────────────────────────────────────────────────────
if $SKIP_MODELS; then
  log "Skipping model download (--skip-models)."
else
  log "Downloading OWLv2 (${OWLV2_REPO}) → ${OWLV2_DIR} …"
  TOKEN_ARG=""
  [[ -n "${HF_TOKEN}" ]] && TOKEN_ARG="--token ${HF_TOKEN}"

  # Run the model_loader CLI helper (part of this package).
  # shellcheck disable=SC2086
  $PYTHON -m detect_track.utils.model_loader \
    --owlv2-dir "${OWLV2_DIR}" \
    --sam2-dir  "${SAM2_DIR}"  \
    --owlv2-repo "${OWLV2_REPO}" \
    --sam2-repo  "${SAM2_REPO}"  \
    ${TOKEN_ARG}

  OWLV2_ABS="$(realpath "${OWLV2_DIR}")"
  SAM2_ABS="$(realpath "${SAM2_DIR}")"

  log "Updating configs/default.yaml with local model paths …"
  # Use Python to safely edit the YAML (avoids sed quoting nightmares).
  $PYTHON - <<PYEOF
import yaml, pathlib
cfg_path = pathlib.Path("configs/default.yaml")
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["models"]["owlv2_path"] = "${OWLV2_ABS}"
cfg["models"]["sam2_checkpoint"] = "${SAM2_ABS}/sam2_hiera_large.pt"
with open(cfg_path, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print(f"  owlv2_path      = ${OWLV2_ABS}")
print(f"  sam2_checkpoint = ${SAM2_ABS}/sam2_hiera_large.pt")
PYEOF
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "Setup complete!"
echo ""
echo "  Run the pipeline:"
echo "    detect-track run --queries 'person' 'car'"
echo ""
echo "  Or with a video file:"
echo "    detect-track run --source /path/to/video.mp4"
echo ""
echo "  The system is now fully offline-capable."
echo "  TRANSFORMERS_OFFLINE=1 is enforced at runtime."
