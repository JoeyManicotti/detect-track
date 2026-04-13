# ─────────────────────────────────────────────────────────────────────────────
# detect-track — multi-stage Docker image
#
# Stage 1 (builder): downloads all model weights while network is available.
# Stage 2 (runtime): copies weights + code; runs fully offline.
#
# Build:
#   docker build -t detect-track:latest .
#
# Run (two GPUs):
#   docker run --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
#              --shm-size=4g detect-track:latest \
#              detect-track run --queries "person" "car"
#
# Run (single GPU, both models on cuda:0):
#   docker run --gpus all --shm-size=4g detect-track:latest \
#              detect-track run \
#              --detector-device cuda:0 --tracker-device cuda:0
#
# Notes:
#   --shm-size=4g  gives /dev/shm enough space for the SAM2 frame ring buffer
#                  (default Docker shm is only 64 MB).
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: model download ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /workspace

# Minimal tools needed to run the download helper.
RUN pip install --no-cache-dir huggingface_hub>=0.23.0 PyYAML>=6.0.1

COPY src/ src/
COPY pyproject.toml .
COPY configs/ configs/

# Install the detect_track package (for the model_loader CLI).
RUN pip install --no-cache-dir -e . --no-deps

# Download models (network available at build time; not at runtime).
ARG OWLV2_REPO=google/owlv2-base-patch16-ensemble
ARG SAM2_REPO=facebook/sam2-hiera-large
ARG HF_TOKEN=""

RUN python -m detect_track.utils.model_loader \
      --owlv2-dir  /models/owlv2 \
      --sam2-dir   /models/sam2  \
      --owlv2-repo "${OWLV2_REPO}" \
      --sam2-repo  "${SAM2_REPO}"  \
      ${HF_TOKEN:+--token "${HF_TOKEN}"}

# ── Stage 2: runtime image ─────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Enforce offline operation — no HF network calls at runtime.
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV HF_HUB_OFFLINE=1
# Silence tokenizer parallelism warning.
ENV TOKENIZERS_PARALLELISM=false

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3-pip \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1

WORKDIR /app

# ── Copy project files ─────────────────────────────────────────────────────────
COPY requirements.txt pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

# ── Install Python dependencies ────────────────────────────────────────────────
# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
      --index-url https://download.pytorch.org/whl/cu121

# SAM2 from source
RUN pip install --no-cache-dir \
      "git+https://github.com/facebookresearch/sam2.git"

# Remaining dependencies + this package
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# ── Copy pre-downloaded model weights from builder ─────────────────────────────
COPY --from=builder /models /models

# Point config at the baked-in model paths.
RUN python3 - <<'PYEOF'
import yaml, pathlib
cfg = pathlib.Path("configs/default.yaml")
with open(cfg) as f:
    d = yaml.safe_load(f)
d["models"]["owlv2_path"] = "/models/owlv2"
d["models"]["sam2_checkpoint"] = "/models/sam2/sam2_hiera_large.pt"
with open(cfg, "w") as f:
    yaml.dump(d, f, default_flow_style=False)
PYEOF

# ── Default entrypoint ─────────────────────────────────────────────────────────
ENTRYPOINT ["detect-track"]
CMD ["run", "--help"]
