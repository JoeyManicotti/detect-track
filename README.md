# detect-track

Async **OWLv2 + SAM2** detection-and-tracking pipeline targeting **15–30 Hz** throughput.

OWLv2 runs zero-shot, open-vocabulary object detection on GPU 0 at 1–3 Hz.  
SAM2 propagates high-quality segmentation masks across every frame on GPU 1 at full frame rate.  
The two models run in completely separate processes, connected only by a zero-copy shared-memory bus and lightweight message queues.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Shared-Memory Ring Buffer                      │
│              (multiprocessing.shared_memory, 128 slots)          │
│              frames written once, read without copy              │
└───────────────┬──────────────────────────────────┬──────────────┘
                │ FrameMetadata (ptr only)          │ FrameMetadata
                ▼                                   ▼
  ┌─────────────────────────┐         ┌─────────────────────────────┐
  │      IngestNode          │         │       DetectorNode           │
  │      (CPU)               │         │       (GPU 0 — OWLv2)        │
  │  cv2.VideoCapture        │─────────►   1–3 Hz detection cadence  │
  │  BGR → RGB               │ det_q   │   bfloat16 precision         │
  │  ~30 Hz capture          │         │   greedy NMS                 │
  └─────────────┬────────────┘         │   priority redetect path     │
                │ tracker_q            └──────────┬──────────────────┘
                │ (every frame)                   │ DetectionResult
                ▼                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    TrackerNode (GPU 1 — SAM2)                   │
  │                                                                 │
  │  Temporal-batch streaming:                                      │
  │  • Accumulate BATCH_SIZE frames → write to /dev/shm as JPEG    │
  │  • init_state + propagate_in_video (one GPU pass per batch)    │
  │  • Transfer last mask per track → next batch anchor frame      │
  │  • Lost track → RedetectRequest → DetectorNode (feedback loop) │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │ TrackResult (per frame)
                                 ▼
                       ┌──────────────────┐
                       │  Consumer        │
                       │  visualise /     │
                       │  record / API    │
                       └──────────────────┘
```

### Throughput math

| Parameter | Value |
|---|---|
| Target fps | 30 |
| OWLv2 cadence | 2 Hz (every 15 frames) |
| SAM2 batch size | 15 frames (0.5 s of video) |
| SAM2 time per batch (A100) | ~200 ms |
| Effective tracker throughput | ~40 fps |
| End-to-end latency | ~500 ms (one batch window) |

Reduce `tracker_batch_size` in `configs/default.yaml` to lower latency at the cost of slightly higher overhead.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Linux (x86-64) | `/dev/shm` required for SAM2 frame staging |
| Python 3.10+ | |
| CUDA 12.1+ | Tested on CUDA 12.1 / 12.4 |
| 2× GPU (recommended) | One GPU works — set both devices to `cuda:0` |
| 8 GB+ VRAM per GPU | SAM2-Hiera-Large + OWLv2-base-patch16 |
| 4 GB+ `/dev/shm` | For SAM2 JPEG frame staging |

> **Docker users:** pass `--shm-size=4g` to give `/dev/shm` enough space.

---

## Installation

### Option A — `setup.sh` (recommended)

```bash
git clone https://github.com/joeymanicotti/detect-track.git
cd detect-track
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:
1. Create a `.venv` virtual environment
2. Install PyTorch with CUDA 12.1
3. Install SAM2 from source (`facebookresearch/sam2`)
4. Install this package and all dependencies
5. Download OWLv2 and SAM2 weights from Hugging Face Hub
6. Update `configs/default.yaml` with the local model paths

After the script completes the system operates **fully offline** — `TRANSFORMERS_OFFLINE=1` is enforced at runtime.

**Options:**

```bash
./setup.sh --no-venv                    # skip venv, use current env
./setup.sh --skip-models                # skip download (dev mode)
./setup.sh --cuda cu118                 # CUDA 11.8 instead of 12.1
./setup.sh --hf-token <HF_TOKEN>        # for gated HF repos
./setup.sh --owlv2-dir /fast/ssd/owlv2  # custom model storage path
```

### Option B — Docker (fully offline after build)

```bash
docker build -t detect-track:latest .
```

Model weights are downloaded **at image build time** and baked in. The container never needs network access at runtime.

```bash
# Two-GPU system
docker run --gpus all --shm-size=4g detect-track:latest \
  detect-track run --queries "person" "car"

# Single-GPU system
docker run --gpus all --shm-size=4g detect-track:latest \
  detect-track run \
  --detector-device cuda:0 \
  --tracker-device cuda:0

# With preview window (requires X11 forwarding)
docker run --gpus all --shm-size=4g \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  detect-track:latest \
  detect-track run --queries "person"
```

### Option C — Manual install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/sam2.git
pip install -e ".[dev]"
detect-track download --owlv2-dir ./models/owlv2 --sam2-dir ./models/sam2
```

---

## Configuration

All settings live in [`configs/default.yaml`](configs/default.yaml). Key sections:

```yaml
models:
  owlv2_path: "./models/owlv2"             # set by setup.sh
  sam2_checkpoint: "./models/sam2/sam2_hiera_large.pt"

devices:
  detector: "cuda:0"   # OWLv2
  tracker:  "cuda:1"   # SAM2 — set to cuda:0 for single-GPU

pipeline:
  target_fps: 30
  detector_fps: 2       # OWLv2 inference rate (Hz)
  tracker_batch_size: 15

detection:
  text_queries:
    - "person"
    - "car"
    - "bicycle"
  score_threshold: 0.15

precision:
  detector_dtype: "bfloat16"
  tracker_dtype:  "bfloat16"

compile:
  enabled: false        # set true on CUDA ≥ 8.0 for max throughput
  mode: "max-autotune"
```

---

## Usage

### Run on webcam

```bash
detect-track run
```

### Run with custom queries

```bash
detect-track run --queries "person" "red backpack" "bicycle"
```

### Run on a video file

```bash
detect-track run --source /path/to/video.mp4
```

### Save annotated output

```bash
detect-track run --source video.mp4 --write-video annotated.mp4 --no-window
```

### Custom device assignment

```bash
# Single GPU
detect-track run --detector-device cuda:0 --tracker-device cuda:0

# Specific GPUs
detect-track run --detector-device cuda:0 --tracker-device cuda:1
```

### Use a custom config file

```bash
detect-track run --config /path/to/my_config.yaml
```

### All CLI options

```
detect-track run --help

  --config,  -c PATH          YAML config file (default: configs/default.yaml)
  --source,  -s SOURCE        Camera index (int) or file/RTSP URL
  --queries, -q TEXT          Text queries (repeatable: -q person -q car)
  --threshold, -t FLOAT       Detection confidence threshold
  --detector-fps FLOAT        OWLv2 inference rate in Hz
  --target-fps FLOAT          Pipeline output target in Hz
  --detector-device TEXT      PyTorch device for OWLv2 (e.g. cuda:0)
  --tracker-device TEXT       PyTorch device for SAM2 (e.g. cuda:1)
  --no-window                 Disable OpenCV preview window
  --write-video PATH          Write annotated video to PATH
  --verbose, -v               Enable debug logging
```

### Download models separately

```bash
detect-track download \
  --owlv2-dir ./models/owlv2 \
  --sam2-dir  ./models/sam2
```

---

## Python API

```python
import yaml
from detect_track.pipeline import Pipeline

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

# Override queries programmatically
config["detection"]["text_queries"] = ["person", "dog"]

with Pipeline(config) as pipeline:
    for result in pipeline.results():
        print(f"Frame {result.frame_id}: {len(result.tracks)} tracks")
        for track in result.tracks:
            print(f"  [{track.track_id}] {track.label}  score={track.score:.2f}  box={track.box}")
```

---

## How it works

### Zero-copy frame bus

The `IngestNode` writes each captured frame into a [`multiprocessing.shared_memory`](https://docs.python.org/3/library/multiprocessing.shared_memory.html) ring buffer with 128 named slots (`dt_frame_<slot>`).  Downstream nodes receive only a lightweight `FrameMetadata` struct (< 200 bytes) through the OS-managed queues; they map the shared block directly to read pixels without any inter-process copy.

### OWLv2 detection cadence

`DetectorNode` pulls one frame every `N` tracker frames (`N = target_fps / detector_fps`, default 15).  When the tracker reports a lost track it sends a `RedetectRequest` which causes the detector to run on the *next* available frame regardless of the cadence — this is the re-detection feedback loop.

### SAM2 batch-streaming

SAM2's `SAM2VideoPredictor` needs all frames available at `init_state` time.  The tracker works around this with a **temporal-batch** approach:

1. Buffer `BATCH_SIZE` frames (default 15) into an in-memory deque.
2. Write them as JPEG files to `/dev/shm` (Linux RAM-backed filesystem, ~2 ms per 1080p frame).
3. Call `predictor.init_state(video_path)` + `predictor.propagate_in_video()` — one GPU forward pass covering all frames in the batch.
4. Keep the last `OVERLAP_FRAMES` (default 5) frames as the anchor for the **next** batch.
5. At the anchor frame, call `predictor.add_new_mask()` for each active track to transfer its last known mask, ensuring continuity across batch boundaries.

### Lost-track detection

After each batch, any track whose mask confidence has been below `tracking.lost_mask_score_threshold` (default 0.30) for `tracking.max_frames_lost` consecutive frames is declared lost.  A `RedetectRequest` is sent back to `DetectorNode`, which re-runs OWLv2 and pushes a fresh `DetectionResult` to re-initialise the tracklet in the next SAM2 batch.

---

## Development

```bash
# Install in editable mode with test dependencies
pip install -e ".[dev]"

# Run tests (no GPU or model weights required)
pytest tests/ -v

# Run a specific test file
pytest tests/test_ipc.py -v
pytest tests/test_pipeline.py -v
```

Test coverage includes:

| Test | What it validates |
|---|---|
| `TestFrameSharedBuffer` | Ring-buffer write/read, slot reuse, cross-process reads |
| `TestNMS` | Greedy NMS correctness (single box, overlap, empty) |
| `TestMaskToXYXY` | Mask → bounding box conversion |
| `TestVisualization` | Drawing functions don't crash; colour stability |
| `TestIngestNode` | Process starts, reads frames, stops cleanly |
| `TestSAM2StreamingTracker` | Batch triggering, detection integration (stub predictor) |
| `TestModelLoader` | `verify_models` raises on missing / passes when present |

---

## OpenCV note

This project uses **`opencv-python-headless`** — the version without Qt/GTK GUI bindings.  This makes it suitable for Docker containers and headless servers.

The `PreviewWindow` class in `utils/visualization.py` calls `cv2.imshow`, which works when a display is available:
- **Local desktop:** works out of the box.
- **Docker with X11 forwarding:** pass `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`.
- **Headless server / CI:** pass `--no-window` to disable the preview entirely.

---

## Troubleshooting

**`FileNotFoundError: Required model weights not found`**  
Run `./setup.sh` or `detect-track download` to fetch the weights, then verify `configs/default.yaml` points to the correct paths.

**`RuntimeError: SAM2 package not found`**  
Install SAM2 from source:  
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

**Out-of-memory on a single GPU**  
Set both devices to `cuda:0` and reduce batch size and/or switch to a smaller model variant. You can also enable `precision.tracker_dtype: float16` instead of `bfloat16`.

**`/dev/shm` full**  
The tracker writes up to `tracker_batch_size` JPEGs at a time.  For 1080p at batch 15, that is ~3 MB. If `/dev/shm` is very small (e.g. default Docker 64 MB), increase it with `--shm-size=4g`.

**Low frame rate**  
- Enable `compile.enabled: true` in the config (requires CUDA ≥ 8.0, adds ~2 min warm-up).
- Reduce input resolution via `cv2.resize` in the ingest node.
- Reduce `tracker_batch_size` — smaller batches have less stall time.
- Use `sam2_hiera_small` or `sam2_hiera_base_plus` instead of `large`.

**Preview window freezes**  
The preview window runs in the main process and blocks on `cv2.waitKey(1)`.  If the output queue is not being consumed fast enough, use `--no-window` and process `TrackResult` objects from the Python API instead.
