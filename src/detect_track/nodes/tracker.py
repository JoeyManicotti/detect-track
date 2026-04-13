"""
Tracker Node — SAM2 video segmentation & tracking on GPU 1.

Architecture: Temporal-batch streaming
───────────────────────────────────────
SAM2's ``SAM2VideoPredictor`` requires all frames to be available at
``init_state`` time; it cannot ingest frames one at a time.  We work around
this with a **temporal-batch** approach:

1. Accumulate ``BATCH_SIZE`` frames in an internal deque.
2. When the deque is full (or when a new OWLv2 detection arrives, whichever
   comes first), write the batch to ``/dev/shm`` as JPEG files and run SAM2.
3. SAM2 propagates segmentation masks through all frames in the batch in a
   single forward pass, yielding per-frame ``TrackResult`` objects.
4. Push all results to the output queue.
5. Keep the last ``OVERLAP_FRAMES`` frames in the deque as the anchor for the
   next batch, and transfer the final mask from each active track to the new
   batch's anchor frame via ``add_new_mask``.  This ensures continuity across
   batch boundaries.

Throughput math (example):
  • BATCH_SIZE = 15 frames, target 30 fps
  • 15 frames / 30 fps = 500 ms of video per batch
  • SAM2-Hiera-Large: ~200 ms/batch on A100 → 40 fps effective throughput
  • End-to-end latency ≈ 500 ms (one batch duration)

Lost-track feedback
────────────────────
After each batch, tracks whose ``frames_lost`` counter exceeds
``max_frames_lost`` are flagged.  A ``RedetectRequest`` is sent to the
Detector node so OWLv2 can re-scan for the lost object.

/dev/shm frame store
─────────────────────
On Linux, ``/dev/shm`` is a RAM-backed tmpfs.  Writing a 1080p JPEG there
takes ~2 ms, well within the 500 ms batch budget.  We clean up old JPEG
files between batches to keep memory consumption bounded.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import shutil
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from detect_track.ipc.frame_buffer import FrameSharedBuffer, attach_reader
from detect_track.ipc.messages import (
    DetectionResult,
    FrameMetadata,
    RedetectRequest,
    StopSignal,
    TrackInfo,
    TrackResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-track bookkeeping
# ---------------------------------------------------------------------------

class _TrackState:
    """Mutable state maintained per active tracklet."""

    __slots__ = (
        "obj_id", "label", "detection_score",
        "last_mask", "last_box", "frames_lost", "active",
    )

    def __init__(self, obj_id: int, label: str, detection_score: float) -> None:
        self.obj_id = obj_id
        self.label = label
        self.detection_score = detection_score
        self.last_mask: Optional[np.ndarray] = None
        self.last_box: Optional[List[float]] = None
        self.frames_lost: int = 0
        self.active: bool = True


# ---------------------------------------------------------------------------
# SAM2 streaming wrapper
# ---------------------------------------------------------------------------

class _SAM2StreamingTracker:
    """
    Stateful wrapper around ``SAM2VideoPredictor`` for online streaming.

    Not thread-safe; must be used within a single process.
    """

    def __init__(self, predictor: Any, device: torch.device, config: dict) -> None:
        self.predictor = predictor
        self.device = device
        self.config = config

        self.batch_size: int = int(config["pipeline"]["tracker_batch_size"])
        self.overlap: int = int(config["pipeline"]["tracker_overlap_frames"])
        self.score_threshold: float = float(
            config["tracking"]["lost_mask_score_threshold"]
        )
        self.max_frames_lost: int = int(config["tracking"]["max_frames_lost"])
        self.max_active_tracks: int = int(config["tracking"]["max_active_tracks"])

        # Frame ring buffer: (abs_frame_id, rgb_ndarray)
        self._frame_deque: deque = deque()
        # Results produced by the last SAM2 batch but not yet popped.
        self._result_queue: deque = deque()

        # Active tracklet state indexed by SAM2 obj_id.
        self._tracks: Dict[int, _TrackState] = {}
        self._next_obj_id: int = 1

        # Detection waiting to be applied at the next batch anchor.
        self._pending_detection: Optional[DetectionResult] = None

        # Create a temporary directory for JPEG frame storage.
        # Prefer /dev/shm (RAM-backed on Linux) for speed, fall back to
        # system temp dir if /dev/shm is not writable.
        self._frame_dir = self._create_frame_dir()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def feed_detection(self, detection: DetectionResult) -> None:
        """Queue a new OWLv2 detection to be applied at the next batch."""
        self._pending_detection = detection

    def ingest_frame(
        self, frame: np.ndarray, frame_id: int
    ) -> Tuple[Optional[TrackResult], List[RedetectRequest]]:
        """
        Accept one RGB frame.

        Parameters
        ----------
        frame:
            RGB uint8 ndarray.
        frame_id:
            The *actual* shared-memory frame identifier from IngestNode.
            Used as the key for ``pipeline.read_frame()`` so the consumer
            can retrieve the correct slot.

        Returns
        -------
        result:
            ``TrackResult`` if one is available from the last processed batch,
            or *None* if the batch is still filling.
        redetect_requests:
            List of ``RedetectRequest`` objects for tracks that were lost
            in the most recently processed batch.
        """
        self._frame_deque.append((frame_id, frame.copy()))

        redetect_requests: List[RedetectRequest] = []

        # Trigger batch when full, or immediately when a new detection arrives
        # (so prompts are applied as soon as possible).
        should_run = (
            len(self._frame_deque) >= self.batch_size
            or (
                self._pending_detection is not None
                and len(self._frame_deque) >= max(1, self.overlap + 1)
            )
        )

        if should_run:
            redetect_requests = self._run_batch()

        # Return one buffered result if available.
        result = self._result_queue.popleft() if self._result_queue else None
        return result, redetect_requests

    def flush_results(self) -> List[TrackResult]:
        """Return all buffered results (call on shutdown)."""
        results = list(self._result_queue)
        self._result_queue.clear()
        return results

    def cleanup(self) -> None:
        shutil.rmtree(str(self._frame_dir), ignore_errors=True)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _run_batch(self) -> List[RedetectRequest]:
        """
        Run SAM2 on the current frame deque.

        1. Write frames as JPEG to temp directory.
        2. ``init_state`` → load all frames.
        3. Transfer masks from previous batch (anchor frame = frame 0).
        4. Apply pending detection prompts (anchor frame = frame 0).
        5. ``propagate_in_video`` → iterate results.
        6. Buffer TrackResult objects.
        7. Trim deque to overlap window.
        8. Return redetect requests for lost tracks.
        """
        frames = list(self._frame_deque)
        if not frames:
            return []

        # If there are no existing tracks and no pending detection, skip
        # the expensive SAM2 call — just emit empty results.
        if not self._tracks and self._pending_detection is None:
            for abs_frame_id, _ in frames:
                self._result_queue.append(
                    TrackResult(frame_id=abs_frame_id, tracks=[])
                )
            self._trim_deque()
            return []

        t0 = time.monotonic()

        # ── 1. Write frames as JPEG ───────────────────────────────────
        self._clear_frame_dir()
        written = 0
        for local_idx, (_, rgb) in enumerate(frames):
            if self._write_jpeg(local_idx, rgb):
                written += 1

        if written == 0:
            logger.error(
                "Failed to write any JPEG frames to %s — skipping batch. "
                "Check disk space and permissions.", self._frame_dir,
            )
            for abs_frame_id, _ in frames:
                self._result_queue.append(
                    TrackResult(frame_id=abs_frame_id, tracks=[])
                )
            self._trim_deque()
            return []

        # Verify at least one file is visible to the filesystem.
        frame_dir = self._frame_dir
        on_disk = [f for f in os.listdir(frame_dir)
                    if os.path.splitext(f)[-1].lower() in (".jpg", ".jpeg")]
        if not on_disk:
            logger.error(
                "Wrote %d JPEGs but os.listdir(%s) sees 0 files. "
                "Filesystem issue — skipping batch.", written, frame_dir,
            )
            for abs_frame_id, _ in frames:
                self._result_queue.append(
                    TrackResult(frame_id=abs_frame_id, tracks=[])
                )
            self._trim_deque()
            return []

        logger.debug(
            "Wrote %d/%d JPEGs to %s", written, len(frames), frame_dir,
        )

        # ── 2. Initialise SAM2 state ───────────────────────────────────
        inference_state = self.predictor.init_state(
            video_path=str(frame_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        self.predictor.reset_state(inference_state)

        anchor_idx = 0  # Anchor frame for new prompts within this batch.

        # ── 3. Transfer masks from previous batch ──────────────────────
        for obj_id, ts in list(self._tracks.items()):
            if ts.last_mask is not None and ts.active:
                try:
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=anchor_idx,
                        obj_id=obj_id,
                        mask=ts.last_mask.astype(bool),
                    )
                except Exception as exc:
                    logger.debug(
                        "Mask transfer failed for track %d: %s", obj_id, exc
                    )

        # ── 4. Apply pending detections ────────────────────────────────
        if self._pending_detection is not None and len(self._tracks) < self.max_active_tracks:
            det = self._pending_detection
            self._pending_detection = None
            for box, label, score in zip(det.boxes, det.labels, det.scores):
                obj_id = self._next_obj_id
                self._next_obj_id += 1
                try:
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=anchor_idx,
                        obj_id=obj_id,
                        box=np.array(box, dtype=np.float32),
                        clear_old_points=True,
                    )
                    self._tracks[obj_id] = _TrackState(
                        obj_id=obj_id,
                        label=label,
                        detection_score=float(score),
                    )
                    logger.debug(
                        "Added track %d for '%s' (score=%.2f)", obj_id, label, score
                    )
                except Exception as exc:
                    logger.warning("Failed to add detection track: %s", exc)

        # ── 5. Propagate through all frames ───────────────────────────
        frame_results: Dict[int, List[TrackInfo]] = {
            local_idx: [] for local_idx in range(len(frames))
        }

        try:
            for out_frame_idx, out_obj_ids, out_mask_logits in (
                self.predictor.propagate_in_video(
                    inference_state, start_frame_idx=0
                )
            ):
                self._process_frame_output(
                    out_frame_idx, out_obj_ids, out_mask_logits,
                    frame_results,
                )
        except Exception as exc:
            logger.error("SAM2 propagation error: %s", exc)

        # ── 6. Buffer TrackResult objects ──────────────────────────────
        for local_idx, (abs_frame_id, _) in enumerate(frames):
            self._result_queue.append(
                TrackResult(
                    frame_id=abs_frame_id,
                    tracks=frame_results.get(local_idx, []),
                )
            )

        # ── 7. Trim deque to overlap ───────────────────────────────────
        self._trim_deque()

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "SAM2 batch: %d frames, %d tracks, %.0f ms",
            len(frames), len(self._tracks), elapsed_ms,
        )

        # ── 8. Emit redetect requests for lost tracks ──────────────────
        return self._collect_redetect_requests(frames[-1][0])

    def _trim_deque(self) -> None:
        """Keep only the last ``overlap`` frames in the deque."""
        overlap_frames = list(self._frame_deque)[-self.overlap :]
        self._frame_deque.clear()
        for item in overlap_frames:
            self._frame_deque.append(item)

    def _process_frame_output(
        self,
        local_idx: int,
        obj_ids: List[int],
        mask_logits: torch.Tensor,
        frame_results: Dict[int, List[TrackInfo]],
    ) -> None:
        """Parse SAM2 output for one frame and update track state."""
        # mask_logits: (N, 1, H, W)
        masks = (mask_logits > 0.0).squeeze(1).cpu().numpy()          # (N, H, W) bool
        scores = torch.sigmoid(mask_logits).squeeze(1).cpu().numpy()   # (N, H, W) float

        for i, obj_id in enumerate(obj_ids):
            obj_id = int(obj_id)
            mask = masks[i]
            mask_score = float(scores[i].max())
            box = _mask_to_xyxy(mask)

            # Update bookkeeping.
            ts = self._tracks.get(obj_id)
            if ts is not None:
                ts.last_mask = mask
                ts.last_box = box
                if mask_score < self.score_threshold:
                    ts.frames_lost += 1
                else:
                    ts.frames_lost = 0

            label = ts.label if ts else "unknown"
            frame_results[local_idx].append(
                TrackInfo(
                    track_id=obj_id,
                    label=label,
                    score=mask_score,
                    box=box,
                )
            )

    def _collect_redetect_requests(
        self, latest_abs_frame_id: int
    ) -> List[RedetectRequest]:
        requests: List[RedetectRequest] = []
        for obj_id, ts in list(self._tracks.items()):
            if ts.frames_lost >= self.max_frames_lost:
                ts.active = False
                requests.append(
                    RedetectRequest(
                        frame_id=latest_abs_frame_id,
                        track_id=obj_id,
                        label=ts.label,
                        reason="low_confidence"
                        if ts.frames_lost < self.max_frames_lost * 2
                        else "disappeared",
                    )
                )
                # Remove the stale track; it will be re-added by the Detector.
                del self._tracks[obj_id]
                logger.info(
                    "Track %d ('%s') declared lost after %d frames.",
                    obj_id, ts.label, ts.frames_lost,
                )
        return requests

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_frame_dir() -> Path:
        """Create a temporary directory for SAM2 JPEG frame storage.

        Uses the system temp directory (/tmp) rather than /dev/shm because
        /dev/shm is already occupied by the shared-memory ring buffer.
        /tmp is backed by disk but the small JPEG files (< 5 MB per batch)
        are negligible compared to GPU processing time.
        """
        d = tempfile.mkdtemp(prefix="sam2_batch_")
        logger.info("SAM2 frame directory: %s", d)
        return Path(d)

    def _clear_frame_dir(self) -> None:
        for p in self._frame_dir.glob("*.jpg"):
            p.unlink(missing_ok=True)

    def _write_jpeg(self, local_idx: int, rgb: np.ndarray) -> bool:
        """Write one frame as JPEG.  Returns True on success."""
        path = str(self._frame_dir / f"{local_idx:06d}.jpg")
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok = cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                return True
            # cv2 failed silently — fall back to PIL.
            logger.debug("cv2.imwrite failed for %s, trying PIL", path)
        except Exception as exc:
            logger.debug("cv2.imwrite raised for %s: %s, trying PIL", path, exc)

        try:
            from PIL import Image as PILImage
            PILImage.fromarray(rgb).save(path, quality=95)
            return True
        except Exception as exc:
            logger.error("Failed to write frame %d: %s", local_idx, exc)
            return False


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _mask_to_xyxy(mask: np.ndarray) -> Optional[List[float]]:
    """Convert a binary mask (H, W) to [x1, y1, x2, y2] in pixel space."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
    return [float(cmin), float(rmin), float(cmax), float(rmax)]


# ---------------------------------------------------------------------------
# Tracker Node (multiprocessing.Process)
# ---------------------------------------------------------------------------

class TrackerNode(mp.Process):
    """
    SAM2 video tracker running in a dedicated process on ``tracker_device``.

    Parameters
    ----------
    frame_buffer_config:
        Dict with ``capacity``, ``frame_shape``, ``dtype``.
    tracker_queue:
        Queue[FrameMetadata | StopSignal] fed by the Ingest node.
    detection_queue:
        Queue[DetectionResult] fed by the Detector node.
    redetect_queue:
        Queue[RedetectRequest] consumed by the Detector node.
    output_queue:
        Queue[TrackResult] consumed by the visualiser / application.
    config:
        Full pipeline config dict.
    stop_event:
        Shared ``multiprocessing.Event``.
    """

    def __init__(
        self,
        frame_buffer_config: dict,
        tracker_queue: Queue,
        detection_queue: Queue,
        redetect_queue: Queue,
        output_queue: Queue,
        config: dict,
        stop_event: mp.Event,
        ready_event: Optional[mp.Event] = None,
    ) -> None:
        super().__init__(name="TrackerNode", daemon=True)
        self.frame_buffer_config = frame_buffer_config
        self.tracker_queue = tracker_queue
        self.detection_queue = detection_queue
        self.redetect_queue = redetect_queue
        self.output_queue = output_queue
        self.config = config
        self.stop_event = stop_event
        self.ready_event = ready_event

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # ── Enforce offline mode ───────────────────────────────────────
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        logger.info("TrackerNode starting on %s", self.config["devices"]["tracker"])

        device = torch.device(self.config["devices"]["tracker"])

        # ── Build SAM2 predictor ───────────────────────────────────────
        predictor = self._build_predictor(device)
        logger.info("SAM2 predictor loaded.")

        # ── Optionally compile the image encoder ──────────────────────
        if self.config.get("compile", {}).get("enabled", False):
            compile_mode = self.config["compile"].get("mode", "default")
            logger.info("torch.compile(mode=%s) …", compile_mode)
            try:
                predictor.image_encoder = torch.compile(
                    predictor.image_encoder, mode=compile_mode
                )
            except Exception as exc:
                logger.warning("torch.compile failed (skipping): %s", exc)

        if self.ready_event is not None:
            self.ready_event.set()
            logger.info("TrackerNode signalled ready.")

        # ── Attach shared-memory reader ────────────────────────────────
        fb_cfg = self.frame_buffer_config
        frame_buffer = attach_reader(
            capacity=fb_cfg["capacity"],
            frame_shape=tuple(fb_cfg["frame_shape"]),
            dtype=np.dtype(fb_cfg["dtype"]),
        )

        streaming_tracker = _SAM2StreamingTracker(predictor, device, self.config)

        frames_processed = 0
        results_emitted = 0
        _log_interval = 30  # log throughput every N frames

        try:
            while not self.stop_event.is_set():
                # ── Check for new detections (non-blocking) ────────────
                try:
                    det: DetectionResult = self.detection_queue.get_nowait()
                    streaming_tracker.feed_detection(det)
                    logger.info(
                        "Tracker received %d detection(s) for frame %d",
                        len(det.boxes), det.frame_id,
                    )
                except queue.Empty:
                    pass

                # ── Get next frame ─────────────────────────────────────
                try:
                    item = self.tracker_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(item, StopSignal):
                    logger.info("TrackerNode received stop signal.")
                    break

                meta: FrameMetadata = item

                try:
                    frame = frame_buffer.read(meta, copy=True)
                except Exception as exc:
                    logger.warning(
                        "SHM read failed for frame %d: %s", meta.frame_id, exc
                    )
                    continue

                # ── Run tracker ────────────────────────────────────────
                try:
                    result, redetect_reqs = streaming_tracker.ingest_frame(
                        frame, meta.frame_id
                    )
                except Exception as exc:
                    logger.error(
                        "Tracker error on frame %d: %s", meta.frame_id, exc,
                        exc_info=True,
                    )
                    # Emit an empty result so the consumer stays in sync.
                    result = TrackResult(frame_id=meta.frame_id, tracks=[])
                    redetect_reqs = []

                # ── Forward output ────────────────────────────────────
                if result is not None:
                    try:
                        self.output_queue.put_nowait(result)
                    except Exception:
                        try:
                            self.output_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.output_queue.put_nowait(result)
                        except Exception:
                            pass

                # ── Forward redetect requests ─────────────────────────
                for req in redetect_reqs:
                    try:
                        self.redetect_queue.put_nowait(req)
                    except Exception:
                        pass

                if result is not None:
                    results_emitted += 1

                frames_processed += 1
                if frames_processed % _log_interval == 0:
                    logger.info(
                        "TrackerNode: %d frames in, %d results out, "
                        "%d active tracks, output_q depth %d",
                        frames_processed,
                        results_emitted,
                        len(streaming_tracker._tracks),
                        self.output_queue.qsize(),
                    )

        except KeyboardInterrupt:
            pass
        finally:
            # Flush any remaining results.
            for result in streaming_tracker.flush_results():
                try:
                    self.output_queue.put_nowait(result)
                except Exception:
                    pass
            streaming_tracker.cleanup()
            frame_buffer.cleanup()
            # Signal the consumer that no more results will arrive.
            try:
                self.output_queue.put_nowait(StopSignal())
            except Exception:
                pass
            logger.info("TrackerNode stopped after %d frames.", frames_processed)

    # ------------------------------------------------------------------
    # SAM2 model loading
    # ------------------------------------------------------------------

    def _build_predictor(self, device: torch.device) -> Any:
        """
        Build a ``SAM2VideoPredictor`` from the local checkpoint.

        Supports both the Facebook SAM2 package (``sam2.build_sam``) and
        a fallback shim for testing without the package installed.
        """
        sam2_config = self.config["models"]["sam2_config"]
        sam2_checkpoint = self.config["models"]["sam2_checkpoint"]
        dtype_str = self.config["precision"]["tracker_dtype"]
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        try:
            from sam2.build_sam import build_sam2_video_predictor

            predictor = build_sam2_video_predictor(
                config_file=sam2_config,
                ckpt_path=sam2_checkpoint,
                device=device,
            )
            # Cast to desired precision.
            predictor = predictor.to(dtype=torch_dtype)
            predictor.eval()
            return predictor

        except ImportError as exc:
            raise RuntimeError(
                "SAM2 package not found.  Run setup.sh to install it:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                f"Original error: {exc}"
            ) from exc
