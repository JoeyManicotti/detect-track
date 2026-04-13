"""
Detector Node — OWLv2 zero-shot open-vocabulary object detection on GPU 0.

Responsibilities
────────────────
1. Load OWLv2 from a local path (offline-safe).
2. Wait for ``FrameMetadata`` on the detector queue.
3. Run OWLv2 with the configured text queries.
4. Push ``DetectionResult`` to the tracker.
5. Listen for ``RedetectRequest`` messages from the tracker and prioritise
   them over the scheduled detection cadence.

Throughput strategy
────────────────────
OWLv2 (base-patch16-ensemble) takes ~150–400 ms per image on a modern GPU —
far too slow for 30 fps.  This node is designed to run at 1–3 Hz.  The Tracker
node runs at full frame rate using SAM2's temporal memory; it only needs OWLv2
for:
  (a) first-frame initialisation, and
  (b) re-detection when a track is lost.

Offline operation
─────────────────
Set ``TRANSFORMERS_OFFLINE=1`` and ``HF_DATASETS_OFFLINE=1`` before import so
Hugging Face never attempts a network call.  ``setup.sh`` downloads all weights
during the installation phase.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
from multiprocessing.queues import Queue
from typing import List, Optional, Tuple

import numpy as np
import torch

from detect_track.ipc.frame_buffer import FrameSharedBuffer, attach_reader
from detect_track.ipc.messages import (
    DetectionResult,
    FrameMetadata,
    RedetectRequest,
    StopSignal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NMS helper (pure NumPy — no torchvision dependency at call site)
# ---------------------------------------------------------------------------

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """
    Greedy non-maximum suppression.  Returns indices of kept boxes sorted
    by descending score.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return keep


# ---------------------------------------------------------------------------
# Detector Node
# ---------------------------------------------------------------------------

class DetectorNode(mp.Process):
    """
    OWLv2 zero-shot detector running in a dedicated process on ``detector_device``.

    Parameters
    ----------
    frame_buffer_config:
        Dict with keys ``capacity``, ``frame_shape``, ``dtype`` needed to
        attach a reader to the shared-memory ring buffer.
    detector_queue:
        Queue[FrameMetadata | StopSignal] fed by the Ingest node.
    detection_queue:
        Queue[DetectionResult] consumed by the Tracker node.
    redetect_queue:
        Queue[RedetectRequest] fed by the Tracker node when a track is lost.
    config:
        Full pipeline config dict.
    stop_event:
        Shared ``multiprocessing.Event``; set to request graceful shutdown.
    """

    def __init__(
        self,
        frame_buffer_config: dict,
        detector_queue: Queue,
        detection_queue: Queue,
        redetect_queue: Queue,
        config: dict,
        stop_event: mp.Event,
    ) -> None:
        super().__init__(name="DetectorNode", daemon=True)
        self.frame_buffer_config = frame_buffer_config
        self.detector_queue = detector_queue
        self.detection_queue = detection_queue
        self.redetect_queue = redetect_queue
        self.config = config
        self.stop_event = stop_event

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        # ── Enforce offline mode before any HF import ──────────────────
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

        # ── Late imports (only needed in this subprocess) ──────────────
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        logger.info("DetectorNode starting on %s", self.config["devices"]["detector"])

        device = torch.device(self.config["devices"]["detector"])
        dtype_str = self.config["precision"]["detector_dtype"]
        torch_dtype = getattr(torch, dtype_str, torch.bfloat16)

        # ── Load model from local path ─────────────────────────────────
        model_path = self.config["models"]["owlv2_path"]
        logger.info("Loading OWLv2 from %s …", model_path)
        processor = Owlv2Processor.from_pretrained(model_path)
        model = Owlv2ForObjectDetection.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(device)
        model.eval()
        logger.info("OWLv2 loaded.")

        # ── Attach shared-memory reader ────────────────────────────────
        fb_cfg = self.frame_buffer_config
        frame_buffer = attach_reader(
            capacity=fb_cfg["capacity"],
            frame_shape=tuple(fb_cfg["frame_shape"]),
            dtype=np.dtype(fb_cfg["dtype"]),
        )

        text_queries: List[str] = self.config["detection"]["text_queries"]
        score_threshold: float = float(self.config["detection"]["score_threshold"])
        nms_iou: float = float(self.config["detection"]["nms_iou_threshold"])

        frames_detected = 0
        redetect_pending: Optional[RedetectRequest] = None

        try:
            while not self.stop_event.is_set():
                # ── Check for priority redetect requests ──────────────
                try:
                    redetect_pending = self.redetect_queue.get_nowait()
                    logger.debug(
                        "Redetect requested for track %d (%s)",
                        redetect_pending.track_id,
                        redetect_pending.label,
                    )
                except queue.Empty:
                    pass

                # ── Pull the next scheduled frame ─────────────────────
                try:
                    item = self.detector_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if isinstance(item, StopSignal):
                    logger.info("DetectorNode received stop signal.")
                    break

                meta: FrameMetadata = item

                try:
                    frame = frame_buffer.read(meta, copy=True)
                except Exception as exc:
                    logger.warning(
                        "SHM read failed for frame %d: %s", meta.frame_id, exc
                    )
                    continue

                # ── Run OWLv2 inference ───────────────────────────────
                try:
                    t0 = time.monotonic()
                    result = self._detect(
                        frame, processor, model, device, torch_dtype,
                        text_queries, score_threshold, nms_iou, meta.frame_id
                    )
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    logger.info(
                        "OWLv2 frame %d: %d detections in %.1f ms",
                        meta.frame_id, len(result.boxes), elapsed_ms,
                    )
                except Exception as exc:
                    logger.error(
                        "OWLv2 inference failed on frame %d: %s",
                        meta.frame_id, exc, exc_info=True,
                    )
                    continue

                if not result.is_empty():
                    try:
                        self.detection_queue.put_nowait(result)
                    except Exception:
                        # Queue full — drop oldest result and retry.
                        try:
                            self.detection_queue.get_nowait()
                        except Exception:
                            pass
                        try:
                            self.detection_queue.put_nowait(result)
                        except Exception:
                            pass

                frames_detected += 1
                redetect_pending = None

        except KeyboardInterrupt:
            pass
        finally:
            frame_buffer.cleanup()
            logger.info("DetectorNode stopped after %d detections.", frames_detected)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @staticmethod
    def _detect(
        frame: np.ndarray,
        processor,
        model,
        device: torch.device,
        torch_dtype: torch.dtype,
        text_queries: List[str],
        score_threshold: float,
        nms_iou: float,
        frame_id: int,
    ) -> DetectionResult:
        """
        Run OWLv2 on a single RGB frame and return a ``DetectionResult``.

        OWLv2 takes a batch of text query lists.  We wrap our flat list
        in an outer list to form a single-image batch:
            ``texts = [[q1, q2, ...]]``
        """
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(frame)
        texts = [text_queries]  # batch dimension

        inputs = processor(
            text=texts,
            images=[pil_image],
            return_tensors="pt",
        ).to(device)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch_dtype):
            outputs = model(**inputs)

        # Post-process: OWLv2 returns boxes in cx,cy,w,h (normalised).
        # post_process_object_detection converts to x1,y1,x2,y2 (pixel).
        target_sizes = torch.tensor(
            [[frame.shape[0], frame.shape[1]]], device=device
        )
        # post_process_object_detection lives on the image_processor,
        # not the top-level Owlv2Processor wrapper.
        results = processor.image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=score_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes_t = results["boxes"].cpu().float()
        scores_t = results["scores"].cpu().float()
        label_ids = results["labels"].cpu().tolist()

        if boxes_t.numel() == 0:
            return DetectionResult(
                frame_id=frame_id, boxes=[], labels=[], scores=[]
            )

        boxes_np = boxes_t.numpy()
        scores_np = scores_t.numpy()

        # Apply NMS to remove duplicates.
        keep = _nms(boxes_np, scores_np, nms_iou)

        boxes_out = boxes_np[keep].tolist()
        scores_out = scores_np[keep].tolist()
        labels_out = [text_queries[label_ids[k]] for k in keep]

        return DetectionResult(
            frame_id=frame_id,
            boxes=boxes_out,
            labels=labels_out,
            scores=scores_out,
        )
