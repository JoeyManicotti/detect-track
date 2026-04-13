"""
Ingest Node — frame capture and distribution.

Responsibilities
────────────────
1. Open a video source (webcam, file, RTSP stream).
2. Capture frames at ``capture_fps``.
3. Write each frame into the shared-memory ring buffer.
4. Push ``FrameMetadata`` pointers to:
   - The *tracker queue* on every frame (SAM2 runs at full fps).
   - The *detector queue* at a reduced rate (OWLv2 runs at ``detector_fps``).

Design notes
────────────
• The node runs in its own ``multiprocessing.Process`` so it cannot block
  the GPU workers.
• If a downstream queue is full the frame is simply dropped for that
  consumer (non-blocking ``put_nowait``).  This is intentional: we never
  block the camera loop waiting for slow consumers.
• The detector queue uses a smaller ``maxsize`` so old pending frames are
  replaced by newer ones via ``drain_and_put``.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from multiprocessing.queues import Queue
from typing import Any, Optional, Union

import cv2
import numpy as np

from detect_track.ipc.frame_buffer import FrameSharedBuffer
from detect_track.ipc.messages import FrameMetadata, StopSignal

logger = logging.getLogger(__name__)


def _drain_and_put(queue: Queue, item: Any) -> None:
    """Replace the oldest item in a full queue with *item* (best-effort)."""
    try:
        queue.put_nowait(item)
    except Exception:  # queue.Full
        # Drain one stale entry, then try again.
        try:
            queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(item)
        except Exception:
            pass


class IngestNode(mp.Process):
    """
    Video capture process.

    Parameters
    ----------
    source:
        Camera index (int) or path / URL (str).
    frame_buffer:
        Owner-mode ``FrameSharedBuffer`` shared with downstream nodes.
    detector_queue:
        Queue[FrameMetadata | StopSignal] consumed by the Detector node.
    tracker_queue:
        Queue[FrameMetadata | StopSignal] consumed by the Tracker node.
    config:
        Sub-dict from the YAML config (``pipeline`` + ``source`` sections).
    stop_event:
        ``multiprocessing.Event`` set by the pipeline to request shutdown.
    """

    def __init__(
        self,
        source: Union[int, str],
        frame_buffer: FrameSharedBuffer,
        detector_queue: Queue,
        tracker_queue: Queue,
        config: dict,
        stop_event: mp.Event,
    ) -> None:
        super().__init__(name="IngestNode", daemon=True)
        self.source = source
        self.frame_buffer = frame_buffer
        self.detector_queue = detector_queue
        self.tracker_queue = tracker_queue
        self.config = config
        self.stop_event = stop_event

    # ------------------------------------------------------------------
    # Process entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("IngestNode started (source=%s)", self.source)

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("Cannot open video source: %s", self.source)
            self._send_stop()
            return

        # Log video properties to help diagnose codec / format issues.
        src_fps    = cap.get(cv2.CAP_PROP_FPS) or 0.0
        src_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
        logger.info(
            "Video: %dx%d @ %.1f fps, %d frames, codec=%r",
            src_width, src_height, src_fps, src_frames, fourcc_str,
        )

        target_fps: float = float(self.config["pipeline"]["target_fps"])
        detector_fps: float = float(self.config["pipeline"]["detector_fps"])
        frame_interval: float = 1.0 / target_fps

        # How many tracker frames between each detector frame.
        detector_period: int = max(1, round(target_fps / detector_fps))

        frame_id = 0
        last_frame_time = time.monotonic()

        try:
            while not self.stop_event.is_set():
                ret, bgr = cap.read()
                if not ret:
                    if frame_id == 0:
                        logger.error(
                            "First frame read failed on %r. "
                            "This is usually a codec issue. "
                            "Try re-encoding: ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4",
                            self.source,
                        )
                    else:
                        logger.info(
                            "Video source exhausted after %d frames.", frame_id
                        )
                    break

                # Convert BGR → RGB for model compatibility.
                frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                # Write to shared memory and get metadata pointer.
                meta: FrameMetadata = self.frame_buffer.write(frame_id, frame)

                # Always deliver to tracker (full fps).
                _drain_and_put(self.tracker_queue, meta)

                # Deliver to detector at reduced rate.
                if frame_id % detector_period == 0:
                    _drain_and_put(self.detector_queue, meta)

                frame_id += 1

                # Pace ourselves to the target fps.
                now = time.monotonic()
                elapsed = now - last_frame_time
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_frame_time = time.monotonic()

        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            logger.info("IngestNode stopped after %d frames.", frame_id)
            self._send_stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_stop(self) -> None:
        """Push StopSignal to all downstream queues."""
        sig = StopSignal()
        for q in (self.detector_queue, self.tracker_queue):
            try:
                q.put_nowait(sig)
            except Exception:
                pass
