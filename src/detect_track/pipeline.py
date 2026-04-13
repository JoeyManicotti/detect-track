"""
Pipeline orchestrator — wires all nodes together and manages their lifecycle.

Topology
────────
                    ┌─────────────────────────────────────┐
                    │          Shared-Memory Buffer        │
                    │   (multiprocessing.shared_memory)    │
                    └──────────┬──────────────┬───────────┘
                               │              │
               ┌───────────────▼──┐    ┌──────▼──────────────┐
               │   IngestNode     │    │                      │
               │   (CPU)          │    │                      │
               │  ┌─────────────┐ │    │                      │
               │  │ detector_q  ├─┼────►  DetectorNode        │
               │  │ (FrameMeta) │ │    │  (OWLv2 – GPU 0)     │
               │  └─────────────┘ │    │          │           │
               │  ┌─────────────┐ │    │  detection_q         │
               │  │ tracker_q   ├─┼────►  (DetectionResult)   │
               │  │ (FrameMeta) │ │    └──────────┬───────────┘
               └──┴─────────────┘                │
                                                  ▼
                                    ┌─────────────────────────┐
                                    │   TrackerNode            │
                                    │   (SAM2 – GPU 1)         │
                                    │         │                │
                                    │    output_q              │
                                    │   (TrackResult)          │
                                    └─────────┬───────────────┘
                                              │
                                              ▼
                                    ┌─────────────────────────┐
                                    │  Consumer (main thread)  │
                                    │  visualise / record      │
                                    └─────────────────────────┘
                                              ▲
                              redetect_q ─────┘
                         (RedetectRequest, GPU1→GPU0)

Usage
─────
::

    from detect_track.pipeline import Pipeline
    import yaml

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = Pipeline(config)
    pipeline.start()

    try:
        for result in pipeline.results():       # blocking generator
            process(result)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from typing import Generator, Optional, Tuple

import numpy as np

from detect_track.ipc.frame_buffer import FrameSharedBuffer
from detect_track.ipc.messages import StopSignal, TrackResult
from detect_track.nodes.detector import DetectorNode
from detect_track.nodes.ingest import IngestNode
from detect_track.nodes.tracker import TrackerNode
from detect_track.utils.model_loader import verify_models

logger = logging.getLogger(__name__)

# Number of shared-memory slots: generous to let the detector always find
# its frame even when running behind.
_SHM_CAPACITY = 128


class Pipeline:
    """
    Manages the full detect-track pipeline lifecycle.

    Parameters
    ----------
    config:
        Parsed YAML configuration dict (see ``configs/default.yaml``).
    verify:
        When *True* (the default), check that all model weights exist on disk
        before spawning any processes.
    """

    def __init__(self, config: dict, *, verify: bool = True) -> None:
        self.config = config

        if verify:
            verify_models(config)

        # ── Resolve source to an absolute path so subprocesses can find it
        # regardless of their CWD.  Camera indices (int) are kept as-is.
        raw_source = config["source"]["input"]
        if isinstance(raw_source, str):
            import os
            config["source"]["input"] = os.path.abspath(raw_source)
        self._source = config["source"]["input"]

        # ── Resolve frame shape from the capture source ────────────────
        self._frame_shape = self._probe_frame_shape(self._source)
        logger.info("Frame shape: %s", self._frame_shape)

        # ── Shared-memory ring buffer ──────────────────────────────────
        self._shm_buffer = FrameSharedBuffer(
            capacity=_SHM_CAPACITY,
            frame_shape=self._frame_shape,
            dtype=np.uint8,
            owner=True,
        )

        # Serialisable config for child processes to attach as readers.
        self._fb_config = {
            "capacity": _SHM_CAPACITY,
            "frame_shape": list(self._frame_shape),
            "dtype": "uint8",
        }

        # ── Queues ─────────────────────────────────────────────────────
        q = config["queues"]
        ctx = mp.get_context("spawn")
        self._detector_queue = ctx.Queue(maxsize=q["ingest_to_detector_maxsize"])
        self._tracker_queue  = ctx.Queue(maxsize=q["ingest_to_tracker_maxsize"])
        self._detection_queue = ctx.Queue(maxsize=q["detection_to_tracker_maxsize"])
        self._redetect_queue  = ctx.Queue(maxsize=q["redetect_maxsize"])
        self._output_queue    = ctx.Queue(maxsize=q["output_maxsize"])

        # ── Shared events ──────────────────────────────────────────────
        self._stop_event = ctx.Event()
        # Each model process sets its ready event when weights are loaded.
        self._detector_ready = ctx.Event()
        self._tracker_ready  = ctx.Event()
        # IngestNode waits on this before reading the first frame.
        self._ingest_start   = ctx.Event()

        # ── Build nodes (not yet started) ──────────────────────────────
        self._ingest = IngestNode(
            source=config["source"]["input"],
            frame_buffer=self._shm_buffer,
            detector_queue=self._detector_queue,
            tracker_queue=self._tracker_queue,
            config=config,
            stop_event=self._stop_event,
            start_event=self._ingest_start,
        )
        self._detector = DetectorNode(
            frame_buffer_config=self._fb_config,
            detector_queue=self._detector_queue,
            detection_queue=self._detection_queue,
            redetect_queue=self._redetect_queue,
            config=config,
            stop_event=self._stop_event,
            ready_event=self._detector_ready,
        )
        self._tracker = TrackerNode(
            frame_buffer_config=self._fb_config,
            tracker_queue=self._tracker_queue,
            detection_queue=self._detection_queue,
            redetect_queue=self._redetect_queue,
            output_queue=self._output_queue,
            config=config,
            stop_event=self._stop_event,
            ready_event=self._tracker_ready,
        )

        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start all pipeline nodes."""
        if self._running:
            raise RuntimeError("Pipeline is already running.")
        logger.info("Starting pipeline …")
        self._tracker.start()
        self._detector.start()
        self._ingest.start()
        self._running = True
        logger.info("Pipeline running.")

        # Background thread: wait for both models, then release IngestNode.
        import threading
        def _release_ingest() -> None:
            self._detector_ready.wait()
            self._tracker_ready.wait()
            logger.info("Both models ready — releasing IngestNode.")
            self._ingest_start.set()
        threading.Thread(target=_release_ingest, daemon=True,
                         name="ModelReadyWatcher").start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Request graceful shutdown and join all processes.

        Parameters
        ----------
        timeout:
            Seconds to wait for each process to exit before force-terminating.
        """
        if not self._running:
            return
        logger.info("Stopping pipeline …")
        self._stop_event.set()

        for node in (self._ingest, self._detector, self._tracker):
            node.join(timeout=timeout)
            if node.is_alive():
                logger.warning("Force-terminating %s", node.name)
                node.terminate()
                node.join(timeout=2.0)

        self._shm_buffer.cleanup()
        self._running = False
        logger.info("Pipeline stopped.")

    def is_running(self) -> bool:
        if not self._running:
            return False
        # Pipeline is "running" as long as any node might still produce output.
        # After ingest finishes (e.g. end of file), the detector and tracker
        # may still be processing queued frames.
        return (
            self._ingest.is_alive()
            or self._detector.is_alive()
            or self._tracker.is_alive()
            or not self._output_queue.empty()
        )

    # ------------------------------------------------------------------
    # Result consumption
    # ------------------------------------------------------------------

    def results(
        self,
        timeout: float = 0.1,
    ) -> Generator[TrackResult, None, None]:
        """
        Blocking generator that yields ``TrackResult`` objects as they
        arrive from the Tracker node.

        Stops when the pipeline is no longer running and the output queue
        is drained, or when a ``StopSignal`` is received.
        """
        while True:
            try:
                item = self._output_queue.get(timeout=timeout)
            except queue.Empty:
                if not self.is_running():
                    break
                continue

            if isinstance(item, StopSignal):
                break
            yield item

    def get_result(self, timeout: float = 0.05) -> Optional[TrackResult]:
        """
        Non-blocking poll; returns *None* if no result is ready.
        Suitable for integration into an existing event loop.
        """
        try:
            item = self._output_queue.get(timeout=timeout)
            if isinstance(item, StopSignal):
                return None
            return item
        except queue.Empty:
            return None

    def read_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Read a raw RGB frame from the shared-memory ring buffer by frame ID.

        Returns *None* if the slot has already been overwritten or an error
        occurs.  The ring buffer has ``_SHM_CAPACITY`` (128) slots, giving a
        window of ~4 seconds at 30 fps before a slot is reused.
        """
        from detect_track.ipc.messages import FrameMetadata

        try:
            name = self._shm_buffer._block_name(frame_id)
            meta = FrameMetadata(
                frame_id=frame_id,
                shm_name=name,
                height=self._frame_shape[0],
                width=self._frame_shape[1],
                channels=self._frame_shape[2],
                dtype="uint8",
            )
            return self._shm_buffer.read(meta, copy=True)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Pipeline":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_frame_shape(
        source: int | str,
        fallback: Tuple[int, int, int] = (720, 1280, 3),
    ) -> Tuple[int, int, int]:
        """
        Open the source briefly to determine (H, W, C) of captured frames.
        Falls back to *fallback* if the source cannot be opened.
        """
        import cv2

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.warning(
                "Cannot probe source %r; falling back to shape %s", source, fallback
            )
            return fallback

        ret, frame = cap.read()
        cap.release()
        if not ret:
            return fallback

        h, w, c = frame.shape
        return (h, w, c)
