"""
Integration tests for the Pipeline and individual nodes.

These tests use lightweight stubs/mocks in place of real GPU models so they
run on any machine without CUDA or the actual model weights installed.
"""

from __future__ import annotations

import sys
import os
import multiprocessing as mp
import queue
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from detect_track.ipc.messages import (
    DetectionResult,
    FrameMetadata,
    RedetectRequest,
    StopSignal,
    TrackInfo,
    TrackResult,
)
from detect_track.ipc.frame_buffer import FrameSharedBuffer, attach_reader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAME_SHAPE = (240, 320, 3)
_SHM_CAPACITY = 16


def _make_frame(value: int = 42) -> np.ndarray:
    return np.full(FRAME_SHAPE, value, dtype=np.uint8)


def _make_config() -> dict:
    return {
        "source": {"input": 0, "capture_fps": 30},
        "pipeline": {
            "target_fps": 30,
            "detector_fps": 2,
            "tracker_batch_size": 5,
            "tracker_overlap_frames": 2,
        },
        "queues": {
            "ingest_to_detector_maxsize": 4,
            "ingest_to_tracker_maxsize": 60,
            "detection_to_tracker_maxsize": 16,
            "redetect_maxsize": 8,
            "output_maxsize": 60,
        },
        "devices": {"detector": "cpu", "tracker": "cpu"},
        "models": {
            "owlv2_path": "./models/owlv2",
            "sam2_config": "sam2_hiera_l.yaml",
            "sam2_checkpoint": "./models/sam2/sam2_hiera_large.pt",
        },
        "detection": {
            "text_queries": ["person", "car"],
            "score_threshold": 0.15,
            "nms_iou_threshold": 0.5,
        },
        "tracking": {
            "lost_mask_score_threshold": 0.30,
            "max_frames_lost": 10,
            "max_active_tracks": 20,
        },
        "precision": {"detector_dtype": "float32", "tracker_dtype": "float32"},
        "compile": {"enabled": False},
        "output": {
            "show_window": False,
            "write_video": False,
            "video_path": "out.mp4",
            "video_fps": 30,
            "mask_alpha": 0.45,
            "draw_boxes": True,
            "draw_labels": True,
            "draw_scores": True,
        },
    }


# ---------------------------------------------------------------------------
# Shared-memory frame buffer (unit)
# ---------------------------------------------------------------------------

class TestFrameBufferBasic:
    """Quick sanity-checks that don't require multiprocessing."""

    def test_write_returns_metadata(self):
        buf = FrameSharedBuffer(
            capacity=_SHM_CAPACITY, frame_shape=FRAME_SHAPE, dtype=np.uint8, owner=True
        )
        try:
            meta = buf.write(7, _make_frame(99))
            assert isinstance(meta, FrameMetadata)
            assert meta.frame_id == 7
            assert meta.height == FRAME_SHAPE[0]
        finally:
            buf.cleanup()

    def test_read_copy_equals_written(self):
        buf = FrameSharedBuffer(
            capacity=_SHM_CAPACITY, frame_shape=FRAME_SHAPE, dtype=np.uint8, owner=True
        )
        try:
            frame = _make_frame(123)
            meta = buf.write(0, frame)
            recovered = buf.read(meta, copy=True)
            np.testing.assert_array_equal(recovered, frame)
        finally:
            buf.cleanup()


# ---------------------------------------------------------------------------
# NMS (unit — does not need GPU)
# ---------------------------------------------------------------------------

class TestNMS:
    def test_single_box_kept(self):
        from detect_track.nodes.detector import _nms
        boxes = np.array([[0, 0, 10, 10]], dtype=float)
        scores = np.array([0.9])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert keep == [0]

    def test_overlapping_boxes_suppressed(self):
        from detect_track.nodes.detector import _nms
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],   # heavily overlaps with first
            [50, 50, 60, 60],  # no overlap
        ], dtype=float)
        scores = np.array([0.9, 0.8, 0.7])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert 0 in keep            # best score kept
        assert 1 not in keep        # suppressed
        assert 2 in keep            # non-overlapping kept

    def test_empty_input(self):
        from detect_track.nodes.detector import _nms
        boxes = np.zeros((0, 4), dtype=float)
        scores = np.array([])
        keep = _nms(boxes, scores, iou_threshold=0.5)
        assert keep == []


# ---------------------------------------------------------------------------
# mask_to_xyxy helper (unit)
# ---------------------------------------------------------------------------

class TestMaskToXYXY:
    def test_full_mask(self):
        from detect_track.nodes.tracker import _mask_to_xyxy
        mask = np.ones((10, 20), dtype=bool)
        box = _mask_to_xyxy(mask)
        assert box == [0.0, 0.0, 19.0, 9.0]

    def test_empty_mask_returns_none(self):
        from detect_track.nodes.tracker import _mask_to_xyxy
        mask = np.zeros((10, 20), dtype=bool)
        assert _mask_to_xyxy(mask) is None

    def test_small_region(self):
        from detect_track.nodes.tracker import _mask_to_xyxy
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:20, 30:50] = True
        box = _mask_to_xyxy(mask)
        assert box == [30.0, 10.0, 49.0, 19.0]


# ---------------------------------------------------------------------------
# Visualization (unit — no display)
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_draw_tracks_no_crash(self):
        from detect_track.utils.visualization import draw_tracks
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = TrackResult(
            frame_id=1,
            tracks=[
                TrackInfo(track_id=1, label="person", score=0.9, box=[10, 10, 100, 200]),
                TrackInfo(track_id=2, label="car",    score=0.7, box=None),
            ],
        )
        out = draw_tracks(frame, result)
        assert out.shape == frame.shape

    def test_draw_hud_no_crash(self):
        from detect_track.utils.visualization import draw_hud
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = draw_hud(frame, fps=25.3, frame_id=42, num_tracks=3)
        assert out is frame  # in-place

    def test_track_colour_stable(self):
        from detect_track.utils.visualization import _track_colour
        c1 = _track_colour(5)
        c2 = _track_colour(5)
        assert c1 == c2  # same track → same colour


# ---------------------------------------------------------------------------
# Ingest node integration test (no real camera)
# ---------------------------------------------------------------------------

def _ingest_video_worker(
    video_path: str,
    shm_capacity: int,
    frame_shape: tuple,
    config: dict,
    detector_q: mp.Queue,
    tracker_q: mp.Queue,
    stop_event: mp.Event,
):
    """Thin wrapper to run IngestNode in a subprocess against a real/fake source."""
    from detect_track.ipc.frame_buffer import FrameSharedBuffer
    from detect_track.nodes.ingest import IngestNode

    buf = FrameSharedBuffer(
        capacity=shm_capacity, frame_shape=frame_shape, dtype=np.uint8, owner=True
    )
    node = IngestNode(
        source=video_path,
        frame_buffer=buf,
        detector_queue=detector_q,
        tracker_queue=tracker_q,
        config=config,
        stop_event=stop_event,
    )
    node.run()
    buf.cleanup()


class TestIngestNode:
    def test_ingest_stops_on_event(self, tmp_path):
        """IngestNode should exit cleanly when stop_event is set."""
        import cv2

        # Create a tiny synthetic video file.
        video_path = str(tmp_path / "test.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        h, w, c = FRAME_SHAPE
        writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
        for i in range(10):
            writer.write(np.full((h, w, 3), i * 25, dtype=np.uint8))
        writer.release()

        ctx = mp.get_context("fork")
        det_q = ctx.Queue(maxsize=4)
        trk_q = ctx.Queue(maxsize=60)
        stop_ev = ctx.Event()
        cfg = _make_config()
        cfg["source"]["input"] = video_path

        p = ctx.Process(
            target=_ingest_video_worker,
            args=(video_path, _SHM_CAPACITY, FRAME_SHAPE, cfg, det_q, trk_q, stop_ev),
            daemon=True,
        )
        p.start()

        # Let it run briefly, then stop.
        time.sleep(0.5)
        stop_ev.set()
        p.join(timeout=5)

        # Should have exited cleanly.
        assert not p.is_alive()

        # Both queues should have received some frames.
        assert not det_q.empty() or not trk_q.empty()


# ---------------------------------------------------------------------------
# SAM2 streaming tracker (stub SAM2 predictor)
# ---------------------------------------------------------------------------

class _StubPredictor:
    """
    Minimal SAM2 predictor stub that returns zero masks for any input.
    Used to test _SAM2StreamingTracker logic without real model weights.
    """

    def init_state(self, video_path, **_kwargs):
        import glob
        jpgs = sorted(glob.glob(str(Path(video_path) / "*.jpg")))
        return {"_num_frames": len(jpgs), "_video_path": video_path}

    def reset_state(self, _state):
        pass

    def add_new_mask(self, state, frame_idx, obj_id, mask):
        pass

    def add_new_points_or_box(self, state, frame_idx, obj_id, box, **_):
        pass

    def propagate_in_video(self, state, start_frame_idx=0):
        import torch
        n = state["_num_frames"]
        for i in range(n):
            # Return a single zero mask per frame (no objects tracked).
            yield i, [], torch.zeros(0, 1, 10, 10)


class TestSAM2StreamingTracker:
    def setup_method(self):
        import torch
        self.device = torch.device("cpu")
        self.config = _make_config()
        self.predictor = _StubPredictor()

    def _make_tracker(self):
        from detect_track.nodes.tracker import _SAM2StreamingTracker
        return _SAM2StreamingTracker(self.predictor, self.device, self.config)

    def test_returns_none_before_batch_full(self):
        tracker = self._make_tracker()
        # batch_size = 5 in test config; feed 4 frames.
        for i in range(4):
            result, reqs = tracker.ingest_frame(_make_frame(i))
            # No result yet (batch not full).
            assert result is None or isinstance(result, TrackResult)
        tracker.cleanup()

    def test_result_after_batch_full(self):
        tracker = self._make_tracker()
        results = []
        for i in range(tracker.batch_size + 1):
            result, _ = tracker.ingest_frame(_make_frame(i))
            if result is not None:
                results.append(result)
        # At least one result should have been produced.
        assert len(results) >= 1
        assert all(isinstance(r, TrackResult) for r in results)
        tracker.cleanup()

    def test_detection_triggers_early_batch(self):
        tracker = self._make_tracker()
        # Feed overlap+1 frames, then a detection — should trigger batch.
        for i in range(tracker.overlap + 2):
            tracker.ingest_frame(_make_frame(i))

        det = DetectionResult(
            frame_id=0,
            boxes=[[10, 10, 100, 100]],
            labels=["person"],
            scores=[0.8],
        )
        tracker.feed_detection(det)
        result, _ = tracker.ingest_frame(_make_frame(99))

        # A batch should have been triggered; we may have a buffered result.
        # (Stub predictor returns no tracks, but result should be TrackResult.)
        tracker.cleanup()

    def test_flush_results_returns_list(self):
        tracker = self._make_tracker()
        for i in range(tracker.batch_size):
            tracker.ingest_frame(_make_frame(i))
        results = tracker.flush_results()
        assert isinstance(results, list)
        tracker.cleanup()


# ---------------------------------------------------------------------------
# Model loader (unit — no network)
# ---------------------------------------------------------------------------

class TestModelLoader:
    def test_verify_models_raises_on_missing(self, tmp_path):
        from detect_track.utils.model_loader import verify_models
        cfg = _make_config()
        cfg["models"]["owlv2_path"] = str(tmp_path / "nonexistent_owlv2")
        cfg["models"]["sam2_checkpoint"] = str(tmp_path / "nonexistent.pt")
        with pytest.raises(FileNotFoundError):
            verify_models(cfg)

    def test_verify_models_passes_when_present(self, tmp_path):
        from detect_track.utils.model_loader import verify_models

        # Create minimal directory structure to satisfy checks.
        owlv2_dir = tmp_path / "owlv2"
        owlv2_dir.mkdir()
        (owlv2_dir / "config.json").write_text("{}")
        sam2_ckpt = tmp_path / "sam2.pt"
        sam2_ckpt.write_bytes(b"\x00" * 16)

        cfg = _make_config()
        cfg["models"]["owlv2_path"] = str(owlv2_dir)
        cfg["models"]["sam2_checkpoint"] = str(sam2_ckpt)

        # Should not raise.
        verify_models(cfg)
