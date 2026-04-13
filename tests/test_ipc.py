"""
Tests for the IPC layer (shared-memory frame buffer and message types).

These tests run without any GPU or model dependencies.
"""

from __future__ import annotations

import sys
import os

# Make the src layout importable without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import multiprocessing as mp
import time

import numpy as np
import pytest

from detect_track.ipc.frame_buffer import FrameSharedBuffer, attach_reader
from detect_track.ipc.messages import (
    DetectionResult,
    FrameMetadata,
    RedetectRequest,
    StopSignal,
    TrackInfo,
    TrackResult,
)


# ---------------------------------------------------------------------------
# FrameSharedBuffer
# ---------------------------------------------------------------------------

class TestFrameSharedBuffer:
    SHAPE = (480, 640, 3)

    def _make_owner(self) -> FrameSharedBuffer:
        return FrameSharedBuffer(
            capacity=8,
            frame_shape=self.SHAPE,
            dtype=np.uint8,
            owner=True,
        )

    def test_write_and_read_roundtrip(self):
        buf = self._make_owner()
        try:
            rng = np.random.default_rng(42)
            frame = rng.integers(0, 255, self.SHAPE, dtype=np.uint8)
            meta = buf.write(0, frame)

            assert isinstance(meta, FrameMetadata)
            assert meta.frame_id == 0
            assert meta.height == self.SHAPE[0]
            assert meta.width == self.SHAPE[1]
            assert meta.channels == self.SHAPE[2]

            recovered = buf.read(meta, copy=True)
            np.testing.assert_array_equal(recovered, frame)
        finally:
            buf.cleanup()

    def test_ring_slot_reuse(self):
        """Frame N and frame N+capacity share the same slot; verify slot wraps."""
        capacity = 4
        buf = FrameSharedBuffer(
            capacity=capacity, frame_shape=self.SHAPE, dtype=np.uint8, owner=True
        )
        try:
            frame_a = np.full(self.SHAPE, 10, dtype=np.uint8)
            frame_b = np.full(self.SHAPE, 20, dtype=np.uint8)

            meta_a = buf.write(0, frame_a)
            # Write capacity more frames so slot 0 is overwritten.
            for i in range(1, capacity):
                buf.write(i, np.zeros(self.SHAPE, dtype=np.uint8))
            # Now overwrite slot 0 (frame_id = capacity maps to slot 0).
            meta_b = buf.write(capacity, frame_b)

            # meta_a and meta_b share the same shm_name.
            assert meta_a.shm_name == meta_b.shm_name

            # Reading now yields frame_b (the overwriter).
            recovered = buf.read(meta_b, copy=True)
            np.testing.assert_array_equal(recovered, frame_b)
        finally:
            buf.cleanup()

    def test_multiple_frames_independent(self):
        """Different frame IDs with capacity > 1 have independent slots."""
        buf = self._make_owner()
        try:
            frames = [
                np.full(self.SHAPE, v, dtype=np.uint8) for v in [11, 22, 33]
            ]
            metas = [buf.write(i, f) for i, f in enumerate(frames)]

            for i, (meta, frame) in enumerate(zip(metas, frames)):
                recovered = buf.read(meta, copy=True)
                np.testing.assert_array_equal(recovered, frame, err_msg=f"frame {i}")
        finally:
            buf.cleanup()

    def test_cross_process_read(self):
        """Owner writes; a reader process in a subprocess reads correctly."""
        buf = self._make_owner()
        try:
            expected = np.full(self.SHAPE, 77, dtype=np.uint8)
            meta = buf.write(0, expected)

            result_queue: mp.Queue = mp.Queue()
            p = mp.Process(
                target=_reader_worker,
                args=(meta, self.SHAPE, result_queue),
                daemon=True,
            )
            p.start()
            p.join(timeout=10)
            assert p.exitcode == 0, f"Reader process failed (exit {p.exitcode})"
            recovered = result_queue.get_nowait()
            np.testing.assert_array_equal(recovered, expected)
        finally:
            buf.cleanup()


def _reader_worker(
    meta: FrameMetadata,
    shape: tuple,
    result_queue: mp.Queue,
) -> None:
    """Subprocess that attaches as a reader and reads a frame."""
    reader = attach_reader(capacity=8, frame_shape=shape, dtype=np.uint8)
    frame = reader.read(meta, copy=True)
    result_queue.put(frame)
    reader.cleanup()


# ---------------------------------------------------------------------------
# Message dataclasses
# ---------------------------------------------------------------------------

class TestMessages:
    def test_detection_result_is_empty(self):
        det = DetectionResult(frame_id=0, boxes=[], labels=[], scores=[])
        assert det.is_empty()

        det2 = DetectionResult(
            frame_id=1,
            boxes=[[0, 0, 10, 10]],
            labels=["cat"],
            scores=[0.9],
        )
        assert not det2.is_empty()

    def test_track_result_has_tracks(self):
        r1 = TrackResult(frame_id=0, tracks=[])
        assert not r1.has_tracks()

        r2 = TrackResult(
            frame_id=1,
            tracks=[TrackInfo(track_id=1, label="dog", score=0.8, box=[0, 0, 5, 5])],
        )
        assert r2.has_tracks()

    def test_redetect_request_fields(self):
        req = RedetectRequest(
            frame_id=100, track_id=3, label="person", reason="low_confidence"
        )
        assert req.track_id == 3
        assert req.reason == "low_confidence"
        assert req.timestamp > 0

    def test_stop_signal_default_reason(self):
        sig = StopSignal()
        assert sig.reason == "shutdown"

    def test_frame_metadata_timestamp(self):
        meta = FrameMetadata(
            frame_id=5, shm_name="dt_frame_5",
            height=480, width=640, channels=3, dtype="uint8"
        )
        assert meta.timestamp > 0


# ---------------------------------------------------------------------------
# attach_reader convenience function
# ---------------------------------------------------------------------------

class TestAttachReader:
    def test_returns_non_owner(self):
        owner = FrameSharedBuffer(
            capacity=4,
            frame_shape=(100, 100, 3),
            dtype=np.uint8,
            owner=True,
        )
        try:
            frame = np.ones((100, 100, 3), dtype=np.uint8)
            meta = owner.write(0, frame)

            reader = attach_reader(
                capacity=4, frame_shape=(100, 100, 3), dtype=np.uint8
            )
            try:
                assert not reader.owner
                recovered = reader.read(meta, copy=True)
                np.testing.assert_array_equal(recovered, frame)
            finally:
                reader.cleanup()
        finally:
            owner.cleanup()
