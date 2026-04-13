"""
Zero-copy shared-memory frame ring buffer.

Architecture
────────────
The Ingest node writes each captured frame into a named
``multiprocessing.shared_memory`` block and pushes only a lightweight
``FrameMetadata`` struct through the inter-process queues.  Downstream nodes
(Detector, Tracker) map the same shared-memory block and read pixels directly
without any copy.

Block naming convention
───────────────────────
Blocks are named  ``dt_frame_<slot>``  where ``slot = frame_id % capacity``.
Because two consecutive frames share the same slot only after ``capacity``
frames, downstream nodes must copy the data before it is overwritten if they
need to retain it past the next ``capacity`` frames.

Thread / process safety
───────────────────────
``FrameSharedBuffer`` is NOT thread-safe on its own.  The Ingest node is the
sole writer; multiple readers are safe as long as they operate on different
frame IDs or hold their own copies.
"""

from __future__ import annotations

import logging
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional, Tuple

import numpy as np

from detect_track.ipc.messages import FrameMetadata

logger = logging.getLogger(__name__)

# Prefix for all shared-memory block names created by this module.
_SHM_PREFIX = "dt_frame_"


class FrameSharedBuffer:
    """
    Ring buffer backed by ``multiprocessing.shared_memory``.

    Parameters
    ----------
    capacity:
        Number of slots in the ring.  A slot is overwritten every
        ``capacity`` frames.  Set this to at least  ``detector_fps *
        detector_latency_s + 2``  so the detector always finds its frame
        intact when it processes the metadata.
    frame_shape:
        ``(height, width, channels)`` tuple.
    dtype:
        NumPy dtype for pixel data (default ``numpy.uint8``).
    owner:
        When *True* the instance creates and owns the SHM blocks (call from
        the Ingest node).  When *False* it attaches to existing blocks (call
        from Detector / Tracker nodes).
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: Tuple[int, int, int],
        dtype: np.dtype = np.uint8,
        *,
        owner: bool,
    ) -> None:
        self.capacity = capacity
        self.frame_shape = frame_shape
        self.dtype = np.dtype(dtype)
        self.owner = owner
        self._frame_nbytes = int(np.prod(frame_shape)) * self.dtype.itemsize
        self._blocks: Dict[str, SharedMemory] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _block_name(self, frame_id: int) -> str:
        return f"{_SHM_PREFIX}{frame_id % self.capacity}"

    def _get_block(self, name: str) -> SharedMemory:
        if name not in self._blocks:
            if self.owner:
                try:
                    shm = SharedMemory(name=name, create=True, size=self._frame_nbytes)
                except FileExistsError:
                    # A previous run left the block open; reuse it.
                    shm = SharedMemory(name=name, create=False)
            else:
                shm = SharedMemory(name=name, create=False)
            self._blocks[name] = shm
        return self._blocks[name]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, frame_id: int, frame: np.ndarray) -> FrameMetadata:
        """
        Write *frame* into the slot for *frame_id* and return a
        ``FrameMetadata`` pointer that can be queued to other processes.

        The caller retains ownership of *frame*; this method copies its
        data into shared memory.
        """
        h, w, c = frame.shape
        name = self._block_name(frame_id)
        shm = self._get_block(name)
        buf = np.ndarray(self.frame_shape, dtype=self.dtype, buffer=shm.buf)
        np.copyto(buf, frame)
        return FrameMetadata(
            frame_id=frame_id,
            shm_name=name,
            height=h,
            width=w,
            channels=c,
            dtype=str(self.dtype),
        )

    def read(self, meta: FrameMetadata, *, copy: bool = True) -> np.ndarray:
        """
        Return a NumPy array view (or copy) of the frame described by *meta*.

        Parameters
        ----------
        meta:
            Metadata struct returned by a previous ``write`` call.
        copy:
            When *True* (the default) return a private copy so the data
            remains valid after the slot is overwritten.  Set *False* only if
            you process the frame immediately and hold no references past the
            next ``capacity`` frames.
        """
        shm = self._get_block(meta.shm_name)
        shape = (meta.height, meta.width, meta.channels)
        dtype = np.dtype(meta.dtype)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr.copy() if copy else arr

    def cleanup(self) -> None:
        """
        Close and, if owner, unlink all shared-memory blocks.

        Should be called from the owner process on shutdown.
        """
        for name, shm in list(self._blocks.items()):
            try:
                shm.close()
                if self.owner:
                    shm.unlink()
            except Exception as exc:  # noqa: BLE001
                logger.debug("SHM cleanup warning for %s: %s", name, exc)
        self._blocks.clear()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Module-level helpers used by non-owner processes
# ---------------------------------------------------------------------------

def attach_reader(
    capacity: int,
    frame_shape: Tuple[int, int, int],
    dtype: np.dtype = np.uint8,
) -> FrameSharedBuffer:
    """Convenience constructor for read-only (non-owner) processes."""
    return FrameSharedBuffer(
        capacity=capacity,
        frame_shape=frame_shape,
        dtype=dtype,
        owner=False,
    )
