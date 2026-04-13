"""
IPC message types shared between pipeline nodes.

All messages are plain dataclasses so they can be safely pickled through
multiprocessing.Queue without pulling in heavy third-party dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Ingest → Detector / Tracker
# ---------------------------------------------------------------------------

@dataclass
class FrameMetadata:
    """
    Pointer to a frame stored in shared memory.

    Nodes never pass raw image arrays through queues; they pass this
    lightweight struct instead and read the pixels directly from the
    shared-memory block whose name is stored here.
    """
    frame_id: int               # Monotonically increasing frame counter
    shm_name: str               # multiprocessing.shared_memory block name
    height: int                 # Frame height in pixels
    width: int                  # Frame width in pixels
    channels: int               # Number of channels (typically 3 for RGB)
    dtype: str                  # numpy dtype string, e.g. "uint8"
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Detector (OWLv2) → Tracker (SAM2)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Bounding-box detections produced by OWLv2 for a single frame.

    ``boxes`` is a list of [x1, y1, x2, y2] coordinates in pixel space.
    ``labels`` are the matched text-query strings.
    ``scores`` are the model confidence values in [0, 1].
    """
    frame_id: int
    boxes: List[List[float]]    # [[x1, y1, x2, y2], ...]
    labels: List[str]           # parallel to boxes
    scores: List[float]         # parallel to boxes
    timestamp: float = field(default_factory=time.monotonic)

    def is_empty(self) -> bool:
        return len(self.boxes) == 0


# ---------------------------------------------------------------------------
# Tracker (SAM2) → Detector (OWLv2)  –  feedback loop
# ---------------------------------------------------------------------------

@dataclass
class RedetectRequest:
    """
    Sent by the Tracker when a track's confidence drops below threshold.

    On receipt, the Detector should run OWLv2 on the next available frame
    and send a fresh DetectionResult to re-initialise the lost tracklet.
    """
    frame_id: int           # Frame at which the track was declared lost
    track_id: int           # Identifier of the lost track
    label: str              # Text label of the lost object
    reason: str             # "low_confidence" | "disappeared"
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Tracker → Output / Consumer
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Per-object tracking data for a single frame."""
    track_id: int
    label: str
    score: float                        # SAM2 mask confidence in [0, 1]
    box: Optional[List[float]]          # [x1, y1, x2, y2] or None if empty mask
    # mask is NOT carried through the queue to avoid serialisation overhead;
    # consumers that need it should be co-located with the Tracker process.


@dataclass
class TrackResult:
    """
    All active tracks for a single frame, produced by the Tracker node.
    """
    frame_id: int
    tracks: List[TrackInfo]
    timestamp: float = field(default_factory=time.monotonic)

    def has_tracks(self) -> bool:
        return len(self.tracks) > 0


# ---------------------------------------------------------------------------
# Control signals
# ---------------------------------------------------------------------------

@dataclass
class StopSignal:
    """Sentinel pushed into any queue to signal graceful shutdown."""
    reason: str = "shutdown"
