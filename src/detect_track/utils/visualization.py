"""
Visualization utilities for the pipeline output.

Renders segmentation masks, bounding boxes, labels, and confidence scores
onto frames for display or video recording.  All functions are pure NumPy /
OpenCV — no GPU dependency — so they can run in the main process while the
GPU nodes operate independently.
"""

from __future__ import annotations

import colorsys
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detect_track.ipc.messages import TrackInfo, TrackResult

logger = logging.getLogger(__name__)

# Colour palette: one distinct colour per track ID, generated on demand.
_COLOUR_CACHE: Dict[int, Tuple[int, int, int]] = {}


def _track_colour(track_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR colour for a given track ID."""
    if track_id not in _COLOUR_CACHE:
        # Space hues evenly using the golden-ratio trick.
        hue = (track_id * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        _COLOUR_CACHE[track_id] = (int(b * 255), int(g * 255), int(r * 255))
    return _COLOUR_CACHE[track_id]


# ---------------------------------------------------------------------------
# Core drawing function
# ---------------------------------------------------------------------------

def draw_tracks(
    frame_bgr: np.ndarray,
    result: TrackResult,
    *,
    masks: Optional[Dict[int, np.ndarray]] = None,
    mask_alpha: float = 0.45,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    draw_scores: bool = True,
) -> np.ndarray:
    """
    Annotate *frame_bgr* (in-place) with tracking results.

    Parameters
    ----------
    frame_bgr:
        BGR image (H, W, 3) uint8.  Modified in-place.
    result:
        ``TrackResult`` from the Tracker node.
    masks:
        Optional dict mapping ``track_id → binary mask (H, W) bool``.
        When provided, a coloured semi-transparent overlay is rendered.
    mask_alpha:
        Mask overlay opacity [0, 1].
    draw_boxes, draw_labels, draw_scores:
        Toggle individual annotation layers.

    Returns
    -------
    np.ndarray
        The annotated frame (same object as *frame_bgr*).
    """
    overlay = frame_bgr.copy() if (masks and mask_alpha > 0) else None

    for track in result.tracks:
        colour = _track_colour(track.track_id)

        # ── Mask overlay ──────────────────────────────────────────────
        if overlay is not None and masks and track.track_id in masks:
            mask = masks[track.track_id]
            if mask is not None and mask.any():
                overlay[mask] = colour

        # ── Bounding box ──────────────────────────────────────────────
        if draw_boxes and track.box is not None:
            x1, y1, x2, y2 = (int(v) for v in track.box)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), colour, thickness=2)

            # ── Label / score ──────────────────────────────────────────
            if draw_labels or draw_scores:
                parts: List[str] = []
                if draw_labels:
                    parts.append(f"[{track.track_id}] {track.label}")
                if draw_scores:
                    parts.append(f"{track.score:.2f}")
                text = "  ".join(parts)

                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                bg_y1 = max(0, y1 - th - baseline - 4)
                cv2.rectangle(
                    frame_bgr,
                    (x1, bg_y1),
                    (x1 + tw + 4, y1),
                    colour,
                    thickness=-1,
                )
                # Choose white or black text based on colour luminance.
                lum = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
                text_colour = (0, 0, 0) if lum > 128 else (255, 255, 255)
                cv2.putText(
                    frame_bgr, text,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_colour, 1,
                    cv2.LINE_AA,
                )

    # Blend mask overlay.
    if overlay is not None:
        cv2.addWeighted(overlay, mask_alpha, frame_bgr, 1 - mask_alpha, 0, frame_bgr)

    return frame_bgr


# ---------------------------------------------------------------------------
# HUD / diagnostics overlay
# ---------------------------------------------------------------------------

def draw_hud(
    frame_bgr: np.ndarray,
    *,
    fps: float = 0.0,
    frame_id: int = 0,
    num_tracks: int = 0,
) -> np.ndarray:
    """
    Draw a heads-up display (pipeline stats) in the top-left corner.
    """
    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_id}",
        f"Tracks: {num_tracks}",
    ]
    y = 20
    for line in lines:
        cv2.putText(
            frame_bgr, line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (0, 255, 0), 1, cv2.LINE_AA,
        )
        y += 20
    return frame_bgr


# ---------------------------------------------------------------------------
# Preview window helper
# ---------------------------------------------------------------------------

class PreviewWindow:
    """
    Manages an OpenCV ``namedWindow`` with FPS tracking.

    Usage::

        win = PreviewWindow("detect-track")
        while running:
            annotated = win.show(frame_bgr, result)
            if annotated is None:  # 'q' pressed
                break
        win.close()
    """

    def __init__(self, title: str = "detect-track") -> None:
        self.title = title
        self._last_ts: float = 0.0
        self._fps: float = 0.0
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(
        self,
        frame_bgr: np.ndarray,
        result: Optional[TrackResult] = None,
        *,
        masks: Optional[Dict[int, np.ndarray]] = None,
        config: Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Annotate and display a frame.

        Returns the annotated frame, or *None* if the user pressed 'q'.
        """
        import time

        now = time.monotonic()
        if self._last_ts:
            alpha = 0.1
            instant_fps = 1.0 / max(now - self._last_ts, 1e-6)
            self._fps = alpha * instant_fps + (1 - alpha) * self._fps
        self._last_ts = now

        annotated = frame_bgr.copy()

        cfg = config or {}
        vis_cfg = cfg.get("output", {})

        if result is not None:
            draw_tracks(
                annotated, result,
                masks=masks,
                mask_alpha=float(vis_cfg.get("mask_alpha", 0.45)),
                draw_boxes=bool(vis_cfg.get("draw_boxes", True)),
                draw_labels=bool(vis_cfg.get("draw_labels", True)),
                draw_scores=bool(vis_cfg.get("draw_scores", True)),
            )
            draw_hud(
                annotated,
                fps=self._fps,
                frame_id=result.frame_id,
                num_tracks=len(result.tracks),
            )

        cv2.imshow(self.title, annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        return annotated

    def close(self) -> None:
        cv2.destroyWindow(self.title)


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

class VideoWriter:
    """Wraps ``cv2.VideoWriter`` for annotated output recording."""

    def __init__(self, path: str, fps: float, frame_size: Tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter at {path!r}")
        logger.info("Recording output to %s at %.1f fps", path, fps)

    def write(self, frame_bgr: np.ndarray) -> None:
        self._writer.write(frame_bgr)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
