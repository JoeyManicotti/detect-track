"""
CLI entry point for detect-track.

Commands
────────
run       Launch the full OWLv2 + SAM2 pipeline.
download  Download model weights for offline use (calls setup logic).

Examples::

    # Run on webcam 0 with default config
    detect-track run

    # Run on a video file with a custom config
    detect-track run --source /path/to/video.mp4 --config my_config.yaml

    # Override text queries on the command line
    detect-track run --queries "person" "bicycle" "car"

    # Download models
    detect-track download --owlv2-dir ./models/owlv2 --sam2-dir ./models/sam2
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import click
import yaml

# Enforce offline mode early — before any transformers import — so that even
# top-level HF imports do not trigger network calls.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _apply_overrides(config: dict, **overrides: object) -> dict:
    """Apply non-None CLI overrides onto the config dict."""
    src = config["source"]
    pip = config["pipeline"]
    det = config["detection"]
    out = config["output"]

    if overrides.get("source") is not None:
        raw = overrides["source"]
        src["input"] = int(raw) if str(raw).isdigit() else raw
    if overrides.get("detector_fps") is not None:
        pip["detector_fps"] = float(overrides["detector_fps"])
    if overrides.get("target_fps") is not None:
        pip["target_fps"] = float(overrides["target_fps"])
    if overrides.get("queries"):
        import re
        # Each flag value may itself contain comma/space-separated tokens, e.g.
        #   -q "person,car"  or  -q "person car"  or  -q person -q car
        merged: list[str] = []
        for token in overrides["queries"]:
            merged.extend(t.strip() for t in re.split(r"[,\s]+", token) if t.strip())
        det["text_queries"] = merged
    if overrides.get("threshold") is not None:
        det["score_threshold"] = float(overrides["threshold"])
    if overrides.get("detector_device") is not None:
        config["devices"]["detector"] = overrides["detector_device"]
    if overrides.get("tracker_device") is not None:
        config["devices"]["tracker"] = overrides["tracker_device"]
    if overrides.get("write_video") is not None:
        out["write_video"] = True
        out["video_path"] = overrides["write_video"]

    return config


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """OWLv2 + SAM2 detect-and-track pipeline."""


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--config", "-c",
    default="configs/default.yaml",
    show_default=True,
    help="Path to YAML config file.",
)
@click.option("--source", "-s", default=None, help="Video source (int or path/URL).")
@click.option(
    "--queries", "-q",
    "queries",
    multiple=True,
    metavar="QUERY",
    help=(
        "Text query for OWLv2. Repeat the flag for multiple objects "
        "(-q person -q car) or pass a comma/space-separated list in one flag "
        "(-q 'person,car,bicycle'). Both forms may be combined."
    ),
)
@click.option("--threshold", "-t", default=None, type=float, help="Detection score threshold.")
@click.option("--detector-fps", default=None, type=float, help="OWLv2 inference rate (Hz).")
@click.option("--target-fps",   default=None, type=float, help="Pipeline output target (Hz).")
@click.option("--detector-device", default=None, help="PyTorch device for OWLv2 (e.g. cuda:0).")
@click.option("--tracker-device",  default=None, help="PyTorch device for SAM2 (e.g. cuda:1).")
@click.option("--port", "-p", default=None, type=int, metavar="PORT",
              help="HTTP streaming server port (default: from config, usually 8080).")
@click.option("--host", default=None, metavar="HOST",
              help="HTTP server bind address (default: 0.0.0.0).")
@click.option("--write-video", default=None, metavar="PATH", help="Write output video to PATH.")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
def run(
    config: str,
    source: Optional[str],
    queries: Tuple[str, ...],
    threshold: Optional[float],
    detector_fps: Optional[float],
    target_fps: Optional[float],
    detector_device: Optional[str],
    tracker_device: Optional[str],
    port: Optional[int],
    host: Optional[str],
    write_video: Optional[str],
    verbose: bool,
) -> None:
    """Launch the pipeline and stream results to a browser via HTTP."""
    _setup_logging(verbose)
    logger = logging.getLogger(__name__)

    cfg = _load_config(config)
    cfg = _apply_overrides(
        cfg,
        source=source,
        queries=queries or None,
        threshold=threshold,
        detector_fps=detector_fps,
        target_fps=target_fps,
        detector_device=detector_device,
        tracker_device=tracker_device,
        write_video=write_video,
    )

    # Streaming server settings (CLI overrides config).
    stream_cfg  = cfg.setdefault("streaming", {})
    serve_host  = host or stream_cfg.get("host", "0.0.0.0")
    serve_port  = port or int(stream_cfg.get("port", 8080))

    logger.info("Text queries: %s", cfg["detection"]["text_queries"])
    logger.info("Detector: %s @ %.1f Hz", cfg["devices"]["detector"],
                cfg["pipeline"]["detector_fps"])
    logger.info("Tracker:  %s @ %.1f Hz target", cfg["devices"]["tracker"],
                cfg["pipeline"]["target_fps"])

    import numpy as np
    import cv2
    from detect_track.pipeline import Pipeline
    from detect_track.nodes.streaming import StreamingServer
    from detect_track.utils.visualization import VideoWriter, draw_tracks, draw_hud

    vis_cfg  = cfg.get("output", {})
    write_out = bool(vis_cfg.get("write_video", False))
    out_path  = vis_cfg.get("video_path", "output.mp4")
    out_fps   = float(vis_cfg.get("video_fps", cfg["pipeline"]["target_fps"]))

    if write_video:
        write_out = True
        out_path  = write_video

    streaming = StreamingServer(host=serve_host, port=serve_port)
    streaming.start()
    click.echo(
        f"\n  View in browser → http://localhost:{serve_port}/\n"
        f"  Press Ctrl-C to stop.\n"
    )

    vw: Optional[VideoWriter] = None

    try:
        with Pipeline(cfg) as pipeline:
            for result in pipeline.results():
                # Read the actual frame from shared memory so the browser
                # sees the real video, not a synthetic canvas.
                frame_rgb = pipeline.read_frame(result.frame_id)
                if frame_rgb is None:
                    # Slot already overwritten — synthesise a placeholder.
                    frame_rgb = np.zeros(pipeline._frame_shape, dtype=np.uint8)

                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Annotate: boxes, labels, HUD.
                draw_tracks(
                    bgr, result,
                    mask_alpha=float(vis_cfg.get("mask_alpha", 0.45)),
                    draw_boxes=bool(vis_cfg.get("draw_boxes", True)),
                    draw_labels=bool(vis_cfg.get("draw_labels", True)),
                    draw_scores=bool(vis_cfg.get("draw_scores", True)),
                )
                draw_hud(bgr, frame_id=result.frame_id,
                         num_tracks=len(result.tracks))

                # Push to MJPEG stream.
                streaming.push_frame(bgr, result)

                # Optionally write to disk.
                if write_out:
                    if vw is None:
                        h, w = bgr.shape[:2]
                        vw = VideoWriter(out_path, out_fps, (w, h))
                    vw.write(bgr)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        streaming.stop()
        if vw:
            vw.release()


# ---------------------------------------------------------------------------
# download command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--owlv2-dir",   default="./models/owlv2",
    show_default=True, help="Destination for OWLv2 weights.",
)
@click.option(
    "--sam2-dir",    default="./models/sam2",
    show_default=True, help="Destination for SAM2 weights.",
)
@click.option(
    "--owlv2-repo",  default="google/owlv2-base-patch16-ensemble",
    show_default=True, help="Hugging Face repo ID for OWLv2.",
)
@click.option(
    "--sam2-repo",   default="facebook/sam2-hiera-large",
    show_default=True, help="Hugging Face repo ID for SAM2.",
)
@click.option("--token", default=None, help="Hugging Face access token.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def download(
    owlv2_dir: str,
    sam2_dir: str,
    owlv2_repo: str,
    sam2_repo: str,
    token: Optional[str],
    verbose: bool,
) -> None:
    """Download model weights to local directories for offline use."""
    _setup_logging(verbose)

    # Remove offline lock for the download command only.
    for var in ("TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_OFFLINE"):
        os.environ.pop(var, None)

    from detect_track.utils.model_loader import download_owlv2, download_sam2

    owlv2_path = download_owlv2(repo_id=owlv2_repo, local_dir=owlv2_dir, token=token)
    sam2_path  = download_sam2(repo_id=sam2_repo,   local_dir=sam2_dir,  token=token)

    click.echo("\nModels downloaded successfully.")
    click.echo(f"  OWLv2:  {owlv2_path}")
    click.echo(f"  SAM2:   {sam2_path}")
    click.echo("\nUpdate configs/default.yaml:")
    click.echo(f"  models.owlv2_path: {owlv2_path}")
    click.echo(f"  models.sam2_checkpoint: {sam2_path}/sam2_hiera_large.pt")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Required for multiprocessing on macOS / Windows ("spawn" start method).
    mp_ctx = __import__("multiprocessing")
    mp_ctx.freeze_support()
    cli()
