"""
Model download and local-path resolution utilities.

Called by ``setup.sh`` (via  ``python -m detect_track.utils.model_loader``)
during the installation phase and by the pipeline at startup to verify that
all weights are present before spawning GPU processes.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hugging Face snapshot downloads
# ---------------------------------------------------------------------------

def download_owlv2(
    repo_id: str = "google/owlv2-base-patch16-ensemble",
    local_dir: str = "./models/owlv2",
    *,
    token: Optional[str] = None,
) -> Path:
    """
    Download OWLv2 weights and processor config from Hugging Face Hub to a
    local directory.  Safe to call repeatedly; skips already-downloaded files.

    Parameters
    ----------
    repo_id:
        Hugging Face repository identifier.
    local_dir:
        Destination directory.  Will be created if it does not exist.
    token:
        Optional HF access token (for gated repos).

    Returns
    -------
    Path
        Absolute path to the populated local directory.
    """
    from huggingface_hub import snapshot_download

    dest = Path(local_dir).resolve()
    logger.info("Downloading OWLv2 (%s) → %s …", repo_id, dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=token,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    logger.info("OWLv2 download complete: %s", dest)
    return dest


def download_sam2(
    repo_id: str = "facebook/sam2-hiera-large",
    local_dir: str = "./models/sam2",
    *,
    token: Optional[str] = None,
) -> Path:
    """
    Download SAM2 weights from Hugging Face Hub.

    The SAM2 checkpoint (``sam2_hiera_large.pt``) and config YAML
    (``sam2_hiera_l.yaml``) are downloaded here; the pipeline config must
    point to these local paths.

    Returns
    -------
    Path
        Absolute path to the populated local directory.
    """
    from huggingface_hub import snapshot_download

    dest = Path(local_dir).resolve()
    logger.info("Downloading SAM2 (%s) → %s …", repo_id, dest)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        token=token,
    )
    logger.info("SAM2 download complete: %s", dest)
    return dest


# ---------------------------------------------------------------------------
# Startup verification
# ---------------------------------------------------------------------------

def verify_models(config: dict) -> None:
    """
    Raise ``FileNotFoundError`` if any model weight is missing.

    Call this before spawning GPU processes so the error is reported cleanly
    rather than inside a subprocess.
    """
    owlv2_path = Path(config["models"]["owlv2_path"])
    sam2_ckpt = Path(config["models"]["sam2_checkpoint"])

    missing: list[str] = []

    if not owlv2_path.exists():
        missing.append(f"OWLv2 directory: {owlv2_path}")
    else:
        # Check for at least a config or model file.
        if not any(owlv2_path.glob("config.json")):
            missing.append(f"OWLv2 config.json missing in {owlv2_path}")

    if not sam2_ckpt.exists():
        missing.append(f"SAM2 checkpoint: {sam2_ckpt}")

    if missing:
        msg = (
            "Required model weights not found:\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\nRun  ./setup.sh  to download models for offline use."
        )
        raise FileNotFoundError(msg)

    logger.info("All model weights found.")


# ---------------------------------------------------------------------------
# CLI entry point for setup.sh
# ---------------------------------------------------------------------------

def _cli_download(argv: list[str] | None = None) -> None:
    """
    Download all models.

    Usage::

        python -m detect_track.utils.model_loader [--owlv2-dir DIR] [--sam2-dir DIR]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OWLv2 and SAM2 models for offline use."
    )
    parser.add_argument(
        "--owlv2-dir", default="./models/owlv2",
        help="Destination for OWLv2 weights (default: ./models/owlv2)"
    )
    parser.add_argument(
        "--sam2-dir", default="./models/sam2",
        help="Destination for SAM2 weights (default: ./models/sam2)"
    )
    parser.add_argument(
        "--owlv2-repo", default="google/owlv2-base-patch16-ensemble",
        help="Hugging Face repo ID for OWLv2"
    )
    parser.add_argument(
        "--sam2-repo", default="facebook/sam2-hiera-large",
        help="Hugging Face repo ID for SAM2"
    )
    parser.add_argument("--token", default=None, help="HF access token (optional)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    download_owlv2(
        repo_id=args.owlv2_repo,
        local_dir=args.owlv2_dir,
        token=args.token,
    )
    download_sam2(
        repo_id=args.sam2_repo,
        local_dir=args.sam2_dir,
        token=args.token,
    )
    print("\nAll models downloaded successfully.")
    print("Set the following in configs/default.yaml:")
    print(f"  models.owlv2_path: {Path(args.owlv2_dir).resolve()}")
    print(f"  models.sam2_checkpoint: {Path(args.sam2_dir).resolve()}/sam2_hiera_large.pt")


if __name__ == "__main__":
    _cli_download(sys.argv[1:])
