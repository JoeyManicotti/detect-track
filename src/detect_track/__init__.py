"""
detect-track: Async OWLv2 + SAM2 detection-and-tracking pipeline at 15–30 Hz.

Quick start::

    from detect_track.pipeline import Pipeline
    import yaml

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    with Pipeline(config) as pipeline:
        for result in pipeline.results():
            print(result.frame_id, [(t.label, t.score) for t in result.tracks])
"""

__version__ = "0.1.0"
