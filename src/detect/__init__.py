"""Detection utilities for stage-one magnetic experiments."""

from .simple_detector import (
    build_detection_score,
    detect_candidates,
    run_simple_detector,
    save_detection_quicklook,
)

__all__ = [
    'build_detection_score',
    'detect_candidates',
    'run_simple_detector',
    'save_detection_quicklook',
]
