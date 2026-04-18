"""Representation builders for stage-one magnetic experiments."""

from .dual_mode import build_dual_mode_representation, build_single_channel_representation
from .real_concat import build_real_concat_representation

__all__ = [
    'build_dual_mode_representation',
    'build_real_concat_representation',
    'build_single_channel_representation',
]
