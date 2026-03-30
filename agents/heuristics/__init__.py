from .click_candidates import ClickCandidate, generate_click_candidates
from .object_extractor import (
    BoundingBox,
    FrameAnalysis,
    FrameDiff,
    FrameObject,
    analyse_frame,
    diff_frames,
    frame_fingerprint,
)
from .strategy import HeuristicBrain

__all__ = [
    "BoundingBox",
    "ClickCandidate",
    "FrameAnalysis",
    "FrameDiff",
    "FrameObject",
    "HeuristicBrain",
    "analyse_frame",
    "diff_frames",
    "frame_fingerprint",
    "generate_click_candidates",
]
