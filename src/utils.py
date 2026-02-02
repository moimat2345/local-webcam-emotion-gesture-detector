"""Shared utility functions for vision analyzers."""

import math

from src.data_models import Landmark


def landmark_distance(p1: Landmark, p2: Landmark) -> float:
    """Calculate 2D Euclidean distance between two landmarks."""
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
