from dataclasses import dataclass
from typing import Tuple

from config import (
    ACTION_UNIT_DISPLAY_NAMES,
    ASYMMETRY_DEVIATION_DISPLAY_NAMES,
    ASYMMETRY_LEFT_DISPLAY_NAMES,
    ASYMMETRY_RIGHT_DISPLAY_NAMES,
    EXPRESSION_DISPLAY_NAMES,
)


@dataclass(frozen=True)
class GraphDefinition:
    """Store node names with shape [N] and directed integer edges with shape [E,2]."""

    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[int, int], ...]


EXPRESSION_GRAPH = GraphDefinition(
    nodes=EXPRESSION_DISPLAY_NAMES,
    edges=(
        (0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
        (0, 4), (4, 0), (0, 5), (5, 0), (0, 6), (6, 0),
        (0, 7), (7, 0), (1, 2), (2, 1), (1, 4), (4, 1),
        (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3),
        (6, 7), (7, 6), (1, 3), (3, 1),
        (5, 6), (6, 5), (5, 7), (7, 5),
    ),
)

ACTION_UNIT_GRAPH = GraphDefinition(
    nodes=ACTION_UNIT_DISPLAY_NAMES,
    edges=(
        (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
        (3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4),
        (16, 5), (5, 16), (6, 7), (7, 6), (7, 8), (8, 7),
        (8, 9), (9, 8), (8, 10), (10, 8), (9, 10), (10, 9),
        (12, 13), (13, 12), (13, 14), (14, 13), (11, 15), (15, 11),
        (11, 7), (7, 11), (14, 15), (15, 14),
    ),
)

FACIAL_ASYMMETRY_GRAPH = GraphDefinition(
    nodes=(
        ASYMMETRY_DEVIATION_DISPLAY_NAMES
        + ASYMMETRY_LEFT_DISPLAY_NAMES
        + ASYMMETRY_RIGHT_DISPLAY_NAMES
    ),
    edges=(
        (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1),
        (3, 4), (4, 3), (5, 6), (6, 5),
        (6, 9), (9, 6), (9, 7), (7, 9), (5, 7), (7, 5),
        (8, 4), (4, 8), (8, 3), (3, 8),
        (10, 11), (11, 10), (12, 13), (13, 12), (12, 16), (16, 12),
        (13, 16), (16, 13), (12, 14), (14, 12), (16, 14), (14, 16),
        (15, 10), (10, 15), (15, 11), (11, 15),
        (17, 18), (18, 17), (19, 20), (20, 19), (19, 23), (23, 19),
        (20, 23), (23, 20), (19, 21), (21, 19), (23, 21), (21, 23),
        (22, 17), (17, 22), (22, 18), (18, 22),
        (0, 10), (10, 0), (0, 17), (17, 0), (1, 10), (10, 1),
        (1, 17), (17, 1), (2, 10), (10, 2), (2, 17), (17, 2),
        (3, 10), (10, 3), (3, 17), (17, 3), (4, 11), (11, 4),
        (4, 18), (18, 4), (5, 12), (12, 5), (5, 19), (19, 5),
        (6, 13), (13, 6), (6, 20), (20, 6), (7, 14), (14, 7),
        (7, 21), (21, 7), (8, 15), (15, 8), (8, 22), (22, 8),
        (9, 16), (16, 9), (9, 23), (23, 9),
        (10, 17), (17, 10), (11, 18), (18, 11), (12, 19), (19, 12),
        (13, 20), (20, 13), (14, 21), (21, 14), (15, 22), (22, 15),
        (16, 23), (23, 16),
    ),
)
