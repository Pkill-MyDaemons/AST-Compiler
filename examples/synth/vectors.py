"""Vector math utilities."""
from typing import List, Optional
import math


class Vector2D:
    """A 2D vector."""

    x: float
    y: float

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def magnitude(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def add(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x + other.x, self.y + other.y)

    def subtract(self, other: "Vector2D") -> "Vector2D":
        return Vector2D(self.x - other.x, self.y - other.y)

    def scale(self, factor: float) -> "Vector2D":
        return Vector2D(self.x * factor, self.y * factor)

    def dot(self, other: "Vector2D") -> float:
        return self.x * other.x + self.y * other.y

    def normalize(self) -> Optional["Vector2D"]:
        mag: float = self.magnitude()
        if mag == 0.0:
            return None
        return Vector2D(self.x / mag, self.y / mag)

    def is_zero(self) -> bool:
        return self.x == 0.0 and self.y == 0.0


def distance(a: Vector2D, b: Vector2D) -> float:
    dx: float = a.x - b.x
    dy: float = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def centroid(points: List[Vector2D]) -> Optional[Vector2D]:
    if not points:
        return None
    sum_x: float = 0.0
    sum_y: float = 0.0
    for p in points:
        sum_x = sum_x + p.x
        sum_y = sum_y + p.y
    n: float = len(points)
    return Vector2D(sum_x / n, sum_y / n)


def closest_point(target: Vector2D, candidates: List[Vector2D]) -> Optional[Vector2D]:
    if not candidates:
        return None
    best = candidates[0]
    best_dist: float = distance(target, best)
    for c in candidates:
        d: float = distance(target, c)
        if d < best_dist:
            best_dist = d
            best = c
    return best
