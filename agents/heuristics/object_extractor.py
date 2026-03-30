from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

GridLike = Sequence[Sequence[int]] | np.ndarray


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left + 1

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def center(self) -> tuple[int, int]:
        return (
            int(round((self.left + self.right) / 2)),
            int(round((self.top + self.bottom) / 2)),
        )


@dataclass(frozen=True, slots=True)
class FrameObject:
    color: int
    area: int
    bbox: BoundingBox
    centroid: tuple[int, int]
    representative_points: tuple[tuple[int, int], ...]
    touches_border: bool


@dataclass(frozen=True, slots=True)
class FrameDiff:
    changed_pixels: int
    changed_ratio: float
    bbox: BoundingBox | None
    hot_points: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class FrameAnalysis:
    grid: np.ndarray
    fingerprint: str
    background_color: int
    objects: tuple[FrameObject, ...]
    palette: tuple[int, ...]
    lattice_step: int


def clamp_point(x: int, y: int) -> tuple[int, int]:
    return (max(0, min(63, x)), max(0, min(63, y)))


def to_numpy_grid(grid: GridLike) -> np.ndarray:
    array = np.asarray(grid, dtype=np.int16)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D frame grid, received shape {array.shape}")
    return array


def frame_fingerprint(grid: np.ndarray) -> str:
    digest = hashlib.blake2b(grid.tobytes(), digest_size=16)
    return digest.hexdigest()


def _most_common(values: Iterable[int], fallback: int = 0) -> int:
    counter = Counter(values)
    if not counter:
        return fallback
    return int(counter.most_common(1)[0][0])


def estimate_background_color(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0

    border = np.concatenate(
        [
            grid[0, :],
            grid[-1, :],
            grid[:, 0],
            grid[:, -1],
        ]
    )
    border_color = _most_common(int(value) for value in border.tolist())
    if np.count_nonzero(grid == border_color) > 0:
        return border_color
    return _most_common(int(value) for value in grid.ravel().tolist())


def _dedupe_points(points: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    seen: set[tuple[int, int]] = set()
    deduped: list[tuple[int, int]] = []
    for x, y in points:
        point = clamp_point(int(x), int(y))
        if point not in seen:
            seen.add(point)
            deduped.append(point)
    return tuple(deduped)


def _extract_component(
    grid: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
) -> list[tuple[int, int]]:
    color = int(grid[start_y, start_x])
    stack = [(start_y, start_x)]
    visited[start_y, start_x] = True
    points: list[tuple[int, int]] = []

    while stack:
        y, x = stack.pop()
        points.append((x, y))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= grid.shape[0] or nx >= grid.shape[1]:
                continue
            if visited[ny, nx] or int(grid[ny, nx]) != color:
                continue
            visited[ny, nx] = True
            stack.append((ny, nx))

    return points


def _representative_points(
    points: Sequence[tuple[int, int]],
    bbox: BoundingBox,
    centroid: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    top_left = (bbox.left, bbox.top)
    top_right = (bbox.right, bbox.top)
    bottom_left = (bbox.left, bbox.bottom)
    bottom_right = (bbox.right, bbox.bottom)
    mid_top = ((bbox.left + bbox.right) // 2, bbox.top)
    mid_bottom = ((bbox.left + bbox.right) // 2, bbox.bottom)
    mid_left = (bbox.left, (bbox.top + bbox.bottom) // 2)
    mid_right = (bbox.right, (bbox.top + bbox.bottom) // 2)
    first_point = points[0]
    return _dedupe_points(
        [
            centroid,
            first_point,
            top_left,
            top_right,
            bottom_left,
            bottom_right,
            mid_top,
            mid_bottom,
            mid_left,
            mid_right,
        ]
    )


def infer_lattice_step(objects: Sequence[FrameObject]) -> int:
    gcd_value = 0
    dimensions: list[int] = []
    for obj in objects:
        for value in (obj.bbox.width, obj.bbox.height):
            if 1 < value <= 16:
                dimensions.append(value)
                gcd_value = value if gcd_value == 0 else math.gcd(gcd_value, value)

    if gcd_value in {2, 3, 4, 5, 6, 7, 8}:
        return gcd_value
    if any(value % 8 == 0 for value in dimensions):
        return 8
    if any(value % 4 == 0 for value in dimensions):
        return 4
    return 4


def analyse_frame(grid_like: GridLike, min_area: int = 2) -> FrameAnalysis:
    grid = to_numpy_grid(grid_like)
    background_color = estimate_background_color(grid)
    visited = np.zeros(grid.shape, dtype=bool)
    objects: list[FrameObject] = []

    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if visited[y, x]:
                continue
            visited[y, x] = True
            color = int(grid[y, x])
            if color == background_color:
                continue

            points = _extract_component(grid, visited, y, x)
            if len(points) < min_area:
                continue

            xs = [point_x for point_x, _ in points]
            ys = [point_y for _, point_y in points]
            bbox = BoundingBox(
                left=min(xs),
                top=min(ys),
                right=max(xs),
                bottom=max(ys),
            )
            centroid = (
                int(round(sum(xs) / len(xs))),
                int(round(sum(ys) / len(ys))),
            )
            touches_border = (
                bbox.left == 0
                or bbox.top == 0
                or bbox.right == width - 1
                or bbox.bottom == height - 1
            )

            objects.append(
                FrameObject(
                    color=color,
                    area=len(points),
                    bbox=bbox,
                    centroid=clamp_point(*centroid),
                    representative_points=_representative_points(points, bbox, centroid),
                    touches_border=touches_border,
                )
            )

    objects.sort(
        key=lambda obj: (
            -obj.area,
            obj.bbox.top,
            obj.bbox.left,
            obj.color,
        )
    )

    palette = tuple(sorted(int(color) for color in np.unique(grid)))
    return FrameAnalysis(
        grid=grid,
        fingerprint=frame_fingerprint(grid),
        background_color=background_color,
        objects=tuple(objects),
        palette=palette,
        lattice_step=infer_lattice_step(objects),
    )


def diff_frames(
    previous_grid: np.ndarray | None,
    current_grid: np.ndarray,
) -> FrameDiff:
    if previous_grid is None or previous_grid.shape != current_grid.shape:
        return FrameDiff(
            changed_pixels=0,
            changed_ratio=0.0,
            bbox=None,
            hot_points=(),
        )

    changed_mask = previous_grid != current_grid
    changed_pixels = int(changed_mask.sum())
    if changed_pixels == 0:
        return FrameDiff(
            changed_pixels=0,
            changed_ratio=0.0,
            bbox=None,
            hot_points=(),
        )

    ys, xs = np.where(changed_mask)
    bbox = BoundingBox(
        left=int(xs.min()),
        top=int(ys.min()),
        right=int(xs.max()),
        bottom=int(ys.max()),
    )
    hot_points = _dedupe_points(
        [
            bbox.center,
            (bbox.left, bbox.top),
            (bbox.right, bbox.top),
            (bbox.left, bbox.bottom),
            (bbox.right, bbox.bottom),
            ((bbox.left + bbox.right) // 2, bbox.top),
            ((bbox.left + bbox.right) // 2, bbox.bottom),
            (bbox.left, (bbox.top + bbox.bottom) // 2),
            (bbox.right, (bbox.top + bbox.bottom) // 2),
        ]
    )
    return FrameDiff(
        changed_pixels=changed_pixels,
        changed_ratio=changed_pixels / float(current_grid.size),
        bbox=bbox,
        hot_points=hot_points,
    )
