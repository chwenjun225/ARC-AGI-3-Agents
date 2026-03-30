from __future__ import annotations

from dataclasses import dataclass

from .object_extractor import FrameAnalysis, FrameDiff, clamp_point


@dataclass(frozen=True, slots=True)
class ClickCandidate:
    x: int
    y: int
    priority: float
    reason: str


def generate_click_candidates(
    analysis: FrameAnalysis,
    diff: FrameDiff | None = None,
    max_candidates: int = 48,
) -> list[ClickCandidate]:
    candidates: dict[tuple[int, int], ClickCandidate] = {}

    def add(x: int, y: int, priority: float, reason: str) -> None:
        point = clamp_point(x, y)
        existing = candidates.get(point)
        if existing is None or priority > existing.priority:
            candidates[point] = ClickCandidate(
                x=point[0],
                y=point[1],
                priority=priority,
                reason=reason,
            )

    if diff and diff.hot_points:
        for index, (x, y) in enumerate(diff.hot_points):
            add(x, y, 110.0 - index, "changed-region")

    for index, obj in enumerate(analysis.objects):
        base_priority = 95.0 - index * 3.0 + min(obj.area, 256) / 24.0
        add(*obj.centroid, base_priority + 1.0, f"object-{index}-centroid")
        for rep_index, (x, y) in enumerate(obj.representative_points):
            add(x, y, base_priority - rep_index * 0.75, f"object-{index}-shape")

        local_step = max(2, min(analysis.lattice_step, 8))
        min_x = max(0, obj.bbox.left - local_step)
        max_x = min(63, obj.bbox.right + local_step)
        min_y = max(0, obj.bbox.top - local_step)
        max_y = min(63, obj.bbox.bottom + local_step)
        for y in range(min_y, max_y + 1, local_step):
            for x in range(min_x, max_x + 1, local_step):
                add(x, y, 42.0 - index, f"object-{index}-local-grid")

    lattice_steps = [max(analysis.lattice_step, 4), 8, 4]
    for step_index, step in enumerate(lattice_steps):
        base_priority = 20.0 - step_index * 4.0
        offset = step // 2
        for y in range(offset, 64, step):
            for x in range(offset, 64, step):
                add(x, y, base_priority, f"global-grid-{step}")

    for x, y in (
        (32, 32),
        (0, 0),
        (63, 0),
        (0, 63),
        (63, 63),
        (32, 0),
        (32, 63),
        (0, 32),
        (63, 32),
    ):
        add(x, y, 8.0, "global-anchor")

    return sorted(
        candidates.values(),
        key=lambda item: (-item.priority, item.y, item.x),
    )[:max_candidates]
