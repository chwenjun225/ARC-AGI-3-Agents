from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Optional

from arcengine import FrameData, GameAction

from .click_candidates import ClickCandidate, generate_click_candidates
from .object_extractor import (
    FrameAnalysis,
    FrameDiff,
    FrameObject,
    analyse_frame,
    diff_frames,
)

MOVEMENT_ACTION_IDS = (1, 2, 3, 4)
PROBE_ORDER = (1, 4, 2, 3, 5)
REVERSE_MOVEMENT = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
}
TURN_LEFT = {
    1: 3,
    2: 4,
    3: 2,
    4: 1,
}
TURN_RIGHT = {
    1: 4,
    2: 3,
    3: 1,
    4: 2,
}
MOVEMENT_DELTAS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}


@dataclass(frozen=True, slots=True)
class ActionProposal:
    action_id: int
    score: float
    reason: str
    data: dict[str, int] | None = None


class HeuristicBrain:
    """Stateful heuristic policy for offline ARC-AGI-3 experimentation."""

    def __init__(self) -> None:
        self.action_success: Counter[int] = Counter()
        self.action_fail: Counter[int] = Counter()
        self.reset_for_new_attempt()

    def reset_for_new_attempt(self) -> None:
        self.state_visits: Counter[str] = Counter()
        self.state_action_attempts: defaultdict[str, Counter[int]] = defaultdict(
            Counter
        )
        self.position_visits: Counter[str] = Counter()
        self.position_action_attempts: defaultdict[str, Counter[int]] = defaultdict(
            Counter
        )
        self.position_transitions: defaultdict[str, dict[int, str]] = defaultdict(dict)
        self.state_click_attempts: defaultdict[str, set[tuple[int, int]]] = defaultdict(
            set
        )
        self.recent_fingerprints: deque[str] = deque(maxlen=12)
        self.candidate_cache: dict[str, list[ClickCandidate]] = {}
        self.planned_movement_actions: deque[int] = deque()

        self.current_analysis: FrameAnalysis | None = None
        self.current_diff: FrameDiff | None = None
        self.avatar_object: FrameObject | None = None
        self.current_position_key: str | None = None
        self.preferred_heading: int | None = None
        self.levels_completed = 0
        self.steps_in_level = 0
        self.no_change_streak = 0
        self.loop_streak = 0
        self.same_action_streak = 0

        self.last_action_id: int | None = None
        self.last_click_target: tuple[int, int] | None = None

    def _reset_level_state(self) -> None:
        self.state_visits.clear()
        self.state_action_attempts.clear()
        self.position_visits.clear()
        self.position_action_attempts.clear()
        self.position_transitions.clear()
        self.state_click_attempts.clear()
        self.recent_fingerprints.clear()
        self.candidate_cache.clear()
        self.planned_movement_actions.clear()
        self.avatar_object = None
        self.current_position_key = None
        self.preferred_heading = None
        self.steps_in_level = 0
        self.no_change_streak = 0
        self.loop_streak = 0
        self.same_action_streak = 0

    def _movement_step_size(self, analysis: FrameAnalysis) -> int:
        step = int(analysis.lattice_step)
        if 1 < step <= 16:
            return step
        return 4

    def _is_hud_like_diff(self, frame_diff: FrameDiff) -> bool:
        bbox = frame_diff.bbox
        if bbox is None:
            return False

        if bbox.height <= 4 and bbox.width >= 8:
            if bbox.top <= 3 or bbox.bottom >= 60:
                return True
        return False

    def _is_significant_movement(
        self,
        action_id: int | None,
        analysis: FrameAnalysis,
        frame_diff: FrameDiff,
    ) -> bool:
        if action_id not in MOVEMENT_ACTION_IDS:
            return False
        if frame_diff.bbox is None:
            return False
        if self._is_hud_like_diff(frame_diff):
            return False

        movement_threshold = max(4, self._movement_step_size(analysis))
        if frame_diff.changed_pixels < movement_threshold:
            return False

        bbox = frame_diff.bbox
        return bbox.width >= 2 or bbox.height >= 2 or frame_diff.changed_pixels >= 8

    def _expected_avatar_center(
        self,
        action_id: int | None,
        analysis: FrameAnalysis,
    ) -> tuple[int, int] | None:
        if action_id not in MOVEMENT_ACTION_IDS or self.avatar_object is None:
            return None

        dx, dy = MOVEMENT_DELTAS[action_id]
        step = self._movement_step_size(analysis)
        x = self.avatar_object.centroid[0] + dx * step
        y = self.avatar_object.centroid[1] + dy * step
        return (max(0, min(63, x)), max(0, min(63, y)))

    def _select_avatar_object(
        self,
        analysis: FrameAnalysis,
        frame_diff: FrameDiff | None,
        action_id: int | None,
        previous_analysis: FrameAnalysis | None = None,
    ) -> FrameObject | None:
        candidates = [
            obj
            for obj in analysis.objects
            if not obj.touches_border and 2 <= obj.area <= 96
        ]
        if not candidates:
            return None

        diff_center = frame_diff.bbox.center if frame_diff and frame_diff.bbox else None
        expected_center = self._expected_avatar_center(action_id, analysis)
        previous_avatar = self.avatar_object
        previous_analysis = previous_analysis or self.current_analysis
        step = self._movement_step_size(analysis)

        def score_object(obj: FrameObject) -> float:
            score = 0.0
            width = obj.bbox.width
            height = obj.bbox.height
            center_x, center_y = obj.centroid

            # Mobile agents are usually compact, interior objects.
            score += max(0.0, 14.0 - obj.area * 0.3)
            if width <= step * 2 and height <= step * 2:
                score += 8.0
            if width <= 12 and height <= 12:
                score += 4.0
            if obj.bbox.bottom >= 60 and obj.bbox.height <= 3:
                score -= 10.0
            if obj.bbox.top <= 3 and obj.bbox.height <= 3:
                score -= 10.0
            if max(width, height) >= max(4, min(width, height) * 4):
                score -= 6.0

            if diff_center is not None:
                if frame_diff is not None and frame_diff.bbox is not None:
                    dx = 0
                    if center_x < frame_diff.bbox.left:
                        dx = frame_diff.bbox.left - center_x
                    elif center_x > frame_diff.bbox.right:
                        dx = center_x - frame_diff.bbox.right

                    dy = 0
                    if center_y < frame_diff.bbox.top:
                        dy = frame_diff.bbox.top - center_y
                    elif center_y > frame_diff.bbox.bottom:
                        dy = center_y - frame_diff.bbox.bottom

                    rect_distance = dx + dy
                    score += max(0.0, 30.0 - rect_distance * 2.6)

                distance = abs(center_x - diff_center[0]) + abs(center_y - diff_center[1])
                score += max(0.0, 28.0 - distance * 1.4)
                if (
                    frame_diff is not None
                    and frame_diff.bbox is not None
                    and frame_diff.bbox.left <= center_x <= frame_diff.bbox.right
                    and frame_diff.bbox.top <= center_y <= frame_diff.bbox.bottom
                ):
                    score += 8.0

            if expected_center is not None:
                distance = abs(center_x - expected_center[0]) + abs(center_y - expected_center[1])
                score += max(0.0, 22.0 - distance * 1.6)

            if previous_analysis is not None:
                previous_candidates = [
                    prev
                    for prev in previous_analysis.objects
                    if not prev.touches_border and prev.color == obj.color and 2 <= prev.area <= 96
                ]
                if previous_candidates:
                    best_previous = min(
                        previous_candidates,
                        key=lambda prev: (
                            abs(prev.area - obj.area) * 0.75
                            + abs(prev.centroid[0] - center_x)
                            + abs(prev.centroid[1] - center_y)
                        ),
                    )
                    shift = abs(best_previous.centroid[0] - center_x) + abs(
                        best_previous.centroid[1] - center_y
                    )
                    area_gap = abs(best_previous.area - obj.area)
                    if 0 < shift <= step * 4 and area_gap <= max(6, obj.area):
                        score += max(0.0, 24.0 - shift * 1.6 - area_gap * 0.4)
                    elif shift == 0:
                        score -= 2.0

            if previous_avatar is not None:
                if obj.color == previous_avatar.color:
                    score += 6.0 if diff_center is not None else 10.0
                area_gap = abs(obj.area - previous_avatar.area)
                area_score = 4.0 if diff_center is not None else 8.0
                score += max(0.0, area_score - area_gap * 0.75)
                continuity = abs(center_x - previous_avatar.centroid[0]) + abs(
                    center_y - previous_avatar.centroid[1]
                )
                continuity_base = 10.0 if diff_center is not None else 18.0
                continuity_scale = 0.45 if diff_center is not None else 0.7
                score += max(0.0, continuity_base - continuity * continuity_scale)
            else:
                center_bias = abs(center_x - 32) + abs(center_y - 32)
                score += max(0.0, 10.0 - center_bias * 0.15)

            return score

        best = max(candidates, key=score_object)
        return best if score_object(best) >= 12.0 else None

    def _position_key(
        self,
        avatar_object: FrameObject | None,
        analysis: FrameAnalysis,
    ) -> str | None:
        if avatar_object is None:
            return None

        step = self._movement_step_size(analysis)
        qx = int(round(avatar_object.centroid[0] / float(step)))
        qy = int(round(avatar_object.centroid[1] / float(step)))
        return f"pos:{avatar_object.color}:{qx}:{qy}"

    def _ordered_movement_actions(self, available_ids: set[int]) -> list[int]:
        movement_ids = [action_id for action_id in MOVEMENT_ACTION_IDS if action_id in available_ids]
        if not movement_ids:
            return []

        if self.preferred_heading in movement_ids:
            preferred_order = [
                self.preferred_heading,
                TURN_LEFT[self.preferred_heading],
                TURN_RIGHT[self.preferred_heading],
                REVERSE_MOVEMENT[self.preferred_heading],
            ]
            ordered = [action_id for action_id in preferred_order if action_id in movement_ids]
            for action_id in PROBE_ORDER:
                if action_id in movement_ids and action_id not in ordered:
                    ordered.append(action_id)
            return ordered

        return [action_id for action_id in PROBE_ORDER if action_id in movement_ids]

    def _has_untried_movement(self, position_key: str, available_ids: set[int]) -> bool:
        movement_ids = [action_id for action_id in MOVEMENT_ACTION_IDS if action_id in available_ids]
        attempted = self.position_action_attempts[position_key]
        return any(attempted[action_id] == 0 for action_id in movement_ids)

    def _path_to_movement_frontier(
        self,
        current_position_key: str,
        available_ids: set[int],
    ) -> list[int]:
        if self._has_untried_movement(current_position_key, available_ids):
            return []

        queue: deque[tuple[str, list[int]]] = deque([(current_position_key, [])])
        seen = {current_position_key}

        while queue:
            position_key, path = queue.popleft()
            transitions = self.position_transitions.get(position_key, {})
            for action_id in self._ordered_movement_actions(available_ids):
                next_position_key = transitions.get(action_id)
                if not next_position_key or next_position_key in seen:
                    continue
                next_path = path + [action_id]
                if self._has_untried_movement(next_position_key, available_ids):
                    return next_path
                seen.add(next_position_key)
                queue.append((next_position_key, next_path))

        return []

    def _movement_reference_key(self, state_fingerprint: str) -> str:
        return self.current_position_key or state_fingerprint

    def observe(self, latest_frame: FrameData) -> tuple[FrameAnalysis, FrameDiff]:
        analysis = analyse_frame(latest_frame.frame[0])
        previous_analysis = self.current_analysis
        previous_position_key = self.current_position_key
        level_delta = latest_frame.levels_completed - self.levels_completed
        frame_diff = diff_frames(
            previous_analysis.grid if previous_analysis is not None else None,
            analysis.grid,
        )
        same_level = not latest_frame.full_reset and level_delta == 0

        if previous_analysis is not None and self.last_action_id is not None:
            previous_fp = previous_analysis.fingerprint
            self.state_action_attempts[previous_fp][self.last_action_id] += 1

            if self.last_action_id == GameAction.ACTION6.value and self.last_click_target:
                self.state_click_attempts[previous_fp].add(self.last_click_target)

            if self.last_action_id in MOVEMENT_ACTION_IDS:
                movement_ref_key = previous_position_key or previous_fp
                self.position_action_attempts[movement_ref_key][self.last_action_id] += 1

            impactful = frame_diff.changed_pixels > 0 or level_delta > 0
            significant_movement = False
            if self.last_action_id in MOVEMENT_ACTION_IDS:
                significant_movement = same_level and self._is_significant_movement(
                    self.last_action_id,
                    analysis,
                    frame_diff,
                )
                impactful = significant_movement or level_delta > 0

            if impactful:
                self.action_success[self.last_action_id] += 1
                self.no_change_streak = 0
            else:
                self.action_fail[self.last_action_id] += 1
                self.no_change_streak += 1

            if significant_movement:
                new_avatar = self._select_avatar_object(
                    analysis,
                    frame_diff,
                    self.last_action_id,
                    previous_analysis=previous_analysis,
                )
                new_position_key = self._position_key(new_avatar, analysis)
                if (
                    previous_position_key
                    and new_position_key
                    and previous_position_key != new_position_key
                ):
                    self.position_transitions[previous_position_key][
                        self.last_action_id
                    ] = new_position_key
                if new_avatar is not None:
                    self.avatar_object = new_avatar
                    self.current_position_key = new_position_key
                    self.preferred_heading = self.last_action_id
                    if (
                        self.planned_movement_actions
                        and self.planned_movement_actions[0] == self.last_action_id
                    ):
                        self.planned_movement_actions.popleft()
                    elif self.planned_movement_actions:
                        self.planned_movement_actions.clear()
            elif self.last_action_id in MOVEMENT_ACTION_IDS and self.planned_movement_actions:
                self.planned_movement_actions.clear()

            if analysis.fingerprint in self.recent_fingerprints:
                self.loop_streak += 1
            else:
                self.loop_streak = 0

            self.steps_in_level += 1

        if latest_frame.full_reset or level_delta > 0:
            self._reset_level_state()

        refresh_diff = None
        if (
            same_level
            and frame_diff.changed_pixels >= max(4, self._movement_step_size(analysis))
            and not self._is_hud_like_diff(frame_diff)
        ):
            refresh_diff = frame_diff
        if self.avatar_object is None:
            if refresh_diff is not None:
                self.avatar_object = self._select_avatar_object(
                    analysis,
                    refresh_diff,
                    self.last_action_id if self.last_action_id in MOVEMENT_ACTION_IDS else None,
                    previous_analysis=previous_analysis,
                )
        else:
            refreshed_avatar = self._select_avatar_object(
                analysis,
                refresh_diff,
                None,
                previous_analysis=previous_analysis,
            )
            if refreshed_avatar is not None:
                self.avatar_object = refreshed_avatar

        self.current_position_key = self._position_key(self.avatar_object, analysis)
        self.current_analysis = analysis
        self.current_diff = frame_diff
        self.levels_completed = latest_frame.levels_completed
        self.recent_fingerprints.append(analysis.fingerprint)
        self.state_visits[analysis.fingerprint] += 1
        if self.current_position_key is not None:
            self.position_visits[self.current_position_key] += 1
        return analysis, frame_diff

    def _candidate_clicks(
        self,
        state_fingerprint: str,
        analysis: FrameAnalysis,
        frame_diff: FrameDiff,
    ) -> list[ClickCandidate]:
        if state_fingerprint not in self.candidate_cache:
            self.candidate_cache[state_fingerprint] = generate_click_candidates(
                analysis,
                diff=frame_diff,
            )

        candidates = self.candidate_cache[state_fingerprint]
        attempted = self.state_click_attempts[state_fingerprint]
        remaining = [
            candidate
            for candidate in candidates
            if (candidate.x, candidate.y) not in attempted
        ]
        return remaining if remaining else candidates[: min(8, len(candidates))]

    def _should_undo(self, available_ids: set[int]) -> bool:
        if GameAction.ACTION7.value not in available_ids:
            return False
        if self.last_action_id in {None, GameAction.ACTION7.value}:
            return False
        if self.steps_in_level < 2:
            return False
        return self.loop_streak >= 2 or self.no_change_streak >= 3

    def _movement_proposals(
        self,
        available_ids: set[int],
        state_fingerprint: str,
    ) -> list[ActionProposal]:
        proposals: list[ActionProposal] = []
        movement_ids = self._ordered_movement_actions(available_ids)
        if not movement_ids:
            return proposals

        reference_key = self._movement_reference_key(state_fingerprint)
        using_position_memory = reference_key != state_fingerprint
        attempted = (
            self.position_action_attempts[reference_key]
            if using_position_memory
            else self.state_action_attempts[state_fingerprint]
        )
        revisit_count = (
            self.position_visits[reference_key]
            if using_position_memory
            else self.state_visits[state_fingerprint]
        )
        untried_here = any(attempted[action_id] == 0 for action_id in movement_ids)

        if using_position_memory and not untried_here and not self.planned_movement_actions:
            frontier_path = self._path_to_movement_frontier(reference_key, available_ids)
            if frontier_path:
                self.planned_movement_actions = deque(frontier_path)

        planned_action_id = (
            self.planned_movement_actions[0] if self.planned_movement_actions else None
        )

        for probe_index, action_id in enumerate(movement_ids):
            score = 22.0
            if self.steps_in_level < len(movement_ids) and movement_ids[self.steps_in_level] == action_id:
                score += 8.0
            score += min(self.action_success[action_id], 12) * 1.25
            score -= min(self.action_fail[action_id], 12) * 0.75
            score -= attempted[action_id] * 4.5
            score -= max(0, revisit_count - 1) * 1.35

            if attempted[action_id] == 0:
                score += 15.0 if using_position_memory else 8.0
            if using_position_memory and self.preferred_heading == action_id:
                score += 4.5
            if planned_action_id == action_id:
                score += 22.0
            elif planned_action_id is not None:
                score -= 4.0

            if self.last_action_id == REVERSE_MOVEMENT.get(action_id) and self.no_change_streak == 0:
                score -= 2.5
            if self.last_action_id == action_id and self.no_change_streak > 0:
                score -= 5.5
            if self.last_action_id == action_id:
                score -= self.same_action_streak * 1.5
            if self.no_change_streak >= 1 and REVERSE_MOVEMENT.get(self.last_action_id) == action_id:
                score += 8.0
            if self.no_change_streak >= 2 and self.last_action_id != action_id:
                score += 4.0
            if self.same_action_streak >= 8 and self.last_action_id == action_id:
                score -= 12.0
            if using_position_memory and untried_here and attempted[action_id] > 0:
                score -= 5.0

            reason = "probe movement action"
            if planned_action_id == action_id:
                reason = "follow known path to frontier"
            elif attempted[action_id] == 0 and using_position_memory:
                reason = "explore untried movement from position"
            proposals.append(
                ActionProposal(
                    action_id=action_id,
                    score=score - probe_index * 0.1,
                    reason=f"{reason} {action_id}",
                )
            )

        return proposals

    def _action5_proposal(
        self,
        available_ids: set[int],
        state_fingerprint: str,
    ) -> list[ActionProposal]:
        if GameAction.ACTION5.value not in available_ids:
            return []

        attempted = self.state_action_attempts[state_fingerprint][GameAction.ACTION5.value]
        score = 16.0
        score += min(self.action_success[GameAction.ACTION5.value], 8) * 1.5
        score -= min(self.action_fail[GameAction.ACTION5.value], 8) * 0.75
        score -= attempted * 6.0

        if not any(action_id in available_ids for action_id in MOVEMENT_ACTION_IDS):
            score += 8.0
        if self.no_change_streak >= 1:
            score += 5.5

        return [
            ActionProposal(
                action_id=GameAction.ACTION5.value,
                score=score,
                reason="try contextual interact action",
            )
        ]

    def _click_proposals(
        self,
        available_ids: set[int],
        state_fingerprint: str,
        analysis: FrameAnalysis,
        frame_diff: FrameDiff,
    ) -> list[ActionProposal]:
        if GameAction.ACTION6.value not in available_ids:
            return []

        candidates = self._candidate_clicks(state_fingerprint, analysis, frame_diff)
        if not candidates:
            return []

        candidate = candidates[0]
        movement_available = any(action_id in available_ids for action_id in MOVEMENT_ACTION_IDS)

        score = 55.0 if not movement_available else 22.0
        if movement_available and self.steps_in_level >= 3:
            score += 14.0
        if self.no_change_streak >= 2:
            score += 8.0

        score += candidate.priority
        score -= len(self.state_click_attempts[state_fingerprint]) * 0.75

        return [
            ActionProposal(
                action_id=GameAction.ACTION6.value,
                score=score,
                reason=f"click {candidate.reason}",
                data={"x": candidate.x, "y": candidate.y},
            )
        ]

    def _fallback_proposal(
        self,
        available_ids: set[int],
        state_fingerprint: str,
        analysis: FrameAnalysis,
        frame_diff: FrameDiff,
    ) -> ActionProposal:
        if GameAction.ACTION6.value in available_ids:
            click_proposals = self._click_proposals(
                available_ids,
                state_fingerprint,
                analysis,
                frame_diff,
            )
            if click_proposals:
                fallback = click_proposals[0]
                return ActionProposal(
                    action_id=fallback.action_id,
                    score=fallback.score,
                    reason=f"{fallback.reason} (fallback)",
                    data=fallback.data,
                )

        action_id = min(available_ids)
        return ActionProposal(
            action_id=action_id,
            score=0.0,
            reason=f"default to available action {action_id}",
        )

    def _materialize_action(self, proposal: ActionProposal) -> GameAction:
        previous_action_id = self.last_action_id
        action = GameAction.from_id(proposal.action_id)
        if proposal.data:
            action.set_data(proposal.data)
            self.last_click_target = (proposal.data["x"], proposal.data["y"])
        else:
            self.last_click_target = None

        if proposal.action_id in MOVEMENT_ACTION_IDS:
            if (
                self.planned_movement_actions
                and self.planned_movement_actions[0] == proposal.action_id
            ):
                self.planned_movement_actions.popleft()
            elif self.planned_movement_actions:
                self.planned_movement_actions.clear()
        elif self.planned_movement_actions:
            self.planned_movement_actions.clear()

        action.reasoning = {
            "policy": "heuristic_baseline",
            "reason": proposal.reason,
            "score": round(proposal.score, 2),
        }
        if previous_action_id == proposal.action_id:
            self.same_action_streak += 1
        else:
            self.same_action_streak = 1
        self.last_action_id = proposal.action_id
        return action

    def decide(self, latest_frame: FrameData) -> GameAction:
        analysis, frame_diff = self.observe(latest_frame)
        available_ids = {int(action_id) for action_id in latest_frame.available_actions}

        if not available_ids:
            fallback = ActionProposal(
                action_id=GameAction.RESET.value,
                score=0.0,
                reason="no available actions, reset",
            )
            return self._materialize_action(fallback)

        if self._should_undo(available_ids):
            return self._materialize_action(
                ActionProposal(
                    action_id=GameAction.ACTION7.value,
                    score=80.0,
                    reason="undo after loop or repeated no-op streak",
                )
            )

        state_fingerprint = analysis.fingerprint
        proposals: list[ActionProposal] = []
        proposals.extend(self._movement_proposals(available_ids, state_fingerprint))
        proposals.extend(self._action5_proposal(available_ids, state_fingerprint))
        proposals.extend(
            self._click_proposals(
                available_ids,
                state_fingerprint,
                analysis,
                frame_diff,
            )
        )

        if not proposals:
            proposals.append(
                self._fallback_proposal(
                    available_ids,
                    state_fingerprint,
                    analysis,
                    frame_diff,
                )
            )

        best = max(
            proposals,
            key=lambda proposal: (proposal.score, -proposal.action_id),
        )
        return self._materialize_action(best)
