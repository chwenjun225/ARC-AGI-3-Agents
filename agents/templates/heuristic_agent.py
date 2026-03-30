from __future__ import annotations

from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent
from ..heuristics import HeuristicBrain


class HeuristicAgent(Agent):
    """Offline-first heuristic baseline for ARC-AGI-3."""

    MAX_ACTIONS = 1800

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.brain = HeuristicBrain()

    @property
    def name(self) -> str:
        return f"{super().name}.offline-heuristic.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        if latest_frame.state in {GameState.NOT_PLAYED, GameState.GAME_OVER}:
            self.brain.reset_for_new_attempt()
            action = GameAction.RESET
            action.reasoning = {
                "policy": "heuristic_baseline",
                "reason": "start or restart the environment",
            }
            return action

        return self.brain.decide(latest_frame)
