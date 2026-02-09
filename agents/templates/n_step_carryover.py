import textwrap
from typing import Any

from arcengine import FrameData, GameAction

from .simple_memory_carryover import SimpleMemoryCarryover


class NStepCarryover(SimpleMemoryCarryover):
    """LLM agent with configurable N-step rolling prompt context."""

    DEFAULT_MODEL = "gemini-3-flash-preview"
    REASONING_EFFORT = "high"
    MODEL_REQUIRES_TOOLS = True

    # Total turns visible to the model each step, including current turn.
    # Example: 2 => previous turn + current turn.
    N_STEP_WINDOW = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._turn_history: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        return f"{super().name}.n-step-carryover"

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if getattr(latest_frame, "full_reset", False):
            self._turn_history.clear()
            self._locked_available_actions = None
        self._trim_turn_history()

        action = super().choose_action(frames, latest_frame)
        self._append_turn_snapshot(latest_frame, action, self.carryover_memory)
        return action

    def _build_system_prompt(self, available_actions: list[GameAction]) -> str:
        choose_rule = "Choose exactly one action from PERSISTENT_AVAILABLE_ACTIONS."
        action6_available = GameAction.ACTION6 in available_actions
        available_names = [action.name for action in available_actions]
        available_actions_block = "\n".join(
            f"- {name}" for name in available_names
        ) or "- <none>"
        coordinate_rule = ""
        if action6_available:
            coordinate_rule = (
                "- If you choose ACTION6, provide x and y "
                "(both integers in [0,63])."
            )

        previous_turns_visible = max(0, self.N_STEP_WINDOW - 1)
        return textwrap.dedent(
            """
You are a turn-based game-playing agent.

Context rules:
- You do NOT have hidden memory.
- You only know what is shown in MEMORY_FROM_PREVIOUS_TURN and PREVIOUS_TURN_WINDOW.
- PREVIOUS_TURN_WINDOW shows up to {previous_turns_visible} prior turns.

PERSISTENT_AVAILABLE_ACTIONS (constant for this game):
{available_actions_block}

Action rules:
- {choose_rule}
{coordinate_rule}

Output rules:
- You must reply by calling submit_action_and_memory exactly once.
            """.format(
                previous_turns_visible=previous_turns_visible,
                available_actions_block=available_actions_block,
                choose_rule=choose_rule,
                coordinate_rule=coordinate_rule,
            )
        ).strip()

    def _build_user_prompt(self, latest_frame: FrameData) -> str:
        memory_text = self.carryover_memory

        return textwrap.dedent(
            """
FRAMES:
{frames}

MEMORY_FROM_PREVIOUS_TURN:
{memory}

PREVIOUS_TURN_WINDOW:
{previous_turn_window}

What action would you like to take?
Frames are formatted as CSV rows for each grid (comma-separated integer values).
Also write all memory you want to persist for the next turn in memory_for_next_turn.
            """.format(
                frames=self._pretty_print_3d(latest_frame.frame),
                memory=memory_text,
                previous_turn_window=self._format_previous_turn_window(),
            )
        ).strip()

    def _format_previous_turn_window(self) -> str:
        if not self._turn_history:
            return "<empty>"

        ordered_turns = sorted(
            self._turn_history, key=lambda turn: int(turn.get("turn_index", 0))
        )
        lines: list[str] = ["(chronological: oldest to newest)"]
        for turn in ordered_turns:
            turn_index = int(turn.get("turn_index", 0))
            lines.extend(
                [
                    f"Turn {turn_index}:",
                    "FRAMES:",
                    turn["frames"],
                    "ACTION_TAKEN:",
                    turn["action"],
                    "MEMORY_SAVED_FOR_NEXT_TURN:",
                    turn["memory"],
                    "",
                ]
            )
        return "\n".join(lines).strip()

    def _append_turn_snapshot(
        self, latest_frame: FrameData, action: GameAction, memory_after: str
    ) -> None:
        action_token = action.name
        if action.name == "ACTION6":
            action_data = action.action_data.model_dump()
            x = action_data.get("x")
            y = action_data.get("y")
            if x is not None and y is not None:
                action_token = f"{action_token} (x={x}, y={y})"

        self._turn_history.append(
            {
                "turn_index": self.action_counter,
                "frames": self._pretty_print_3d(latest_frame.frame) or "<empty>",
                "action": action_token,
                "memory": memory_after if memory_after else "<empty>",
            }
        )
        self._trim_turn_history()

    def _trim_turn_history(self) -> None:
        max_previous_turns = max(0, self.N_STEP_WINDOW - 1)
        if max_previous_turns == 0:
            self._turn_history.clear()
            return
        if len(self._turn_history) > max_previous_turns:
            self._turn_history = self._turn_history[-max_previous_turns:]

    def _build_action_reasoning(
        self,
        *,
        parse_status: str,
        action_note: str,
        response: Any | None,
        parsed_args: dict[str, Any] | None,
        memory_after: str,
        error: str | None = None,
        parse_failures_in_row: int | None = None,
        memory_cleared: bool | None = None,
    ) -> dict[str, Any]:
        payload = super()._build_action_reasoning(
            parse_status=parse_status,
            action_note=action_note,
            response=response,
            parsed_args=parsed_args,
            memory_after=memory_after,
            error=error,
            parse_failures_in_row=parse_failures_in_row,
            memory_cleared=memory_cleared,
        )
        payload["agent_type"] = "n_step_carryover"
        payload["n_step_window"] = self.N_STEP_WINDOW
        payload["previous_turns_in_prompt"] = len(self._turn_history)
        return payload
