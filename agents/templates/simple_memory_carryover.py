import base64
import io
import json
import logging
import os
import re
import textwrap
import uuid
from typing import Any

from arcengine import FrameData, GameAction, GameState
from openai import AuthenticationError, BadRequestError, OpenAI as OpenAIClient

from ..agent import Agent

logger = logging.getLogger(__name__)


class SimpleMemoryCarryover(Agent):
    """LLM agent with explicit one-turn memory carryover only."""

    MAX_ACTIONS = 50
    ACTION6_COORD_RETRY_COUNT = 2
    ACTION_MAPPINGS = {
        "RESET": "RESET - Reset current level",
        "ACTION1": "ACTION1 - Up arrow key, W",
        "ACTION2": "ACTION2 - Down arrow key, S",
        "ACTION3": "ACTION3 - Left arrow key, A",
        "ACTION4": "ACTION4 - Right arrow key, D",
        "ACTION5": "ACTION5 - Spacebar",
        "ACTION6": "ACTION6 - Click",
        "ACTION7": "ACTION7 (special action, Undo) - Z",
    }
    ARC_COLOR_PALETTE = (
        (255, 255, 255),  # 0 White
        (204, 204, 204),  # 1 Light Gray
        (153, 153, 153),  # 2 Medium Gray
        (102, 102, 102),  # 3 Dark Gray
        (51, 51, 51),     # 4 Very Dark Gray
        (0, 0, 0),        # 5 Black
        (229, 58, 163),   # 6 Pink
        (255, 123, 204),  # 7 Light Pink
        (249, 60, 49),    # 8 Red
        (30, 147, 255),   # 9 Blue
        (136, 216, 241),  # 10 Light Blue
        (255, 220, 0),    # 11 Yellow
        (255, 133, 27),   # 12 Orange
        (146, 18, 49),    # 13 Dark Red
        (79, 204, 48),    # 14 Green
        (163, 86, 214),   # 15 Purple
    )

    RECORDINGS_DIR_ENV = "RECORDINGS_DIR"
    BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = "google/gemini-3-flash-preview"
    REASONING_EFFORT = "high"

    # Multimodal
    USE_IMAGE_INPUT = False
    IMAGE_INPUT_SIZE = 512

    ENABLE_CHAT_LOG = False # Log to a readable .md file

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing required env var: OPENROUTER_API_KEY")
        self.base_url = self.BASE_URL
        self.model = self.MODEL
        self.carryover_memory = ""
        self._pending_reasoning: dict[str, Any] | None = None
        self.consecutive_parse_failures = 0
        self._last_levels_completed: int | None = None
        self._log_conversation = self.ENABLE_CHAT_LOG
        self._chat_log_path: str | None = None
        self._conversation_log_session_id = uuid.uuid4().hex
        self._conversation_log_announced = False
        self.client = OpenAIClient(
            api_key=api_key,
            base_url=self.base_url,
        )
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        model_name = getattr(self, "model", self.MODEL)
        sanitized_model_name = model_name.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}.memory-carryover"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        available_actions = self._available_actions(latest_frame)
        fallback_action = self._deterministic_fallback_action(latest_frame)
        max_attempts = 1 + max(0, int(self.ACTION6_COORD_RETRY_COUNT))
        system_prompt = self._build_system_prompt(
            available_actions, latest_state=latest_frame.state
        )
        user_prompt = self._build_user_prompt(latest_frame)
        response: Any | None = None
        request_payload: dict[str, Any] | None = None
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.REASONING_EFFORT:
            create_kwargs["extra_body"] = {
                "reasoning": {"effort": self.REASONING_EFFORT}
            }
        request_payload = create_kwargs

        last_error: str | None = None
        for attempt_index in range(max_attempts):
            try:
                response = self.client.chat.completions.create(**create_kwargs)
                args = self._extract_action_and_memory_from_response(response)

                action, action_note = self._action_from_arguments(
                    args=args,
                    fallback_action=fallback_action,
                    available_actions=available_actions,
                )
                retry_error = self._retry_error_from_action_note(action_note)
                if retry_error is not None:
                    if attempt_index < (max_attempts - 1):
                        raise ValueError(retry_error)
                    raise ValueError(f"{retry_error} after retries.")

                self.carryover_memory = self._extract_memory(args)
                self.consecutive_parse_failures = 0
                self._pending_reasoning = self._build_action_reasoning(
                    parse_status="ok",
                    action_note=action_note,
                    response=response,
                    parsed_args=args,
                    memory_after=self.carryover_memory,
                )
                self._log_turn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    parsed_args=args,
                    action=action,
                    error=None,
                    memory_after=self.carryover_memory,
                    request_payload=request_payload,
                )
                return action
            except AuthenticationError as e:
                error = f"AuthenticationError: {e}"
                self._log_turn(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=response,
                    parsed_args=None,
                    action=fallback_action,
                    error=error,
                    memory_after=self.carryover_memory,
                    request_payload=request_payload,
                )
                raise RuntimeError(
                    "OpenRouter authentication failed. Check OPENROUTER_API_KEY."
                ) from e
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, BadRequestError) as e:
                last_error = f"{type(e).__name__}: {e}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            if attempt_index < (max_attempts - 1):
                logger.warning(
                    "Action selection failed; retrying with the same prompt (attempt %s/%s). Error: %s",
                    attempt_index + 1,
                    max_attempts,
                    last_error,
                )
                continue

        error = last_error or "Action selection failed after retries."
        action = self._handle_parse_failure(fallback_action, error, response=response)
        self._log_turn(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            parsed_args=None,
            action=action,
            error=error,
            memory_after=self.carryover_memory,
            request_payload=request_payload,
        )
        return action

    def _build_system_prompt(
        self, available_actions: list[GameAction], latest_state: GameState
    ) -> str:
        available_names = [action.name for action in available_actions]
        available_actions_inline = ", ".join(available_names) or "<none>"
        non_click_action_names = [
            action_name
            for action_name in available_names
            if action_name not in (GameAction.ACTION6.name, GameAction.RESET.name)
        ]
        non_click_actions_inline = ", ".join(non_click_action_names) or "<none>"
        action_mappings = self._available_action_mappings(available_actions)
        action6_available = GameAction.ACTION6 in available_actions
        game_state = latest_state.name if isinstance(latest_state, GameState) else str(latest_state)
        game_state_rule = (
            "- Current state is GAME_OVER. The previous run is lost.\n"
            "- In GAME_OVER, RESET is the only allowed action.\n"
            if latest_state is GameState.GAME_OVER
            else ""
        )
        action6_rule = ""
        action6_format_rule = ""
        non_click_format_rule = ""
        if non_click_action_names:
            non_click_format_rule = (
                f"- For non-click actions, use `<action>ACTION_NAME</action>` where ACTION_NAME is one of ({non_click_actions_inline})\n"
            )
        if action6_available:
            action6_rule = (
                "- If you choose ACTION6, you must also provide integer x and y in [0,63].\n"
            )
            action6_format_rule = (
                "- For ACTION6, provide coordinates as `<action>ACTION6 x=<int> y=<int></action>`\n"
            )
        return textwrap.dedent(
            """
You are a turn-based game-playing agent. Your task is to analyze the current game state, choose an appropriate action, and manage your memory effectively across turns.

## Critical Memory Constraint

You do NOT have persistent hidden memory. Your memory works as follows:
- You only have access to information explicitly provided in MEMORY_FROM_PREVIOUS_TURN
- When you write `<memory>...</memory>`, it COMPLETELY REPLACES all previous memory
- If you need to compare the current state to a previous state, you must have stored the relevant previous state information in your memory during the last turn
- Any information not included in your `<memory>` output will be permanently lost

## Your Task

1. Analyze the current game state (provided as frames)
2. Review your memory from the previous turn
3. Choose exactly one action from the available options
4. Determine what information you want to remember for future turns
5. Submit your chosen action and memory

## Available Actions

Choose exactly ONE action from ({available_actions_inline}).
Available action mappings:
{action_mappings}
Current game state: {game_state}
{game_state_rule}
{action6_rule}
## Instructions for Your Response

Before making your final decision, work through your reasoning in <game_analysis> tags inside your thinking block:

1. Current State Understanding: Parse the frames data and note key observations.
2. Context Review: Review the information you stored in memory from the previous turn.
3. State Comparison: If memory contains previous-state information, identify what changed and what stayed the same.
4. Memory Planning: Decide what to keep, discard, and add to your next-turn `<memory>`.
5. Action Selection: Choose the single most appropriate action.

Your final output must be plain text containing exactly:
1) one <action>...</action> block
2) one <memory>...</memory> block

Formatting rules:
{non_click_format_rule}
{action6_format_rule}
- Put all carryover memory in `<memory>...</memory>` (this fully replaces previous memory)
- Do not use tool calls or function calls
- Do not output pseudo-code or plain text action descriptions outside these tags
            """.format(
                available_actions_inline=available_actions_inline,
                action_mappings=action_mappings,
                game_state=game_state,
                game_state_rule=game_state_rule,
                action6_rule=action6_rule,
                non_click_format_rule=non_click_format_rule,
                action6_format_rule=action6_format_rule,
            )
        ).strip()

    def _build_user_prompt(self, latest_frame: FrameData) -> str | list[dict[str, Any]]:
        memory_text = self.carryover_memory
        turn_data_text = self._build_turn_data_text(latest_frame)
        if self.USE_IMAGE_INPUT:
            prompt_content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": f"TURN_DATA:\n{turn_data_text}",
                },
                {
                    "type": "text",
                    "text": "FRAMES (attached as images in Grid order):",
                }
            ]
            for index, grid in enumerate(latest_frame.frame):
                prompt_content.append({"type": "text", "text": f"Grid {index}:"})
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._grid_to_data_url(grid)},
                    }
                )
            prompt_content.append(
                {
                    "type": "text",
                    "text": f"MEMORY_FROM_PREVIOUS_TURN:\n{memory_text}",
                }
            )
            return prompt_content

        return textwrap.dedent(
            """
TURN_DATA:
{turn_data}

FRAMES:
{frames}

MEMORY_FROM_PREVIOUS_TURN:
{memory}
            """.format(
                turn_data=turn_data_text,
                frames=self._pretty_print_3d(latest_frame.frame),
                memory=memory_text,
            )
        ).strip()

    def _build_turn_data_text(
        self, latest_frame: FrameData
    ) -> str:
        state_name = self._frame_state_name(latest_frame)
        levels_completed = self._coerce_optional_int(
            getattr(latest_frame, "levels_completed", None)
        )
        win_levels = self._coerce_optional_int(getattr(latest_frame, "win_levels", None))
        current_level_index = self._current_level_index(levels_completed, win_levels)

        lines = [
            f"- game_state: {state_name}",
            f"- current_level_index: {self._format_optional_int(current_level_index)}",
        ]
        progress_event = self._detect_progress_event(levels_completed)
        if progress_event == "level":
            lines.append("New level reached!")
        return "\n".join(lines)

    def _frame_state_name(self, frame: FrameData) -> str:
        state = getattr(frame, "state", None)
        if isinstance(state, GameState):
            return state.name
        return str(state)

    def _coerce_optional_int(self, value: Any) -> int | None:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _format_optional_int(self, value: int | None) -> str:
        if value is None:
            return "unknown"
        return str(value)

    def _current_level_index(
        self, levels_completed: int | None, win_levels: int | None
    ) -> int | None:
        if levels_completed is None:
            return None
        candidate = levels_completed + 1
        if win_levels is not None and win_levels > 0:
            return min(candidate, win_levels)
        return candidate

    def _detect_progress_event(self, levels_completed: int | None) -> str | None:
        progress_event: str | None = None
        if (
            levels_completed is not None
            and self._last_levels_completed is not None
            and levels_completed > self._last_levels_completed
        ):
            progress_event = "level"

        self._last_levels_completed = levels_completed
        return progress_event

    def _parse_action_token(self, action_token: str) -> GameAction:
        token = action_token.strip().upper()
        if not token:
            raise ValueError("missing action_name")
        return GameAction.from_name(token)

    def _available_action_mappings(self, available_actions: list[GameAction]) -> str:
        lines: list[str] = []
        for action in available_actions:
            action_name = action.name
            label = self.ACTION_MAPPINGS.get(action_name)
            if label is None:
                lines.append(f"- {action_name}")
            else:
                lines.append(f"- {label}")
        if not lines:
            return "- <none>"
        return "\n".join(lines)

    def _extract_action_and_memory_from_response(self, response: Any) -> dict[str, Any]:
        response_text = self._extract_response_text(response)
        action_text = self._extract_tag_content(response_text, "action")
        memory_text = self._extract_tag_content(response_text, "memory")
        args = self._parse_action_text(action_text)
        args["memory_for_next_turn"] = memory_text
        return args

    def _extract_response_text(self, response: Any) -> str:
        if not response.choices:
            raise ValueError("No choices returned from model.")
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if content is None:
            raise ValueError("Model response content was empty.")

        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for item in content:
                item_text: str | None = None
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        raw_text = item.get("text")
                        item_text = str(raw_text) if raw_text is not None else None
                    elif item.get("text") is not None:
                        item_text = str(item.get("text"))
                else:
                    raw_text = getattr(item, "text", None)
                    if raw_text is not None:
                        item_text = str(raw_text)
                if item_text:
                    parts.append(item_text)
            text = "\n".join(parts)
        else:
            text = str(content)

        text = text.strip()
        if not text:
            raise ValueError("Model response text was empty.")
        return text

    def _extract_tag_content(self, response_text: str, tag_name: str) -> str:
        pattern = re.compile(
            rf"<{tag_name}>(.*?)</{tag_name}>",
            re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.findall(response_text)
        if not matches:
            raise ValueError(f"Missing <{tag_name}>...</{tag_name}> block in model output.")
        return matches[-1].strip()

    def _parse_action_text(self, action_text: str) -> dict[str, Any]:
        payload = action_text.strip()
        if not payload:
            raise ValueError("empty <action> tag")

        if payload.startswith("{"):
            parsed = json.loads(payload)
            if not isinstance(parsed, dict):
                raise ValueError("<action> JSON must decode to an object.")
            return parsed

        action_match = re.search(r"\b(ACTION[1-7]|RESET)\b", payload, re.IGNORECASE)
        if action_match is None:
            raise ValueError(f"Could not parse action name from <action>: {payload!r}")
        action_name = action_match.group(1).upper()
        parsed_action: dict[str, Any] = {"action_name": action_name}

        x_match = re.search(r"\bx\s*[:=]\s*(-?\d+)", payload, re.IGNORECASE)
        y_match = re.search(r"\by\s*[:=]\s*(-?\d+)", payload, re.IGNORECASE)
        if x_match is not None and y_match is not None:
            parsed_action["x"] = int(x_match.group(1))
            parsed_action["y"] = int(y_match.group(1))
            return parsed_action

        if action_name == GameAction.ACTION6.name:
            remainder = payload[action_match.end() :]
            coords = re.findall(r"-?\d+", remainder)
            if len(coords) >= 2:
                parsed_action["x"] = int(coords[0])
                parsed_action["y"] = int(coords[1])
        return parsed_action

    def _extract_memory(self, args: dict[str, Any]) -> str:
        memory = args.get("memory_for_next_turn", "")
        if isinstance(memory, str):
            return memory.strip()
        return json.dumps(memory, ensure_ascii=True).strip()

    def _action_from_arguments(
        self,
        args: dict[str, Any],
        fallback_action: GameAction,
        available_actions: list[GameAction],
    ) -> tuple[GameAction, str]:
        action_name = str(
            args.get("action_name", args.get("action", args.get("name", "")))
        ).strip()
        try:
            requested_action = self._parse_action_token(action_name)
        except ValueError:
            return fallback_action, f"invalid_action_name={action_name}"

        if requested_action not in available_actions:
            return fallback_action, f"action_not_available={requested_action.name}"

        if requested_action.is_complex():
            x, y = self._parse_coordinates(args)
            if x is None or y is None:
                return fallback_action, "invalid_coordinates_for_action6"
            requested_action.set_data({"x": x, "y": y})
            return requested_action, "ok_action6"

        return requested_action, "ok_simple"

    def _retry_error_from_action_note(self, action_note: str) -> str | None:
        if action_note == "invalid_coordinates_for_action6":
            return "ACTION6 was selected without valid coordinates"
        if action_note.startswith("invalid_action_name="):
            return f"Model selected an invalid action token ({action_note})"
        if action_note.startswith("action_not_available="):
            return f"Model selected an unavailable action ({action_note})"
        return None

    def _parse_coordinates(self, args: dict[str, Any]) -> tuple[int | None, int | None]:
        try:
            x = int(args.get("x"))
            y = int(args.get("y"))
        except (TypeError, ValueError):
            return None, None
        if not (0 <= x <= 63 and 0 <= y <= 63):
            return None, None
        return x, y

    def _handle_parse_failure(
        self, fallback_action: GameAction, error: str, response: Any | None = None
    ) -> GameAction:
        self.consecutive_parse_failures += 1

        self._pending_reasoning = self._build_action_reasoning(
            parse_status="failed",
            action_note="fallback_after_parse_failure",
            response=response,
            parsed_args=None,
            memory_after=self.carryover_memory,
            error=error,
            parse_failures_in_row=self.consecutive_parse_failures,
        )
        return fallback_action

    def do_action_request(self, action: GameAction) -> FrameData:
        """Submit action with this agent's pending reasoning payload."""
        data = action.action_data.model_dump()
        reasoning_payload = self._pending_reasoning
        try:
            raw = None
            try:
                raw = self.arc_env.step(
                    action,
                    data=data,
                    reasoning=reasoning_payload,
                )
            except Exception as e:
                logger.warning(
                    "Action request failed for %s with data=%s; falling back to latest observation. Error: %s",
                    action.name,
                    json.dumps(data, ensure_ascii=True),
                    e,
                )
            if raw is None:
                raw = getattr(self.arc_env, "observation_space", None)
            if raw is None:
                raise RuntimeError(
                    "Action request failed and no observation fallback was available."
                )
            if (
                raw is not None
                and reasoning_payload is not None
                and getattr(raw, "action_input", None) is not None
                and getattr(raw.action_input, "reasoning", None) is None
            ):
                raw.action_input.reasoning = reasoning_payload
            return self._convert_raw_frame_data(raw)
        finally:
            self._pending_reasoning = None

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
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_type": "simple_memory_carryover",
            "model": self.model,
            "parse_status": parse_status,
            "action_note": action_note,
            "memory_chars": len(memory_after),
            "assistant_response": self._conversation_response_payload(response),
            "parsed_output": parsed_args,
            "tool_args": parsed_args,
        }
        if error is not None:
            payload["error"] = error
        if parse_failures_in_row is not None:
            payload["parse_failures_in_row"] = parse_failures_in_row
        return payload

    def _available_actions(self, latest_frame: FrameData) -> list[GameAction]:
        if latest_frame.state is GameState.GAME_OVER:
            return [GameAction.RESET]
        raw_actions = getattr(latest_frame, "available_actions", None) or []
        parsed: list[GameAction] = []
        for raw in raw_actions:
            action = self._coerce_action(raw)
            if action is not None:
                parsed.append(action)
        if GameAction.RESET not in parsed:
            parsed.append(GameAction.RESET)
        return parsed

    def _deterministic_fallback_action(self, latest_frame: FrameData) -> GameAction:
        available_actions = self._available_actions(latest_frame)
        non_reset_actions = [a for a in available_actions if a != GameAction.RESET]

        # Prefer any non-reset simple action first.
        for action in non_reset_actions:
            if action.is_simple():
                return action

        # If only non-reset complex actions exist, use deterministic coordinates.
        for action in non_reset_actions:
            if action.is_complex():
                action.set_data({"x": 0, "y": 0})
                return action

        # RESET is a last resort only when no other action exists.
        if GameAction.RESET in available_actions:
            return GameAction.RESET
        return GameAction.RESET

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(",".join(str(value) for value in row))
            lines.append("")
        return "\n".join(lines).strip()

    def _grid_to_data_url(self, grid: list[list[Any]]) -> str:
        try:
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "USE_IMAGE_INPUT=True requires Pillow. Install dependency 'pillow'."
            ) from e

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        if height == 0 or width == 0:
            raise ValueError("Cannot render empty grid image.")
        if any(len(row) != width for row in grid):
            raise ValueError("Grid rows must all have the same width.")

        img = Image.new("RGB", (width, height), "black")
        pixels = img.load()
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                try:
                    index = int(value) % len(self.ARC_COLOR_PALETTE)
                except (TypeError, ValueError):
                    index = 0
                pixels[x, y] = self.ARC_COLOR_PALETTE[index]

        img = img.resize((self.IMAGE_INPUT_SIZE, self.IMAGE_INPUT_SIZE), Image.NEAREST)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{payload}"

    def _coerce_action(self, raw: Any) -> GameAction | None:
        if isinstance(raw, GameAction):
            return raw
        if isinstance(raw, str):
            try:
                return GameAction.from_name(raw)
            except ValueError:
                return None
        if isinstance(raw, int):
            try:
                return GameAction.from_id(raw)
            except ValueError:
                return None
        return None

    def _resolve_chat_log_path(self) -> str:
        directory = os.getenv(self.RECORDINGS_DIR_ENV, "").strip()
        if not directory:
            directory = "recordings"
        os.makedirs(directory, exist_ok=True)

        filename = (
            f"{self.name}.{self._conversation_log_session_id}.conversation.chat.md"
        )
        return os.path.join(directory, filename)

    def _format_chat_turn(
        self,
        *,
        turn_index: int,
        system_prompt: str,
        user_prompt: str | list[dict[str, Any]],
        parsed_args: dict[str, Any] | None,
        response: Any | None,
        action: GameAction,
        error: str | None,
        memory_after: str,
        request_payload: dict[str, Any] | None,
    ) -> str:
        response_payload = self._raw_response_payload(response)
        if isinstance(user_prompt, str):
            user_prompt_fence = "text"
            user_prompt_text = user_prompt
        else:
            user_prompt_fence = "json"
            user_prompt_text = json.dumps(user_prompt, ensure_ascii=True, indent=2)
        action_data = action.action_data.model_dump()
        action6_lines: list[str] = []
        if action.name == "ACTION6":
            action6_lines = [
                f"- ACTION6 coordinates: x={action_data.get('x')}, y={action_data.get('y')}",
            ]
        lines = [
            f"## Turn {turn_index}",
            "",
            "### System Prompt",
            "```text",
            system_prompt,
            "```",
            "",
            "### User Prompt",
            f"```{user_prompt_fence}",
            user_prompt_text,
            "```",
            "",
            "### API Request (Raw)",
            "```json",
            json.dumps(request_payload, ensure_ascii=True, indent=2),
            "```",
            "",
            "### API Response (Raw)",
            "```json",
            json.dumps(response_payload, ensure_ascii=True, indent=2),
            "```",
            "",
            "### Parsed Output (Derived)",
            "```json",
            json.dumps(parsed_args, ensure_ascii=True, indent=2),
            "```",
            "",
            "### Decision",
            f"- selected_action: {action.name} ({action.value})",
            *action6_lines,
            "Memory for next turn:",
            "```text",
            memory_after if memory_after else "<empty>",
            "```",
            f"- parse_failures_in_row: {self.consecutive_parse_failures}",
            f"- error: {error or '<none>'}",
            "",
            "---",
            "",
        ]
        return "\n".join(lines)

    def _raw_response_payload(self, response: Any | None) -> Any:
        if response is None:
            return None
        if hasattr(response, "model_dump"):
            try:
                return self._redact_thought_signatures(response.model_dump())
            except Exception:
                pass
        if hasattr(response, "to_dict"):
            try:
                return self._redact_thought_signatures(response.to_dict())
            except Exception:
                pass
        return {"repr": repr(response)}

    def _redact_thought_signatures(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            if payload.get("type") == "reasoning.encrypted":
                # Drop encrypted reasoning blocks from logs to keep transcripts concise.
                return None

            redacted: dict[str, Any] = {}
            for key, value in payload.items():
                if key == "thought_signature":
                    redacted[key] = "(hidden for brevity)"
                else:
                    sanitized_value = self._redact_thought_signatures(value)
                    if sanitized_value is not None:
                        redacted[key] = sanitized_value
            return redacted
        if isinstance(payload, list):
            sanitized_items = []
            for item in payload:
                sanitized = self._redact_thought_signatures(item)
                if sanitized is not None:
                    sanitized_items.append(sanitized)
            return sanitized_items
        return payload

    def _conversation_response_payload(self, response: Any | None) -> dict[str, Any] | None:
        if response is None:
            return None

        payload: dict[str, Any] = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
        }

        choices = getattr(response, "choices", None) or []
        if choices:
            message = getattr(choices[0], "message", None)
            if message is not None:
                payload["content"] = self._redact_thought_signatures(
                    getattr(message, "content", None)
                )
                function_call = getattr(message, "function_call", None)
                if function_call is not None:
                    payload["function_call"] = {
                        "name": getattr(function_call, "name", None),
                        "arguments": getattr(function_call, "arguments", None),
                    }
                tool_calls = getattr(message, "tool_calls", None) or []
                payload["tool_calls"] = [
                    {
                        "id": getattr(tc, "id", None),
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(
                            getattr(tc, "function", None), "arguments", None
                        ),
                    }
                    for tc in tool_calls
                ]

        usage = getattr(response, "usage", None)
        if usage is not None:
            payload["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }

        return payload

    def _log_turn(
        self,
        *,
        system_prompt: str,
        user_prompt: str | list[dict[str, Any]],
        response: Any | None,
        parsed_args: dict[str, Any] | None,
        action: GameAction,
        error: str | None,
        memory_after: str,
        request_payload: dict[str, Any] | None,
    ) -> None:
        if not self._log_conversation:
            return

        if self._chat_log_path is None:
            self._chat_log_path = self._resolve_chat_log_path()
            if not self._conversation_log_announced:
                logger.info(
                    "SimpleMemoryCarryover chat log: %s",
                    self._chat_log_path,
                )
                self._conversation_log_announced = True

        chat_turn = self._format_chat_turn(
            turn_index=self.action_counter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parsed_args=parsed_args,
            response=response,
            action=action,
            error=error,
            memory_after=memory_after,
            request_payload=request_payload,
        )
        with open(self._chat_log_path, "a", encoding="utf-8") as f:
            f.write(chat_turn)
