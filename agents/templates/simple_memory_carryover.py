import json
import logging
import os
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Any

import openai
from arcengine import FrameData, GameAction, GameState
from openai import OpenAI as OpenAIClient

from ..agent import Agent

logger = logging.getLogger(__name__)


class SimpleMemoryCarryover(Agent):
    """LLM agent with explicit one-turn memory carryover only."""

    MAX_ACTIONS = 30

    GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
    RECORDINGS_DIR_ENV = "RECORDINGS_DIR"

    DEFAULT_MODEL = "gemini-3-flash-preview"
    REASONING_EFFORT = "high"

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    PARSE_FAILURE_CLEAR_THRESHOLD = 0
    ENABLE_CHAT_LOG = True

    TOOL_NAME = "submit_action_and_memory"
    # Prefer modern tool-calling by default; keep legacy function-calling fallback.
    MODEL_REQUIRES_TOOLS = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.api_key = os.getenv(self.GEMINI_API_KEY_ENV, "").strip()
        self.base_url = self.DEFAULT_BASE_URL
        self.model = self.DEFAULT_MODEL
        self.carryover_memory = ""
        self._locked_available_actions: list[GameAction] | None = None
        self._pending_reasoning: dict[str, Any] | None = None
        self.consecutive_parse_failures = 0
        self.parse_failure_clear_threshold = self.PARSE_FAILURE_CLEAR_THRESHOLD
        self._log_conversation = self.ENABLE_CHAT_LOG
        self._chat_log_path: str | None = None
        self._conversation_log_session_id = uuid.uuid4().hex
        self._conversation_log_announced = False
        self.client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)
        super().__init__(*args, **kwargs)

    @property
    def name(self) -> str:
        model_name = getattr(self, "model", self.DEFAULT_MODEL)
        sanitized_model_name = model_name.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}.memory-carryover"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        available_actions = self._available_actions(latest_frame)
        if self._locked_available_actions is None and available_actions:
            # Game action set is constant; lock once to keep tool schema stable.
            self._locked_available_actions = list(available_actions)
        tool_actions = self._locked_available_actions or available_actions
        fallback_action = self._deterministic_fallback_action(latest_frame)
        system_prompt = self._build_system_prompt(tool_actions)
        user_prompt = self._build_user_prompt(latest_frame)
        memory_before = self.carryover_memory
        response: Any | None = None
        request_payload: dict[str, Any] | None = None

        try:
            create_kwargs: dict[str, Any] = {
                "model": self.model,
                "reasoning_effort": self.REASONING_EFFORT,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if self.MODEL_REQUIRES_TOOLS:
                create_kwargs["tools"] = self._build_tools(tool_actions)
                create_kwargs["tool_choice"] = "required"
            else:
                create_kwargs["functions"] = self._build_functions(tool_actions)
                create_kwargs["function_call"] = {"name": self.TOOL_NAME}

            request_payload = create_kwargs
            response = self.client.chat.completions.create(**create_kwargs)
            args = self._extract_action_and_memory_arguments(response)

            action, action_note = self._action_from_arguments(
                args=args,
                latest_frame=latest_frame,
                fallback_action=fallback_action,
                available_actions=available_actions,
            )

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
                latest_frame=latest_frame,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                parsed_args=args,
                action=action,
                error=None,
                memory_before=memory_before,
                memory_after=self.carryover_memory,
                request_payload=request_payload,
            )
            return action

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"
            action = self._handle_parse_failure(fallback_action, error, response=response)
            self._log_turn(
                latest_frame=latest_frame,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                parsed_args=None,
                action=action,
                error=error,
                memory_before=memory_before,
                memory_after=self.carryover_memory,
                request_payload=request_payload,
            )
            return action
        except openai.BadRequestError as e:
            error = f"BadRequestError: {e}"
            action = self._handle_parse_failure(fallback_action, error, response=response)
            self._log_turn(
                latest_frame=latest_frame,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                parsed_args=None,
                action=action,
                error=error,
                memory_before=memory_before,
                memory_after=self.carryover_memory,
                request_payload=request_payload,
            )
            return action
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            action = self._handle_parse_failure(
                fallback_action, error, response=response
            )
            self._log_turn(
                latest_frame=latest_frame,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response=response,
                parsed_args=None,
                action=action,
                error=error,
                memory_before=memory_before,
                memory_after=self.carryover_memory,
                request_payload=request_payload,
            )
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
                "(both integers in [0,63]). where (0,0) refers to the top left"
            )
        return textwrap.dedent(
            """
You are a turn-based game-playing agent.

Critical memory constraint:
- You do NOT have hidden memory.
- You only remember text shown in MEMORY_FROM_PREVIOUS_TURN.
- The text you return in memory_for_next_turn fully replaces prior memory.
- If you would like to compare the current state to the previous state, you have to store all relevant information about the current state in your memory for use during your next turn.

PERSISTENT_AVAILABLE_ACTIONS (constant for this game):
{available_actions_block}

Action rules:
- {choose_rule}
{coordinate_rule}

Output rules:
- You must reply by calling submit_action_and_memory exactly once.
            """.format(
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

What action would you like to take?
Frames are formatted as CSV rows for each grid (comma-separated integer values).
Also write all memory you want to persist for the next turn in memory_for_next_turn.
            """.format(
                frames=self._pretty_print_3d(latest_frame.frame),
                memory=memory_text,
            )
        ).strip()

    def _build_tools(self, available_actions: list[GameAction]) -> list[dict[str, Any]]:
        action_enum = self._tool_action_enum(available_actions)
        action6_available = GameAction.ACTION6 in available_actions
        action_desc = "One selected action from AVAILABLE_ACTIONS."
        properties: dict[str, Any] = {
            "action_name": {
                "type": "string",
                "enum": action_enum,
                "description": action_desc,
            },
            "memory_for_next_turn": {
                "type": "string",
                "description": "All memory text to persist to the next turn.",
            },
        }
        if action6_available:
            properties["x"] = {
                "type": "integer",
                "minimum": 0,
                "maximum": 63,
                "description": "Required only for ACTION6.",
            }
            properties["y"] = {
                "type": "integer",
                "minimum": 0,
                "maximum": 63,
                "description": "Required only for ACTION6.",
            }
        return [
            {
                "type": "function",
                "function": {
                    "name": self.TOOL_NAME,
                    "description": "Submit the next action and the full memory to persist to next turn.",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": ["action_name", "memory_for_next_turn"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def _build_functions(self, available_actions: list[GameAction]) -> list[dict[str, Any]]:
        tools = self._build_tools(available_actions)
        return [tools[0]["function"]]

    def _action_token_for_tool(self, action: GameAction) -> str:
        return action.name

    def _tool_action_enum(self, available_actions: list[GameAction]) -> list[str]:
        tokens: list[str] = []
        for action in available_actions:
            token = self._action_token_for_tool(action)
            if token not in tokens:
                tokens.append(token)
        return tokens

    def _parse_action_token(self, action_token: str) -> GameAction:
        token = action_token.strip()
        if not token:
            raise ValueError("missing action_name")
        return GameAction.from_name(token)

    def _extract_tool_arguments(self, response: Any) -> dict[str, Any]:
        if not response.choices:
            raise ValueError("No choices returned from model.")
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if len(tool_calls) == 0:
            raise ValueError("Model did not call a tool.")
        tool_call = tool_calls[0]
        if tool_call.function.name != self.TOOL_NAME:
            raise ValueError(f"Unexpected tool name: {tool_call.function.name}")
        raw_arguments = tool_call.function.arguments or "{}"
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments were not a JSON object.")
        return parsed

    def _extract_function_arguments(self, response: Any) -> dict[str, Any]:
        if not response.choices:
            raise ValueError("No choices returned from model.")
        message = response.choices[0].message
        function_call = getattr(message, "function_call", None)
        if function_call is None:
            raise ValueError("Model did not call a function.")
        function_name = getattr(function_call, "name", None)
        if function_name != self.TOOL_NAME:
            raise ValueError(f"Unexpected function name: {function_name}")
        raw_arguments = getattr(function_call, "arguments", None) or "{}"
        parsed = json.loads(raw_arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Function arguments were not a JSON object.")
        return parsed

    def _extract_action_and_memory_arguments(self, response: Any) -> dict[str, Any]:
        if self.MODEL_REQUIRES_TOOLS:
            return self._extract_tool_arguments(response)
        return self._extract_function_arguments(response)

    def _extract_memory(self, args: dict[str, Any]) -> str:
        memory = args.get("memory_for_next_turn", "")
        if isinstance(memory, str):
            return memory.strip()
        return json.dumps(memory, ensure_ascii=True).strip()

    def _action_from_arguments(
        self,
        args: dict[str, Any],
        latest_frame: FrameData,
        fallback_action: GameAction,
        available_actions: list[GameAction],
    ) -> tuple[GameAction, str]:
        action_name = str(args.get("action_name", args.get("action", ""))).strip()
        try:
            requested_action = self._parse_action_token(action_name)
        except ValueError:
            return fallback_action, f"invalid_action_name={action_name}"

        if requested_action not in available_actions:
            return fallback_action, f"action_not_available={requested_action.name}"

        if requested_action.is_complex():
            x, y = self._parse_coordinates(args)
            if x is None or y is None:
                safe_fallback = self._fallback_non_complex_action(
                    latest_frame, available_actions
                )
                return safe_fallback, "invalid_coordinates_for_action6"
            requested_action.set_data({"x": x, "y": y})
            return requested_action, "ok_action6"

        return requested_action, "ok_simple"

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
        cleared_memory = False
        if (
            self.parse_failure_clear_threshold > 0
            and self.consecutive_parse_failures >= self.parse_failure_clear_threshold
        ):
            self.carryover_memory = ""
            cleared_memory = True

        self._pending_reasoning = self._build_action_reasoning(
            parse_status="failed",
            action_note="fallback_after_parse_failure",
            response=response,
            parsed_args=None,
            memory_after=self.carryover_memory,
            error=error,
            parse_failures_in_row=self.consecutive_parse_failures,
            memory_cleared=cleared_memory,
        )
        return fallback_action

    def do_action_request(self, action: GameAction) -> FrameData:
        """Submit action with this agent's pending reasoning payload."""
        data = action.action_data.model_dump()
        reasoning_payload = self._pending_reasoning
        try:
            raw = self.arc_env.step(
                action,
                data=data,
                reasoning=reasoning_payload,
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
        memory_cleared: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "agent_type": "simple_memory_carryover",
            "model": self.model,
            "parse_status": parse_status,
            "action_note": action_note,
            "memory_chars": len(memory_after),
            # This is what the ARC site displays under Reasoning Log.
            "assistant_response": self._conversation_response_payload(response),
            "tool_args": parsed_args,
        }
        if error is not None:
            payload["error"] = error
        if parse_failures_in_row is not None:
            payload["parse_failures_in_row"] = parse_failures_in_row
        if memory_cleared is not None:
            payload["memory_cleared"] = memory_cleared
        return payload

    def _available_actions(self, latest_frame: FrameData) -> list[GameAction]:
        raw_actions = getattr(latest_frame, "available_actions", None) or []
        parsed: list[GameAction] = []
        for raw in raw_actions:
            action = self._coerce_action(raw)
            if action is not None:
                parsed.append(action)
        return parsed

    def _deterministic_fallback_action(self, latest_frame: FrameData) -> GameAction:
        available_actions = self._available_actions(latest_frame)
        if (
            latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
            and GameAction.RESET in available_actions
        ):
            return GameAction.RESET
        for action in available_actions:
            if action.is_simple():
                return action
        if available_actions:
            # If only complex actions are available, use deterministic coordinates.
            action = available_actions[0]
            if action.is_complex():
                action.set_data({"x": 0, "y": 0})
            return action
        return GameAction.RESET

    def _fallback_non_complex_action(
        self, latest_frame: FrameData, available_actions: list[GameAction]
    ) -> GameAction:
        for action in available_actions:
            if action.is_simple():
                return action
        return self._deterministic_fallback_action(latest_frame)

    def _pretty_print_3d(self, array_3d: list[list[list[Any]]]) -> str:
        lines = []
        for i, block in enumerate(array_3d):
            lines.append(f"Grid {i}:")
            for row in block:
                lines.append(",".join(str(value) for value in row))
            lines.append("")
        return "\n".join(lines).strip()

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

    def _split_user_prompt_sections(self, user_prompt: str) -> tuple[str, str, str]:
        memory_header = "MEMORY_FROM_PREVIOUS_TURN:\n"
        frames_header = "\n\nFRAMES:\n"
        available_actions_header = "\n\nAVAILABLE_ACTIONS:\n"

        memory_start = user_prompt.find(memory_header)
        frames_start = user_prompt.find(frames_header)
        available_actions_start = user_prompt.find(available_actions_header)

        if (
            memory_start == 0
            and frames_start > memory_start
            and available_actions_start > frames_start
        ):
            memory_text_start = memory_start + len(memory_header)
            frames_text_start = frames_start + len(frames_header)
            # Keep "AVAILABLE_ACTIONS:" label in the instructions block.
            instructions_text_start = available_actions_start + 2

            memory_text = user_prompt[memory_text_start:frames_start]
            frame_text = user_prompt[frames_text_start:available_actions_start]
            instructions_text = user_prompt[instructions_text_start:]
            return memory_text, frame_text, instructions_text

        # Fallback keeps all original text instead of inventing a summary.
        return "<unparsed>", user_prompt, user_prompt

    def _format_chat_turn(
        self,
        *,
        timestamp: str,
        turn_index: int,
        latest_frame: FrameData,
        available_actions: list[str],
        system_prompt: str,
        user_prompt: str,
        parsed_args: dict[str, Any] | None,
        response: Any | None,
        action: GameAction,
        error: str | None,
        memory_before: str,
        memory_after: str,
        state_name: str,
        request_payload: dict[str, Any] | None,
    ) -> str:
        response_payload = self._raw_response_payload(response)
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
            "```text",
            user_prompt,
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
            "### Parsed Tool Arguments (Derived)",
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
                return response.model_dump()
            except Exception:
                pass
        if hasattr(response, "to_dict"):
            try:
                return response.to_dict()
            except Exception:
                pass
        return {"repr": repr(response)}

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
                payload["content"] = getattr(message, "content", None)
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
        latest_frame: FrameData,
        system_prompt: str,
        user_prompt: str,
        response: Any | None,
        parsed_args: dict[str, Any] | None,
        action: GameAction,
        error: str | None,
        memory_before: str,
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

        available_actions = [a.name for a in self._available_actions(latest_frame)]
        frame_state = getattr(latest_frame, "state", None)
        if hasattr(frame_state, "name"):
            state_name = frame_state.name
        else:
            state_name = str(frame_state)

        timestamp = datetime.now(timezone.utc).isoformat()
        chat_turn = self._format_chat_turn(
            timestamp=timestamp,
            turn_index=self.action_counter,
            latest_frame=latest_frame,
            available_actions=available_actions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parsed_args=parsed_args,
            response=response,
            action=action,
            error=error,
            memory_before=memory_before,
            memory_after=memory_after,
            state_name=state_name,
            request_payload=request_payload,
        )
        with open(self._chat_log_path, "a", encoding="utf-8") as f:
            f.write(chat_turn)
