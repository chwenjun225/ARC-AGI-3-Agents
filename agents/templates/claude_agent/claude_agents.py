import asyncio
import json
import logging
import os
import textwrap
import traceback
import uuid
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, ToolUseBlock, ResultMessage, SystemMessage

from ...agent import Agent
from .claude_tools import create_arc_tools_server
from .claude_recorder import ClaudeCodeRecorder

logger = logging.getLogger()


class ClaudeCodeAgent(Agent):
    MAX_ACTIONS: int = 80
    MODEL: str = "claude-sonnet-4-5-20250929"
    MAX_CONSECUTIVE_ERRORS: int = 3
    
    token_counter: int
    step_counter: int
    mcp_server: Any
    latest_reasoning: str
    latest_reasoning_dict: dict[str, Any]
    claude_recorder: Optional[ClaudeCodeRecorder]
    captured_messages: list[Any]
    current_prompt: str
    result_message: Optional[Any]
    current_frame: Optional[FrameData]
    session_id: Optional[str]
    consecutive_errors: int
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.token_counter = 0
        self.step_counter = 0
        self.cumulative_cost_usd = 0.0
        self.latest_reasoning = ""
        self.latest_reasoning_dict = {}
        self.current_frame = None
        self.session_id = None
        self.consecutive_errors = 0
        self.mcp_server = create_arc_tools_server(self)
        self.notes_session_id = str(uuid.uuid4())
        self.notes_dir = os.path.abspath(f"./game_notes/{self.game_id}_{self.notes_session_id}")
        os.makedirs(self.notes_dir, exist_ok=True)
        self.notes_path = os.path.join(self.notes_dir, "notes.md")
        with open(self.notes_path, 'w') as f:
            f.write("")
        logger.info(f"Created notes file: {self.notes_path}")
        self.captured_messages = []
        self.current_prompt = ""
        self.result_message = None
        
        if kwargs.get("record", False):
            self.claude_recorder = ClaudeCodeRecorder(
                game_id=kwargs.get("game_id", "unknown"),
                agent_name=self.agent_name,
                session_id=self.notes_session_id
            )
        else:
            self.claude_recorder = None
        
        logging.getLogger("anthropic").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

        # Persistent async infrastructure — single Claude Code subprocess for all turns
        self._loop = asyncio.new_event_loop()
        self._client: Optional[ClaudeSDKClient] = None
        self._client_connected = False
    
    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any([
            latest_frame.state is GameState.WIN,
        ])
    
    def do_action_request(self, action: GameAction) -> FrameData:
        data = action.action_data.model_dump()
        
        if self.latest_reasoning_dict:
            data["reasoning"] = self.latest_reasoning_dict
            logger.info(f"Added reasoning to action request: {len(str(self.latest_reasoning_dict))} chars")
        
        raw = self.arc_env.step(
            action,
            data=data,
            reasoning=data.get("reasoning", {}),
        )
        return self._convert_raw_frame_data(raw)
    
    def _format_grid(self, frame: FrameData) -> str:
        try:
            if frame.frame and len(frame.frame) > 0:
                first_layer = frame.frame[0]
                return "\n".join(
                    [" ".join([str(cell).rjust(2) for cell in row]) for row in first_layer]
                )
            return ""
        except Exception as e:
            logger.error(f"Failed to format grid: {e}")
            return ""

    def build_game_prompt(self, frames: list[FrameData], latest_frame: FrameData) -> str:
        grid_str = self._format_grid(latest_frame) or "No grid data available"

        try:
            available_actions_str = ", ".join(
                [f"ACTION{a}" if a > 0 else "RESET" for a in latest_frame.available_actions]
            )
            if not available_actions_str:
                logger.warning("No available actions in frame")
        except Exception as e:
            available_actions_str = "ERROR"
            logger.error(f"Failed to format available actions: {e}")

        # Build animation history from all previous frames that have grid data
        history_frames = [(i, f) for i, f in enumerate(frames) if f.frame and len(f.frame) > 0]
        animation_section = ""
        if history_frames:
            parts = []
            for idx, (_, f) in enumerate(history_frames):
                g = self._format_grid(f)
                if g:
                    parts.append(f"--- Frame {idx + 1} (State: {f.state.value}, Levels: {f.levels_completed}) ---\n{g}")
            if parts:
                animation_section = (
                    "\n\nAnimation History (all previous frames received, in order):\n"
                    + "\n\n".join(parts)
                    + "\n\n--- End of Animation History ---"
                )

        prompt = textwrap.dedent(f"""
            You are playing an ARC-AGI-3 game. Your goal is to solve the puzzle.

            Game: {self.game_id}
            Current State: {latest_frame.state.value}
            Levels Completed: {latest_frame.levels_completed}
            Available Actions: {available_actions_str}
            {animation_section}

            Current Grid (64x64, values 0-15):
            {grid_str}
            
            You have access to the following game action tools:
            - reset_game: Reset the game to start over
            - action1_move_up: Execute ACTION1
            - action2_move_down: Execute ACTION2
            - action3_move_left: Execute ACTION3
            - action4_move_right: Execute ACTION4
            - action5_interact: Execute ACTION5
            - action6_click: Execute ACTION6 with coordinates (x, y) in range 0-63
            - action7_undo: Execute ACTION7 (undo)

            PERSISTENT SCRATCH PAD: You have a notes file at: {self.notes_path}
            You can use the built-in Read, Edit, and Write tools to manage this file.
            Recommended workflow each turn:
            1. First, Read your notes file to recall your previous insights and strategy.
            2. Then, Edit the notes file to update with any new observations (use targeted edits, don't rewrite the whole file).
               Use Write only for the initial creation of the notes file.
            3. Finally, call exactly ONE game action tool to make your move.
            Only call game action tools that are in the available_actions list.

            Before calling a game action tool, explain your reasoning.
        """).strip()

        strategy_prompt = os.getenv("STRATEGY_PROMPT", "").strip()
        if strategy_prompt:
            prompt += f"\n\n## Strategy Prompt\n{strategy_prompt}"

        return prompt
    
    def _ensure_client_connected(self) -> None:
        """Connect the ClaudeSDKClient if not already connected.

        Uses a single persistent subprocess across all game turns.
        """
        if self._client_connected:
            return

        async def _connect():
            # Clean up stale client from a previous failed connection
            if self._client:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass

            options = ClaudeAgentOptions(
                model=self.MODEL,
                mcp_servers={"arc-game-tools": self.mcp_server},
                permission_mode="bypassPermissions",
                cwd=self.notes_dir,
                tools=["Read", "Edit", "Write"],
                system_prompt={
                    "type": "preset",
                    "preset": "claude_code",
                    "append": textwrap.dedent("""
                        IMPORTANT - INTERRUPT BEHAVIOR: After you call a game action tool,
                        the system will interrupt you to process the action and advance the game.
                        This is expected and normal — do not try to prevent or work around interrupts.
                        Each turn, you should:
                        1. Optionally read/update your notes file for strategy tracking
                        2. Analyze the current game state
                        3. Call exactly ONE game action tool
                        You will then be interrupted, and the next game state will be provided.
                    """).strip()
                },
            )
            self._client = ClaudeSDKClient(options=options)
            await self._client.connect()
            self._client_connected = True
            logger.info("ClaudeSDKClient connected (persistent subprocess)")

        self._loop.run_until_complete(_connect())

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        self.step_counter += 1
        logger.info(f"Step {self.step_counter}: Choosing action...")
        
        if self.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
            logger.error(f"FATAL: {self.consecutive_errors} consecutive errors, stopping agent")
            raise RuntimeError(f"Too many consecutive errors ({self.consecutive_errors}), cannot continue")
        
        self.current_frame = latest_frame
        self.latest_reasoning = ""
        self.latest_reasoning_dict = {}
        action_taken: Optional[GameAction] = None
        self.captured_messages = []
        self.current_prompt = self.build_game_prompt(frames, latest_frame)
        self.result_message = None

        # Ensure the persistent client is connected (single subprocess for all turns)
        self._ensure_client_connected()

        async def _run_turn():
            nonlocal action_taken
            reasoning_parts = []

            try:
                # Send prompt to the existing persistent session
                await self._client.query(self.current_prompt)

                # receive_response() yields messages and auto-terminates after ResultMessage
                async for message in self._client.receive_response():
                    self.captured_messages.append(message)

                    if isinstance(message, SystemMessage) and message.subtype == 'init':
                        if not self.session_id:
                            self.session_id = message.data.get('session_id')
                            logger.info(f"Session started: {self.session_id}")
                        else:
                            resumed_session = message.data.get('session_id')
                            if resumed_session != self.session_id:
                                logger.warning(f"Session ID mismatch: expected {self.session_id}, got {resumed_session}")

                    if isinstance(message, ResultMessage):
                        self.result_message = message
                        if message.is_error:
                            logger.error(f"ResultMessage indicates error occurred during query")

                    if isinstance(message, AssistantMessage) and not action_taken:
                        for block in message.content:
                            if hasattr(block, "text") and block.text:
                                reasoning_parts.append(block.text)
                                logger.info(f"Claude reasoning: {block.text[:100]}...")

                                if "credit balance is too low" in block.text.lower():
                                    logger.error("FATAL: Credit balance too low - stopping immediately")
                                    print("\n" + "="*80)
                                    print("\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m")
                                    print("Please add credits to your Anthropic account to continue.")
                                    print("="*80 + "\n")
                                    os._exit(1)

                            if isinstance(block, ToolUseBlock):
                                tool_name = block.name
                                logger.info(f"Claude calling tool: {tool_name}")

                                if reasoning_parts:
                                    self.latest_reasoning = " ".join(reasoning_parts)

                                action_taken = self.parse_action_from_tool(tool_name, block.input)

                                if action_taken:
                                    if latest_frame.available_actions and action_taken.value not in latest_frame.available_actions:
                                        logger.warning(f"Action {action_taken.name} (value={action_taken.value}) not in available_actions: {latest_frame.available_actions}")
                                    logger.info(f"Action found: {action_taken.name}, sending interrupt")
                                    try:
                                        await self._client.interrupt()
                                        logger.debug("Interrupt sent successfully")
                                    except Exception as e:
                                        logger.debug(f"Interrupt error (may be expected): {e}")
                                    # Break inner for-loop; outer async-for continues
                                    # to drain remaining messages until ResultMessage
                                    break
            except Exception as e:
                if "credit balance" in str(e).lower():
                    raise
                logger.error(f"Error during query execution: {e}")
                logger.debug(traceback.format_exc())
                # Mark disconnected so next turn reconnects
                self._client_connected = False

        try:
            self._loop.run_until_complete(_run_turn())
        except RuntimeError as e:
            if "credit balance" in str(e).lower():
                print("\n" + "="*80)
                print("\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m")
                print("Please add credits to your Anthropic account to continue.")
                print("="*80 + "\n")
                os._exit(1)
            raise
        except Exception as e:
            if "credit balance" in str(e).lower():
                print("\n" + "="*80)
                print("\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m")
                print("Please add credits to your Anthropic account to continue.")
                print("="*80 + "\n")
                os._exit(1)
            logger.error(f"Error running event loop: {e}")
            logger.debug(traceback.format_exc())
        
        if action_taken:
            self.consecutive_errors = 0
            if not self.latest_reasoning:
                logger.warning("Action taken but no reasoning captured")
            
            if self.claude_recorder and not self.is_playback:
                parsed_action = {
                    "action": action_taken.value,
                    "reasoning": self.latest_reasoning
                }
                
                cost_usd = 0.0
                if self.result_message and hasattr(self.result_message, 'total_cost_usd'):
                    cost_usd = self.result_message.total_cost_usd or 0.0
                    logger.debug(f"Cost from API: ${cost_usd:.6f}")
                else:
                    logger.warning("No total_cost_usd in ResultMessage")
                
                self.cumulative_cost_usd += cost_usd
                
                if self.result_message:
                    try:
                        self.track_tokens_from_result(self.result_message)
                    except Exception as e:
                        logger.error(f"Failed to track tokens: {e}")
                
                try:
                    self.claude_recorder.save_step(
                        step=self.step_counter,
                        prompt=self.current_prompt,
                        messages=self.captured_messages,
                        parsed_action=parsed_action,
                        total_cost_usd=cost_usd
                    )
                except Exception as e:
                    logger.error(f"Failed to save step recording: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            return action_taken
        
        self.consecutive_errors += 1
        logger.warning(f"No action was taken by Claude (consecutive errors: {self.consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}), defaulting to RESET")
        if not self.captured_messages:
            logger.error("No messages captured at all - query may have failed completely")
            if self.session_id:
                logger.error(f"Session may be corrupted: {self.session_id}")
        else:
            logger.warning(f"Captured {len(self.captured_messages)} messages but no valid action found")
        return GameAction.RESET
    
    def cleanup(self, scorecard=None):
        """Clean up the persistent client and event loop."""
        if self._client_connected and self._client:
            try:
                self._loop.run_until_complete(self._client.disconnect())
                logger.info("ClaudeSDKClient disconnected")
            except Exception as e:
                logger.warning(f"Error disconnecting client: {e}")
            self._client = None
            self._client_connected = False

        try:
            pending = asyncio.all_tasks(self._loop)
            if pending:
                for task in pending:
                    task.cancel()
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
        except Exception as e:
            logger.warning(f"Error closing event loop: {e}")

        super().cleanup(scorecard)

    def parse_action_from_tool(self, tool_name: str, tool_input: dict[str, Any]) -> Optional[GameAction]:
        tool_map = {
            "mcp__arc-game-tools__reset_game": GameAction.RESET,
            "mcp__arc-game-tools__action1_move_up": GameAction.ACTION1,
            "mcp__arc-game-tools__action2_move_down": GameAction.ACTION2,
            "mcp__arc-game-tools__action3_move_left": GameAction.ACTION3,
            "mcp__arc-game-tools__action4_move_right": GameAction.ACTION4,
            "mcp__arc-game-tools__action5_interact": GameAction.ACTION5,
            "mcp__arc-game-tools__action6_click": GameAction.ACTION6,
            "mcp__arc-game-tools__action7_undo": GameAction.ACTION7,
        }
        
        if tool_name in tool_map:
            action = tool_map[tool_name]
            
            if self.current_frame and self.current_frame.available_actions:
                if action.value not in self.current_frame.available_actions:
                    logger.warning(f"Rejecting {action.name} (value={action.value}): not in available_actions {self.current_frame.available_actions}")
                    return None
            
            if action == GameAction.ACTION6:
                x = tool_input.get("x", 0)
                y = tool_input.get("y", 0)
                
                if not (0 <= x <= 63 and 0 <= y <= 63):
                    logger.warning(f"ACTION6 coordinates out of range: x={x}, y={y}")
                    return None
                
                action.action_data.x = x
                action.action_data.y = y
            
            action.action_data.game_id = self.game_id
            
            if self.latest_reasoning:
                self.latest_reasoning_dict = {
                    "thought": self.latest_reasoning[:16000]
                }
                logger.info(f"Prepared reasoning for action ({len(self.latest_reasoning)} chars)")
            else:
                self.latest_reasoning_dict = {}
                logger.warning("No reasoning captured for action - reasoning logs will not appear in replay")
            
            return action
        
        logger.debug(f"Non-action tool called: {tool_name}")
        return None
    
    def track_tokens_from_result(self, result_message: Any) -> None:
        if not hasattr(result_message, 'usage') or not result_message.usage:
            logger.debug("No usage data in ResultMessage")
            return
        
        try:
            usage = result_message.usage
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("cache_read_input_tokens", 0)
            total = input_tokens + output_tokens
            self.token_counter += total
            
            logger.info(
                f"Token usage: +{total} (in: {input_tokens}, out: {output_tokens}, cached: {cached_tokens}), total: {self.token_counter}"
            )
        except Exception as e:
            logger.error(f"Error tracking tokens: {e}")
        
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record({
                "tokens": total,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": self.token_counter,
            })
