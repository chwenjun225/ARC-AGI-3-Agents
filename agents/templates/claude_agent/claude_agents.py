import json
import logging
import os
import textwrap
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ToolUseBlock, ResultMessage, SystemMessage

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
        self.current_frame = None
        self.session_id = None
        self.consecutive_errors = 0
        self.mcp_server = create_arc_tools_server(self)
        self.captured_messages = []
        self.current_prompt = ""
        self.result_message = None
        
        if kwargs.get("record", False):
            self.claude_recorder = ClaudeCodeRecorder(
                game_id=kwargs.get("game_id", "unknown"),
                agent_name=self.agent_name
            )
        else:
            self.claude_recorder = None
        
        logging.getLogger("anthropic").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
    
    @property
    def name(self) -> str:
        sanitized_model_name = self.MODEL.replace("/", "-").replace(":", "-")
        return f"{super().name}.{sanitized_model_name}"
    
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any([
            latest_frame.state is GameState.WIN,
        ])
    
    def build_game_prompt(self, latest_frame: FrameData) -> str:
        try:
            if latest_frame.frame and len(latest_frame.frame) > 0:
                first_layer = latest_frame.frame[0]
                grid_str = "\n".join(
                    [" ".join([str(cell).rjust(2) for cell in row]) for row in first_layer]
                )
            else:
                grid_str = "No grid data available"
                logger.warning("Frame has no grid data")
        except Exception as e:
            grid_str = "Error formatting grid data"
            logger.error(f"Failed to format grid: {e}")
        
        try:
            available_actions_str = ", ".join(
                [f"ACTION{a}" if a > 0 else "RESET" for a in latest_frame.available_actions]
            )
            if not available_actions_str:
                logger.warning("No available actions in frame")
        except Exception as e:
            available_actions_str = "ERROR"
            logger.error(f"Failed to format available actions: {e}")
        
        prompt = textwrap.dedent(f"""
            You are playing an ARC-AGI-3 game. Your goal is to solve the puzzle.
            
            Game: {self.game_id}
            Current State: {latest_frame.state.value}
            Levels Completed: {latest_frame.levels_completed}
            Available Actions: {available_actions_str}
            
            Current Grid (64x64, values 0-15):
            {grid_str}
            
            You have access to the following tools:
            - reset_game: Reset the game to start over
            - action1_move_up: Execute ACTION1
            - action2_move_down: Execute ACTION2
            - action3_move_left: Execute ACTION3
            - action4_move_right: Execute ACTION4
            - action5_interact: Execute ACTION5
            - action6_click: Execute ACTION6 with coordinates (x, y) in range 0-63
            - action7_undo: Execute ACTION7 (undo)
            - read_notes: Read persistent notes about patterns/insights discovered
            - write_notes: Write persistent notes to remember across turns
            
            PERSISTENT MEMORY: You have the option to use read_notes/write_notes to maintain insights across turns.
            Track patterns, hypotheses, strategies, and what works/doesn't work. 
            A recommendation is to write the notes upon the initial analysis of the game, if you choose to analyze the game.
            
            Before calling a tool, explain your reasoning. Then call exactly ONE tool.
            Only call tools that are in the available_actions list.
        """).strip()
        
        return prompt
    
    async def prompt_generator(self, latest_frame: FrameData):
        game_prompt = self.build_game_prompt(latest_frame)
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": game_prompt
            }
        }
    
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
        action_taken: Optional[GameAction] = None
        self.captured_messages = []
        self.current_prompt = self.build_game_prompt(latest_frame)
        self.result_message = None
        
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def run_query():
            nonlocal action_taken
            
            reasoning_parts = []
            query_gen = None
            
            try:
                options = ClaudeAgentOptions(
                    model=self.MODEL,
                    mcp_servers={"arc-game-tools": self.mcp_server},
                    permission_mode="bypassPermissions",
                )
                
                if self.session_id:
                    options.resume = self.session_id
                    logger.debug(f"Resuming session: {self.session_id}")
                
                query_gen = query(
                    prompt=self.prompt_generator(latest_frame),
                    options=options
                )
                
                async for message in query_gen:
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
                                    import os
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
                                    logger.info(f"Parsed action: {action_taken.name}")
                                    break
                                else:
                                    logger.warning(f"Failed to parse action from tool: {tool_name}")
            except Exception as e:
                if "credit balance" in str(e).lower():
                    raise
                logger.error(f"Error during query execution: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            finally:
                if query_gen is not None:
                    try:
                        await query_gen.aclose()
                    except Exception as e:
                        logger.debug(f"Error closing query generator: {e}")
        
        try:
            loop.run_until_complete(run_query())
            
            pending = asyncio.all_tasks(loop)
            if pending:
                logger.debug(f"Waiting for {len(pending)} pending tasks to complete")
                results = loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Pending task {i} raised exception: {result}")
        except RuntimeError as e:
            if "credit balance" in str(e).lower():
                import os
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                try:
                    loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
                except:
                    pass
                print("\n" + "="*80)
                print("\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m")
                print("Please add credits to your Anthropic account to continue.")
                print("="*80 + "\n")
                os._exit(1)
            raise
        except Exception as e:
            if "credit balance" in str(e).lower():
                import os
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                try:
                    loop.run_until_complete(asyncio.gather(*asyncio.all_tasks(loop), return_exceptions=True))
                except:
                    pass
                print("\n" + "="*80)
                print("\033[91m" + "ERROR: Insufficient Anthropic API Credits" + "\033[0m")
                print("Please add credits to your Anthropic account to continue.")
                print("="*80 + "\n")
                os._exit(1)
            logger.error(f"Error running event loop: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        finally:
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"Error closing event loop: {e}")
        
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
                action.action_data.__dict__["reasoning"] = {
                    "text": self.latest_reasoning[:16000]
                }
            else:
                logger.debug("No reasoning captured for action")
            
            return action
        
        logger.warning(f"Unknown tool name: {tool_name}")
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
