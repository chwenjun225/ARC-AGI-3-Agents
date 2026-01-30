import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_agent_sdk import AssistantMessage, ToolUseBlock, ResultMessage, UserMessage, SystemMessage

logger = logging.getLogger()


class ClaudeCodeRecorder:
    
    ANTHROPIC_PRICING = {
        "input_tokens": 0.003 / 1000,
        "output_tokens": 0.015 / 1000,
    }
    
    def __init__(self, game_id: str, agent_name: str):
        self.game_id = game_id
        self.agent_name = agent_name
        
        recordings_dir = os.getenv("RECORDINGS_DIR", "recordings")
        self.output_dir = Path(recordings_dir) / f"{game_id}_{agent_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ClaudeCodeRecorder initialized: {self.output_dir}")
    
    def save_step(
        self,
        step: int,
        prompt: str,
        messages: list[Any],
        parsed_action: dict[str, Any],
        total_cost_usd: float,
        cumulative_cost_usd: float
    ) -> None:
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            formatted_messages = self.format_messages(messages)
            
            step_data = {
                "step": step,
                "timestamp": timestamp,
                "prompt": prompt,
                "messages": formatted_messages,
                "parsed_action": parsed_action,
                "cost_usd": total_cost_usd,
                "cumulative_cost_usd": cumulative_cost_usd
            }
            
            step_filename = self.output_dir / f"step_{step:03d}.json"
            with open(step_filename, "w", encoding="utf-8") as f:
                json.dump(step_data, f, indent=2)
            
            logger.info(f"Saved step {step} to {step_filename}")
        except Exception as e:
            logger.error(f"Failed to save step {step}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def format_messages(self, messages: list[Any]) -> list[dict[str, Any]]:
        formatted = []
        tool_id_to_name = {}
        message_index = 0
        
        for i, msg in enumerate(messages):
            try:
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, "text") and block.text:
                            formatted.append({
                                "message": message_index,
                                "type": "text",
                                "content": block.text
                            })
                            message_index += 1
                        
                        if isinstance(block, ToolUseBlock):
                            tool_id_to_name[block.id] = block.name
                            formatted.append({
                                "message": message_index,
                                "type": "tool_call",
                                "tool_name": block.name,
                                "tool_input": block.input
                            })
                            message_index += 1
                
                elif isinstance(msg, UserMessage):
                    if hasattr(msg, 'content') and isinstance(msg.content, list):
                        for block in msg.content:
                            if hasattr(block, 'tool_use_id'):
                                content_text = ""
                                if hasattr(block, 'content') and isinstance(block.content, list):
                                    for item in block.content:
                                        if isinstance(item, dict) and 'text' in item:
                                            content_text = item['text']
                                            break
                                
                                formatted.append({
                                    "message": message_index,
                                    "type": "tool_result",
                                    "tool_use_id": block.tool_use_id,
                                    "content": content_text,
                                    "is_error": getattr(block, 'is_error', None)
                                })
                                message_index += 1
                
                elif isinstance(msg, ResultMessage):
                    usage = msg.usage or {}
                    formatted.append({
                        "message": message_index,
                        "type": "result",
                        "duration_ms": msg.duration_ms,
                        "duration_api_ms": msg.duration_api_ms,
                        "is_error": msg.is_error,
                        "tokens": {
                            "input": usage.get("input_tokens", 0),
                            "output": usage.get("output_tokens", 0),
                        }
                    })
                    message_index += 1
            
            except Exception as e:
                logger.warning(f"Error formatting message {i} (type={type(msg).__name__}): {e}")
                formatted.append({
                    "message": message_index,
                    "type": "error",
                    "error": str(e),
                    "msg_type": type(msg).__name__
                })
                message_index += 1
        
        return formatted
    
    def aggregate_responses(self, formatted_messages: list[dict[str, Any]]) -> str:
        text_parts = []
        
        for msg in formatted_messages:
            if msg.get("type") == "text":
                text_parts.append(msg.get("content", ""))
        
        return "".join(text_parts)
    
    def calculate_cost_from_usage(self, usage: dict[str, Any]) -> float:
        try:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            
            if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
                logger.warning(f"Invalid token counts: input={input_tokens}, output={output_tokens}")
                return 0.0
            
            cost_usd = (
                input_tokens * self.ANTHROPIC_PRICING["input_tokens"] +
                output_tokens * self.ANTHROPIC_PRICING["output_tokens"]
            )
            
            logger.debug(f"Calculated cost from usage: {input_tokens} in, {output_tokens} out = ${cost_usd}")
            return cost_usd
        except Exception as e:
            logger.error(f"Error calculating cost: {e}")
            return 0.0
