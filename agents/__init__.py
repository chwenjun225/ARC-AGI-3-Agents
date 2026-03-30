import logging
from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.heuristic_agent import HeuristicAgent
from .templates.random_agent import Random

load_dotenv()
logger = logging.getLogger(__name__)

OPTIONAL_EXPORTS: list[str] = []

try:
    from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly

    OPTIONAL_EXPORTS.extend(["LangGraphFunc", "LangGraphTextOnly"])
except ModuleNotFoundError as exc:
    logger.debug("Skipping langgraph functional agents: %s", exc)

try:
    from .templates.langgraph_random_agent import LangGraphRandom

    OPTIONAL_EXPORTS.append("LangGraphRandom")
except ModuleNotFoundError as exc:
    logger.debug("Skipping langgraph random agent: %s", exc)

try:
    from .templates.langgraph_thinking import LangGraphThinking

    OPTIONAL_EXPORTS.append("LangGraphThinking")
except ModuleNotFoundError as exc:
    logger.debug("Skipping langgraph thinking agent: %s", exc)

try:
    from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM

    OPTIONAL_EXPORTS.extend(["LLM", "FastLLM", "GuidedLLM", "ReasoningLLM"])
except ModuleNotFoundError as exc:
    logger.debug("Skipping OpenAI-backed LLM agents: %s", exc)

try:
    from .templates.multimodal import MultiModalLLM

    OPTIONAL_EXPORTS.append("MultiModalLLM")
except ModuleNotFoundError as exc:
    logger.debug("Skipping multimodal agent: %s", exc)

try:
    from .templates.reasoning_agent import ReasoningAgent

    OPTIONAL_EXPORTS.append("ReasoningAgent")
except ModuleNotFoundError as exc:
    logger.debug("Skipping reasoning agent: %s", exc)

try:
    from .templates.smolagents import SmolCodingAgent, SmolVisionAgent

    OPTIONAL_EXPORTS.extend(["SmolCodingAgent", "SmolVisionAgent"])
except ModuleNotFoundError as exc:
    logger.debug("Skipping smolagents integrations: %s", exc)

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
if "ReasoningAgent" in globals():
    AVAILABLE_AGENTS["reasoningagent"] = cast(Type[Agent], ReasoningAgent)

__all__ = [
    "Swarm",
    "Random",
    "HeuristicAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
]
__all__.extend(OPTIONAL_EXPORTS)
