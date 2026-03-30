from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DEPS = PROJECT_ROOT / ".deps312"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env.example")
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ENVIRONMENTS_DIR", str(PROJECT_ROOT / "environment_files"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RECORDINGS_DIR", str(BASE_DIR / "recordings"))

from arc_agi import Arcade, OperationMode

from agents import AVAILABLE_AGENTS

ROOT_URL = "http://offline.local"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_games(arcade: Arcade, filters: Iterable[str], limit: int | None) -> list[str]:
    available = sorted(environment.game_id for environment in arcade.get_environments())
    filters = [value for value in filters if value]
    if filters:
        selected = [
            game_id
            for game_id in available
            if any(game_id.startswith(prefix) for prefix in filters)
        ]
    else:
        selected = available

    if limit is not None:
        selected = selected[:limit]
    return selected


def format_scorecard(scorecard: object) -> str:
    environments = getattr(scorecard, "environments", [])
    if not environments:
        return "No environment scores were produced."

    rows = [
        (
            environment.id,
            environment.score,
            environment.levels_completed,
            environment.level_count,
            environment.actions,
            environment.resets,
        )
        for environment in environments
    ]
    rows.sort(key=lambda row: (-row[1], row[0]))

    lines = [
        "game_id\tscore\tlevels\tactions\tresets",
    ]
    for game_id, score, levels_completed, level_count, actions, resets in rows:
        lines.append(
            f"{game_id}\t{score:.2f}\t{levels_completed}/{level_count}\t{actions}\t{resets}"
        )
    return "\n".join(lines)


def main() -> int:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run ARC-AGI-3 agents locally in offline mode.")
    parser.add_argument(
        "--agent",
        default="heuristicagent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Agent to run against offline environments.",
    )
    parser.add_argument(
        "--games",
        default="",
        help="Comma-separated list of game_id prefixes, for example 'ls20,ft09'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of games to run.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable agent-side JSONL recordings.",
    )
    parser.add_argument(
        "--tags",
        default="offline,benchmark",
        help="Comma-separated tags stored on the local scorecard.",
    )
    args = parser.parse_args()

    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    selected_games = resolve_games(
        arcade,
        filters=[value.strip() for value in args.games.split(",")],
        limit=args.limit,
    )

    if not selected_games:
        logging.error("No offline games matched the provided filters.")
        return 1

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    agent_class = AVAILABLE_AGENTS[args.agent]
    logging.info("Running %s on %d game(s): %s", args.agent, len(selected_games), selected_games)
    if args.record:
        logging.info(
            "Recording gameplay traces into %s",
            Path(os.environ["RECORDINGS_DIR"]).resolve(),
        )

    card_id = arcade.open_scorecard(tags=tags)
    failures: list[tuple[str, str]] = []

    try:
        for game_id in selected_games:
            logging.info("Starting %s", game_id)
            environment = arcade.make(
                game_id,
                scorecard_id=card_id,
                save_recording=False,
            )
            if environment is None:
                failures.append((game_id, "environment could not be created"))
                continue

            agent = agent_class(
                card_id=card_id,
                game_id=game_id,
                agent_name=args.agent,
                ROOT_URL=ROOT_URL,
                record=args.record,
                arc_env=environment,
                tags=tags,
            )

            try:
                agent.main()
            except Exception as exc:  # noqa: BLE001
                logging.exception("Agent failed on %s", game_id)
                failures.append((game_id, str(exc)))
    finally:
        scorecard = arcade.close_scorecard(card_id)

    if scorecard is None:
        logging.error("Local scorecard could not be closed.")
        return 1

    print(format_scorecard(scorecard))
    print(f"\nOverall score: {scorecard.score:.2f}")

    if failures:
        print("\nFailures:")
        for game_id, message in failures:
            print(f"- {game_id}: {message}")

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
