from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
LOCAL_DEPS = PROJECT_ROOT / ".deps312"
HEURISTICS_DIR = BASE_DIR / "agents" / "heuristics"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(HEURISTICS_DIR))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from arcengine import GameAction

from object_extractor import FrameAnalysis, FrameDiff, analyse_frame, diff_frames

RECORDING_SUFFIX = ".recording.jsonl"
DEFAULT_RECORDINGS_DIR = BASE_DIR / "recordings"


@dataclass(slots=True)
class StepReport:
    index: int
    action_id: int | None
    action_name: str
    state_before: str
    state_after: str
    levels_before: int
    levels_after: int
    level_delta: int
    changed_pixels: int
    changed_ratio: float
    bbox: tuple[int, int, int, int] | None
    hud_like: bool
    fingerprint_changed: bool
    substantial_change: bool
    object_count: int
    palette_size: int
    click_target: tuple[int, int] | None
    reason: str


@dataclass(slots=True)
class ActionAggregate:
    attempts: int = 0
    fingerprint_changes: int = 0
    substantial_changes: int = 0
    level_ups: int = 0
    total_changed_pixels: int = 0
    max_changed_pixels: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    click_targets: Counter[str] = field(default_factory=Counter)

    def add(self, step: StepReport) -> None:
        self.attempts += 1
        if step.fingerprint_changed:
            self.fingerprint_changes += 1
        if step.substantial_change:
            self.substantial_changes += 1
        if step.level_delta > 0:
            self.level_ups += step.level_delta
        self.total_changed_pixels += step.changed_pixels
        self.max_changed_pixels = max(self.max_changed_pixels, step.changed_pixels)
        if step.reason:
            self.reasons[step.reason] += 1
        if step.click_target is not None:
            self.click_targets[f"{step.click_target[0]},{step.click_target[1]}"] += 1

    def to_dict(self, action_name: str) -> dict[str, Any]:
        return {
            "action_name": action_name,
            "attempts": self.attempts,
            "fingerprint_changes": self.fingerprint_changes,
            "substantial_changes": self.substantial_changes,
            "level_ups": self.level_ups,
            "avg_changed_pixels": (
                round(self.total_changed_pixels / self.attempts, 2) if self.attempts else 0.0
            ),
            "max_changed_pixels": self.max_changed_pixels,
            "top_reasons": [reason for reason, _ in self.reasons.most_common(3)],
            "top_click_targets": [
                {"target": target, "count": count}
                for target, count in self.click_targets.most_common(3)
            ],
        }


def normalize_action_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        if "." in text:
            text = text.rsplit(".", maxsplit=1)[-1]
        text = text.upper()
        if text in GameAction.__members__:
            return int(GameAction[text].value)
    return None


def action_name_from_id(action_id: int | None) -> str:
    if action_id is None:
        return "UNKNOWN"
    try:
        return GameAction.from_id(action_id).name
    except (KeyError, ValueError):
        try:
            return GameAction(action_id).name
        except ValueError:
            return f"ACTION_{action_id}"


def extract_reason(reasoning: Any) -> str:
    if isinstance(reasoning, str):
        return reasoning.strip()
    if isinstance(reasoning, dict):
        if not reasoning:
            return ""
        for key in ("reason", "summary", "thought"):
            value = reasoning.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        compact = json.dumps(reasoning, ensure_ascii=False, sort_keys=True)
        return compact if len(compact) <= 120 else compact[:117] + "..."
    return ""


def is_hud_like_diff(frame_diff: FrameDiff) -> bool:
    bbox = frame_diff.bbox
    if bbox is None:
        return False
    if bbox.height <= 4 and bbox.width >= 8:
        if bbox.top <= 3 or bbox.bottom >= 60:
            return True
    return False


def frame_step_threshold(analysis: FrameAnalysis) -> int:
    step = int(analysis.lattice_step)
    if 1 < step <= 16:
        return max(4, step)
    return 4


def format_bbox(bbox: tuple[int, int, int, int] | None) -> str:
    if bbox is None:
        return "-"
    left, top, right, bottom = bbox
    return f"{left},{top}->{right},{bottom}"


def resolve_recording_path(
    recording: str | None,
    recordings_dir: Path,
    latest_prefix: str | None,
) -> Path:
    search_dirs = [recordings_dir, DEFAULT_RECORDINGS_DIR, Path.cwd() / "recordings"]
    env_recordings_dir = os.environ.get("RECORDINGS_DIR")
    if env_recordings_dir:
        search_dirs.insert(0, Path(env_recordings_dir))
    seen_dirs: list[Path] = []
    for directory in search_dirs:
        resolved = directory.resolve()
        if resolved not in seen_dirs:
            seen_dirs.append(resolved)

    if recording:
        candidate = Path(recording)
        if candidate.is_file():
            return candidate.resolve()
        for directory in seen_dirs:
            nested = directory / recording
            if nested.is_file():
                return nested.resolve()
        raise FileNotFoundError(f"Recording not found: {recording}")

    candidates: list[Path] = []
    for directory in seen_dirs:
        if not directory.exists():
            continue
        candidates.extend(
            path for path in directory.glob(f"*{RECORDING_SUFFIX}") if path.is_file()
        )

    if latest_prefix:
        candidates = [path for path in candidates if path.name.startswith(latest_prefix)]

    if not candidates:
        search_root = ", ".join(str(directory) for directory in seen_dirs)
        raise FileNotFoundError(
            f"No recordings found in [{search_root}] matching prefix {latest_prefix!r}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_recording_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number} in {path}") from exc
            if isinstance(payload, dict):
                events.append(payload)
    return events


def extract_frame_payloads(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for event in events:
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        frame = data.get("frame")
        if not isinstance(frame, list) or not frame:
            continue
        if not isinstance(frame[0], list):
            continue
        frames.append(data)
    return frames


def summarize_recording(path: Path) -> dict[str, Any]:
    events = load_recording_events(path)
    frames = extract_frame_payloads(events)
    if not frames:
        raise ValueError(f"No frame events found in recording: {path}")

    action_stats: defaultdict[str, ActionAggregate] = defaultdict(ActionAggregate)
    step_reports: list[StepReport] = []
    unique_fingerprints: set[str] = set()

    previous_payload: dict[str, Any] | None = None
    previous_analysis: FrameAnalysis | None = None
    longest_no_substantial_streak = 0
    current_no_substantial_streak = 0

    for index, payload in enumerate(frames):
        analysis = analyse_frame(payload["frame"][0])
        frame_diff = diff_frames(
            previous_analysis.grid if previous_analysis is not None else None,
            analysis.grid,
        )
        action_input = payload.get("action_input")
        if not isinstance(action_input, dict):
            action_input = {}

        action_id = normalize_action_id(action_input.get("id"))
        click_target = None
        action_data = action_input.get("data")
        if isinstance(action_data, dict):
            x = action_data.get("x")
            y = action_data.get("y")
            if isinstance(x, int) and isinstance(y, int):
                click_target = (x, y)

        state_before = (
            str(previous_payload.get("state")) if previous_payload is not None else "NOT_PLAYED"
        )
        state_after = str(payload.get("state", "UNKNOWN"))
        levels_before = (
            int(previous_payload.get("levels_completed", 0))
            if previous_payload is not None
            else 0
        )
        levels_after = int(payload.get("levels_completed", 0))
        level_delta = levels_after - levels_before
        fingerprint_changed = (
            previous_analysis is None or analysis.fingerprint != previous_analysis.fingerprint
        )
        hud_like = is_hud_like_diff(frame_diff)
        substantial_change = level_delta > 0 or (
            frame_diff.changed_pixels >= frame_step_threshold(analysis) and not hud_like
        )

        if substantial_change:
            current_no_substantial_streak = 0
        else:
            current_no_substantial_streak += 1
            longest_no_substantial_streak = max(
                longest_no_substantial_streak,
                current_no_substantial_streak,
            )

        bbox = None
        if frame_diff.bbox is not None:
            bbox = (
                frame_diff.bbox.left,
                frame_diff.bbox.top,
                frame_diff.bbox.right,
                frame_diff.bbox.bottom,
            )

        step = StepReport(
            index=index,
            action_id=action_id,
            action_name=action_name_from_id(action_id),
            state_before=state_before,
            state_after=state_after,
            levels_before=levels_before,
            levels_after=levels_after,
            level_delta=level_delta,
            changed_pixels=frame_diff.changed_pixels,
            changed_ratio=round(frame_diff.changed_ratio, 6),
            bbox=bbox,
            hud_like=hud_like,
            fingerprint_changed=fingerprint_changed,
            substantial_change=substantial_change,
            object_count=len(analysis.objects),
            palette_size=len(analysis.palette),
            click_target=click_target,
            reason=extract_reason(action_input.get("reasoning")),
        )
        step_reports.append(step)
        action_stats[step.action_name].add(step)
        unique_fingerprints.add(analysis.fingerprint)

        previous_payload = payload
        previous_analysis = analysis

    last_payload = frames[-1]
    summary = {
        "recording_path": str(path),
        "event_count": len(events),
        "frame_count": len(frames),
        "game_id": str(last_payload.get("game_id", "")),
        "final_state": str(last_payload.get("state", "UNKNOWN")),
        "levels_completed": int(last_payload.get("levels_completed", 0)),
        "win_levels": int(last_payload.get("win_levels", 0)),
        "unique_fingerprints": len(unique_fingerprints),
        "longest_no_substantial_streak": longest_no_substantial_streak,
    }
    action_rows = [
        aggregate.to_dict(action_name)
        for action_name, aggregate in sorted(
            action_stats.items(),
            key=lambda item: (-item[1].attempts, item[0]),
        )
    ]
    return {
        "summary": summary,
        "actions": action_rows,
        "steps": [asdict(step) for step in step_reports],
    }


def print_report(
    report: dict[str, Any],
    steps_to_show: int,
    impactful_only: bool,
) -> None:
    summary = report["summary"]
    print(f"Recording: {summary['recording_path']}")
    print(
        "Game: "
        f"{summary['game_id']} | Frames: {summary['frame_count']} | "
        f"Levels: {summary['levels_completed']}/{summary['win_levels']} | "
        f"Final state: {summary['final_state']}"
    )
    print(
        "Unique states: "
        f"{summary['unique_fingerprints']} | "
        f"Longest no-substantial streak: {summary['longest_no_substantial_streak']}"
    )

    print("\nAction Summary")
    print("action\tattempts\tfp_change\tsubstantial\tlevel_up\tavg_px\tmax_px\ttop_reason")
    for action in report["actions"]:
        top_reason = action["top_reasons"][0] if action["top_reasons"] else "-"
        print(
            f"{action['action_name']}\t{action['attempts']}\t"
            f"{action['fingerprint_changes']}\t{action['substantial_changes']}\t"
            f"{action['level_ups']}\t{action['avg_changed_pixels']:.2f}\t"
            f"{action['max_changed_pixels']}\t{top_reason}"
        )

    print("\nTrace")
    rows = report["steps"]
    if impactful_only:
        rows = [
            row
            for row in rows
            if row["substantial_change"] or row["level_delta"] > 0 or row["changed_pixels"] > 0
        ]
    for row in rows[:steps_to_show]:
        click_text = ""
        if row["click_target"] is not None:
            click_text = f" click={row['click_target'][0]},{row['click_target'][1]}"
        reason_text = f" reason={row['reason']}" if row["reason"] else ""
        print(
            f"#{row['index']:04d} {row['action_name']} "
            f"px={row['changed_pixels']} "
            f"bbox={format_bbox(row['bbox'])} "
            f"levels={row['levels_before']}->{row['levels_after']} "
            f"state={row['state_before']}->{row['state_after']} "
            f"substantial={'yes' if row['substantial_change'] else 'no'}"
            f"{click_text}{reason_text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ARC-AGI-3 agent replay recordings.")
    parser.add_argument(
        "recording",
        nargs="?",
        help="Path to a specific *.recording.jsonl file. If omitted, the latest recording is used.",
    )
    parser.add_argument(
        "--latest-prefix",
        default=None,
        help="Optional filename prefix used when auto-selecting the latest recording.",
    )
    parser.add_argument(
        "--recordings-dir",
        default=str(DEFAULT_RECORDINGS_DIR),
        help="Directory searched for recordings when a direct path is not provided.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="How many replay steps to print in the trace section.",
    )
    parser.add_argument(
        "--impactful-only",
        action="store_true",
        help="Only print steps that changed the frame or progressed the level.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the full analysis report as JSON.",
    )
    args = parser.parse_args()

    recording_path = resolve_recording_path(
        recording=args.recording,
        recordings_dir=Path(args.recordings_dir),
        latest_prefix=args.latest_prefix,
    )
    report = summarize_recording(recording_path)
    print_report(
        report,
        steps_to_show=max(0, args.steps),
        impactful_only=args.impactful_only,
    )

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
