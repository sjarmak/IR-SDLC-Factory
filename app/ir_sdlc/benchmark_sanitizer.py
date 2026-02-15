"""Utilities for patching benchmark files with missing instruction fields."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass
class AugmentResult:
    """Encapsulate the result of augmenting instructions."""

    tasks_processed: int
    tasks_modified: int


class BenchmarkInstructionAugmenter:
    """Adds instruction fields to benchmark tasks when missing."""

    def __init__(self, fallback_fields: Sequence[str] | None = None):
        self.fallback_fields = fallback_fields or (
            "scenario",
            "vague_prompt",
            "query",
            "title",
        )

    def ensure_instruction(self, task: dict) -> bool:
        """Ensure a single task contains an instruction field.

        Returns True if the task was modified.
        """
        existing = task.get("instruction")
        if isinstance(existing, str) and existing.strip():
            return False

        instruction = self._select_instruction_text(task)
        task["instruction"] = instruction
        return True

    def augment_tasks(self, tasks: Iterable[dict]) -> AugmentResult:
        processed = 0
        modified = 0
        for task in tasks:
            processed += 1
            if self.ensure_instruction(task):
                modified += 1
        return AugmentResult(tasks_processed=processed, tasks_modified=modified)

    def augment_file(self, path: Path, *, dry_run: bool = False) -> AugmentResult:
        """Add missing instruction fields in-place."""
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle if line.strip()]

        tasks = [json.loads(line) for line in lines]
        result = self.augment_tasks(tasks)

        if not dry_run and result.tasks_modified:
            serialized = [json.dumps(task, ensure_ascii=True) for task in tasks]
            content = "\n".join(serialized) + "\n"
            path.write_text(content, encoding="utf-8")

        return result

    def _select_instruction_text(self, task: dict) -> str:
        for field in self.fallback_fields:
            value = task.get(field)
            if isinstance(value, str):
                candidate = value.strip()
                if candidate:
                    return candidate
        raise ValueError(
            f"Unable to infer instruction text for task {task.get('task_id', '<unknown>')}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add instruction fields to benchmarks")
    parser.add_argument(
        "--tasks-file",
        action="append",
        required=True,
        help="Path to a benchmark JSONL file. Repeat for multiple files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many tasks would change without editing files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    augmenter = BenchmarkInstructionAugmenter()

    for file_path in args.tasks_file:
        path = Path(file_path)
        result = augmenter.augment_file(path, dry_run=args.dry_run)
        logger.info(
            "Processed %s: %d tasks, %d modified%s",
            path,
            result.tasks_processed,
            result.tasks_modified,
            " (dry run)" if args.dry_run else "",
        )


if __name__ == "__main__":
    main()
