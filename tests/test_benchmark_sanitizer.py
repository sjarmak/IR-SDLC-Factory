import json
import pytest
from pathlib import Path

from app.ir_sdlc.benchmark_sanitizer import BenchmarkInstructionAugmenter


@pytest.fixture
def augmenter():
    return BenchmarkInstructionAugmenter()


def write_tasks(path: Path, tasks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task) + "\n")


def read_tasks(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_adds_instruction_from_scenario(tmp_path, augmenter):
    file_path = tmp_path / "tasks.jsonl"
    write_tasks(
        file_path,
        [
            {"task_id": "task-1", "scenario": "Detailed scenario text."},
            {"task_id": "task-2", "instruction": "Existing text"},
        ],
    )

    result = augmenter.augment_file(file_path)
    assert result.tasks_modified == 1

    tasks = read_tasks(file_path)
    assert tasks[0]["instruction"] == "Detailed scenario text."
    assert tasks[1]["instruction"] == "Existing text"


def test_fallback_to_vague_prompt(tmp_path, augmenter):
    file_path = tmp_path / "tasks.jsonl"
    write_tasks(
        file_path,
        [{"task_id": "task-1", "vague_prompt": "Use fallback"}],
    )

    result = augmenter.augment_file(file_path)
    assert result.tasks_modified == 1
    assert read_tasks(file_path)[0]["instruction"] == "Use fallback"


def test_fallback_to_query_and_title(augmenter):
    task = {"task_id": "task-1", "query": "Use query"}
    assert augmenter.ensure_instruction(task) is True
    assert task["instruction"] == "Use query"

    task = {"task_id": "task-2", "title": "Title fallback"}
    assert augmenter.ensure_instruction(task) is True
    assert task["instruction"] == "Title fallback"


def test_raises_when_no_text(augmenter):
    with pytest.raises(ValueError):
        augmenter.ensure_instruction({"task_id": "task-1"})


def test_dry_run_does_not_modify_file(tmp_path, augmenter):
    file_path = tmp_path / "tasks.jsonl"
    write_tasks(file_path, [{"task_id": "task-1", "scenario": "Text"}])

    result = augmenter.augment_file(file_path, dry_run=True)
    assert result.tasks_modified == 1
    assert "instruction" not in read_tasks(file_path)[0]
