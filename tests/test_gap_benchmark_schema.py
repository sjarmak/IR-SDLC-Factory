import json
from pathlib import Path

GAP_FILE = Path(__file__).parent.parent / "benchmarks" / "ir-sdlc-gap-filling.jsonl"


def load_tasks():
    with GAP_FILE.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_gap_tasks_include_repo_urls_and_commits():
    tasks = load_tasks()
    assert tasks, "gap benchmark should contain tasks"
    for task in tasks:
        assert "repo_url" in task and task["repo_url"].startswith("https://github.com/"), task
        assert task.get("commit_hash"), task


def test_gap_tasks_have_instruction_text():
    tasks = load_tasks()
    for task in tasks:
        instruction = task.get("instruction", "").strip()
        assert instruction, task
