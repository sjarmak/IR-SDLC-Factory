"""
Harbor Adapter for IR-SDLC-Bench.

This module generates Harbor-compatible task directories from IR tasks,
enabling seamless integration with the Harbor evaluation framework.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tomlkit

from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    GroundTruth,
    RetrievalResult,
    IREvaluationResult,
)
from app.ir_sdlc.metrics import IRMetrics


# Harbor task.toml template
TASK_TOML_TEMPLATE = """version = "1.0"

[metadata]
author_name = "{author_name}"
author_email = "{author_email}"
difficulty = "{difficulty}"
category = "information_retrieval"
task_type = "{task_type}"
repo_name = "{repo_name}"
tags = {tags}

[verifier]
timeout_sec = {verifier_timeout}

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = {build_timeout}
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
"""


# Dockerfile template for IR tasks
DOCKERFILE_TEMPLATE = """FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Clone the target repository at the specific commit
RUN git clone {repo_url} /repo && \\
    cd /repo && \\
    git checkout {commit_hash}

# Install evaluation dependencies
RUN pip install --no-cache-dir \\
    requests \\
    pyyaml

# Copy task files
COPY . /task

# Set environment variables
ENV REPO_PATH=/repo
ENV TASK_PATH=/task
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["bash"]
"""


# Test script template
TEST_SCRIPT_TEMPLATE = """#!/bin/bash
set -e

# IR-SDLC-Bench Evaluation Script
# This script evaluates IR tool retrieval results against ground truth

REPO_PATH="${{REPO_PATH:-/repo}}"
TASK_PATH="${{TASK_PATH:-/task}}"
RESULTS_FILE="${{RESULTS_FILE:-/app/retrieval_results.json}}"
REWARD_DIR="/logs/verifier"

mkdir -p "$REWARD_DIR"

# Check if results file exists
if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Retrieval results file not found at $RESULTS_FILE"
    echo '{{"error": "no_results", "mrr": 0, "recall_at_10": 0}}' > "$REWARD_DIR/reward.json"
    exit 0
fi

# Run the evaluation script
python3 /task/tests/evaluate_retrieval.py \\
    --results "$RESULTS_FILE" \\
    --ground-truth /task/ground_truth.json \\
    --output "$REWARD_DIR/reward.json"

# Also output primary score to reward.txt for compatibility
python3 -c "
import json
with open('$REWARD_DIR/reward.json') as f:
    reward = json.load(f)
    primary = reward.get('mrr', 0)
    print(primary)
" > "$REWARD_DIR/reward.txt"

echo "Evaluation complete. Results written to $REWARD_DIR/"
"""


# Python evaluation script
EVALUATE_RETRIEVAL_SCRIPT = '''#!/usr/bin/env python3
"""
Evaluation script for IR-SDLC-Bench tasks.

Computes retrieval metrics from results and ground truth.
"""

import argparse
import json
import math
import sys
from pathlib import Path


def compute_precision_at_k(retrieved_files, ground_truth_files, k):
    """Compute Precision@K."""
    if k <= 0:
        return 0.0
    top_k = retrieved_files[:k]
    if not top_k:
        return 0.0
    gt_set = set(ground_truth_files)
    relevant = sum(1 for f in top_k if f in gt_set)
    return relevant / k


def compute_recall_at_k(retrieved_files, ground_truth_files, k):
    """Compute Recall@K."""
    if not ground_truth_files:
        return 1.0
    top_k = retrieved_files[:k]
    gt_set = set(ground_truth_files)
    found = sum(1 for f in top_k if f in gt_set)
    return found / len(gt_set)


def compute_mrr(retrieved_files, ground_truth_files):
    """Compute Mean Reciprocal Rank."""
    gt_set = set(ground_truth_files)
    for i, f in enumerate(retrieved_files):
        if f in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(retrieved_files, ground_truth_files, k):
    """Compute NDCG@K."""
    if k <= 0 or not ground_truth_files:
        return 0.0

    gt_set = set(ground_truth_files)
    top_k = retrieved_files[:k]

    # DCG
    dcg = 0.0
    for i, f in enumerate(top_k):
        if f in gt_set:
            dcg += 1.0 / math.log2(i + 2)

    # IDCG
    idcg = 0.0
    for i in range(min(k, len(ground_truth_files))):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate(results_path, ground_truth_path, output_path):
    """Run evaluation and output metrics."""

    # Load results
    with open(results_path) as f:
        results = json.load(f)

    # Load ground truth
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    # Extract file paths from results
    retrieved_files = []
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                loc = item.get("location", item)
                retrieved_files.append(loc.get("file_path", ""))
            elif isinstance(item, str):
                retrieved_files.append(item)
    elif isinstance(results, dict):
        if "retrieved_results" in results:
            for item in results["retrieved_results"]:
                loc = item.get("location", item)
                retrieved_files.append(loc.get("file_path", ""))
        elif "files" in results:
            retrieved_files = results["files"]

    # Extract ground truth files
    gt_files = []
    if isinstance(ground_truth, dict):
        locations = ground_truth.get("locations", [])
        for loc in locations:
            gt_files.append(loc.get("file_path", ""))
    elif isinstance(ground_truth, list):
        for item in ground_truth:
            if isinstance(item, dict):
                gt_files.append(item.get("file_path", ""))
            elif isinstance(item, str):
                gt_files.append(item)

    # Compute metrics
    metrics = {
        "precision_at_1": compute_precision_at_k(retrieved_files, gt_files, 1),
        "precision_at_5": compute_precision_at_k(retrieved_files, gt_files, 5),
        "precision_at_10": compute_precision_at_k(retrieved_files, gt_files, 10),
        "recall_at_1": compute_recall_at_k(retrieved_files, gt_files, 1),
        "recall_at_5": compute_recall_at_k(retrieved_files, gt_files, 5),
        "recall_at_10": compute_recall_at_k(retrieved_files, gt_files, 10),
        "mrr": compute_mrr(retrieved_files, gt_files),
        "ndcg_at_10": compute_ndcg(retrieved_files, gt_files, 10),
    }

    # Write output
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics computed: MRR={metrics['mrr']:.4f}, Recall@10={metrics['recall_at_10']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate IR retrieval results")
    parser.add_argument("--results", required=True, help="Path to retrieval results JSON")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")

    args = parser.parse_args()
    evaluate(args.results, args.ground_truth, args.output)


if __name__ == "__main__":
    main()
'''


# Solution script template (oracle)
SOLUTION_SCRIPT_TEMPLATE = """#!/bin/bash
# Oracle solution for IR-SDLC-Bench
# This script outputs the ground truth as the retrieval result

TASK_PATH="${{TASK_PATH:-/task}}"

# Copy ground truth to results
cat "$TASK_PATH/ground_truth.json" | python3 -c "
import json, sys
gt = json.load(sys.stdin)
results = []
for loc in gt.get('locations', []):
    results.append({
        'location': loc,
        'score': 1.0
    })
print(json.dumps({'retrieved_results': results}, indent=2))
" > /app/retrieval_results.json

echo "Oracle solution: Ground truth copied to retrieval results"
"""


@dataclass
class HarborConfig:
    """Configuration for Harbor task generation."""
    author_name: str = "IR-SDLC-Bench"
    author_email: str = "ir-sdlc-bench@example.com"
    verifier_timeout: float = 300.0
    agent_timeout: float = 1800.0
    build_timeout: float = 600.0
    cpus: int = 2
    memory_mb: int = 4096
    storage_mb: int = 20480


class HarborTaskGenerator:
    """
    Generates Harbor-compatible task directories from IR tasks.

    Creates the following structure:
    task_id/
    ├── task.toml
    ├── instruction.md
    ├── ground_truth.json
    ├── environment/
    │   └── Dockerfile
    ├── solution/
    │   └── solve.sh
    └── tests/
        ├── test.sh
        └── evaluate_retrieval.py
    """

    def __init__(self, config: Optional[HarborConfig] = None):
        self.config = config or HarborConfig()

    def generate_task_directory(
        self,
        task: IRTask,
        output_dir: Path,
    ) -> Path:
        """
        Generate a complete Harbor task directory for an IR task.

        Args:
            task: The IR task to convert
            output_dir: Base directory for output

        Returns:
            Path to the generated task directory
        """
        task_dir = output_dir / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate task.toml
        self._generate_task_toml(task, task_dir)

        # Generate instruction.md
        self._generate_instruction(task, task_dir)

        # Generate ground_truth.json
        self._generate_ground_truth(task, task_dir)

        # Generate environment/
        self._generate_environment(task, task_dir)

        # Generate solution/
        self._generate_solution(task, task_dir)

        # Generate tests/
        self._generate_tests(task, task_dir)

        return task_dir

    def _generate_task_toml(self, task: IRTask, task_dir: Path) -> None:
        """Generate task.toml configuration file."""
        tags_str = json.dumps(task.tags)

        content = TASK_TOML_TEMPLATE.format(
            author_name=self.config.author_name,
            author_email=self.config.author_email,
            difficulty=task.difficulty,
            task_type=task.task_type,
            repo_name=task.repo_name,
            tags=tags_str,
            verifier_timeout=self.config.verifier_timeout,
            agent_timeout=self.config.agent_timeout,
            build_timeout=self.config.build_timeout,
            cpus=self.config.cpus,
            memory_mb=self.config.memory_mb,
            storage_mb=self.config.storage_mb,
        )

        (task_dir / "task.toml").write_text(content)

    def _generate_instruction(self, task: IRTask, task_dir: Path) -> None:
        """Generate instruction.md file."""
        lines = [
            f"# {task.task_type.replace('_', ' ').title()} Task",
            "",
            f"**Repository:** {task.repo_name}",
            f"**Commit:** {task.commit_hash}",
            f"**Difficulty:** {task.difficulty}",
            "",
            "## Task Description",
            "",
            task.query,
            "",
            "## Instructions",
            "",
            "Your task is to retrieve the most relevant code locations from the repository "
            "that are related to the query above.",
            "",
            "### Output Format",
            "",
            "Save your retrieval results to `/app/retrieval_results.json` with the following format:",
            "",
            "```json",
            "{",
            '  "retrieved_results": [',
            "    {",
            '      "location": {',
            '        "file_path": "path/to/file.py",',
            '        "start_line": 10,',
            '        "end_line": 50,',
            '        "function_name": "function_name"  // optional',
            "      },",
            '      "score": 0.95,',
            '      "snippet": "..."  // optional',
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "### Evaluation",
            "",
            "Your retrieval results will be evaluated using standard IR metrics:",
            "- Precision@K",
            "- Recall@K",
            "- Mean Reciprocal Rank (MRR)",
            "- NDCG@K",
            "",
            "### Repository Location",
            "",
            f"The repository is cloned to `/repo` at commit `{task.commit_hash}`.",
            "",
        ]

        # Add context if available
        if task.context:
            lines.extend([
                "## Additional Context",
                "",
                "```json",
                json.dumps(task.context, indent=2),
                "```",
                "",
            ])

        (task_dir / "instruction.md").write_text("\n".join(lines))

    def _generate_ground_truth(self, task: IRTask, task_dir: Path) -> None:
        """Generate ground_truth.json file."""
        if task.ground_truth:
            gt_data = task.ground_truth.to_dict()
        else:
            gt_data = {"locations": [], "granularity": "file"}

        (task_dir / "ground_truth.json").write_text(json.dumps(gt_data, indent=2))

    def _generate_environment(self, task: IRTask, task_dir: Path) -> None:
        """Generate environment/ directory with Dockerfile."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(exist_ok=True)

        dockerfile_content = DOCKERFILE_TEMPLATE.format(
            repo_url=task.repo_url,
            commit_hash=task.commit_hash,
        )

        (env_dir / "Dockerfile").write_text(dockerfile_content)

    def _generate_solution(self, task: IRTask, task_dir: Path) -> None:
        """Generate solution/ directory with solve.sh."""
        solution_dir = task_dir / "solution"
        solution_dir.mkdir(exist_ok=True)

        (solution_dir / "solve.sh").write_text(SOLUTION_SCRIPT_TEMPLATE)
        os.chmod(solution_dir / "solve.sh", 0o755)

    def _generate_tests(self, task: IRTask, task_dir: Path) -> None:
        """Generate tests/ directory with test.sh and evaluation script."""
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(exist_ok=True)

        # test.sh
        (tests_dir / "test.sh").write_text(TEST_SCRIPT_TEMPLATE)
        os.chmod(tests_dir / "test.sh", 0o755)

        # evaluate_retrieval.py
        (tests_dir / "evaluate_retrieval.py").write_text(EVALUATE_RETRIEVAL_SCRIPT)
        os.chmod(tests_dir / "evaluate_retrieval.py", 0o755)

    def generate_dataset(
        self,
        dataset: IRDataset,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate Harbor task directories for all tasks in a dataset.

        Args:
            dataset: The IR dataset to convert
            output_dir: Base directory for output

        Returns:
            List of paths to generated task directories
        """
        dataset_dir = output_dir / dataset.name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        generated_paths = []
        for task in dataset.tasks:
            task_path = self.generate_task_directory(task, dataset_dir)
            generated_paths.append(task_path)

        # Generate dataset-level metadata
        self._generate_dataset_readme(dataset, dataset_dir)

        return generated_paths

    def _generate_dataset_readme(self, dataset: IRDataset, dataset_dir: Path) -> None:
        """Generate README.md for the dataset."""
        lines = [
            f"# {dataset.name}",
            "",
            f"**Version:** {dataset.version}",
            "",
            dataset.description,
            "",
            "## Tasks",
            "",
            f"This dataset contains {len(dataset.tasks)} IR evaluation tasks.",
            "",
            "### Task Types",
            "",
        ]

        # Count tasks by type
        type_counts = {}
        for task in dataset.tasks:
            task_type = task.task_type
            type_counts[task_type] = type_counts.get(task_type, 0) + 1

        for task_type, count in sorted(type_counts.items()):
            lines.append(f"- **{task_type}**: {count} tasks")

        lines.extend([
            "",
            "## Usage",
            "",
            "### With Harbor CLI",
            "",
            "```bash",
            f"# Run on entire dataset",
            f'harbor jobs start -p datasets/{dataset.name} -a <agent> -m <model>',
            "",
            "# Run on single task",
            f'harbor trials start -p datasets/{dataset.name}/<task-id> -a <agent> -m <model>',
            "```",
            "",
            "### From Registry (after registration)",
            "",
            "```bash",
            f'harbor jobs start -d {dataset.name}@{dataset.version} -a <agent> -m <model>',
            "```",
            "",
        ])

        (dataset_dir / "README.md").write_text("\n".join(lines))


def generate_harbor_task(task: IRTask, output_dir: Path, config: Optional[HarborConfig] = None) -> Path:
    """
    Convenience function to generate a Harbor task from an IR task.

    Args:
        task: The IR task to convert
        output_dir: Output directory
        config: Optional Harbor configuration

    Returns:
        Path to the generated task directory
    """
    generator = HarborTaskGenerator(config)
    return generator.generate_task_directory(task, output_dir)


def generate_harbor_dataset(
    dataset: IRDataset,
    output_dir: Path,
    config: Optional[HarborConfig] = None,
) -> list[Path]:
    """
    Convenience function to generate Harbor tasks for an entire dataset.

    Args:
        dataset: The IR dataset to convert
        output_dir: Output directory
        config: Optional Harbor configuration

    Returns:
        List of paths to generated task directories
    """
    generator = HarborTaskGenerator(config)
    return generator.generate_dataset(dataset, output_dir)


def generate_registry_entry(
    dataset: IRDataset,
    git_url: str,
    git_commit_id: str = "head",
    base_path: str = "datasets",
) -> dict:
    """
    Generate a Harbor registry.json entry for a dataset.

    Args:
        dataset: The IR dataset
        git_url: Git URL for the harbor-datasets repository
        git_commit_id: Commit ID (use "head" for latest)
        base_path: Base path within the repository

    Returns:
        Dictionary suitable for inclusion in registry.json
    """
    return dataset.to_harbor_registry(git_url, git_commit_id, base_path)
