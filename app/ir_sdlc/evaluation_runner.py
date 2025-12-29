#!/usr/bin/env python3
"""
IR Evaluation Runner for IR-SDLC-Bench.

This module provides the main evaluation loop for running IR tools
on benchmark tasks and computing metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    RetrievalResult,
    IREvaluationResult,
)
from app.ir_sdlc.metrics import IRMetrics, aggregate_metrics
from app.ir_sdlc.ir_tool_interface import (
    IRToolInterface,
    IRToolConfig,
    TimedIRTool,
    get_ir_tool,
)
from app.ir_sdlc.harbor_adapter import (
    HarborTaskGenerator,
    HarborConfig,
    generate_harbor_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation run."""
    tool_name: str
    tool_config: Optional[dict] = None

    # Evaluation settings
    top_k_values: Optional[list[int]] = None  # K values for P@K, R@K, etc.
    primary_metric: str = "mrr"

    # Execution settings
    num_workers: int = 1
    timeout_per_task_sec: float = 300.0

    # Output settings
    output_dir: str = "results"
    save_detailed_results: bool = True
    generate_harbor_output: bool = True

    def __post_init__(self):
        if self.top_k_values is None:
            self.top_k_values = [1, 5, 10, 20]
        if self.tool_config is None:
            self.tool_config = {}


class IREvaluationRunner:
    """
    Main evaluation runner for IR-SDLC-Bench.

    Orchestrates the evaluation of IR tools on benchmark tasks,
    computes metrics, and generates reports.
    """

    def __init__(
        self,
        ir_tool: IRToolInterface,
        config: EvaluationConfig,
    ):
        self.ir_tool = TimedIRTool(ir_tool)
        self.config = config
        self.results: list[IREvaluationResult] = []
        self._start_time: Optional[datetime] = None

    def evaluate_task(
        self,
        task: IRTask,
        repo_path: str,
    ) -> IREvaluationResult:
        """
        Evaluate the IR tool on a single task.

        Args:
            task: The IR task to evaluate
            repo_path: Path to the cloned repository

        Returns:
            Evaluation result with metrics
        """
        logger.info(f"Evaluating task: {task.task_id}")

        # Index the repository if not already done
        if self.ir_tool._indexed_repo != repo_path:
            logger.info(f"Indexing repository: {repo_path}")
            self.ir_tool.index_repository(repo_path)

        # Run retrieval
        start_time = time.perf_counter()

        if task.context:
            retrieved = self.ir_tool.retrieve_with_context(
                query=task.query,
                context=task.context,
                top_k=max(self.config.top_k_values) if self.config.top_k_values else 20,
            )
        else:
            retrieved = self.ir_tool.retrieve(
                query=task.query,
                top_k=max(self.config.top_k_values) if self.config.top_k_values else 20,
            )

        retrieval_time_ms = (time.perf_counter() - start_time) * 1000

        # Compute metrics
        metrics = {}
        if task.ground_truth:
            ir_metrics = IRMetrics.compute_all(
                retrieved=retrieved,
                ground_truth=task.ground_truth,
                retrieval_time_ms=retrieval_time_ms,
            )
            metrics = ir_metrics.to_dict()
        else:
            metrics = {"retrieval_time_ms": retrieval_time_ms}

        result = IREvaluationResult(
            task_id=task.task_id,
            tool_name=self.ir_tool.name,
            retrieved_results=retrieved,
            metrics=metrics,
            retrieval_time_ms=retrieval_time_ms,
            indexing_time_ms=self.ir_tool._index_time_ms,
            tool_config=self.config.tool_config or {},
        )

        return result

    def evaluate_dataset(
        self,
        dataset: IRDataset,
        repos_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Evaluate the IR tool on an entire dataset.

        Args:
            dataset: The IR dataset to evaluate
            repos_dir: Base directory containing cloned repositories
            progress_callback: Optional callback for progress updates

        Returns:
            Aggregated results dictionary
        """
        self._start_time = datetime.now()
        self.results = []

        total_tasks = len(dataset.tasks)
        logger.info(f"Evaluating {total_tasks} tasks from dataset: {dataset.name}")

        for i, task in enumerate(dataset.tasks):
            try:
                # Determine repository path
                repo_slug = task.repo_name.replace("/", "__")
                repo_path = os.path.join(repos_dir, repo_slug)

                if not os.path.exists(repo_path):
                    logger.warning(f"Repository not found: {repo_path}")
                    continue

                result = self.evaluate_task(task, repo_path)
                self.results.append(result)

                if progress_callback:
                    progress_callback(i + 1, total_tasks)

            except Exception as e:
                logger.error(f"Error evaluating task {task.task_id}: {e}")
                continue

        # Aggregate results
        aggregated = self._aggregate_results()

        # Save results
        self._save_results(dataset, aggregated)

        return aggregated

    def _aggregate_results(self) -> dict:
        """Aggregate results across all tasks."""
        if not self.results:
            return {}

        # Collect all metrics
        all_metrics = []
        for result in self.results:
            if result.metrics:
                metrics = IRMetrics(
                    precision_at_1=result.metrics.get("precision@1", 0),
                    precision_at_5=result.metrics.get("precision@5", 0),
                    precision_at_10=result.metrics.get("precision@10", 0),
                    recall_at_10=result.metrics.get("recall@10", 0),
                    mrr=result.metrics.get("mrr", 0),
                    ndcg_at_10=result.metrics.get("ndcg@10", 0),
                    retrieval_time_ms=result.metrics.get("retrieval_time_ms", 0),
                )
                all_metrics.append(metrics)

        aggregated = aggregate_metrics(all_metrics)

        return {
            "tool_name": self.ir_tool.name,
            "num_tasks": len(self.results),
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": datetime.now().isoformat(),
            "metrics": aggregated,
        }

    def _save_results(self, dataset: IRDataset, aggregated: dict) -> None:
        """Save evaluation results to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_dir / f"{self.ir_tool.name}_{timestamp}"
        run_dir.mkdir(exist_ok=True)

        # Save aggregated results
        with open(run_dir / "summary.json", "w") as f:
            json.dump(aggregated, f, indent=2)

        # Save detailed results if enabled
        if self.config.save_detailed_results:
            detailed = [result.to_dict() for result in self.results]
            with open(run_dir / "detailed_results.jsonl", "w") as f:
                for result in detailed:
                    f.write(json.dumps(result) + "\n")

        # Generate Harbor-compatible output if enabled
        if self.config.generate_harbor_output:
            self._generate_harbor_output(run_dir, dataset)

        logger.info(f"Results saved to: {run_dir}")

    def _generate_harbor_output(self, run_dir: Path, dataset: IRDataset) -> None:
        """Generate Harbor-compatible reward files."""
        harbor_dir = run_dir / "harbor_results"
        harbor_dir.mkdir(exist_ok=True)

        for result in self.results:
            task_dir = harbor_dir / result.task_id
            task_dir.mkdir(exist_ok=True)

            # Generate reward.json
            reward = result.to_harbor_reward()
            with open(task_dir / "reward.json", "w") as f:
                json.dump(reward, f, indent=2)

            # Generate reward.txt (primary metric)
            primary = reward.get(self.config.primary_metric, reward.get("mrr", 0))
            with open(task_dir / "reward.txt", "w") as f:
                f.write(str(primary))


def run_ir_evaluation(
    tasks_file: str,
    ir_tool_name: str,
    repos_dir: str,
    output_dir: str,
    tool_config: Optional[dict] = None,
    num_workers: int = 1,
    generate_harbor: bool = True,
) -> dict:
    """
    Convenience function to run IR evaluation.

    Args:
        tasks_file: Path to JSONL file with IR tasks
        ir_tool_name: Name of the IR tool to use
        repos_dir: Directory containing cloned repositories
        output_dir: Directory for output results
        tool_config: Optional tool configuration
        num_workers: Number of parallel workers
        generate_harbor: Whether to generate Harbor-compatible output

    Returns:
        Aggregated evaluation results
    """
    # Load dataset
    dataset = IRDataset.load_jsonl(Path(tasks_file))

    # Create IR tool
    config = IRToolConfig(
        name=ir_tool_name,
        parameters=tool_config or {},
    )
    ir_tool = get_ir_tool(ir_tool_name, config)

    # Create evaluation config
    eval_config = EvaluationConfig(
        tool_name=ir_tool_name,
        tool_config=tool_config or {},
        output_dir=output_dir,
        num_workers=num_workers,
        generate_harbor_output=generate_harbor,
    )

    # Run evaluation
    runner = IREvaluationRunner(ir_tool, eval_config)
    results = runner.evaluate_dataset(dataset, repos_dir)

    return results


def generate_harbor_tasks(
    tasks_file: str,
    output_dir: str,
    harbor_config: Optional[HarborConfig] = None,
) -> list[Path]:
    """
    Generate Harbor-compatible task directories from IR tasks.

    Args:
        tasks_file: Path to JSONL file with IR tasks
        output_dir: Directory for output Harbor tasks
        harbor_config: Optional Harbor configuration

    Returns:
        List of generated task directory paths
    """
    # Load dataset
    dataset = IRDataset.load_jsonl(Path(tasks_file))

    # Generate Harbor tasks
    return generate_harbor_dataset(dataset, Path(output_dir), harbor_config)
