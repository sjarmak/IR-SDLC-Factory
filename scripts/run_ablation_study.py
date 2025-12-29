#!/usr/bin/env python3
"""
Ablation Study Runner for IR-SDLC-Bench.

Runs A/B experiments comparing baseline agents vs MCP-enhanced agents.
This script validates the benchmark framework end-to-end by:

1. Loading benchmark tasks from JSONL files
2. Running multiple agent configurations
3. Collecting metrics (success rate, token efficiency, time)
4. Generating comparison reports with statistical analysis

Ablation configurations:
- Baseline (no MCP) vs Deep Search MCP
- Baseline vs Full Sourcegraph MCP
- Deep Search only vs Full toolkit

Usage:
    python scripts/run_ablation_study.py --tasks 25 --output outputs/ablation
    python scripts/run_ablation_study.py --benchmark benchmarks/ir-sdlc-advanced-reasoning.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ir_sdlc.agent_metrics import (
    AgentExecutionTrace,
    AgentMetrics,
    TaskCompletionStatus,
    CodeCorrectnessLevel,
)
from app.ir_sdlc.comparative_analysis import (
    ABComparator,
    ComparisonConfig,
    ComparisonReport,
    ComparisonType,
    StatisticalAnalysis,
    create_comparison_from_runs,
    compare_multiple_tools,
    rank_tools_by_metric,
)
from app.ir_sdlc.dashboard_schema import (
    IRSDLCBenchmarkRun,
    IRSDLCTaskResult,
    IRRetrievalMetrics,
    AgentExecutionMetrics,
    IRToolType,
    SDLCTaskType,
    LLMJudgeScore,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Task Loading
# =============================================================================

def load_benchmark_tasks(benchmark_path: Path, max_tasks: Optional[int] = None) -> List[dict]:
    """Load benchmark tasks from JSONL file.
    
    Args:
        benchmark_path: Path to JSONL benchmark file
        max_tasks: Maximum number of tasks to load
        
    Returns:
        List of task dictionaries
    """
    tasks = []
    with open(benchmark_path, "r") as f:
        for line in f:
            if line.strip():
                task = json.loads(line)
                tasks.append(task)
                if max_tasks and len(tasks) >= max_tasks:
                    break
    
    logger.info(f"Loaded {len(tasks)} tasks from {benchmark_path}")
    return tasks


# =============================================================================
# Simulated Agent Runners
# =============================================================================

@dataclass
class SimulatedAgentConfig:
    """Configuration for a simulated agent."""
    name: str
    ir_tool_type: IRToolType
    
    # Performance characteristics (relative to baseline)
    base_success_rate: float = 0.6
    success_rate_boost: float = 0.0  # Bonus from MCP tools
    
    base_tokens: int = 5000
    token_reduction_pct: float = 0.0  # Reduction from better retrieval
    
    base_time_sec: float = 120.0
    time_reduction_pct: float = 0.0  # Reduction from faster search
    
    # IR-specific characteristics
    avg_ir_queries: int = 10
    ir_query_reduction: float = 0.0  # Reduction from better search
    
    # Variance (for realistic simulation)
    variance_factor: float = 0.2


# Agent configurations for ablation study
AGENT_CONFIGS = {
    "baseline": SimulatedAgentConfig(
        name="BaselineAgent",
        ir_tool_type=IRToolType.BASELINE,
        base_success_rate=0.55,
        base_tokens=6000,
        base_time_sec=150.0,
        avg_ir_queries=15,
    ),
    "deep_search": SimulatedAgentConfig(
        name="DeepSearchAgent",
        ir_tool_type=IRToolType.DEEP_SEARCH,
        base_success_rate=0.55,
        success_rate_boost=0.15,  # +15% from semantic search
        base_tokens=6000,
        token_reduction_pct=0.20,  # 20% fewer tokens
        base_time_sec=150.0,
        time_reduction_pct=0.25,  # 25% faster
        avg_ir_queries=15,
        ir_query_reduction=0.30,  # 30% fewer queries needed
    ),
    "sourcegraph_full": SimulatedAgentConfig(
        name="SourcegraphFullAgent",
        ir_tool_type=IRToolType.SOURCEGRAPH_FULL,
        base_success_rate=0.55,
        success_rate_boost=0.25,  # +25% from full toolkit
        base_tokens=6000,
        token_reduction_pct=0.30,  # 30% fewer tokens
        base_time_sec=150.0,
        time_reduction_pct=0.35,  # 35% faster
        avg_ir_queries=15,
        ir_query_reduction=0.40,  # 40% fewer queries needed
    ),
    "keyword_only": SimulatedAgentConfig(
        name="KeywordOnlyAgent",
        ir_tool_type=IRToolType.KEYWORD_ONLY,
        base_success_rate=0.55,
        success_rate_boost=0.05,  # Only slight improvement
        base_tokens=6000,
        token_reduction_pct=0.10,  # 10% fewer tokens
        base_time_sec=150.0,
        time_reduction_pct=0.15,  # 15% faster
        avg_ir_queries=15,
        ir_query_reduction=0.10,  # 10% fewer queries
    ),
}


class SimulatedAgentRunner:
    """Simulates agent execution for testing the comparison framework.
    
    Generates realistic-looking execution traces with configurable
    performance characteristics. Used for end-to-end testing before
    integrating real agents.
    """
    
    def __init__(self, config: SimulatedAgentConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = random.Random(seed)
    
    def run_task(
        self,
        task_id: str,
        task_data: dict,
        agent_import_path: str,
        model_name: str,
        timeout_seconds: int,
    ) -> AgentExecutionTrace:
        """Simulate running an agent on a task.
        
        Generates a realistic-looking execution trace based on:
        - Task difficulty (from task_data)
        - Agent configuration
        - Random variance
        """
        # Get task difficulty
        difficulty = task_data.get("difficulty", "medium")
        difficulty_multiplier = {
            "easy": 0.8,
            "medium": 1.0,
            "hard": 1.3,
            "expert": 1.6,
        }.get(difficulty, 1.0)
        
        # Calculate success probability
        success_prob = self.config.base_success_rate + self.config.success_rate_boost
        success_prob /= difficulty_multiplier  # Harder tasks reduce success
        success_prob = max(0.1, min(0.95, success_prob))  # Clamp
        
        # Add variance
        success_prob += self.rng.gauss(0, 0.05)
        success_prob = max(0.0, min(1.0, success_prob))
        
        # Determine success
        succeeded = self.rng.random() < success_prob
        
        # Calculate tokens
        base_tokens = self.config.base_tokens * difficulty_multiplier
        tokens = base_tokens * (1 - self.config.token_reduction_pct)
        tokens *= self.rng.gauss(1.0, self.config.variance_factor)
        tokens = max(500, int(tokens))
        
        # Calculate time
        base_time = self.config.base_time_sec * difficulty_multiplier
        time_sec = base_time * (1 - self.config.time_reduction_pct)
        time_sec *= self.rng.gauss(1.0, self.config.variance_factor)
        time_sec = max(10.0, time_sec)
        
        # Calculate IR queries
        base_queries = self.config.avg_ir_queries * difficulty_multiplier
        ir_queries = base_queries * (1 - self.config.ir_query_reduction)
        ir_queries *= self.rng.gauss(1.0, self.config.variance_factor * 0.5)
        ir_queries = max(1, int(ir_queries))
        
        # Determine completion status and correctness
        if succeeded:
            status = TaskCompletionStatus.SUCCESS
            correctness = CodeCorrectnessLevel.CORRECT
            tests_passed = 10
            tests_total = 10
        elif self.rng.random() < 0.3:
            status = TaskCompletionStatus.PARTIAL
            correctness = CodeCorrectnessLevel.MOSTLY_CORRECT
            tests_passed = self.rng.randint(6, 9)
            tests_total = 10
        else:
            status = TaskCompletionStatus.FAILURE
            correctness = CodeCorrectnessLevel.INCORRECT
            tests_passed = self.rng.randint(0, 5)
            tests_total = 10
        
        # Generate LLM judge score (simulated)
        if succeeded:
            llm_score = LLMJudgeScore(
                tests_pass=0.9 + self.rng.gauss(0, 0.05),
                code_changes=0.85 + self.rng.gauss(0, 0.08),
                architecture=0.8 + self.rng.gauss(0, 0.1),
                overall=0.85 + self.rng.gauss(0, 0.05),
            )
        else:
            llm_score = LLMJudgeScore(
                tests_pass=0.3 + self.rng.gauss(0, 0.15),
                code_changes=0.4 + self.rng.gauss(0, 0.15),
                architecture=0.5 + self.rng.gauss(0, 0.15),
                overall=0.4 + self.rng.gauss(0, 0.12),
            )
        
        # Clamp scores to valid range
        for attr in ["tests_pass", "code_changes", "architecture", "overall"]:
            val = getattr(llm_score, attr)
            setattr(llm_score, attr, max(0.0, min(1.0, val)))
        
        # Build trace
        trace = AgentExecutionTrace(
            task_id=task_id,
            agent_name=self.config.name,
            ir_tool_name=self.config.ir_tool_type.value,
            completed=succeeded,
            completion_status=status,
            compiles=status != TaskCompletionStatus.FAILURE,
            tests_total=tests_total,
            tests_passed=tests_passed,
            correctness_level=correctness,
            input_tokens=int(tokens * 0.7),
            output_tokens=int(tokens * 0.3),
            total_tokens=tokens,
            wall_clock_time_sec=time_sec,
            context_tokens=int(tokens * 0.4),
            context_files=ir_queries,
            llm_judge_score=llm_score,
            metadata={
                "sdlc_type": task_data.get("task_type", "bug_triage"),
                "difficulty": difficulty,
                "ir_queries": ir_queries,
            },
        )
        
        return trace


# =============================================================================
# Ablation Study Runner
# =============================================================================

@dataclass
class AblationStudyConfig:
    """Configuration for the full ablation study."""
    # Comparison experiments to run
    experiments: List[tuple]  # List of (baseline_config, treatment_config) pairs
    
    # Task selection
    benchmark_path: Path
    max_tasks: int = 50
    
    # Output
    output_dir: Path = Path("outputs/ablation")
    
    # Reproducibility
    seed: int = 42


def run_single_comparison(
    config_name: str,
    baseline_config: SimulatedAgentConfig,
    treatment_config: SimulatedAgentConfig,
    tasks: List[dict],
    seed: int,
    output_dir: Path,
) -> ComparisonReport:
    """Run a single A/B comparison experiment.
    
    Args:
        config_name: Name for this comparison
        baseline_config: Configuration for baseline agent
        treatment_config: Configuration for treatment agent
        tasks: List of tasks to run
        seed: Random seed for reproducibility
        output_dir: Directory to save results
        
    Returns:
        ComparisonReport with results
    """
    logger.info(f"Running comparison: {config_name}")
    logger.info(f"  Baseline: {baseline_config.name}")
    logger.info(f"  Treatment: {treatment_config.name}")
    logger.info(f"  Tasks: {len(tasks)}")
    
    # Create runners
    baseline_runner = SimulatedAgentRunner(baseline_config, seed=seed)
    treatment_runner = SimulatedAgentRunner(treatment_config, seed=seed + 1)
    
    # Configure comparison
    comparison_config = ComparisonConfig(
        experiment_id=config_name.replace(" ", "_").lower(),
        experiment_name=config_name,
        description=f"Comparison of {baseline_config.name} vs {treatment_config.name}",
        comparison_type=ComparisonType.BASELINE_VS_MCP,
        baseline_tool_type=baseline_config.ir_tool_type,
        baseline_agent_name=baseline_config.name,
        treatment_tool_type=treatment_config.ir_tool_type,
        treatment_agent_name=treatment_config.name,
        output_dir=str(output_dir),
    )
    
    # Create comparator
    comparator = ABComparator(
        config=comparison_config,
        baseline_runner=baseline_runner,
        treatment_runner=treatment_runner,
    )
    
    # Run comparison
    def progress_callback(task_id: str, current: int, total: int):
        if current % 10 == 0 or current == total:
            logger.info(f"  Progress: {current}/{total} tasks")
    
    report = comparator.run_comparison(tasks, progress_callback)
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = comparator.save_report()
    logger.info(f"  Report saved to: {report_path}")
    
    return report


def run_ablation_study(config: AblationStudyConfig) -> Dict[str, ComparisonReport]:
    """Run the full ablation study.
    
    Args:
        config: Study configuration
        
    Returns:
        Dict mapping experiment name to ComparisonReport
    """
    logger.info("=" * 60)
    logger.info("IR-SDLC-Bench Ablation Study")
    logger.info("=" * 60)
    
    # Load tasks
    tasks = load_benchmark_tasks(config.benchmark_path, config.max_tasks)
    
    # Run each experiment
    reports = {}
    for i, (baseline_name, treatment_name) in enumerate(config.experiments):
        baseline_config = AGENT_CONFIGS[baseline_name]
        treatment_config = AGENT_CONFIGS[treatment_name]
        
        experiment_name = f"{baseline_name}_vs_{treatment_name}"
        
        report = run_single_comparison(
            config_name=experiment_name,
            baseline_config=baseline_config,
            treatment_config=treatment_config,
            tasks=tasks,
            seed=config.seed + i * 1000,
            output_dir=config.output_dir / experiment_name,
        )
        
        reports[experiment_name] = report
    
    return reports


def generate_summary_report(
    reports: Dict[str, ComparisonReport],
    output_path: Path,
) -> None:
    """Generate a summary report across all experiments.
    
    Args:
        reports: Dict of experiment name to ComparisonReport
        output_path: Path to save summary
    """
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": {},
        "rankings": {},
    }
    
    for name, report in reports.items():
        summary["experiments"][name] = report.get_summary()
    
    # Rank by key metrics
    for metric in ["success_rate", "llm_score", "tokens"]:
        rankings = []
        for name, report in reports.items():
            if metric in report.metric_comparisons:
                mc = report.metric_comparisons[metric]
                treatment_name = report.config.treatment_agent_name
                rankings.append({
                    "agent": treatment_name,
                    "experiment": name,
                    "improvement_pct": mc.percent_improvement,
                    "is_significant": mc.hypothesis_test.is_significant if mc.hypothesis_test else None,
                })
        
        rankings.sort(key=lambda x: x["improvement_pct"], reverse=True)
        summary["rankings"][metric] = rankings
    
    # Save summary
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    
    for name, exp_summary in summary["experiments"].items():
        print(f"\n{name}:")
        print(f"  Winner: {exp_summary['overall_winner']}")
        print(f"  Treatment win rate: {exp_summary['treatment_win_rate']:.1%}")
        print(f"  Baseline win rate: {exp_summary['baseline_win_rate']:.1%}")
        
        if exp_summary.get("significant_improvements"):
            print(f"  Significant improvements: {', '.join(exp_summary['significant_improvements'])}")
        if exp_summary.get("significant_regressions"):
            print(f"  Significant regressions: {', '.join(exp_summary['significant_regressions'])}")
    
    print("\n" + "-" * 60)
    print("RANKINGS BY METRIC")
    print("-" * 60)
    
    for metric, rankings in summary["rankings"].items():
        print(f"\n{metric}:")
        for i, r in enumerate(rankings[:5], 1):
            sig = "***" if r["is_significant"] else ""
            print(f"  {i}. {r['agent']}: {r['improvement_pct']:+.1f}% {sig}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run IR-SDLC-Bench ablation study"
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("benchmarks/ir-sdlc-advanced-reasoning.jsonl"),
        help="Path to benchmark JSONL file",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=25,
        help="Maximum number of tasks to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ablation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baseline:deep_search", "baseline:sourcegraph_full", "deep_search:sourcegraph_full"],
        help="Experiments to run as baseline:treatment pairs",
    )
    
    args = parser.parse_args()
    
    # Parse experiments
    experiments = []
    for exp in args.experiments:
        parts = exp.split(":")
        if len(parts) != 2:
            logger.error(f"Invalid experiment format: {exp}. Use baseline:treatment")
            sys.exit(1)
        experiments.append((parts[0], parts[1]))
    
    # Validate experiment configs
    for baseline, treatment in experiments:
        if baseline not in AGENT_CONFIGS:
            logger.error(f"Unknown agent config: {baseline}")
            logger.info(f"Available configs: {list(AGENT_CONFIGS.keys())}")
            sys.exit(1)
        if treatment not in AGENT_CONFIGS:
            logger.error(f"Unknown agent config: {treatment}")
            logger.info(f"Available configs: {list(AGENT_CONFIGS.keys())}")
            sys.exit(1)
    
    # Validate benchmark file
    if not args.benchmark.exists():
        logger.error(f"Benchmark file not found: {args.benchmark}")
        sys.exit(1)
    
    # Run study
    study_config = AblationStudyConfig(
        experiments=experiments,
        benchmark_path=args.benchmark,
        max_tasks=args.tasks,
        output_dir=args.output,
        seed=args.seed,
    )
    
    reports = run_ablation_study(study_config)
    
    # Generate summary
    generate_summary_report(
        reports,
        args.output / "ablation_summary.json",
    )
    
    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
