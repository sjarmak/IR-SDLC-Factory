"""
Agent Output Quality Metrics for IR-SDLC-Bench.

Defines metrics to measure how IR tool choice impacts coding agent output quality.
These metrics complement the base IR metrics (IRMetrics) by measuring downstream
effects on agent performance.

Metrics covered:
1. Task completion rate - Binary success/failure
2. Code correctness - Test pass rate, compilation success
3. Token efficiency - Tokens used per task
4. Time efficiency - Wall clock time
5. Retrieval quality correlation - How IR metrics predict agent success
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.ir_sdlc.metrics import IRMetrics


class TaskCompletionStatus(Enum):
    """Possible completion statuses for an agent task."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ERROR = "error"


class CodeCorrectnessLevel(Enum):
    """Levels of code correctness for agent output."""
    CORRECT = "correct"           # All tests pass, compiles
    MOSTLY_CORRECT = "mostly_correct"  # Minor issues, >80% tests pass
    PARTIALLY_CORRECT = "partially_correct"  # Some tests pass, 50-80%
    INCORRECT = "incorrect"       # <50% tests pass or doesn't compile
    NO_OUTPUT = "no_output"       # Agent produced no code


@dataclass
class AgentExecutionTrace:
    """
    Captures the execution trace of a coding agent on a task.
    
    This is the input for computing agent output quality metrics.
    """
    # Task identification
    task_id: str
    agent_name: str
    ir_tool_name: str
    
    # Completion metrics
    completed: bool = False
    completion_status: TaskCompletionStatus = TaskCompletionStatus.FAILURE
    
    # Code correctness metrics
    compiles: bool = False
    tests_total: int = 0
    tests_passed: int = 0
    correctness_level: CodeCorrectnessLevel = CodeCorrectnessLevel.NO_OUTPUT
    
    # Efficiency metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    wall_clock_time_sec: float = 0.0
    
    # IR context provided to agent
    context_tokens: int = 0  # Tokens in retrieval context
    context_files: int = 0   # Number of files in context
    
    # Associated IR metrics (if available)
    ir_metrics: Optional[IRMetrics] = None
    
    # Metadata
    error_message: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


def compute_task_completion_rate(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute task completion rate across a set of agent executions.
    
    Returns the fraction of tasks where the agent successfully completed
    the task (status == SUCCESS).
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Completion rate as a float between 0.0 and 1.0
    """
    if not traces:
        return 0.0
    
    successful = sum(
        1 for t in traces 
        if t.completion_status == TaskCompletionStatus.SUCCESS
    )
    return successful / len(traces)


def compute_partial_completion_rate(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute rate of at least partial task completion.
    
    Returns the fraction of tasks where the agent achieved at least
    partial success (SUCCESS or PARTIAL).
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Partial completion rate as a float between 0.0 and 1.0
    """
    if not traces:
        return 0.0
    
    at_least_partial = sum(
        1 for t in traces 
        if t.completion_status in (
            TaskCompletionStatus.SUCCESS, 
            TaskCompletionStatus.PARTIAL
        )
    )
    return at_least_partial / len(traces)


def compute_test_pass_rate(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute overall test pass rate across all agent executions.
    
    Aggregates tests_passed / tests_total across all tasks that have tests.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Test pass rate as a float between 0.0 and 1.0
    """
    total_tests = sum(t.tests_total for t in traces)
    total_passed = sum(t.tests_passed for t in traces)
    
    if total_tests == 0:
        return 0.0
    
    return total_passed / total_tests


def compute_compilation_rate(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute the fraction of tasks where agent output compiles successfully.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Compilation success rate as a float between 0.0 and 1.0
    """
    if not traces:
        return 0.0
    
    # Only count traces that produced output
    traces_with_output = [
        t for t in traces 
        if t.correctness_level != CodeCorrectnessLevel.NO_OUTPUT
    ]
    
    if not traces_with_output:
        return 0.0
    
    compiled = sum(1 for t in traces_with_output if t.compiles)
    return compiled / len(traces_with_output)


def compute_code_correctness_score(trace: AgentExecutionTrace) -> float:
    """
    Compute a normalized code correctness score for a single execution.
    
    Score is based on:
    - Compilation: 0.2 points
    - Test pass rate: 0.8 points (scaled by tests passed / tests total)
    
    Args:
        trace: A single agent execution trace
        
    Returns:
        Correctness score as a float between 0.0 and 1.0
    """
    if trace.correctness_level == CodeCorrectnessLevel.NO_OUTPUT:
        return 0.0
    
    score = 0.0
    
    # Compilation contributes 20%
    if trace.compiles:
        score += 0.2
    
    # Test pass rate contributes 80%
    if trace.tests_total > 0:
        test_score = trace.tests_passed / trace.tests_total
        score += 0.8 * test_score
    elif trace.compiles:
        # If compiles but no tests, give partial credit
        score += 0.3
    
    return score


def compute_average_correctness_score(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute average code correctness score across all executions.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Average correctness score as a float between 0.0 and 1.0
    """
    if not traces:
        return 0.0
    
    scores = [compute_code_correctness_score(t) for t in traces]
    return statistics.mean(scores)


def compute_tokens_per_task(traces: list[AgentExecutionTrace]) -> dict:
    """
    Compute token usage statistics per task.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Dictionary with mean, median, std, min, max token usage
    """
    if not traces:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0, "max": 0}
    
    tokens = [t.total_tokens for t in traces]
    
    return {
        "mean": statistics.mean(tokens),
        "median": statistics.median(tokens),
        "std": statistics.stdev(tokens) if len(tokens) > 1 else 0.0,
        "min": min(tokens),
        "max": max(tokens),
    }


def compute_tokens_per_successful_task(traces: list[AgentExecutionTrace]) -> dict:
    """
    Compute token usage for successful tasks only.
    
    This measures the "cost" of a successful task completion.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Dictionary with token statistics for successful tasks
    """
    successful = [
        t for t in traces 
        if t.completion_status == TaskCompletionStatus.SUCCESS
    ]
    
    return compute_tokens_per_task(successful)


def compute_token_efficiency(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute token efficiency: successful completions per 1000 tokens.
    
    Higher is better - more completions for fewer tokens.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Efficiency score (successful completions per 1000 tokens)
    """
    if not traces:
        return 0.0
    
    total_tokens = sum(t.total_tokens for t in traces)
    if total_tokens == 0:
        return 0.0
    
    successful = sum(
        1 for t in traces 
        if t.completion_status == TaskCompletionStatus.SUCCESS
    )
    
    # Normalize to per 1000 tokens
    return (successful / total_tokens) * 1000


def compute_context_efficiency(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute context efficiency: how much of provided context is used effectively.
    
    Measures successful completions relative to context tokens provided.
    Higher is better - more success per context token.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Context efficiency score
    """
    if not traces:
        return 0.0
    
    total_context_tokens = sum(t.context_tokens for t in traces)
    if total_context_tokens == 0:
        return 0.0
    
    successful = sum(
        1 for t in traces 
        if t.completion_status == TaskCompletionStatus.SUCCESS
    )
    
    return (successful / total_context_tokens) * 1000


def compute_time_per_task(traces: list[AgentExecutionTrace]) -> dict:
    """
    Compute wall clock time statistics per task.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Dictionary with mean, median, std, min, max time in seconds
    """
    if not traces:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    times = [t.wall_clock_time_sec for t in traces]
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
    }


def compute_timeout_rate(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute the fraction of tasks that timed out.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Timeout rate as a float between 0.0 and 1.0
    """
    if not traces:
        return 0.0
    
    timeouts = sum(
        1 for t in traces 
        if t.completion_status == TaskCompletionStatus.TIMEOUT
    )
    return timeouts / len(traces)


def compute_time_efficiency(traces: list[AgentExecutionTrace]) -> float:
    """
    Compute time efficiency: successful completions per minute.
    
    Higher is better - more completions in less time.
    
    Args:
        traces: List of agent execution traces
        
    Returns:
        Efficiency score (successful completions per minute)
    """
    if not traces:
        return 0.0
    
    total_time_sec = sum(t.wall_clock_time_sec for t in traces)
    if total_time_sec == 0:
        return 0.0
    
    successful = sum(
        1 for t in traces 
        if t.completion_status == TaskCompletionStatus.SUCCESS
    )
    
    # Convert to per minute
    total_time_min = total_time_sec / 60
    return successful / total_time_min


def compute_ir_success_correlation(traces: list[AgentExecutionTrace]) -> dict:
    """
    Compute correlation between IR metrics and agent success.
    
    Calculates Pearson correlation coefficient between various IR metrics
    and task completion success. This helps identify which IR metrics
    best predict agent success.
    
    Args:
        traces: List of agent execution traces (must have ir_metrics populated)
        
    Returns:
        Dictionary mapping IR metric names to their correlation with success
    """
    # Filter traces with IR metrics
    traces_with_ir = [t for t in traces if t.ir_metrics is not None]
    
    if len(traces_with_ir) < 3:
        return {}  # Not enough data for meaningful correlation
    
    # Convert success to binary
    success_values = [
        1.0 if t.completion_status == TaskCompletionStatus.SUCCESS else 0.0
        for t in traces_with_ir
    ]
    
    correlations = {}
    
    # Key IR metrics to correlate
    ir_metric_extractors = {
        "mrr": lambda m: m.mrr,
        "recall@10": lambda m: m.recall_at_10,
        "precision@10": lambda m: m.precision_at_10,
        "ndcg@10": lambda m: m.ndcg_at_10,
        "hit@5": lambda m: m.hit_at_5,
        "context_efficiency": lambda m: m.context_efficiency,
        "file_level_recall": lambda m: m.file_level_recall,
        "cross_module_coverage": lambda m: m.cross_module_coverage,
    }
    
    for metric_name, extractor in ir_metric_extractors.items():
        try:
            metric_values = [extractor(t.ir_metrics) for t in traces_with_ir]
            corr = _pearson_correlation(metric_values, success_values)
            if corr is not None:
                correlations[metric_name] = corr
        except (AttributeError, TypeError):
            continue
    
    return correlations


def _pearson_correlation(x: list[float], y: list[float]) -> Optional[float]:
    """
    Compute Pearson correlation coefficient between two lists.
    
    Args:
        x: First list of values
        y: Second list of values
        
    Returns:
        Pearson correlation coefficient, or None if cannot compute
    """
    if len(x) != len(y) or len(x) < 2:
        return None
    
    n = len(x)
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    # Compute covariance and standard deviations
    covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    
    if std_x == 0 or std_y == 0:
        return None
    
    return covariance / (std_x * std_y)


def compute_ir_correctness_correlation(traces: list[AgentExecutionTrace]) -> dict:
    """
    Compute correlation between IR metrics and code correctness score.
    
    This measures how well IR retrieval quality predicts code correctness,
    not just binary success.
    
    Args:
        traces: List of agent execution traces (must have ir_metrics populated)
        
    Returns:
        Dictionary mapping IR metric names to their correlation with correctness
    """
    traces_with_ir = [t for t in traces if t.ir_metrics is not None]
    
    if len(traces_with_ir) < 3:
        return {}
    
    correctness_scores = [
        compute_code_correctness_score(t) for t in traces_with_ir
    ]
    
    correlations = {}
    
    ir_metric_extractors = {
        "mrr": lambda m: m.mrr,
        "recall@10": lambda m: m.recall_at_10,
        "precision@10": lambda m: m.precision_at_10,
        "ndcg@10": lambda m: m.ndcg_at_10,
        "hit@5": lambda m: m.hit_at_5,
        "context_efficiency": lambda m: m.context_efficiency,
        "file_level_recall": lambda m: m.file_level_recall,
    }
    
    for metric_name, extractor in ir_metric_extractors.items():
        try:
            metric_values = [extractor(t.ir_metrics) for t in traces_with_ir]
            corr = _pearson_correlation(metric_values, correctness_scores)
            if corr is not None:
                correlations[metric_name] = corr
        except (AttributeError, TypeError):
            continue
    
    return correlations


@dataclass
class AgentMetrics:
    """
    Container for computing and storing agent output quality metrics.
    
    Provides a convenient interface for computing all metrics at once
    and converting to various output formats.
    """
    # Task completion metrics
    task_completion_rate: float = 0.0
    partial_completion_rate: float = 0.0
    
    # Code correctness metrics
    test_pass_rate: float = 0.0
    compilation_rate: float = 0.0
    average_correctness_score: float = 0.0
    
    # Token efficiency metrics
    tokens_per_task: dict = field(default_factory=dict)
    tokens_per_successful_task: dict = field(default_factory=dict)
    token_efficiency: float = 0.0
    context_efficiency: float = 0.0
    
    # Time efficiency metrics
    time_per_task: dict = field(default_factory=dict)
    timeout_rate: float = 0.0
    time_efficiency: float = 0.0
    
    # IR correlation metrics
    ir_success_correlation: dict = field(default_factory=dict)
    ir_correctness_correlation: dict = field(default_factory=dict)
    
    # Task counts
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    
    @classmethod
    def compute_all(
        cls,
        traces: list[AgentExecutionTrace],
    ) -> "AgentMetrics":
        """
        Compute all agent metrics from execution traces.
        
        Args:
            traces: List of agent execution traces
            
        Returns:
            AgentMetrics instance with all metrics computed
        """
        if not traces:
            return cls()
        
        successful = sum(
            1 for t in traces 
            if t.completion_status == TaskCompletionStatus.SUCCESS
        )
        failed = sum(
            1 for t in traces 
            if t.completion_status in (
                TaskCompletionStatus.FAILURE, 
                TaskCompletionStatus.ERROR
            )
        )
        
        return cls(
            # Task completion
            task_completion_rate=compute_task_completion_rate(traces),
            partial_completion_rate=compute_partial_completion_rate(traces),
            
            # Code correctness
            test_pass_rate=compute_test_pass_rate(traces),
            compilation_rate=compute_compilation_rate(traces),
            average_correctness_score=compute_average_correctness_score(traces),
            
            # Token efficiency
            tokens_per_task=compute_tokens_per_task(traces),
            tokens_per_successful_task=compute_tokens_per_successful_task(traces),
            token_efficiency=compute_token_efficiency(traces),
            context_efficiency=compute_context_efficiency(traces),
            
            # Time efficiency
            time_per_task=compute_time_per_task(traces),
            timeout_rate=compute_timeout_rate(traces),
            time_efficiency=compute_time_efficiency(traces),
            
            # IR correlation
            ir_success_correlation=compute_ir_success_correlation(traces),
            ir_correctness_correlation=compute_ir_correctness_correlation(traces),
            
            # Counts
            total_tasks=len(traces),
            successful_tasks=successful,
            failed_tasks=failed,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            # Task completion
            "task_completion_rate": self.task_completion_rate,
            "partial_completion_rate": self.partial_completion_rate,
            
            # Code correctness
            "test_pass_rate": self.test_pass_rate,
            "compilation_rate": self.compilation_rate,
            "average_correctness_score": self.average_correctness_score,
            
            # Token efficiency
            "tokens_per_task": self.tokens_per_task,
            "tokens_per_successful_task": self.tokens_per_successful_task,
            "token_efficiency": self.token_efficiency,
            "context_efficiency": self.context_efficiency,
            
            # Time efficiency
            "time_per_task": self.time_per_task,
            "timeout_rate": self.timeout_rate,
            "time_efficiency": self.time_efficiency,
            
            # IR correlation
            "ir_success_correlation": self.ir_success_correlation,
            "ir_correctness_correlation": self.ir_correctness_correlation,
            
            # Counts
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
        }

    def to_summary_dict(self) -> dict:
        """Convert to a summary dictionary with key metrics only."""
        return {
            "task_completion_rate": self.task_completion_rate,
            "test_pass_rate": self.test_pass_rate,
            "average_correctness_score": self.average_correctness_score,
            "token_efficiency": self.token_efficiency,
            "time_efficiency": self.time_efficiency,
            "tokens_per_task_mean": self.tokens_per_task.get("mean", 0.0),
            "time_per_task_mean": self.time_per_task.get("mean", 0.0),
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
        }

    def get_primary_score(self, metric: str = "task_completion_rate") -> float:
        """Get a single primary score for ranking agents/IR tools."""
        metrics_map = {
            "task_completion_rate": self.task_completion_rate,
            "test_pass_rate": self.test_pass_rate,
            "average_correctness_score": self.average_correctness_score,
            "token_efficiency": self.token_efficiency,
            "time_efficiency": self.time_efficiency,
        }
        return metrics_map.get(metric, self.task_completion_rate)


def compare_ir_tool_impact(
    traces_by_tool: dict[str, list[AgentExecutionTrace]],
) -> dict:
    """
    Compare the impact of different IR tools on agent performance.
    
    Computes metrics for each IR tool and provides comparative analysis.
    
    Args:
        traces_by_tool: Dictionary mapping IR tool names to their execution traces
        
    Returns:
        Comparative analysis dictionary
    """
    results = {}
    
    for tool_name, traces in traces_by_tool.items():
        metrics = AgentMetrics.compute_all(traces)
        results[tool_name] = {
            "metrics": metrics.to_dict(),
            "summary": metrics.to_summary_dict(),
        }
    
    # Compute rankings for each key metric
    rankings = {}
    key_metrics = [
        "task_completion_rate",
        "test_pass_rate",
        "average_correctness_score",
        "token_efficiency",
        "time_efficiency",
    ]
    
    for metric in key_metrics:
        tool_scores = [
            (tool, results[tool]["summary"].get(metric, 0.0))
            for tool in traces_by_tool.keys()
        ]
        # Sort by score descending
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        rankings[metric] = [tool for tool, _ in tool_scores]
    
    return {
        "by_tool": results,
        "rankings": rankings,
    }


def aggregate_agent_metrics(metrics_list: list[AgentMetrics]) -> dict:
    """
    Aggregate agent metrics across multiple evaluation runs.
    
    Args:
        metrics_list: List of AgentMetrics from different runs
        
    Returns:
        Aggregated metrics with mean, std, min, max for each metric
    """
    if not metrics_list:
        return {}
    
    scalar_metrics = [
        "task_completion_rate",
        "partial_completion_rate",
        "test_pass_rate",
        "compilation_rate",
        "average_correctness_score",
        "token_efficiency",
        "context_efficiency",
        "timeout_rate",
        "time_efficiency",
    ]
    
    aggregates = {}
    
    for metric in scalar_metrics:
        values = [getattr(m, metric) for m in metrics_list]
        aggregates[metric] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    
    # Aggregate counts
    aggregates["total_tasks"] = sum(m.total_tasks for m in metrics_list)
    aggregates["successful_tasks"] = sum(m.successful_tasks for m in metrics_list)
    aggregates["failed_tasks"] = sum(m.failed_tasks for m in metrics_list)
    
    return aggregates
