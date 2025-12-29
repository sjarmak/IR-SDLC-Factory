"""
Tests for Agent Output Quality Metrics.

Validates the agent_metrics module that measures how IR tool choice
impacts coding agent output quality.
"""

import pytest
import math

from app.ir_sdlc.agent_metrics import (
    TaskCompletionStatus,
    CodeCorrectnessLevel,
    AgentExecutionTrace,
    AgentMetrics,
    compute_task_completion_rate,
    compute_partial_completion_rate,
    compute_test_pass_rate,
    compute_compilation_rate,
    compute_code_correctness_score,
    compute_average_correctness_score,
    compute_tokens_per_task,
    compute_tokens_per_successful_task,
    compute_token_efficiency,
    compute_context_efficiency,
    compute_time_per_task,
    compute_timeout_rate,
    compute_time_efficiency,
    compute_ir_success_correlation,
    compute_ir_correctness_correlation,
    compare_ir_tool_impact,
    aggregate_agent_metrics,
    _pearson_correlation,
)
from app.ir_sdlc.metrics import IRMetrics


class TestAgentExecutionTrace:
    """Tests for AgentExecutionTrace dataclass."""
    
    def test_trace_creation_minimal(self):
        """Test creating a trace with minimal fields."""
        trace = AgentExecutionTrace(
            task_id="task-001",
            agent_name="test-agent",
            ir_tool_name="grep",
        )
        
        assert trace.task_id == "task-001"
        assert trace.agent_name == "test-agent"
        assert trace.ir_tool_name == "grep"
        assert trace.completed is False
        assert trace.completion_status == TaskCompletionStatus.FAILURE
    
    def test_trace_total_tokens_computed(self):
        """Test that total_tokens is auto-computed from input + output."""
        trace = AgentExecutionTrace(
            task_id="task-001",
            agent_name="test-agent",
            ir_tool_name="grep",
            input_tokens=1000,
            output_tokens=500,
        )
        
        assert trace.total_tokens == 1500
    
    def test_trace_total_tokens_not_overwritten(self):
        """Test that explicit total_tokens is preserved."""
        trace = AgentExecutionTrace(
            task_id="task-001",
            agent_name="test-agent",
            ir_tool_name="grep",
            input_tokens=1000,
            output_tokens=500,
            total_tokens=2000,  # Explicitly set
        )
        
        # Explicit value preserved
        assert trace.total_tokens == 2000


class TestTaskCompletionMetrics:
    """Tests for task completion rate metrics."""
    
    def test_completion_rate_all_successful(self):
        """Test completion rate when all tasks succeed."""
        traces = [
            AgentExecutionTrace(
                task_id=f"task-{i}",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            )
            for i in range(10)
        ]
        
        assert compute_task_completion_rate(traces) == 1.0
    
    def test_completion_rate_all_failed(self):
        """Test completion rate when all tasks fail."""
        traces = [
            AgentExecutionTrace(
                task_id=f"task-{i}",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
            )
            for i in range(10)
        ]
        
        assert compute_task_completion_rate(traces) == 0.0
    
    def test_completion_rate_mixed(self):
        """Test completion rate with mixed results."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-4",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.TIMEOUT,
            ),
        ]
        
        # 2 successful out of 4
        assert compute_task_completion_rate(traces) == 0.5
    
    def test_completion_rate_empty(self):
        """Test completion rate with empty list."""
        assert compute_task_completion_rate([]) == 0.0
    
    def test_partial_completion_rate(self):
        """Test partial completion rate includes PARTIAL status."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.PARTIAL,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
            ),
        ]
        
        # SUCCESS + PARTIAL = 2/3
        rate = compute_partial_completion_rate(traces)
        assert abs(rate - 2/3) < 0.001


class TestCodeCorrectnessMetrics:
    """Tests for code correctness metrics."""
    
    def test_test_pass_rate(self):
        """Test aggregate test pass rate calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                tests_total=10,
                tests_passed=8,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                tests_total=5,
                tests_passed=5,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                tests_total=5,
                tests_passed=2,
            ),
        ]
        
        # 15 passed out of 20 total
        assert compute_test_pass_rate(traces) == 0.75
    
    def test_test_pass_rate_no_tests(self):
        """Test test pass rate when no tests exist."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                tests_total=0,
                tests_passed=0,
            ),
        ]
        
        assert compute_test_pass_rate(traces) == 0.0
    
    def test_compilation_rate(self):
        """Test compilation success rate."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                compiles=True,
                correctness_level=CodeCorrectnessLevel.CORRECT,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                compiles=False,
                correctness_level=CodeCorrectnessLevel.INCORRECT,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                compiles=True,
                correctness_level=CodeCorrectnessLevel.MOSTLY_CORRECT,
            ),
        ]
        
        # 2 compiled out of 3
        rate = compute_compilation_rate(traces)
        assert abs(rate - 2/3) < 0.001
    
    def test_compilation_rate_excludes_no_output(self):
        """Test that NO_OUTPUT traces are excluded from compilation rate."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                compiles=True,
                correctness_level=CodeCorrectnessLevel.CORRECT,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                compiles=False,
                correctness_level=CodeCorrectnessLevel.NO_OUTPUT,
            ),
        ]
        
        # Only 1 trace with output, it compiles
        assert compute_compilation_rate(traces) == 1.0
    
    def test_code_correctness_score_perfect(self):
        """Test correctness score for perfect output."""
        trace = AgentExecutionTrace(
            task_id="task-1",
            agent_name="test",
            ir_tool_name="grep",
            compiles=True,
            tests_total=10,
            tests_passed=10,
            correctness_level=CodeCorrectnessLevel.CORRECT,
        )
        
        # 0.2 (compiles) + 0.8 * 1.0 (all tests pass) = 1.0
        assert compute_code_correctness_score(trace) == 1.0
    
    def test_code_correctness_score_no_output(self):
        """Test correctness score for no output."""
        trace = AgentExecutionTrace(
            task_id="task-1",
            agent_name="test",
            ir_tool_name="grep",
            correctness_level=CodeCorrectnessLevel.NO_OUTPUT,
        )
        
        assert compute_code_correctness_score(trace) == 0.0
    
    def test_code_correctness_score_partial(self):
        """Test correctness score for partial correctness."""
        trace = AgentExecutionTrace(
            task_id="task-1",
            agent_name="test",
            ir_tool_name="grep",
            compiles=True,
            tests_total=10,
            tests_passed=5,
            correctness_level=CodeCorrectnessLevel.PARTIALLY_CORRECT,
        )
        
        # 0.2 (compiles) + 0.8 * 0.5 (50% tests) = 0.6
        assert abs(compute_code_correctness_score(trace) - 0.6) < 0.001
    
    def test_average_correctness_score(self):
        """Test average correctness across multiple traces."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                compiles=True,
                tests_total=10,
                tests_passed=10,
                correctness_level=CodeCorrectnessLevel.CORRECT,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                compiles=True,
                tests_total=10,
                tests_passed=5,
                correctness_level=CodeCorrectnessLevel.PARTIALLY_CORRECT,
            ),
        ]
        
        # (1.0 + 0.6) / 2 = 0.8
        assert compute_average_correctness_score(traces) == 0.8


class TestTokenEfficiencyMetrics:
    """Tests for token efficiency metrics."""
    
    def test_tokens_per_task(self):
        """Test token usage statistics computation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=1000,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=2000,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=3000,
            ),
        ]
        
        stats = compute_tokens_per_task(traces)
        
        assert stats["mean"] == 2000.0
        assert stats["median"] == 2000.0
        assert stats["min"] == 1000
        assert stats["max"] == 3000
    
    def test_tokens_per_successful_task(self):
        """Test token stats for successful tasks only."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=1000,
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=5000,  # This is a failure, should be excluded
                completion_status=TaskCompletionStatus.FAILURE,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=2000,
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
        ]
        
        stats = compute_tokens_per_successful_task(traces)
        
        # Only includes the two successful tasks
        assert stats["mean"] == 1500.0
        assert stats["min"] == 1000
        assert stats["max"] == 2000
    
    def test_token_efficiency(self):
        """Test token efficiency calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=1000,
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                total_tokens=1000,
                completion_status=TaskCompletionStatus.FAILURE,
            ),
        ]
        
        # 1 successful / 2000 tokens * 1000 = 0.5 completions per 1000 tokens
        assert compute_token_efficiency(traces) == 0.5
    
    def test_context_efficiency(self):
        """Test context efficiency calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                context_tokens=500,
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                context_tokens=500,
                completion_status=TaskCompletionStatus.FAILURE,
            ),
        ]
        
        # 1 successful / 1000 context tokens * 1000 = 1.0
        assert compute_context_efficiency(traces) == 1.0


class TestTimeEfficiencyMetrics:
    """Tests for time efficiency metrics."""
    
    def test_time_per_task(self):
        """Test time statistics computation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                wall_clock_time_sec=10.0,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                wall_clock_time_sec=20.0,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                wall_clock_time_sec=30.0,
            ),
        ]
        
        stats = compute_time_per_task(traces)
        
        assert stats["mean"] == 20.0
        assert stats["median"] == 20.0
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
    
    def test_timeout_rate(self):
        """Test timeout rate calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.TIMEOUT,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-4",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.TIMEOUT,
            ),
        ]
        
        assert compute_timeout_rate(traces) == 0.5
    
    def test_time_efficiency(self):
        """Test time efficiency calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                wall_clock_time_sec=60.0,  # 1 minute
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                wall_clock_time_sec=60.0,
                completion_status=TaskCompletionStatus.SUCCESS,
            ),
        ]
        
        # 2 successful / 2 minutes = 1.0 completions per minute
        assert compute_time_efficiency(traces) == 1.0


class TestIRCorrelationMetrics:
    """Tests for IR quality correlation metrics."""
    
    def test_pearson_correlation_perfect_positive(self):
        """Test Pearson correlation for perfect positive relationship."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        
        corr = _pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr - 1.0) < 0.001
    
    def test_pearson_correlation_perfect_negative(self):
        """Test Pearson correlation for perfect negative relationship."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 8.0, 6.0, 4.0, 2.0]
        
        corr = _pearson_correlation(x, y)
        assert corr is not None
        assert abs(corr - (-1.0)) < 0.001
    
    def test_pearson_correlation_no_correlation(self):
        """Test Pearson correlation when no relationship exists."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [3.0, 3.0, 3.0, 3.0, 3.0]  # Constant, std = 0
        
        corr = _pearson_correlation(x, y)
        assert corr is None  # Can't compute when std = 0
    
    def test_ir_success_correlation(self):
        """Test IR-to-success correlation calculation."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
                ir_metrics=IRMetrics(mrr=0.9, recall_at_10=0.8),
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
                ir_metrics=IRMetrics(mrr=0.8, recall_at_10=0.7),
            ),
            AgentExecutionTrace(
                task_id="task-3",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
                ir_metrics=IRMetrics(mrr=0.2, recall_at_10=0.1),
            ),
            AgentExecutionTrace(
                task_id="task-4",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
                ir_metrics=IRMetrics(mrr=0.1, recall_at_10=0.05),
            ),
        ]
        
        correlations = compute_ir_success_correlation(traces)
        
        # High MRR/recall should correlate with success
        assert "mrr" in correlations
        assert correlations["mrr"] > 0.5  # Positive correlation
    
    def test_ir_success_correlation_insufficient_data(self):
        """Test that correlation returns empty dict with insufficient data."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
                ir_metrics=IRMetrics(mrr=0.9),
            ),
        ]
        
        correlations = compute_ir_success_correlation(traces)
        assert correlations == {}


class TestAgentMetrics:
    """Tests for AgentMetrics container class."""
    
    def test_compute_all_basic(self):
        """Test computing all metrics from traces."""
        traces = [
            AgentExecutionTrace(
                task_id="task-1",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.SUCCESS,
                compiles=True,
                tests_total=10,
                tests_passed=10,
                correctness_level=CodeCorrectnessLevel.CORRECT,
                total_tokens=1000,
                wall_clock_time_sec=30.0,
            ),
            AgentExecutionTrace(
                task_id="task-2",
                agent_name="test",
                ir_tool_name="grep",
                completion_status=TaskCompletionStatus.FAILURE,
                compiles=False,
                tests_total=10,
                tests_passed=0,
                correctness_level=CodeCorrectnessLevel.INCORRECT,
                total_tokens=2000,
                wall_clock_time_sec=60.0,
            ),
        ]
        
        metrics = AgentMetrics.compute_all(traces)
        
        assert metrics.task_completion_rate == 0.5
        assert metrics.test_pass_rate == 0.5  # 10/20
        assert metrics.total_tasks == 2
        assert metrics.successful_tasks == 1
        assert metrics.failed_tasks == 1
    
    def test_compute_all_empty(self):
        """Test computing metrics from empty list."""
        metrics = AgentMetrics.compute_all([])
        
        assert metrics.task_completion_rate == 0.0
        assert metrics.total_tasks == 0
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AgentMetrics(
            task_completion_rate=0.75,
            test_pass_rate=0.80,
            total_tasks=100,
        )
        
        d = metrics.to_dict()
        
        assert d["task_completion_rate"] == 0.75
        assert d["test_pass_rate"] == 0.80
        assert d["total_tasks"] == 100
    
    def test_to_summary_dict(self):
        """Test conversion to summary dictionary."""
        metrics = AgentMetrics(
            task_completion_rate=0.75,
            test_pass_rate=0.80,
            average_correctness_score=0.7,
            token_efficiency=2.5,
            time_efficiency=1.5,
            tokens_per_task={"mean": 1500},
            time_per_task={"mean": 45.0},
            total_tasks=100,
            successful_tasks=75,
        )
        
        summary = metrics.to_summary_dict()
        
        assert summary["task_completion_rate"] == 0.75
        assert summary["tokens_per_task_mean"] == 1500
        assert "ir_success_correlation" not in summary  # Not in summary
    
    def test_get_primary_score(self):
        """Test getting primary score for ranking."""
        metrics = AgentMetrics(
            task_completion_rate=0.75,
            test_pass_rate=0.80,
        )
        
        assert metrics.get_primary_score("task_completion_rate") == 0.75
        assert metrics.get_primary_score("test_pass_rate") == 0.80
        assert metrics.get_primary_score("unknown") == 0.75  # Falls back to completion


class TestIRToolComparison:
    """Tests for comparing IR tool impact."""
    
    def test_compare_ir_tool_impact(self):
        """Test comparing multiple IR tools."""
        traces_by_tool = {
            "grep": [
                AgentExecutionTrace(
                    task_id="task-1",
                    agent_name="test",
                    ir_tool_name="grep",
                    completion_status=TaskCompletionStatus.SUCCESS,
                    total_tokens=1000,
                ),
                AgentExecutionTrace(
                    task_id="task-2",
                    agent_name="test",
                    ir_tool_name="grep",
                    completion_status=TaskCompletionStatus.FAILURE,
                    total_tokens=2000,
                ),
            ],
            "semantic": [
                AgentExecutionTrace(
                    task_id="task-1",
                    agent_name="test",
                    ir_tool_name="semantic",
                    completion_status=TaskCompletionStatus.SUCCESS,
                    total_tokens=1500,
                ),
                AgentExecutionTrace(
                    task_id="task-2",
                    agent_name="test",
                    ir_tool_name="semantic",
                    completion_status=TaskCompletionStatus.SUCCESS,
                    total_tokens=1500,
                ),
            ],
        }
        
        results = compare_ir_tool_impact(traces_by_tool)
        
        assert "by_tool" in results
        assert "rankings" in results
        assert "grep" in results["by_tool"]
        assert "semantic" in results["by_tool"]
        
        # Semantic should rank higher for completion rate
        assert results["rankings"]["task_completion_rate"][0] == "semantic"


class TestAggregateMetrics:
    """Tests for aggregating metrics across runs."""
    
    def test_aggregate_agent_metrics(self):
        """Test aggregating metrics from multiple runs."""
        metrics_list = [
            AgentMetrics(
                task_completion_rate=0.70,
                test_pass_rate=0.80,
                total_tasks=100,
                successful_tasks=70,
            ),
            AgentMetrics(
                task_completion_rate=0.80,
                test_pass_rate=0.90,
                total_tasks=100,
                successful_tasks=80,
            ),
            AgentMetrics(
                task_completion_rate=0.60,
                test_pass_rate=0.70,
                total_tasks=100,
                successful_tasks=60,
            ),
        ]
        
        aggregates = aggregate_agent_metrics(metrics_list)
        
        assert "task_completion_rate" in aggregates
        assert aggregates["task_completion_rate"]["mean"] == 0.70
        assert aggregates["task_completion_rate"]["min"] == 0.60
        assert aggregates["task_completion_rate"]["max"] == 0.80
        assert aggregates["total_tasks"] == 300
        assert aggregates["successful_tasks"] == 210
