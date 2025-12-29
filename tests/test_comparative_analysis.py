"""
Unit tests for the comparative analysis framework.

Tests cover:
1. Statistical analysis functions (Cohen's d, Welch's t-test, McNemar's test)
2. MetricComparison computation
3. TaskLevelComparison computation
4. ComparisonReport aggregation
5. ABComparator with mock runners
6. Utility functions for creating comparisons from runs
"""

import math
import pytest
import statistics
from typing import List
from unittest.mock import Mock, MagicMock

from app.ir_sdlc.comparative_analysis import (
    ABComparator,
    ComparisonConfig,
    ComparisonReport,
    ComparisonType,
    EffectSize,
    HypothesisTestResult,
    MetricComparison,
    StatisticalAnalysis,
    TaskLevelComparison,
    create_comparison_from_runs,
    compare_multiple_tools,
    rank_tools_by_metric,
)
from app.ir_sdlc.agent_metrics import (
    AgentExecutionTrace,
    TaskCompletionStatus,
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


# =============================================================================
# Statistical Analysis Tests
# =============================================================================

class TestCohensD:
    """Tests for Cohen's d effect size calculation."""
    
    def test_cohens_d_no_difference(self):
        """Cohen's d should be 0 for identical groups."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = StatisticalAnalysis.cohens_d(group_a, group_b)
        
        assert result.cohens_d == 0.0
        assert result.interpretation == "negligible"
    
    def test_cohens_d_small_effect(self):
        """Cohen's d should detect small effect size."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.5, 2.5, 3.5, 4.5, 5.5]  # Small shift (~0.32 std)
        
        result = StatisticalAnalysis.cohens_d(group_a, group_b)
        
        assert 0.2 <= abs(result.cohens_d) < 0.5
        assert result.interpretation == "small"
    
    def test_cohens_d_large_effect(self):
        """Cohen's d should detect large effect size."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [5.0, 6.0, 7.0, 8.0, 9.0]  # Large shift
        
        result = StatisticalAnalysis.cohens_d(group_a, group_b)
        
        assert abs(result.cohens_d) >= 0.8
        assert result.interpretation == "large"
    
    def test_cohens_d_empty_groups(self):
        """Cohen's d should handle empty groups."""
        result = StatisticalAnalysis.cohens_d([], [1.0, 2.0])
        assert result.interpretation == "invalid"
        
        result = StatisticalAnalysis.cohens_d([1.0, 2.0], [])
        assert result.interpretation == "invalid"
    
    def test_cohens_d_single_sample(self):
        """Cohen's d should handle single sample per group."""
        result = StatisticalAnalysis.cohens_d([1.0], [2.0])
        assert result.interpretation == "insufficient_data"
    
    def test_cohens_d_confidence_interval(self):
        """Cohen's d should compute confidence interval."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [3.0, 4.0, 5.0, 6.0, 7.0]
        
        result = StatisticalAnalysis.cohens_d(group_a, group_b)
        
        ci_lower, ci_upper = result.confidence_interval
        assert ci_lower < result.cohens_d < ci_upper


class TestWelchsTTest:
    """Tests for Welch's t-test."""
    
    def test_welchs_t_test_significant_difference(self):
        """Welch's t-test should detect significant difference."""
        group_a = [1.0, 1.5, 2.0, 2.5, 3.0]
        group_b = [4.0, 4.5, 5.0, 5.5, 6.0]
        
        result = StatisticalAnalysis.welchs_t_test(group_a, group_b)
        
        assert result.test_name == "welchs_t_test"
        assert result.p_value < 0.05
        assert result.is_significant
        assert result.effect_size is not None
        assert result.effect_size.interpretation == "large"
    
    def test_welchs_t_test_no_difference(self):
        """Welch's t-test should not find significance for same groups."""
        group_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        group_b = [1.1, 1.9, 3.1, 4.1, 4.9]  # Nearly identical
        
        result = StatisticalAnalysis.welchs_t_test(group_a, group_b)
        
        assert result.p_value > 0.05
        assert not result.is_significant
    
    def test_welchs_t_test_insufficient_samples(self):
        """Welch's t-test should handle insufficient samples."""
        result = StatisticalAnalysis.welchs_t_test([1.0], [2.0])
        
        assert result.p_value == 1.0
        assert not result.is_significant
        assert "Insufficient sample size" in result.interpretation
    
    def test_welchs_t_test_sample_sizes_recorded(self):
        """Welch's t-test should record sample sizes."""
        group_a = [1.0, 2.0, 3.0]
        group_b = [4.0, 5.0, 6.0, 7.0]
        
        result = StatisticalAnalysis.welchs_t_test(group_a, group_b)
        
        assert result.sample_sizes == (3, 4)


class TestMcNemarTest:
    """Tests for McNemar's test for paired nominal data."""
    
    def test_mcnemar_treatment_better(self):
        """McNemar should detect when treatment is better."""
        # Baseline fails, treatment succeeds more often
        baseline = [False, False, False, True, True]
        treatment = [True, True, True, True, True]
        
        result = StatisticalAnalysis.mcnemar_test(baseline, treatment)
        
        assert "treatment improved" in result.interpretation
    
    def test_mcnemar_no_difference(self):
        """McNemar should detect no difference for identical results."""
        baseline = [True, False, True, False, True]
        treatment = [True, False, True, False, True]
        
        result = StatisticalAnalysis.mcnemar_test(baseline, treatment)
        
        assert "identical performance" in result.interpretation
    
    def test_mcnemar_mismatched_lengths(self):
        """McNemar should raise error for mismatched lengths."""
        with pytest.raises(ValueError, match="same length"):
            StatisticalAnalysis.mcnemar_test([True, False], [True])


class TestBootstrapCI:
    """Tests for bootstrap confidence interval."""
    
    def test_bootstrap_ci_mean(self):
        """Bootstrap CI should work for mean."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        lower, point, upper = StatisticalAnalysis.bootstrap_confidence_interval(data)
        
        assert lower <= point <= upper
        assert point == pytest.approx(statistics.mean(data))
    
    def test_bootstrap_ci_empty_data(self):
        """Bootstrap CI should handle empty data."""
        result = StatisticalAnalysis.bootstrap_confidence_interval([])
        assert result == (0.0, 0.0, 0.0)
    
    def test_bootstrap_ci_single_value(self):
        """Bootstrap CI should handle single value."""
        lower, point, upper = StatisticalAnalysis.bootstrap_confidence_interval([5.0])
        assert lower == point == upper == 5.0


# =============================================================================
# MetricComparison Tests
# =============================================================================

class TestMetricComparison:
    """Tests for MetricComparison class."""
    
    def test_compute_comparison_basic(self):
        """MetricComparison should compute basic statistics."""
        mc = MetricComparison(metric_name="success_rate")
        mc.baseline_values = [0.5, 0.6, 0.7, 0.8]
        mc.treatment_values = [0.7, 0.8, 0.9, 1.0]
        
        mc.compute_comparison()
        
        assert mc.baseline_mean == pytest.approx(0.65)
        assert mc.treatment_mean == pytest.approx(0.85)
        assert mc.absolute_diff == pytest.approx(0.2)
        assert mc.percent_improvement > 0
    
    def test_compute_comparison_with_hypothesis_test(self):
        """MetricComparison should run hypothesis test with enough data."""
        mc = MetricComparison(metric_name="tokens")
        mc.baseline_values = [100, 120, 130, 140, 150]
        mc.treatment_values = [80, 90, 100, 110, 120]
        
        mc.compute_comparison()
        
        assert mc.hypothesis_test is not None
        assert mc.hypothesis_test.test_name == "welchs_t_test"
    
    def test_to_dict_serialization(self):
        """MetricComparison should serialize to dict."""
        mc = MetricComparison(metric_name="test_metric")
        mc.baseline_values = [1.0, 2.0]
        mc.treatment_values = [3.0, 4.0]
        mc.compute_comparison()
        
        result = mc.to_dict()
        
        assert result["metric_name"] == "test_metric"
        assert "baseline" in result
        assert "treatment" in result
        assert "comparison" in result


# =============================================================================
# TaskLevelComparison Tests
# =============================================================================

class TestTaskLevelComparison:
    """Tests for TaskLevelComparison class."""
    
    def test_compute_deltas_treatment_wins(self):
        """TaskLevelComparison should detect treatment win."""
        comparison = TaskLevelComparison(
            task_id="test_task",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            baseline_success=False,
            treatment_success=True,
            baseline_tokens=1000,
            treatment_tokens=800,
            baseline_duration_sec=60.0,
            treatment_duration_sec=45.0,
        )
        
        comparison.compute_deltas()
        
        assert comparison.winner == "treatment"
        assert comparison.token_efficiency > 0  # Treatment used fewer tokens
        assert comparison.time_efficiency > 0  # Treatment was faster
    
    def test_compute_deltas_baseline_wins(self):
        """TaskLevelComparison should detect baseline win."""
        comparison = TaskLevelComparison(
            task_id="test_task",
            sdlc_type=SDLCTaskType.CODE_REVIEW,
            baseline_success=True,
            treatment_success=False,
            baseline_reward=1.0,
            treatment_reward=0.0,
        )
        
        comparison.compute_deltas()
        
        assert comparison.winner == "baseline"
    
    def test_compute_deltas_tie(self):
        """TaskLevelComparison should detect tie."""
        comparison = TaskLevelComparison(
            task_id="test_task",
            sdlc_type=SDLCTaskType.SECURITY_AUDIT,
            baseline_success=True,
            treatment_success=True,
            baseline_reward=0.8,
            treatment_reward=0.85,  # Within 0.1 threshold
        )
        
        comparison.compute_deltas()
        
        assert comparison.winner == "tie"
    
    def test_to_ir_comparison_conversion(self):
        """TaskLevelComparison should convert to IRComparison."""
        comparison = TaskLevelComparison(
            task_id="convert_test",
            sdlc_type=SDLCTaskType.REFACTORING_ANALYSIS,
            baseline_success=True,
            treatment_success=True,
            baseline_tokens=500,
            treatment_tokens=400,
        )
        comparison.compute_deltas()
        
        ir_comp = comparison.to_ir_comparison()
        
        assert ir_comp.task_id == "convert_test"
        assert ir_comp.baseline_tokens == 500
        assert ir_comp.ir_enhanced_tokens == 400


# =============================================================================
# ComparisonReport Tests
# =============================================================================

class TestComparisonReport:
    """Tests for ComparisonReport class."""
    
    def _create_sample_task_comparisons(self) -> List[TaskLevelComparison]:
        """Create sample task comparisons for testing."""
        comparisons = []
        
        # Treatment wins 3 out of 5
        for i in range(5):
            comp = TaskLevelComparison(
                task_id=f"task_{i}",
                sdlc_type=SDLCTaskType.BUG_TRIAGE,
                baseline_success=i >= 3,  # 2 baseline successes
                treatment_success=i < 3,  # 3 treatment successes
                baseline_tokens=1000 + i * 100,
                treatment_tokens=800 + i * 100,
                baseline_duration_sec=60.0,
                treatment_duration_sec=45.0,
            )
            comp.compute_deltas()
            comparisons.append(comp)
        
        return comparisons
    
    def test_compute_aggregates_win_rates(self):
        """ComparisonReport should compute win rates."""
        config = ComparisonConfig(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
        )
        report = ComparisonReport(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            config=config,
        )
        report.task_comparisons = self._create_sample_task_comparisons()
        
        report.compute_aggregates()
        
        assert report.treatment_win_rate == pytest.approx(0.6)
        assert report.baseline_win_rate == pytest.approx(0.4)
        assert report.tie_rate == pytest.approx(0.0)
    
    def test_compute_aggregates_metric_comparisons(self):
        """ComparisonReport should build metric comparisons."""
        config = ComparisonConfig(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
        )
        report = ComparisonReport(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            config=config,
        )
        report.task_comparisons = self._create_sample_task_comparisons()
        
        report.compute_aggregates()
        
        assert "success_rate" in report.metric_comparisons
        assert "tokens" in report.metric_comparisons
        assert "duration_sec" in report.metric_comparisons
    
    def test_get_summary(self):
        """ComparisonReport should generate summary."""
        config = ComparisonConfig(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
        )
        report = ComparisonReport(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            config=config,
        )
        report.task_comparisons = self._create_sample_task_comparisons()
        report.compute_aggregates()
        
        summary = report.get_summary()
        
        assert summary["experiment_id"] == "test_exp"
        assert summary["total_tasks"] == 5
        assert "key_metrics" in summary
    
    def test_to_json_serialization(self):
        """ComparisonReport should serialize to JSON."""
        config = ComparisonConfig(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
        )
        report = ComparisonReport(
            experiment_id="test_exp",
            experiment_name="Test Experiment",
            config=config,
        )
        report.task_comparisons = self._create_sample_task_comparisons()
        report.compute_aggregates()
        
        json_str = report.to_json()
        
        import json
        parsed = json.loads(json_str)
        assert parsed["experiment_id"] == "test_exp"


# =============================================================================
# ABComparator Tests
# =============================================================================

class TestABComparator:
    """Tests for ABComparator class."""
    
    def _create_mock_trace(
        self,
        task_id: str,
        success: bool,
        tokens: int = 1000,
        duration: float = 60.0,
    ) -> AgentExecutionTrace:
        """Create a mock execution trace."""
        return AgentExecutionTrace(
            task_id=task_id,
            agent_name="TestAgent",
            ir_tool_name="test_tool",
            completed=success,
            completion_status=TaskCompletionStatus.SUCCESS if success else TaskCompletionStatus.FAILURE,
            total_tokens=tokens,
            wall_clock_time_sec=duration,
            tests_total=10 if success else 10,
            tests_passed=10 if success else 3,
        )
    
    def test_run_comparison_from_traces(self):
        """ABComparator should build comparison from traces."""
        config = ComparisonConfig(
            experiment_id="trace_comparison",
            experiment_name="Trace Comparison Test",
        )
        comparator = ABComparator(config)
        
        baseline_traces = [
            self._create_mock_trace("task_1", False, 1000),
            self._create_mock_trace("task_2", True, 1200),
            self._create_mock_trace("task_3", False, 900),
        ]
        
        treatment_traces = [
            self._create_mock_trace("task_1", True, 800),
            self._create_mock_trace("task_2", True, 1000),
            self._create_mock_trace("task_3", True, 700),
        ]
        
        report = comparator.run_comparison_from_traces(baseline_traces, treatment_traces)
        
        assert report is not None
        assert len(report.task_comparisons) == 3
        assert report.treatment_win_rate > report.baseline_win_rate
    
    def test_run_comparison_from_traces_mismatched_count(self):
        """ABComparator should raise error for mismatched trace counts."""
        config = ComparisonConfig(
            experiment_id="error_test",
            experiment_name="Error Test",
        )
        comparator = ABComparator(config)
        
        with pytest.raises(ValueError, match="Trace count mismatch"):
            comparator.run_comparison_from_traces(
                [self._create_mock_trace("task_1", True)],
                [
                    self._create_mock_trace("task_1", True),
                    self._create_mock_trace("task_2", True),
                ],
            )
    
    def test_comparator_with_mock_runners(self):
        """ABComparator should work with mock agent runners."""
        config = ComparisonConfig(
            experiment_id="mock_runner_test",
            experiment_name="Mock Runner Test",
        )
        
        # Create mock runners
        baseline_runner = Mock()
        treatment_runner = Mock()
        
        baseline_runner.run_task = Mock(
            return_value=self._create_mock_trace("test", False, 1000)
        )
        treatment_runner.run_task = Mock(
            return_value=self._create_mock_trace("test", True, 800)
        )
        
        comparator = ABComparator(
            config,
            baseline_runner=baseline_runner,
            treatment_runner=treatment_runner,
        )
        
        tasks = [
            {"task_id": "task_1", "sdlc_type": "bug_triage"},
            {"task_id": "task_2", "sdlc_type": "code_review"},
        ]
        
        report = comparator.run_comparison(tasks)
        
        assert report is not None
        assert len(report.task_comparisons) == 2
        assert baseline_runner.run_task.call_count == 2
        assert treatment_runner.run_task.call_count == 2


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def _create_mock_run(
        self,
        run_id: str,
        ir_tool_type: IRToolType,
        task_results: List[IRSDLCTaskResult],
    ) -> IRSDLCBenchmarkRun:
        """Create a mock benchmark run."""
        return IRSDLCBenchmarkRun(
            run_id=run_id,
            ir_tool_type=ir_tool_type,
            agent_name=f"{ir_tool_type.value}_agent",
            task_results=task_results,
        )
    
    def _create_mock_task_result(
        self,
        task_id: str,
        success: bool,
        tokens: int = 1000,
    ) -> IRSDLCTaskResult:
        """Create a mock task result."""
        return IRSDLCTaskResult(
            task_id=task_id,
            task_title=f"Task {task_id}",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            repo_name="test/repo",
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
            difficulty="medium",
            ir_tool_type=IRToolType.BASELINE,
            ir_tool_name="TestTool",
            agent_import_path="test.agent",
            model_name="test-model",
            execution_metrics=AgentExecutionMetrics(
                success=success,
                reward=1.0 if success else 0.0,
                total_tokens=tokens,
                duration_sec=60.0,
            ),
            ir_metrics=IRRetrievalMetrics(
                total_queries=5,
                context_tokens_retrieved=tokens // 2,
            ),
        )
    
    def test_create_comparison_from_runs(self):
        """Should create comparison from two benchmark runs."""
        baseline_results = [
            self._create_mock_task_result("task_1", False),
            self._create_mock_task_result("task_2", True),
        ]
        treatment_results = [
            self._create_mock_task_result("task_1", True),
            self._create_mock_task_result("task_2", True),
        ]
        
        baseline_run = self._create_mock_run(
            "baseline_run", IRToolType.BASELINE, baseline_results
        )
        treatment_run = self._create_mock_run(
            "treatment_run", IRToolType.DEEP_SEARCH, treatment_results
        )
        
        report = create_comparison_from_runs(baseline_run, treatment_run)
        
        assert report is not None
        assert len(report.task_comparisons) == 2
        assert report.experiment_id == "baseline_run_vs_treatment_run"
    
    def test_compare_multiple_tools(self):
        """Should compare multiple tools against baseline."""
        baseline_results = [self._create_mock_task_result("task_1", False)]
        deep_search_results = [self._create_mock_task_result("task_1", True)]
        full_results = [self._create_mock_task_result("task_1", True)]
        
        runs = {
            "baseline": self._create_mock_run("baseline", IRToolType.BASELINE, baseline_results),
            "deep_search": self._create_mock_run("deep_search", IRToolType.DEEP_SEARCH, deep_search_results),
            "full": self._create_mock_run("full", IRToolType.SOURCEGRAPH_FULL, full_results),
        }
        
        comparisons = compare_multiple_tools(runs, baseline_tool="baseline")
        
        assert "deep_search" in comparisons
        assert "full" in comparisons
        assert "baseline" not in comparisons  # Baseline not compared to itself
    
    def test_compare_multiple_tools_missing_baseline(self):
        """Should raise error if baseline not in runs."""
        runs = {"tool_a": Mock()}
        
        with pytest.raises(ValueError, match="not in runs"):
            compare_multiple_tools(runs, baseline_tool="missing_baseline")
    
    def test_rank_tools_by_metric(self):
        """Should rank tools by metric improvement."""
        # Create mock comparisons
        config1 = ComparisonConfig(experiment_id="t1", experiment_name="T1")
        report1 = ComparisonReport(experiment_id="t1", experiment_name="T1", config=config1)
        mc1 = MetricComparison(metric_name="success_rate")
        mc1.baseline_values = [0.5, 0.5]
        mc1.treatment_values = [0.8, 0.8]
        mc1.compute_comparison()
        report1.metric_comparisons["success_rate"] = mc1
        
        config2 = ComparisonConfig(experiment_id="t2", experiment_name="T2")
        report2 = ComparisonReport(experiment_id="t2", experiment_name="T2", config=config2)
        mc2 = MetricComparison(metric_name="success_rate")
        mc2.baseline_values = [0.5, 0.5]
        mc2.treatment_values = [0.6, 0.6]
        mc2.compute_comparison()
        report2.metric_comparisons["success_rate"] = mc2
        
        comparisons = {"tool_a": report1, "tool_b": report2}
        
        rankings = rank_tools_by_metric(comparisons, "success_rate")
        
        assert len(rankings) == 2
        assert rankings[0][0] == "tool_a"  # Higher improvement first
        assert rankings[0][1] > rankings[1][1]


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_comparison_report(self):
        """ComparisonReport should handle empty task comparisons."""
        config = ComparisonConfig(experiment_id="empty", experiment_name="Empty")
        report = ComparisonReport(experiment_id="empty", experiment_name="Empty", config=config)
        
        report.compute_aggregates()
        
        assert report.treatment_win_rate == 0.0
        assert report.baseline_win_rate == 0.0
        assert len(report.metric_comparisons) == 0
    
    def test_zero_baseline_values(self):
        """MetricComparison should handle zero baseline values."""
        mc = MetricComparison(metric_name="test")
        mc.baseline_values = [0.0, 0.0]
        mc.treatment_values = [1.0, 1.0]
        
        mc.compute_comparison()
        
        # Should not crash, relative diff may be undefined
        assert mc.baseline_mean == 0.0
        assert mc.treatment_mean == 1.0
    
    def test_single_task_comparison(self):
        """Should work with a single task."""
        config = ComparisonConfig(experiment_id="single", experiment_name="Single")
        comparator = ABComparator(config)
        
        baseline = AgentExecutionTrace(
            task_id="only_task",
            agent_name="base",
            ir_tool_name="base",
            completion_status=TaskCompletionStatus.SUCCESS,
            completed=True,
            total_tokens=1000,
        )
        
        treatment = AgentExecutionTrace(
            task_id="only_task",
            agent_name="treat",
            ir_tool_name="treat",
            completion_status=TaskCompletionStatus.SUCCESS,
            completed=True,
            total_tokens=800,
        )
        
        report = comparator.run_comparison_from_traces([baseline], [treatment])
        
        assert len(report.task_comparisons) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
