"""
Comparative Analysis Framework for IR-SDLC Benchmark.

This module provides A/B comparison capabilities between baseline agents
and MCP-enhanced agents. It tracks metrics like token usage, success rate,
IR query efficiency, and produces statistical comparison reports.

Key components:
1. ComparisonConfig - Configuration for A/B comparison runs
2. ABComparator - Runner for A/B comparison experiments
3. StatisticalAnalysis - Statistical significance testing
4. ComparisonReport - Report generation and visualization

Reference: CodeContextBench runners/compare_results.py and aggregator.py
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

from app.ir_sdlc.agent_metrics import (
    AgentExecutionTrace,
    AgentMetrics,
    TaskCompletionStatus,
    compute_task_completion_rate,
    compute_test_pass_rate,
    compute_token_efficiency,
    compute_time_efficiency,
)
from app.ir_sdlc.dashboard_schema import (
    IRComparison,
    IRRetrievalMetrics,
    IRSDLCBenchmarkRun,
    IRSDLCTaskResult,
    IRToolType,
    LLMJudgeScore,
    SDLCTaskType,
)


# =============================================================================
# Configuration
# =============================================================================

class ComparisonType(str, Enum):
    """Type of A/B comparison experiment."""
    BASELINE_VS_MCP = "baseline_vs_mcp"  # Baseline vs any MCP tool
    DEEP_SEARCH_VS_FULL = "deep_search_vs_full"  # Deep Search vs Full Sourcegraph
    KEYWORD_VS_SEMANTIC = "keyword_vs_semantic"  # Keyword-only vs semantic search
    MULTI_TOOL = "multi_tool"  # Compare multiple tools at once


@dataclass
class ComparisonConfig:
    """Configuration for an A/B comparison experiment.
    
    Defines which agents/tools to compare and evaluation parameters.
    """
    # Experiment identification
    experiment_id: str
    experiment_name: str
    description: str = ""
    
    # Comparison type
    comparison_type: ComparisonType = ComparisonType.BASELINE_VS_MCP
    
    # Baseline configuration
    baseline_tool_type: IRToolType = IRToolType.BASELINE
    baseline_agent_name: str = "BaselineAgent"
    baseline_import_path: str = ""
    
    # Treatment (MCP-enhanced) configuration
    treatment_tool_type: IRToolType = IRToolType.DEEP_SEARCH
    treatment_agent_name: str = "DeepSearchAgent"
    treatment_import_path: str = ""
    
    # Model configuration (same model for fair comparison)
    model_name: str = "anthropic/claude-haiku-4-5-20251001"
    
    # Evaluation parameters
    tasks_to_run: List[str] = field(default_factory=list)  # Empty = all tasks
    num_trials_per_task: int = 1  # Number of repeated trials per task
    timeout_seconds: int = 300
    
    # Statistical thresholds
    significance_level: float = 0.05  # Alpha for hypothesis testing
    min_effect_size: float = 0.1  # Minimum meaningful difference
    
    # Output configuration
    output_dir: str = "outputs/comparisons"
    save_trajectories: bool = True
    
    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "comparison_type": self.comparison_type.value,
            "baseline": {
                "tool_type": self.baseline_tool_type.value,
                "agent_name": self.baseline_agent_name,
                "import_path": self.baseline_import_path,
            },
            "treatment": {
                "tool_type": self.treatment_tool_type.value,
                "agent_name": self.treatment_agent_name,
                "import_path": self.treatment_import_path,
            },
            "model_name": self.model_name,
            "tasks_to_run": self.tasks_to_run,
            "num_trials_per_task": self.num_trials_per_task,
            "timeout_seconds": self.timeout_seconds,
            "significance_level": self.significance_level,
            "min_effect_size": self.min_effect_size,
        }


# =============================================================================
# Agent Runner Protocol
# =============================================================================

class AgentRunner(Protocol):
    """Protocol for running agents on tasks.
    
    Implementations should handle Harbor agent invocation,
    context retrieval, and result collection.
    """
    
    def run_task(
        self,
        task_id: str,
        task_data: dict,
        agent_import_path: str,
        model_name: str,
        timeout_seconds: int,
    ) -> AgentExecutionTrace:
        """Run an agent on a single task.
        
        Args:
            task_id: Unique task identifier
            task_data: Task specification (prompt, repo, etc.)
            agent_import_path: Import path for the agent class
            model_name: Model to use
            timeout_seconds: Maximum execution time
            
        Returns:
            Execution trace with metrics
        """
        ...


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class EffectSize:
    """Effect size calculation results."""
    cohens_d: float
    interpretation: str  # "negligible", "small", "medium", "large"
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class HypothesisTestResult:
    """Results from a statistical hypothesis test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: Optional[EffectSize] = None
    sample_sizes: Tuple[int, int] = (0, 0)
    interpretation: str = ""


class StatisticalAnalysis:
    """Statistical analysis utilities for A/B comparison.
    
    Provides hypothesis testing and effect size calculations.
    """
    
    @staticmethod
    def cohens_d(
        group_a: List[float],
        group_b: List[float],
    ) -> EffectSize:
        """Calculate Cohen's d effect size.
        
        Cohen's d measures the standardized difference between two means.
        
        Args:
            group_a: First group of measurements
            group_b: Second group of measurements
            
        Returns:
            EffectSize with Cohen's d and interpretation
        """
        if not group_a or not group_b:
            return EffectSize(0.0, "invalid", (0.0, 0.0))
        
        mean_a = statistics.mean(group_a)
        mean_b = statistics.mean(group_b)
        
        # Pooled standard deviation
        n_a, n_b = len(group_a), len(group_b)
        
        if n_a == 1 and n_b == 1:
            # Can't compute std with single samples
            return EffectSize(0.0, "insufficient_data", (0.0, 0.0))
        
        var_a = statistics.variance(group_a) if n_a > 1 else 0
        var_b = statistics.variance(group_b) if n_b > 1 else 0
        
        # Pooled variance
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10
        
        d = (mean_b - mean_a) / pooled_std
        
        # Interpretation (Cohen's conventions)
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        # Approximate 95% CI using Hedges and Olkin (1985)
        se = math.sqrt((n_a + n_b) / (n_a * n_b) + (d ** 2) / (2 * (n_a + n_b)))
        ci_lower = d - 1.96 * se
        ci_upper = d + 1.96 * se
        
        return EffectSize(d, interpretation, (ci_lower, ci_upper))
    
    @staticmethod
    def welchs_t_test(
        group_a: List[float],
        group_b: List[float],
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """Perform Welch's t-test for independent samples.
        
        Welch's t-test doesn't assume equal variances, making it more robust.
        
        Args:
            group_a: First group (baseline)
            group_b: Second group (treatment)
            alpha: Significance level
            
        Returns:
            HypothesisTestResult with t-statistic and p-value
        """
        if len(group_a) < 2 or len(group_b) < 2:
            return HypothesisTestResult(
                test_name="welchs_t_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_sizes=(len(group_a), len(group_b)),
                interpretation="Insufficient sample size (need at least 2 per group)",
            )
        
        n_a, n_b = len(group_a), len(group_b)
        mean_a, mean_b = statistics.mean(group_a), statistics.mean(group_b)
        var_a, var_b = statistics.variance(group_a), statistics.variance(group_b)
        
        # Welch's t-statistic
        se = math.sqrt(var_a / n_a + var_b / n_b)
        if se == 0:
            return HypothesisTestResult(
                test_name="welchs_t_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_sizes=(n_a, n_b),
                interpretation="Zero variance in both groups",
            )
        
        t_stat = (mean_b - mean_a) / se
        
        # Welch-Satterthwaite degrees of freedom
        num = (var_a / n_a + var_b / n_b) ** 2
        denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
        df = num / denom if denom > 0 else 1
        
        # Approximate p-value using normal distribution for large samples
        # For small samples, this is an approximation
        p_value = 2 * (1 - StatisticalAnalysis._normal_cdf(abs(t_stat)))
        
        effect = StatisticalAnalysis.cohens_d(group_a, group_b)
        is_sig = p_value < alpha
        
        interpretation = (
            f"Treatment {'significantly' if is_sig else 'does not significantly'} "
            f"differ from baseline (t={t_stat:.3f}, p={p_value:.4f}, d={effect.cohens_d:.3f})"
        )
        
        return HypothesisTestResult(
            test_name="welchs_t_test",
            statistic=t_stat,
            p_value=p_value,
            is_significant=is_sig,
            effect_size=effect,
            sample_sizes=(n_a, n_b),
            interpretation=interpretation,
        )
    
    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Approximate the standard normal CDF using error function."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def bootstrap_confidence_interval(
        data: List[float],
        statistic_fn: Callable[[List[float]], float] = statistics.mean,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Calculate bootstrap confidence interval for a statistic.
        
        Args:
            data: Sample data
            statistic_fn: Function to compute the statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (default: 95%)
            
        Returns:
            Tuple of (lower_bound, point_estimate, upper_bound)
        """
        import random
        
        if not data:
            return (0.0, 0.0, 0.0)
        
        point_estimate = statistic_fn(data)
        
        if len(data) == 1:
            return (point_estimate, point_estimate, point_estimate)
        
        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = random.choices(data, k=len(data))
            bootstrap_stats.append(statistic_fn(sample))
        
        bootstrap_stats.sort()
        
        # Percentile method
        alpha = 1 - confidence
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap) - 1
        
        return (
            bootstrap_stats[lower_idx],
            point_estimate,
            bootstrap_stats[upper_idx],
        )
    
    @staticmethod
    def mcnemar_test(
        baseline_success: List[bool],
        treatment_success: List[bool],
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """McNemar's test for paired nominal data.
        
        Useful for comparing success rates on the same tasks.
        
        Args:
            baseline_success: List of success/failure for baseline
            treatment_success: List of success/failure for treatment
            alpha: Significance level
            
        Returns:
            HypothesisTestResult
        """
        if len(baseline_success) != len(treatment_success):
            raise ValueError("Baseline and treatment must have same length")
        
        n = len(baseline_success)
        if n == 0:
            return HypothesisTestResult(
                test_name="mcnemar_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_sizes=(0, 0),
                interpretation="Empty samples",
            )
        
        # Count discordant pairs
        # b: baseline failed, treatment succeeded
        # c: baseline succeeded, treatment failed
        b = sum(1 for bs, ts in zip(baseline_success, treatment_success) if not bs and ts)
        c = sum(1 for bs, ts in zip(baseline_success, treatment_success) if bs and not ts)
        
        if b + c == 0:
            return HypothesisTestResult(
                test_name="mcnemar_test",
                statistic=0.0,
                p_value=1.0,
                is_significant=False,
                sample_sizes=(n, n),
                interpretation="No discordant pairs - identical performance",
            )
        
        # McNemar's chi-squared statistic with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        
        # Approximate p-value (chi-squared with 1 df)
        p_value = 1 - StatisticalAnalysis._chi2_cdf(chi2, 1)
        is_sig = p_value < alpha
        
        # Determine direction
        if b > c:
            direction = "treatment improved"
        elif c > b:
            direction = "baseline better"
        else:
            direction = "no difference"
        
        interpretation = (
            f"McNemar's test: {direction} "
            f"(b={b}, c={c}, χ²={chi2:.3f}, p={p_value:.4f})"
        )
        
        return HypothesisTestResult(
            test_name="mcnemar_test",
            statistic=chi2,
            p_value=p_value,
            is_significant=is_sig,
            sample_sizes=(n, n),
            interpretation=interpretation,
        )
    
    @staticmethod
    def _chi2_cdf(x: float, df: int) -> float:
        """Approximate chi-squared CDF for df=1."""
        if df != 1:
            raise ValueError("Only df=1 is supported")
        # For df=1, chi2 CDF = 2 * normal CDF - 1
        return 2 * StatisticalAnalysis._normal_cdf(math.sqrt(x)) - 1


# =============================================================================
# Comparison Results
# =============================================================================

@dataclass
class MetricComparison:
    """Comparison of a single metric between baseline and treatment."""
    metric_name: str
    
    # Baseline statistics
    baseline_mean: float = 0.0
    baseline_std: float = 0.0
    baseline_median: float = 0.0
    baseline_values: List[float] = field(default_factory=list)
    
    # Treatment statistics
    treatment_mean: float = 0.0
    treatment_std: float = 0.0
    treatment_median: float = 0.0
    treatment_values: List[float] = field(default_factory=list)
    
    # Comparison
    absolute_diff: float = 0.0  # treatment - baseline
    relative_diff: float = 0.0  # (treatment - baseline) / baseline
    percent_improvement: float = 0.0
    
    # Statistical tests
    hypothesis_test: Optional[HypothesisTestResult] = None
    confidence_interval: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    def compute_comparison(self) -> None:
        """Compute comparison statistics."""
        if self.baseline_values:
            self.baseline_mean = statistics.mean(self.baseline_values)
            self.baseline_std = statistics.stdev(self.baseline_values) if len(self.baseline_values) > 1 else 0
            self.baseline_median = statistics.median(self.baseline_values)
        
        if self.treatment_values:
            self.treatment_mean = statistics.mean(self.treatment_values)
            self.treatment_std = statistics.stdev(self.treatment_values) if len(self.treatment_values) > 1 else 0
            self.treatment_median = statistics.median(self.treatment_values)
        
        self.absolute_diff = self.treatment_mean - self.baseline_mean
        
        if self.baseline_mean != 0:
            self.relative_diff = self.absolute_diff / self.baseline_mean
            self.percent_improvement = self.relative_diff * 100
        
        # Run hypothesis test
        if len(self.baseline_values) >= 2 and len(self.treatment_values) >= 2:
            self.hypothesis_test = StatisticalAnalysis.welchs_t_test(
                self.baseline_values, self.treatment_values
            )
            
            # Bootstrap CI for the difference
            diff_samples = [t - b for b, t in zip(self.baseline_values, self.treatment_values)]
            if diff_samples:
                self.confidence_interval = StatisticalAnalysis.bootstrap_confidence_interval(diff_samples)
    
    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "baseline": {
                "mean": self.baseline_mean,
                "std": self.baseline_std,
                "median": self.baseline_median,
                "n": len(self.baseline_values),
            },
            "treatment": {
                "mean": self.treatment_mean,
                "std": self.treatment_std,
                "median": self.treatment_median,
                "n": len(self.treatment_values),
            },
            "comparison": {
                "absolute_diff": self.absolute_diff,
                "relative_diff": self.relative_diff,
                "percent_improvement": self.percent_improvement,
            },
            "statistical_test": {
                "test_name": self.hypothesis_test.test_name if self.hypothesis_test else None,
                "p_value": self.hypothesis_test.p_value if self.hypothesis_test else None,
                "is_significant": self.hypothesis_test.is_significant if self.hypothesis_test else None,
                "effect_size": self.hypothesis_test.effect_size.cohens_d if self.hypothesis_test and self.hypothesis_test.effect_size else None,
                "interpretation": self.hypothesis_test.interpretation if self.hypothesis_test else None,
            },
            "confidence_interval": {
                "lower": self.confidence_interval[0],
                "estimate": self.confidence_interval[1],
                "upper": self.confidence_interval[2],
            },
        }


@dataclass
class TaskLevelComparison:
    """Per-task comparison between baseline and treatment."""
    task_id: str
    sdlc_type: SDLCTaskType
    
    # Results
    baseline_success: bool = False
    treatment_success: bool = False
    
    baseline_reward: float = 0.0
    treatment_reward: float = 0.0
    
    baseline_tokens: int = 0
    treatment_tokens: int = 0
    
    baseline_duration_sec: float = 0.0
    treatment_duration_sec: float = 0.0
    
    baseline_ir_queries: int = 0
    treatment_ir_queries: int = 0
    
    # LLM judge scores
    baseline_llm_score: Optional[float] = None
    treatment_llm_score: Optional[float] = None
    
    # Computed deltas
    reward_delta: float = 0.0
    token_efficiency: float = 0.0
    time_efficiency: float = 0.0
    query_efficiency: float = 0.0
    
    # Analysis
    winner: str = "tie"  # "baseline", "treatment", "tie"
    advantage_category: Optional[str] = None
    advantage_explanation: Optional[str] = None
    
    def compute_deltas(self) -> None:
        """Compute comparative metrics."""
        self.reward_delta = self.treatment_reward - self.baseline_reward
        
        if self.baseline_tokens > 0:
            self.token_efficiency = (self.baseline_tokens - self.treatment_tokens) / self.baseline_tokens
        
        if self.baseline_duration_sec > 0:
            self.time_efficiency = (self.baseline_duration_sec - self.treatment_duration_sec) / self.baseline_duration_sec
        
        if self.baseline_ir_queries > 0:
            self.query_efficiency = (self.baseline_ir_queries - self.treatment_ir_queries) / self.baseline_ir_queries
        
        # Determine winner
        if self.treatment_success and not self.baseline_success:
            self.winner = "treatment"
        elif self.baseline_success and not self.treatment_success:
            self.winner = "baseline"
        elif self.reward_delta > 0.1:
            self.winner = "treatment"
        elif self.reward_delta < -0.1:
            self.winner = "baseline"
        else:
            self.winner = "tie"
    
    def to_ir_comparison(self) -> IRComparison:
        """Convert to IRComparison for dashboard export."""
        comparison = IRComparison(
            task_id=self.task_id,
            sdlc_type=self.sdlc_type,
            baseline_success=self.baseline_success,
            baseline_reward=self.baseline_reward,
            baseline_tokens=self.baseline_tokens,
            baseline_duration_sec=self.baseline_duration_sec,
            baseline_ir_queries=self.baseline_ir_queries,
            ir_enhanced_success=self.treatment_success,
            ir_enhanced_reward=self.treatment_reward,
            ir_enhanced_tokens=self.treatment_tokens,
            ir_enhanced_duration_sec=self.treatment_duration_sec,
            ir_enhanced_queries=self.treatment_ir_queries,
            baseline_llm_score=self.baseline_llm_score,
            ir_enhanced_llm_score=self.treatment_llm_score,
            mcp_advantage_category=self.advantage_category,
            mcp_advantage_explanation=self.advantage_explanation,
        )
        comparison.compute_deltas()
        return comparison
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "sdlc_type": self.sdlc_type.value,
            "baseline": {
                "success": self.baseline_success,
                "reward": self.baseline_reward,
                "tokens": self.baseline_tokens,
                "duration_sec": self.baseline_duration_sec,
                "ir_queries": self.baseline_ir_queries,
                "llm_score": self.baseline_llm_score,
            },
            "treatment": {
                "success": self.treatment_success,
                "reward": self.treatment_reward,
                "tokens": self.treatment_tokens,
                "duration_sec": self.treatment_duration_sec,
                "ir_queries": self.treatment_ir_queries,
                "llm_score": self.treatment_llm_score,
            },
            "deltas": {
                "reward_delta": self.reward_delta,
                "token_efficiency": self.token_efficiency,
                "time_efficiency": self.time_efficiency,
                "query_efficiency": self.query_efficiency,
            },
            "winner": self.winner,
            "advantage_category": self.advantage_category,
            "advantage_explanation": self.advantage_explanation,
        }


@dataclass
class ComparisonReport:
    """Complete A/B comparison report.
    
    Aggregates task-level and metric-level comparisons with
    statistical analysis and visualization-ready data.
    """
    # Experiment info
    experiment_id: str
    experiment_name: str
    config: ComparisonConfig
    
    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    
    # Task-level results
    task_comparisons: List[TaskLevelComparison] = field(default_factory=list)
    
    # Metric-level comparisons
    metric_comparisons: Dict[str, MetricComparison] = field(default_factory=dict)
    
    # Aggregate metrics
    baseline_metrics: Optional[AgentMetrics] = None
    treatment_metrics: Optional[AgentMetrics] = None
    
    # Overall winner analysis
    overall_winner: str = "undetermined"  # "baseline", "treatment", "tie"
    treatment_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    tie_rate: float = 0.0
    
    # Statistical summary
    significant_improvements: List[str] = field(default_factory=list)
    significant_regressions: List[str] = field(default_factory=list)
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from task comparisons."""
        if not self.task_comparisons:
            return
        
        n_tasks = len(self.task_comparisons)
        
        # Win rates
        treatment_wins = sum(1 for t in self.task_comparisons if t.winner == "treatment")
        baseline_wins = sum(1 for t in self.task_comparisons if t.winner == "baseline")
        ties = sum(1 for t in self.task_comparisons if t.winner == "tie")
        
        self.treatment_win_rate = treatment_wins / n_tasks
        self.baseline_win_rate = baseline_wins / n_tasks
        self.tie_rate = ties / n_tasks
        
        # Overall winner
        if self.treatment_win_rate > self.baseline_win_rate + 0.1:
            self.overall_winner = "treatment"
        elif self.baseline_win_rate > self.treatment_win_rate + 0.1:
            self.overall_winner = "baseline"
        else:
            self.overall_winner = "tie"
        
        # Build metric comparisons
        self._compute_metric_comparisons()
        
        # Find significant results
        for name, mc in self.metric_comparisons.items():
            if mc.hypothesis_test and mc.hypothesis_test.is_significant:
                if mc.absolute_diff > 0:
                    self.significant_improvements.append(name)
                else:
                    self.significant_regressions.append(name)
        
        self.completed_at = datetime.now(timezone.utc).isoformat()
    
    def _compute_metric_comparisons(self) -> None:
        """Build metric comparisons from task data."""
        metrics_to_compare = [
            ("success_rate", lambda t: 1.0 if t.baseline_success else 0.0, lambda t: 1.0 if t.treatment_success else 0.0),
            ("reward", lambda t: t.baseline_reward, lambda t: t.treatment_reward),
            ("tokens", lambda t: float(t.baseline_tokens), lambda t: float(t.treatment_tokens)),
            ("duration_sec", lambda t: t.baseline_duration_sec, lambda t: t.treatment_duration_sec),
            ("ir_queries", lambda t: float(t.baseline_ir_queries), lambda t: float(t.treatment_ir_queries)),
        ]
        
        for name, baseline_fn, treatment_fn in metrics_to_compare:
            mc = MetricComparison(metric_name=name)
            mc.baseline_values = [baseline_fn(t) for t in self.task_comparisons]
            mc.treatment_values = [treatment_fn(t) for t in self.task_comparisons]
            mc.compute_comparison()
            self.metric_comparisons[name] = mc
        
        # LLM score comparison (only for tasks with scores)
        llm_baseline = [t.baseline_llm_score for t in self.task_comparisons if t.baseline_llm_score is not None]
        llm_treatment = [t.treatment_llm_score for t in self.task_comparisons if t.treatment_llm_score is not None]
        
        if llm_baseline and llm_treatment:
            mc = MetricComparison(metric_name="llm_score")
            mc.baseline_values = llm_baseline
            mc.treatment_values = llm_treatment
            mc.compute_comparison()
            self.metric_comparisons["llm_score"] = mc
    
    def get_summary(self) -> dict:
        """Get a concise summary of the comparison."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "total_tasks": len(self.task_comparisons),
            "overall_winner": self.overall_winner,
            "treatment_win_rate": self.treatment_win_rate,
            "baseline_win_rate": self.baseline_win_rate,
            "tie_rate": self.tie_rate,
            "significant_improvements": self.significant_improvements,
            "significant_regressions": self.significant_regressions,
            "key_metrics": {
                name: {
                    "baseline_mean": mc.baseline_mean,
                    "treatment_mean": mc.treatment_mean,
                    "percent_improvement": mc.percent_improvement,
                    "is_significant": mc.hypothesis_test.is_significant if mc.hypothesis_test else None,
                }
                for name, mc in self.metric_comparisons.items()
            },
        }
    
    def to_dict(self) -> dict:
        """Full serialization for export."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "config": self.config.to_dict(),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "task_comparisons": [t.to_dict() for t in self.task_comparisons],
            "metric_comparisons": {n: m.to_dict() for n, m in self.metric_comparisons.items()},
            "aggregate": {
                "overall_winner": self.overall_winner,
                "treatment_win_rate": self.treatment_win_rate,
                "baseline_win_rate": self.baseline_win_rate,
                "tie_rate": self.tie_rate,
                "significant_improvements": self.significant_improvements,
                "significant_regressions": self.significant_regressions,
            },
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def export_to_dashboard(self, exporter) -> Path:
        """Export comparison to dashboard format.
        
        Args:
            exporter: CodeContextBenchExporter instance
            
        Returns:
            Path to the exported comparison file
        """
        ir_comparisons = [t.to_ir_comparison() for t in self.task_comparisons]
        return exporter.export_comparison(ir_comparisons, f"{self.experiment_id}_comparison.json")


# =============================================================================
# A/B Comparator
# =============================================================================

class ABComparator:
    """Runs A/B comparison experiments between agents.
    
    Handles task execution, metric collection, and comparison report generation.
    """
    
    def __init__(
        self,
        config: ComparisonConfig,
        baseline_runner: Optional[AgentRunner] = None,
        treatment_runner: Optional[AgentRunner] = None,
    ):
        """Initialize comparator.
        
        Args:
            config: Comparison experiment configuration
            baseline_runner: Runner for baseline agent (can be set later)
            treatment_runner: Runner for treatment agent (can be set later)
        """
        self.config = config
        self.baseline_runner = baseline_runner
        self.treatment_runner = treatment_runner
        
        self.baseline_traces: List[AgentExecutionTrace] = []
        self.treatment_traces: List[AgentExecutionTrace] = []
        
        self.report: Optional[ComparisonReport] = None
    
    def run_comparison(
        self,
        tasks: List[dict],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> ComparisonReport:
        """Run A/B comparison on a set of tasks.
        
        Args:
            tasks: List of task specifications to run
            progress_callback: Optional callback(task_id, current, total)
            
        Returns:
            ComparisonReport with all results
        """
        if not self.baseline_runner or not self.treatment_runner:
            raise ValueError("Both baseline and treatment runners must be set")
        
        self.report = ComparisonReport(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.experiment_name,
            config=self.config,
        )
        
        total_tasks = len(tasks)
        
        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{i}")
            
            if progress_callback:
                progress_callback(task_id, i + 1, total_tasks)
            
            # Run baseline
            baseline_trace = self._run_with_retries(
                self.baseline_runner,
                task_id,
                task,
                self.config.baseline_import_path,
            )
            self.baseline_traces.append(baseline_trace)
            
            # Run treatment
            treatment_trace = self._run_with_retries(
                self.treatment_runner,
                task_id,
                task,
                self.config.treatment_import_path,
            )
            self.treatment_traces.append(treatment_trace)
            
            # Build task comparison
            comparison = self._build_task_comparison(
                task_id,
                task,
                baseline_trace,
                treatment_trace,
            )
            self.report.task_comparisons.append(comparison)
        
        # Compute aggregates
        self.report.baseline_metrics = AgentMetrics.compute_all(self.baseline_traces)
        self.report.treatment_metrics = AgentMetrics.compute_all(self.treatment_traces)
        self.report.compute_aggregates()
        
        return self.report
    
    def run_comparison_from_traces(
        self,
        baseline_traces: List[AgentExecutionTrace],
        treatment_traces: List[AgentExecutionTrace],
    ) -> ComparisonReport:
        """Build comparison from pre-collected traces.
        
        Useful for comparing previously run experiments.
        
        Args:
            baseline_traces: Traces from baseline agent runs
            treatment_traces: Traces from treatment agent runs
            
        Returns:
            ComparisonReport with comparison results
        """
        if len(baseline_traces) != len(treatment_traces):
            raise ValueError(
                f"Trace count mismatch: baseline={len(baseline_traces)}, "
                f"treatment={len(treatment_traces)}"
            )
        
        self.baseline_traces = baseline_traces
        self.treatment_traces = treatment_traces
        
        self.report = ComparisonReport(
            experiment_id=self.config.experiment_id,
            experiment_name=self.config.experiment_name,
            config=self.config,
        )
        
        for baseline, treatment in zip(baseline_traces, treatment_traces):
            comparison = self._build_task_comparison_from_traces(baseline, treatment)
            self.report.task_comparisons.append(comparison)
        
        self.report.baseline_metrics = AgentMetrics.compute_all(baseline_traces)
        self.report.treatment_metrics = AgentMetrics.compute_all(treatment_traces)
        self.report.compute_aggregates()
        
        return self.report
    
    def _run_with_retries(
        self,
        runner: AgentRunner,
        task_id: str,
        task: dict,
        agent_import_path: str,
    ) -> AgentExecutionTrace:
        """Run task with retry logic for multiple trials."""
        traces = []
        
        for trial in range(self.config.num_trials_per_task):
            try:
                trace = runner.run_task(
                    task_id=f"{task_id}_trial{trial}",
                    task_data=task,
                    agent_import_path=agent_import_path,
                    model_name=self.config.model_name,
                    timeout_seconds=self.config.timeout_seconds,
                )
                traces.append(trace)
            except Exception as e:
                # Create a failed trace
                trace = AgentExecutionTrace(
                    task_id=f"{task_id}_trial{trial}",
                    agent_name=agent_import_path,
                    ir_tool_name=agent_import_path,
                    completed=False,
                    completion_status=TaskCompletionStatus.ERROR,
                    error_message=str(e),
                )
                traces.append(trace)
        
        # Return the best trace if multiple trials
        if len(traces) == 1:
            return traces[0]
        
        # Pick the most successful trial
        successful = [t for t in traces if t.completion_status == TaskCompletionStatus.SUCCESS]
        if successful:
            return successful[0]
        
        partial = [t for t in traces if t.completion_status == TaskCompletionStatus.PARTIAL]
        if partial:
            return partial[0]
        
        return traces[0]
    
    def _build_task_comparison(
        self,
        task_id: str,
        task: dict,
        baseline: AgentExecutionTrace,
        treatment: AgentExecutionTrace,
    ) -> TaskLevelComparison:
        """Build TaskLevelComparison from task and traces."""
        sdlc_type = SDLCTaskType(task.get("sdlc_type", "bug_triage"))
        
        comparison = TaskLevelComparison(
            task_id=task_id,
            sdlc_type=sdlc_type,
            baseline_success=baseline.completion_status == TaskCompletionStatus.SUCCESS,
            treatment_success=treatment.completion_status == TaskCompletionStatus.SUCCESS,
            baseline_reward=baseline.tests_passed / max(baseline.tests_total, 1) if baseline.tests_total else (1.0 if baseline.completed else 0.0),
            treatment_reward=treatment.tests_passed / max(treatment.tests_total, 1) if treatment.tests_total else (1.0 if treatment.completed else 0.0),
            baseline_tokens=baseline.total_tokens,
            treatment_tokens=treatment.total_tokens,
            baseline_duration_sec=baseline.wall_clock_time_sec,
            treatment_duration_sec=treatment.wall_clock_time_sec,
        )
        
        # Add LLM judge scores if available
        if baseline.llm_judge_score:
            comparison.baseline_llm_score = baseline.llm_judge_score.overall
        if treatment.llm_judge_score:
            comparison.treatment_llm_score = treatment.llm_judge_score.overall
        
        comparison.compute_deltas()
        return comparison
    
    def _build_task_comparison_from_traces(
        self,
        baseline: AgentExecutionTrace,
        treatment: AgentExecutionTrace,
    ) -> TaskLevelComparison:
        """Build comparison directly from traces."""
        # Infer SDLC type from metadata if available
        sdlc_type_str = baseline.metadata.get("sdlc_type", "bug_triage")
        try:
            sdlc_type = SDLCTaskType(sdlc_type_str)
        except ValueError:
            sdlc_type = SDLCTaskType.BUG_TRIAGE
        
        comparison = TaskLevelComparison(
            task_id=baseline.task_id,
            sdlc_type=sdlc_type,
            baseline_success=baseline.completion_status == TaskCompletionStatus.SUCCESS,
            treatment_success=treatment.completion_status == TaskCompletionStatus.SUCCESS,
            baseline_reward=baseline.tests_passed / max(baseline.tests_total, 1) if baseline.tests_total else (1.0 if baseline.completed else 0.0),
            treatment_reward=treatment.tests_passed / max(treatment.tests_total, 1) if treatment.tests_total else (1.0 if treatment.completed else 0.0),
            baseline_tokens=baseline.total_tokens,
            treatment_tokens=treatment.total_tokens,
            baseline_duration_sec=baseline.wall_clock_time_sec,
            treatment_duration_sec=treatment.wall_clock_time_sec,
        )
        
        if baseline.llm_judge_score:
            comparison.baseline_llm_score = baseline.llm_judge_score.overall
        if treatment.llm_judge_score:
            comparison.treatment_llm_score = treatment.llm_judge_score.overall
        
        comparison.compute_deltas()
        return comparison
    
    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """Save comparison report to JSON file.
        
        Args:
            output_path: Path to save to (default: config.output_dir)
            
        Returns:
            Path to saved file
        """
        if not self.report:
            raise ValueError("No report available - run comparison first")
        
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{self.config.experiment_id}_report.json"
        
        with open(output_path, "w") as f:
            f.write(self.report.to_json())
        
        return output_path


# =============================================================================
# Comparison Utilities
# =============================================================================

def create_comparison_from_runs(
    baseline_run: IRSDLCBenchmarkRun,
    treatment_run: IRSDLCBenchmarkRun,
    experiment_name: str = "run_comparison",
) -> ComparisonReport:
    """Create a comparison report from two benchmark runs.
    
    Args:
        baseline_run: Baseline benchmark run
        treatment_run: Treatment/MCP-enhanced run
        experiment_name: Name for the comparison
        
    Returns:
        ComparisonReport
    """
    config = ComparisonConfig(
        experiment_id=f"{baseline_run.run_id}_vs_{treatment_run.run_id}",
        experiment_name=experiment_name,
        baseline_tool_type=baseline_run.ir_tool_type,
        baseline_agent_name=baseline_run.agent_name or "baseline",
        treatment_tool_type=treatment_run.ir_tool_type,
        treatment_agent_name=treatment_run.agent_name or "treatment",
        model_name=baseline_run.model_name,
    )
    
    # Convert task results to traces
    baseline_traces = [_task_result_to_trace(r) for r in baseline_run.task_results]
    treatment_traces = [_task_result_to_trace(r) for r in treatment_run.task_results]
    
    comparator = ABComparator(config)
    return comparator.run_comparison_from_traces(baseline_traces, treatment_traces)


def _task_result_to_trace(result: IRSDLCTaskResult) -> AgentExecutionTrace:
    """Convert IRSDLCTaskResult to AgentExecutionTrace."""
    trace = AgentExecutionTrace(
        task_id=result.task_id,
        agent_name=result.agent_import_path,
        ir_tool_name=result.ir_tool_name,
        completed=result.execution_metrics.success,
        completion_status=(
            TaskCompletionStatus.SUCCESS if result.execution_metrics.success 
            else TaskCompletionStatus.FAILURE
        ),
        total_tokens=result.execution_metrics.total_tokens,
        wall_clock_time_sec=result.execution_metrics.duration_sec,
        context_tokens=result.ir_metrics.context_tokens_retrieved,
        context_files=len(result.retrieved_files),
    )
    
    # Add metadata for SDLC type
    trace.metadata["sdlc_type"] = result.sdlc_type.value
    
    # Add LLM judge score if available
    if result.llm_judge:
        trace.llm_judge_score = result.llm_judge
    
    return trace


def compare_multiple_tools(
    runs_by_tool: Dict[str, IRSDLCBenchmarkRun],
    baseline_tool: str = "baseline",
) -> Dict[str, ComparisonReport]:
    """Compare multiple IR tools against a baseline.
    
    Args:
        runs_by_tool: Dict mapping tool name to benchmark run
        baseline_tool: Name of the baseline tool to compare against
        
    Returns:
        Dict mapping tool name to ComparisonReport vs baseline
    """
    if baseline_tool not in runs_by_tool:
        raise ValueError(f"Baseline tool '{baseline_tool}' not in runs")
    
    baseline_run = runs_by_tool[baseline_tool]
    comparisons = {}
    
    for tool_name, run in runs_by_tool.items():
        if tool_name == baseline_tool:
            continue
        
        comparisons[tool_name] = create_comparison_from_runs(
            baseline_run,
            run,
            experiment_name=f"{baseline_tool}_vs_{tool_name}",
        )
    
    return comparisons


def rank_tools_by_metric(
    comparisons: Dict[str, ComparisonReport],
    metric: str = "success_rate",
) -> List[Tuple[str, float, bool]]:
    """Rank tools by a specific metric improvement.
    
    Args:
        comparisons: Dict of tool name to ComparisonReport
        metric: Metric to rank by
        
    Returns:
        List of (tool_name, improvement_percent, is_significant) sorted by improvement
    """
    rankings = []
    
    for tool_name, report in comparisons.items():
        if metric in report.metric_comparisons:
            mc = report.metric_comparisons[metric]
            is_sig = mc.hypothesis_test.is_significant if mc.hypothesis_test else False
            rankings.append((tool_name, mc.percent_improvement, is_sig))
    
    # Sort by improvement descending
    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings
