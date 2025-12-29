"""
IR Impact Telemetry Capture for IR-SDLC-Bench.

This module defines telemetry capture to measure how IR quality impacts
agent output quality. It instruments agents to collect IR-specific metrics
that predict downstream task success.

Key signals captured:
1. Retrieval Accuracy - Did the agent find the right files?
2. Context Utilization - How much retrieved context was used?
3. Navigation Efficiency - Time/tokens spent searching vs implementing
4. First-Retrieval Success - Was the first search attempt productive?

Integrates with:
- Agent execution traces (AgentExecutionTrace)
- IR metrics (IRMetrics)
- Dashboard export (IRRetrievalMetrics)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.ir_sdlc.data_structures import (
    CodeLocation,
    GroundTruth,
    RetrievalGranularity,
    RetrievalResult,
)
from app.ir_sdlc.metrics import IRMetrics


# =============================================================================
# IR Event Types
# =============================================================================

class IREventType(str, Enum):
    """Types of IR events to capture."""
    SEARCH_QUERY = "search_query"
    SEARCH_RESULT = "search_result"
    FILE_READ = "file_read"
    FILE_SKIP = "file_skip"
    CONTEXT_USED = "context_used"
    CONTEXT_DISCARDED = "context_discarded"
    NAVIGATION_START = "navigation_start"
    NAVIGATION_END = "navigation_end"
    IMPLEMENTATION_START = "implementation_start"
    IMPLEMENTATION_END = "implementation_end"


class SearchType(str, Enum):
    """Types of search operations."""
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    NLS = "nls"  # Natural Language Search
    DEEP_SEARCH = "deep_search"
    GREP = "grep"
    GLOB = "glob"
    SYMBOL = "symbol"
    REFERENCE = "reference"


# =============================================================================
# Telemetry Events
# =============================================================================

@dataclass
class IREvent:
    """A single IR telemetry event.
    
    Records an IR-related action taken by the agent.
    """
    event_type: IREventType
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Search-specific fields
    search_type: Optional[SearchType] = None
    query: Optional[str] = None
    results_count: Optional[int] = None
    relevant_results_count: Optional[int] = None
    
    # File-specific fields
    file_path: Optional[str] = None
    file_tokens: Optional[int] = None
    file_was_relevant: Optional[bool] = None
    
    # Context-specific fields
    context_tokens: Optional[int] = None
    context_lines: Optional[int] = None
    
    # Timing
    duration_ms: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SearchAttempt:
    """A single search attempt with results.
    
    Tracks whether a search was productive.
    """
    attempt_number: int
    search_type: SearchType
    query: str
    
    # Results
    results_count: int = 0
    relevant_results_count: int = 0
    first_relevant_rank: Optional[int] = None  # Rank of first relevant result (1-indexed)
    
    # Productivity assessment
    was_productive: bool = False  # Did it lead to useful context?
    led_to_implementation: bool = False  # Did agent proceed to implementation?
    
    # Cost
    tokens_used: int = 0
    time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Telemetry Session
# =============================================================================

@dataclass
class IRTelemetrySession:
    """Telemetry session for a single agent task execution.
    
    Aggregates all IR events and computes derived metrics.
    """
    task_id: str
    agent_name: str
    ir_tool_name: str
    
    # Session timing
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    
    # Raw events
    events: List[IREvent] = field(default_factory=list)
    search_attempts: List[SearchAttempt] = field(default_factory=list)
    
    # Files accessed
    files_read: List[str] = field(default_factory=list)
    files_relevant: List[str] = field(default_factory=list)  # Files actually used
    files_discarded: List[str] = field(default_factory=list)  # Files read but not used
    
    # Ground truth (if known)
    ground_truth_files: List[str] = field(default_factory=list)
    
    # Phase timing (in seconds)
    navigation_time_sec: float = 0.0
    implementation_time_sec: float = 0.0
    total_time_sec: float = 0.0
    
    # Phase tokens
    navigation_tokens: int = 0
    implementation_tokens: int = 0
    total_tokens: int = 0
    
    # Computed metrics (call compute_metrics to populate)
    _metrics_computed: bool = False
    
    def record_event(self, event: IREvent) -> None:
        """Record an IR event."""
        self.events.append(event)
    
    def record_search(self, attempt: SearchAttempt) -> None:
        """Record a search attempt."""
        self.search_attempts.append(attempt)
        
        # Also create an event
        self.record_event(IREvent(
            event_type=IREventType.SEARCH_QUERY,
            search_type=attempt.search_type,
            query=attempt.query,
            results_count=attempt.results_count,
            relevant_results_count=attempt.relevant_results_count,
            duration_ms=attempt.time_ms,
        ))
    
    def record_file_read(
        self,
        file_path: str,
        tokens: int,
        was_relevant: bool = False,
    ) -> None:
        """Record a file read operation."""
        self.files_read.append(file_path)
        if was_relevant:
            self.files_relevant.append(file_path)
        
        self.record_event(IREvent(
            event_type=IREventType.FILE_READ,
            file_path=file_path,
            file_tokens=tokens,
            file_was_relevant=was_relevant,
        ))
    
    def end_session(self, total_tokens: int = 0, total_time_sec: float = 0.0) -> None:
        """End the session and record final metrics."""
        self.end_time = datetime.now(timezone.utc).isoformat()
        self.total_tokens = total_tokens
        self.total_time_sec = total_time_sec
    
    def compute_metrics(self) -> "IRImpactMetrics":
        """Compute impact metrics from the session data."""
        metrics = IRImpactMetrics.from_session(self)
        self._metrics_computed = True
        return metrics
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "ir_tool_name": self.ir_tool_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events_count": len(self.events),
            "search_attempts_count": len(self.search_attempts),
            "files_read_count": len(self.files_read),
            "files_relevant_count": len(self.files_relevant),
            "navigation_time_sec": self.navigation_time_sec,
            "implementation_time_sec": self.implementation_time_sec,
            "navigation_tokens": self.navigation_tokens,
            "implementation_tokens": self.implementation_tokens,
        }


# =============================================================================
# IR Impact Metrics
# =============================================================================

@dataclass
class RetrievalAccuracyMetrics:
    """Metrics measuring whether the agent found the right files.
    
    Key question: Did IR help the agent find relevant code?
    """
    # File-level accuracy
    files_found_precision: float = 0.0  # Relevant files found / all files read
    files_found_recall: float = 0.0  # Relevant files found / all relevant files
    files_found_f1: float = 0.0
    
    # Ranking quality
    first_relevant_file_rank: Optional[int] = None  # Rank of first ground truth file
    mrr: float = 0.0  # Mean reciprocal rank
    
    # Ground truth coverage
    ground_truth_coverage: float = 0.0  # % of ground truth files found
    false_positive_rate: float = 0.0  # Irrelevant files / all files read
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ContextUtilizationMetrics:
    """Metrics measuring how much retrieved context was used.
    
    Key question: Was the retrieved context actually useful?
    """
    # Token utilization
    context_tokens_retrieved: int = 0
    context_tokens_used: int = 0
    context_utilization_ratio: float = 0.0  # used / retrieved
    
    # File utilization
    files_retrieved: int = 0
    files_used: int = 0
    file_utilization_ratio: float = 0.0  # used / retrieved
    
    # Waste metrics
    wasted_tokens: int = 0  # Tokens in unused context
    waste_ratio: float = 0.0  # wasted / retrieved
    
    # Quality indicators
    high_utilization: bool = False  # utilization > 0.5
    context_was_sufficient: bool = False  # Agent didn't need more searches
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NavigationEfficiencyMetrics:
    """Metrics measuring time/tokens spent searching vs implementing.
    
    Key question: Did IR reduce navigation overhead?
    """
    # Time split
    navigation_time_sec: float = 0.0
    implementation_time_sec: float = 0.0
    navigation_time_ratio: float = 0.0  # nav / (nav + impl)
    
    # Token split
    navigation_tokens: int = 0
    implementation_tokens: int = 0
    navigation_token_ratio: float = 0.0  # nav / (nav + impl)
    
    # Efficiency indicators
    low_navigation_overhead: bool = False  # ratio < 0.3
    time_efficient: bool = False  # ratio < 0.25
    token_efficient: bool = False  # ratio < 0.4
    
    # Search iterations
    search_iterations: int = 0  # Number of search-refine cycles
    searches_before_success: int = 0  # Searches before first useful result
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FirstRetrievalMetrics:
    """Metrics measuring first search attempt productivity.
    
    Key question: Was the first search attempt productive?
    """
    # First attempt success
    first_search_productive: bool = False  # First search returned relevant results
    first_search_led_to_implementation: bool = False  # Went to impl after first search
    
    # First attempt quality
    first_search_precision: float = 0.0  # Relevant / total in first search
    first_search_recall: float = 0.0  # Relevant found / all relevant
    first_relevant_rank: Optional[int] = None  # Rank of first relevant in first search
    
    # Recovery metrics
    needed_refinement: bool = False  # Had to refine query
    refinement_count: int = 0  # Number of query refinements
    refinement_improved: bool = False  # Refinement led to better results
    
    # One-shot success
    one_shot_success: bool = False  # Completed task with single search
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class IRImpactMetrics:
    """Complete IR impact metrics for a task.
    
    Aggregates all impact signals to measure how IR quality
    affected agent success on this task.
    """
    task_id: str
    agent_name: str
    ir_tool_name: str
    
    # Component metrics
    retrieval_accuracy: RetrievalAccuracyMetrics = field(default_factory=RetrievalAccuracyMetrics)
    context_utilization: ContextUtilizationMetrics = field(default_factory=ContextUtilizationMetrics)
    navigation_efficiency: NavigationEfficiencyMetrics = field(default_factory=NavigationEfficiencyMetrics)
    first_retrieval: FirstRetrievalMetrics = field(default_factory=FirstRetrievalMetrics)
    
    # Aggregate scores (0-1 scale)
    retrieval_quality_score: float = 0.0
    efficiency_score: float = 0.0
    overall_ir_impact_score: float = 0.0
    
    # Outcome prediction
    predicted_success_probability: float = 0.0  # Based on IR metrics
    actual_success: Optional[bool] = None
    prediction_was_correct: Optional[bool] = None
    
    @classmethod
    def from_session(cls, session: IRTelemetrySession) -> "IRImpactMetrics":
        """Compute IR impact metrics from a telemetry session."""
        metrics = cls(
            task_id=session.task_id,
            agent_name=session.agent_name,
            ir_tool_name=session.ir_tool_name,
        )
        
        # Compute retrieval accuracy
        metrics.retrieval_accuracy = cls._compute_retrieval_accuracy(session)
        
        # Compute context utilization
        metrics.context_utilization = cls._compute_context_utilization(session)
        
        # Compute navigation efficiency
        metrics.navigation_efficiency = cls._compute_navigation_efficiency(session)
        
        # Compute first retrieval metrics
        metrics.first_retrieval = cls._compute_first_retrieval(session)
        
        # Compute aggregate scores
        metrics._compute_aggregate_scores()
        
        return metrics
    
    @staticmethod
    def _compute_retrieval_accuracy(session: IRTelemetrySession) -> RetrievalAccuracyMetrics:
        """Compute retrieval accuracy metrics."""
        acc = RetrievalAccuracyMetrics()
        
        files_read = set(session.files_read)
        files_relevant = set(session.files_relevant)
        ground_truth = set(session.ground_truth_files)
        
        if files_read:
            acc.files_found_precision = len(files_relevant) / len(files_read)
            acc.false_positive_rate = len(files_read - files_relevant) / len(files_read)
        
        if ground_truth:
            files_found = files_read & ground_truth
            acc.files_found_recall = len(files_found) / len(ground_truth)
            acc.ground_truth_coverage = acc.files_found_recall
        
        if acc.files_found_precision + acc.files_found_recall > 0:
            acc.files_found_f1 = (
                2 * acc.files_found_precision * acc.files_found_recall /
                (acc.files_found_precision + acc.files_found_recall)
            )
        
        # Compute MRR from search attempts
        for i, attempt in enumerate(session.search_attempts):
            if attempt.first_relevant_rank is not None:
                acc.first_relevant_file_rank = attempt.first_relevant_rank
                acc.mrr = 1.0 / attempt.first_relevant_rank
                break
        
        return acc
    
    @staticmethod
    def _compute_context_utilization(session: IRTelemetrySession) -> ContextUtilizationMetrics:
        """Compute context utilization metrics."""
        util = ContextUtilizationMetrics()
        
        # Count tokens from file read events
        file_events = [e for e in session.events if e.event_type == IREventType.FILE_READ]
        util.context_tokens_retrieved = sum(e.file_tokens or 0 for e in file_events)
        
        relevant_events = [e for e in file_events if e.file_was_relevant]
        util.context_tokens_used = sum(e.file_tokens or 0 for e in relevant_events)
        
        if util.context_tokens_retrieved > 0:
            util.context_utilization_ratio = util.context_tokens_used / util.context_tokens_retrieved
            util.wasted_tokens = util.context_tokens_retrieved - util.context_tokens_used
            util.waste_ratio = util.wasted_tokens / util.context_tokens_retrieved
        
        util.files_retrieved = len(session.files_read)
        util.files_used = len(session.files_relevant)
        
        if util.files_retrieved > 0:
            util.file_utilization_ratio = util.files_used / util.files_retrieved
        
        util.high_utilization = util.context_utilization_ratio > 0.5
        util.context_was_sufficient = len(session.search_attempts) <= 2
        
        return util
    
    @staticmethod
    def _compute_navigation_efficiency(session: IRTelemetrySession) -> NavigationEfficiencyMetrics:
        """Compute navigation efficiency metrics."""
        eff = NavigationEfficiencyMetrics()
        
        eff.navigation_time_sec = session.navigation_time_sec
        eff.implementation_time_sec = session.implementation_time_sec
        eff.navigation_tokens = session.navigation_tokens
        eff.implementation_tokens = session.implementation_tokens
        
        total_time = eff.navigation_time_sec + eff.implementation_time_sec
        if total_time > 0:
            eff.navigation_time_ratio = eff.navigation_time_sec / total_time
        
        total_tokens = eff.navigation_tokens + eff.implementation_tokens
        if total_tokens > 0:
            eff.navigation_token_ratio = eff.navigation_tokens / total_tokens
        
        eff.low_navigation_overhead = eff.navigation_time_ratio < 0.3
        eff.time_efficient = eff.navigation_time_ratio < 0.25
        eff.token_efficient = eff.navigation_token_ratio < 0.4
        
        eff.search_iterations = len(session.search_attempts)
        
        # Count searches before first productive one
        for i, attempt in enumerate(session.search_attempts):
            if attempt.was_productive:
                eff.searches_before_success = i
                break
        else:
            eff.searches_before_success = len(session.search_attempts)
        
        return eff
    
    @staticmethod
    def _compute_first_retrieval(session: IRTelemetrySession) -> FirstRetrievalMetrics:
        """Compute first retrieval metrics."""
        first = FirstRetrievalMetrics()
        
        if not session.search_attempts:
            return first
        
        first_attempt = session.search_attempts[0]
        
        first.first_search_productive = first_attempt.was_productive
        first.first_search_led_to_implementation = first_attempt.led_to_implementation
        
        if first_attempt.results_count > 0:
            first.first_search_precision = first_attempt.relevant_results_count / first_attempt.results_count
        
        first.first_relevant_rank = first_attempt.first_relevant_rank
        
        # Check for refinements
        first.needed_refinement = len(session.search_attempts) > 1
        first.refinement_count = max(0, len(session.search_attempts) - 1)
        
        # Check if refinement improved
        if first.needed_refinement and len(session.search_attempts) >= 2:
            second_attempt = session.search_attempts[1]
            first.refinement_improved = (
                second_attempt.relevant_results_count > first_attempt.relevant_results_count or
                (second_attempt.first_relevant_rank is not None and
                 first_attempt.first_relevant_rank is not None and
                 second_attempt.first_relevant_rank < first_attempt.first_relevant_rank)
            )
        
        # One-shot success
        first.one_shot_success = (
            len(session.search_attempts) == 1 and
            first_attempt.led_to_implementation
        )
        
        return first
    
    def _compute_aggregate_scores(self) -> None:
        """Compute aggregate scores from component metrics."""
        # Retrieval quality score (0-1)
        self.retrieval_quality_score = (
            0.3 * self.retrieval_accuracy.files_found_f1 +
            0.3 * self.retrieval_accuracy.mrr +
            0.2 * self.retrieval_accuracy.ground_truth_coverage +
            0.2 * (1 - self.retrieval_accuracy.false_positive_rate)
        )
        
        # Efficiency score (0-1)
        self.efficiency_score = (
            0.25 * (1 - self.navigation_efficiency.navigation_time_ratio) +
            0.25 * (1 - self.navigation_efficiency.navigation_token_ratio) +
            0.25 * self.context_utilization.context_utilization_ratio +
            0.25 * (1 if self.first_retrieval.first_search_productive else 0)
        )
        
        # Overall IR impact score (0-1)
        self.overall_ir_impact_score = (
            0.5 * self.retrieval_quality_score +
            0.5 * self.efficiency_score
        )
        
        # Predicted success probability (simple heuristic)
        self.predicted_success_probability = (
            0.4 * self.overall_ir_impact_score +
            0.3 * (1 if self.first_retrieval.one_shot_success else 0.3) +
            0.3 * self.retrieval_accuracy.ground_truth_coverage
        )
    
    def set_actual_outcome(self, success: bool) -> None:
        """Record the actual task outcome."""
        self.actual_success = success
        
        # Check if prediction was correct
        predicted = self.predicted_success_probability > 0.5
        self.prediction_was_correct = (predicted == success)
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "ir_tool_name": self.ir_tool_name,
            "retrieval_accuracy": self.retrieval_accuracy.to_dict(),
            "context_utilization": self.context_utilization.to_dict(),
            "navigation_efficiency": self.navigation_efficiency.to_dict(),
            "first_retrieval": self.first_retrieval.to_dict(),
            "aggregate": {
                "retrieval_quality_score": self.retrieval_quality_score,
                "efficiency_score": self.efficiency_score,
                "overall_ir_impact_score": self.overall_ir_impact_score,
                "predicted_success_probability": self.predicted_success_probability,
            },
            "outcome": {
                "actual_success": self.actual_success,
                "prediction_was_correct": self.prediction_was_correct,
            },
        }


# =============================================================================
# Telemetry Collector
# =============================================================================

class IRTelemetryCollector:
    """Collector for IR telemetry across multiple task executions.
    
    Provides methods to instrument agent execution and aggregate metrics.
    """
    
    def __init__(self, agent_name: str, ir_tool_name: str):
        self.agent_name = agent_name
        self.ir_tool_name = ir_tool_name
        
        self.sessions: List[IRTelemetrySession] = []
        self.metrics: List[IRImpactMetrics] = []
        
        self._active_session: Optional[IRTelemetrySession] = None
    
    def start_session(self, task_id: str) -> IRTelemetrySession:
        """Start a new telemetry session for a task."""
        session = IRTelemetrySession(
            task_id=task_id,
            agent_name=self.agent_name,
            ir_tool_name=self.ir_tool_name,
        )
        self._active_session = session
        return session
    
    def end_session(
        self,
        total_tokens: int = 0,
        total_time_sec: float = 0.0,
        success: Optional[bool] = None,
    ) -> IRImpactMetrics:
        """End the current session and compute metrics."""
        if self._active_session is None:
            raise ValueError("No active session to end")
        
        session = self._active_session
        session.end_session(total_tokens, total_time_sec)
        
        self.sessions.append(session)
        
        metrics = session.compute_metrics()
        if success is not None:
            metrics.set_actual_outcome(success)
        
        self.metrics.append(metrics)
        self._active_session = None
        
        return metrics
    
    @property
    def active_session(self) -> Optional[IRTelemetrySession]:
        """Get the current active session."""
        return self._active_session
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics across all sessions."""
        if not self.metrics:
            return {}
        
        # Compute averages for key metrics
        agg = {
            "total_sessions": len(self.metrics),
            "agent_name": self.agent_name,
            "ir_tool_name": self.ir_tool_name,
        }
        
        # Retrieval accuracy
        agg["avg_files_found_f1"] = statistics.mean(
            m.retrieval_accuracy.files_found_f1 for m in self.metrics
        )
        agg["avg_mrr"] = statistics.mean(
            m.retrieval_accuracy.mrr for m in self.metrics
        )
        agg["avg_ground_truth_coverage"] = statistics.mean(
            m.retrieval_accuracy.ground_truth_coverage for m in self.metrics
        )
        
        # Context utilization
        agg["avg_context_utilization"] = statistics.mean(
            m.context_utilization.context_utilization_ratio for m in self.metrics
        )
        agg["high_utilization_rate"] = statistics.mean(
            1 if m.context_utilization.high_utilization else 0 for m in self.metrics
        )
        
        # Navigation efficiency
        agg["avg_navigation_time_ratio"] = statistics.mean(
            m.navigation_efficiency.navigation_time_ratio for m in self.metrics
        )
        agg["avg_navigation_token_ratio"] = statistics.mean(
            m.navigation_efficiency.navigation_token_ratio for m in self.metrics
        )
        agg["time_efficient_rate"] = statistics.mean(
            1 if m.navigation_efficiency.time_efficient else 0 for m in self.metrics
        )
        
        # First retrieval
        agg["first_search_productive_rate"] = statistics.mean(
            1 if m.first_retrieval.first_search_productive else 0 for m in self.metrics
        )
        agg["one_shot_success_rate"] = statistics.mean(
            1 if m.first_retrieval.one_shot_success else 0 for m in self.metrics
        )
        agg["avg_refinement_count"] = statistics.mean(
            m.first_retrieval.refinement_count for m in self.metrics
        )
        
        # Aggregate scores
        agg["avg_retrieval_quality_score"] = statistics.mean(
            m.retrieval_quality_score for m in self.metrics
        )
        agg["avg_efficiency_score"] = statistics.mean(
            m.efficiency_score for m in self.metrics
        )
        agg["avg_overall_ir_impact_score"] = statistics.mean(
            m.overall_ir_impact_score for m in self.metrics
        )
        
        # Prediction accuracy
        metrics_with_outcomes = [m for m in self.metrics if m.actual_success is not None]
        if metrics_with_outcomes:
            agg["prediction_accuracy"] = statistics.mean(
                1 if m.prediction_was_correct else 0 for m in metrics_with_outcomes
            )
            agg["actual_success_rate"] = statistics.mean(
                1 if m.actual_success else 0 for m in metrics_with_outcomes
            )
        
        return agg
    
    def get_correlation_with_success(self) -> Dict[str, float]:
        """Compute correlation between IR metrics and task success."""
        metrics_with_outcomes = [m for m in self.metrics if m.actual_success is not None]
        
        if len(metrics_with_outcomes) < 5:
            return {}
        
        success_values = [1.0 if m.actual_success else 0.0 for m in metrics_with_outcomes]
        
        correlations = {}
        
        # Key metrics to correlate
        metric_extractors = {
            "retrieval_quality_score": lambda m: m.retrieval_quality_score,
            "efficiency_score": lambda m: m.efficiency_score,
            "overall_ir_impact_score": lambda m: m.overall_ir_impact_score,
            "files_found_f1": lambda m: m.retrieval_accuracy.files_found_f1,
            "mrr": lambda m: m.retrieval_accuracy.mrr,
            "context_utilization": lambda m: m.context_utilization.context_utilization_ratio,
            "navigation_time_ratio": lambda m: m.navigation_efficiency.navigation_time_ratio,
            "first_search_productive": lambda m: 1.0 if m.first_retrieval.first_search_productive else 0.0,
        }
        
        for metric_name, extractor in metric_extractors.items():
            metric_values = [extractor(m) for m in metrics_with_outcomes]
            corr = self._pearson_correlation(metric_values, success_values)
            if corr is not None:
                correlations[metric_name] = corr
        
        return correlations
    
    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> Optional[float]:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return None
        
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
        std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
        
        if std_x == 0 or std_y == 0:
            return None
        
        return covariance / (std_x * std_y)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_collector_for_agent(
    agent_name: str,
    ir_tool_name: str,
) -> IRTelemetryCollector:
    """Create a telemetry collector for an agent.
    
    Args:
        agent_name: Name of the agent
        ir_tool_name: Name of the IR tool being used
        
    Returns:
        Configured IRTelemetryCollector
    """
    return IRTelemetryCollector(agent_name, ir_tool_name)


def compute_ir_impact_from_trace(
    task_id: str,
    agent_name: str,
    ir_tool_name: str,
    files_read: List[str],
    files_relevant: List[str],
    ground_truth_files: List[str],
    navigation_time_sec: float,
    implementation_time_sec: float,
    navigation_tokens: int,
    implementation_tokens: int,
    search_attempts: List[SearchAttempt],
    success: Optional[bool] = None,
) -> IRImpactMetrics:
    """Compute IR impact metrics from trace data.
    
    Convenience function for computing metrics without using the collector.
    
    Args:
        task_id: Task identifier
        agent_name: Agent name
        ir_tool_name: IR tool name
        files_read: List of files read by agent
        files_relevant: List of files that were relevant
        ground_truth_files: Ground truth file list
        navigation_time_sec: Time spent navigating
        implementation_time_sec: Time spent implementing
        navigation_tokens: Tokens spent on navigation
        implementation_tokens: Tokens spent on implementation
        search_attempts: List of search attempts
        success: Optional actual task success
        
    Returns:
        Computed IRImpactMetrics
    """
    session = IRTelemetrySession(
        task_id=task_id,
        agent_name=agent_name,
        ir_tool_name=ir_tool_name,
    )
    
    session.files_read = files_read
    session.files_relevant = files_relevant
    session.ground_truth_files = ground_truth_files
    session.navigation_time_sec = navigation_time_sec
    session.implementation_time_sec = implementation_time_sec
    session.navigation_tokens = navigation_tokens
    session.implementation_tokens = implementation_tokens
    session.search_attempts = search_attempts
    
    metrics = session.compute_metrics()
    
    if success is not None:
        metrics.set_actual_outcome(success)
    
    return metrics
