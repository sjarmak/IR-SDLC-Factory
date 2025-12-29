"""
Tests for IR impact telemetry capture.

Validates:
1. Telemetry session recording
2. Retrieval accuracy metrics computation
3. Context utilization metrics computation
4. Navigation efficiency metrics computation
5. First retrieval metrics computation
6. Aggregate score computation
7. Correlation with success
"""

import pytest
from typing import List

from app.ir_sdlc.ir_telemetry import (
    IREventType,
    SearchType,
    IREvent,
    SearchAttempt,
    IRTelemetrySession,
    IRImpactMetrics,
    RetrievalAccuracyMetrics,
    ContextUtilizationMetrics,
    NavigationEfficiencyMetrics,
    FirstRetrievalMetrics,
    IRTelemetryCollector,
    create_collector_for_agent,
    compute_ir_impact_from_trace,
)


class TestIREvent:
    """Tests for IR event recording."""
    
    def test_event_creation(self):
        """Events should be created with timestamp."""
        event = IREvent(event_type=IREventType.SEARCH_QUERY)
        
        assert event.event_type == IREventType.SEARCH_QUERY
        assert event.timestamp is not None
    
    def test_event_with_search_details(self):
        """Search events should capture search details."""
        event = IREvent(
            event_type=IREventType.SEARCH_QUERY,
            search_type=SearchType.SEMANTIC,
            query="find authentication handler",
            results_count=10,
            relevant_results_count=3,
            duration_ms=150.0,
        )
        
        assert event.search_type == SearchType.SEMANTIC
        assert event.query == "find authentication handler"
        assert event.results_count == 10
        assert event.relevant_results_count == 3
    
    def test_event_to_dict(self):
        """Events should serialize to dict without None values."""
        event = IREvent(
            event_type=IREventType.FILE_READ,
            file_path="src/auth.py",
            file_tokens=500,
        )
        
        d = event.to_dict()
        
        assert "event_type" in d
        assert "file_path" in d
        assert "query" not in d  # None values excluded


class TestSearchAttempt:
    """Tests for search attempt recording."""
    
    def test_search_attempt_creation(self):
        """Search attempts should capture all details."""
        attempt = SearchAttempt(
            attempt_number=1,
            search_type=SearchType.DEEP_SEARCH,
            query="authentication middleware",
            results_count=15,
            relevant_results_count=5,
            first_relevant_rank=2,
            was_productive=True,
            led_to_implementation=True,
            tokens_used=200,
            time_ms=100.0,
        )
        
        assert attempt.attempt_number == 1
        assert attempt.search_type == SearchType.DEEP_SEARCH
        assert attempt.was_productive
        assert attempt.first_relevant_rank == 2


class TestIRTelemetrySession:
    """Tests for telemetry session management."""
    
    def test_session_creation(self):
        """Sessions should be created with metadata."""
        session = IRTelemetrySession(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        assert session.task_id == "test-001"
        assert session.start_time is not None
        assert session.end_time is None
    
    def test_record_search(self):
        """Session should record search attempts."""
        session = IRTelemetrySession(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        attempt = SearchAttempt(
            attempt_number=1,
            search_type=SearchType.SEMANTIC,
            query="auth handler",
        )
        session.record_search(attempt)
        
        assert len(session.search_attempts) == 1
        assert len(session.events) == 1
        assert session.events[0].event_type == IREventType.SEARCH_QUERY
    
    def test_record_file_read(self):
        """Session should record file reads."""
        session = IRTelemetrySession(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        session.record_file_read("src/auth.py", tokens=500, was_relevant=True)
        session.record_file_read("src/utils.py", tokens=200, was_relevant=False)
        
        assert len(session.files_read) == 2
        assert len(session.files_relevant) == 1
        assert "src/auth.py" in session.files_relevant
    
    def test_end_session(self):
        """Session should record end time and totals."""
        session = IRTelemetrySession(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        session.end_session(total_tokens=1000, total_time_sec=60.0)
        
        assert session.end_time is not None
        assert session.total_tokens == 1000
        assert session.total_time_sec == 60.0
    
    def test_compute_metrics(self):
        """Session should compute metrics."""
        session = IRTelemetrySession(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        session.files_read = ["a.py", "b.py", "c.py"]
        session.files_relevant = ["a.py", "b.py"]
        session.ground_truth_files = ["a.py", "d.py"]
        session.navigation_time_sec = 30.0
        session.implementation_time_sec = 90.0
        
        metrics = session.compute_metrics()
        
        assert isinstance(metrics, IRImpactMetrics)
        assert metrics.task_id == "test-001"


class TestRetrievalAccuracyMetrics:
    """Tests for retrieval accuracy computation."""
    
    def test_precision_calculation(self):
        """Precision should be relevant / read."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.files_read = ["a.py", "b.py", "c.py", "d.py"]
        session.files_relevant = ["a.py", "b.py"]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.retrieval_accuracy.files_found_precision == 0.5
    
    def test_recall_calculation(self):
        """Recall should be found / ground truth."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.files_read = ["a.py", "b.py"]
        session.ground_truth_files = ["a.py", "c.py", "d.py"]
        
        metrics = IRImpactMetrics.from_session(session)
        
        # Only a.py from ground truth was found
        assert metrics.retrieval_accuracy.files_found_recall == pytest.approx(1/3, rel=0.01)
    
    def test_f1_calculation(self):
        """F1 should be harmonic mean of precision and recall."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.files_read = ["a.py", "b.py"]
        session.files_relevant = ["a.py"]
        session.ground_truth_files = ["a.py", "c.py"]
        
        metrics = IRImpactMetrics.from_session(session)
        
        # Precision = 0.5, Recall = 0.5, F1 = 0.5
        assert metrics.retrieval_accuracy.files_found_f1 == 0.5
    
    def test_mrr_from_search_attempts(self):
        """MRR should be computed from first relevant rank."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.search_attempts = [
            SearchAttempt(
                attempt_number=1,
                search_type=SearchType.SEMANTIC,
                query="test",
                first_relevant_rank=3,
            )
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.retrieval_accuracy.mrr == pytest.approx(1/3, rel=0.01)
        assert metrics.retrieval_accuracy.first_relevant_file_rank == 3


class TestContextUtilizationMetrics:
    """Tests for context utilization computation."""
    
    def test_utilization_ratio(self):
        """Utilization ratio should be used / retrieved."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        
        # Record file reads
        session.record_event(IREvent(
            event_type=IREventType.FILE_READ,
            file_path="a.py",
            file_tokens=1000,
            file_was_relevant=True,
        ))
        session.record_event(IREvent(
            event_type=IREventType.FILE_READ,
            file_path="b.py",
            file_tokens=1000,
            file_was_relevant=False,
        ))
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.context_utilization.context_tokens_retrieved == 2000
        assert metrics.context_utilization.context_tokens_used == 1000
        assert metrics.context_utilization.context_utilization_ratio == 0.5
    
    def test_high_utilization_flag(self):
        """High utilization should be set when ratio > 0.5."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        
        session.record_event(IREvent(
            event_type=IREventType.FILE_READ,
            file_path="a.py",
            file_tokens=800,
            file_was_relevant=True,
        ))
        session.record_event(IREvent(
            event_type=IREventType.FILE_READ,
            file_path="b.py",
            file_tokens=200,
            file_was_relevant=False,
        ))
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.context_utilization.high_utilization is True


class TestNavigationEfficiencyMetrics:
    """Tests for navigation efficiency computation."""
    
    def test_time_ratio(self):
        """Time ratio should be nav / (nav + impl)."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.navigation_time_sec = 20.0  # 20% of total - efficient
        session.implementation_time_sec = 80.0
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.navigation_efficiency.navigation_time_ratio == 0.2
        assert metrics.navigation_efficiency.time_efficient is True  # < 0.25
    
    def test_token_ratio(self):
        """Token ratio should be nav / (nav + impl)."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.navigation_tokens = 300  # 30% of total - efficient
        session.implementation_tokens = 700
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.navigation_efficiency.navigation_token_ratio == 0.3
        assert metrics.navigation_efficiency.token_efficient is True  # < 0.4
    
    def test_search_iterations_count(self):
        """Search iterations should count attempts."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.search_attempts = [
            SearchAttempt(1, SearchType.SEMANTIC, "query 1", was_productive=False),
            SearchAttempt(2, SearchType.SEMANTIC, "query 2", was_productive=True),
            SearchAttempt(3, SearchType.SEMANTIC, "query 3", was_productive=True),
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.navigation_efficiency.search_iterations == 3
        assert metrics.navigation_efficiency.searches_before_success == 1


class TestFirstRetrievalMetrics:
    """Tests for first retrieval metrics computation."""
    
    def test_first_search_productive(self):
        """Should detect productive first search."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.search_attempts = [
            SearchAttempt(
                1, SearchType.DEEP_SEARCH, "query",
                was_productive=True,
                led_to_implementation=True,
            )
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.first_retrieval.first_search_productive is True
        assert metrics.first_retrieval.first_search_led_to_implementation is True
    
    def test_one_shot_success(self):
        """Should detect one-shot success."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.search_attempts = [
            SearchAttempt(
                1, SearchType.DEEP_SEARCH, "query",
                led_to_implementation=True,
            )
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.first_retrieval.one_shot_success is True
        assert metrics.first_retrieval.needed_refinement is False
    
    def test_refinement_detection(self):
        """Should detect query refinement."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="test",
            ir_tool_name="test",
        )
        session.search_attempts = [
            SearchAttempt(
                1, SearchType.SEMANTIC, "auth",
                results_count=10, relevant_results_count=1,
            ),
            SearchAttempt(
                2, SearchType.SEMANTIC, "authentication handler",
                results_count=5, relevant_results_count=4,
            ),
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert metrics.first_retrieval.needed_refinement is True
        assert metrics.first_retrieval.refinement_count == 1
        assert metrics.first_retrieval.refinement_improved is True


class TestIRImpactMetrics:
    """Tests for aggregate IR impact metrics."""
    
    def test_aggregate_scores(self):
        """Should compute aggregate scores."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        session.files_read = ["a.py", "b.py"]
        session.files_relevant = ["a.py"]
        session.ground_truth_files = ["a.py"]
        session.navigation_time_sec = 20.0
        session.implementation_time_sec = 80.0
        session.search_attempts = [
            SearchAttempt(1, SearchType.DEEP_SEARCH, "query", was_productive=True)
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        
        assert 0 <= metrics.retrieval_quality_score <= 1
        assert 0 <= metrics.efficiency_score <= 1
        assert 0 <= metrics.overall_ir_impact_score <= 1
    
    def test_outcome_tracking(self):
        """Should track actual outcome and prediction accuracy."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        session.files_read = ["a.py"]
        session.files_relevant = ["a.py"]
        session.ground_truth_files = ["a.py"]
        session.search_attempts = [
            SearchAttempt(1, SearchType.DEEP_SEARCH, "query", was_productive=True)
        ]
        
        metrics = IRImpactMetrics.from_session(session)
        metrics.set_actual_outcome(True)
        
        assert metrics.actual_success is True
        assert metrics.prediction_was_correct is not None
    
    def test_to_dict(self):
        """Should serialize to comprehensive dict."""
        session = IRTelemetrySession(
            task_id="test",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
        )
        
        metrics = IRImpactMetrics.from_session(session)
        d = metrics.to_dict()
        
        assert "task_id" in d
        assert "retrieval_accuracy" in d
        assert "context_utilization" in d
        assert "navigation_efficiency" in d
        assert "first_retrieval" in d
        assert "aggregate" in d
        assert "outcome" in d


class TestIRTelemetryCollector:
    """Tests for the telemetry collector."""
    
    def test_session_lifecycle(self):
        """Collector should manage session lifecycle."""
        collector = IRTelemetryCollector("TestAgent", "deep_search")
        
        session = collector.start_session("task-001")
        assert collector.active_session is session
        
        session.files_read = ["a.py"]
        metrics = collector.end_session(total_tokens=1000, total_time_sec=60.0)
        
        assert collector.active_session is None
        assert len(collector.sessions) == 1
        assert len(collector.metrics) == 1
        assert isinstance(metrics, IRImpactMetrics)
    
    def test_aggregate_metrics(self):
        """Collector should aggregate metrics across sessions."""
        collector = IRTelemetryCollector("TestAgent", "deep_search")
        
        for i in range(5):
            session = collector.start_session(f"task-{i}")
            session.files_read = ["a.py", "b.py"]
            session.files_relevant = ["a.py"] if i % 2 == 0 else ["a.py", "b.py"]
            session.ground_truth_files = ["a.py"]
            session.navigation_time_sec = 20.0 + i * 5
            session.implementation_time_sec = 80.0
            collector.end_session(success=(i % 2 == 0))
        
        agg = collector.aggregate_metrics()
        
        assert agg["total_sessions"] == 5
        assert "avg_files_found_f1" in agg
        assert "avg_context_utilization" in agg
        assert "avg_navigation_time_ratio" in agg
        assert "actual_success_rate" in agg
    
    def test_correlation_with_success(self):
        """Collector should compute correlation with success."""
        collector = IRTelemetryCollector("TestAgent", "deep_search")
        
        # Create sessions with varying metrics and outcomes
        for i in range(10):
            session = collector.start_session(f"task-{i}")
            
            # High performing sessions
            if i < 5:
                session.files_read = ["a.py"]
                session.files_relevant = ["a.py"]
                session.ground_truth_files = ["a.py"]
                session.search_attempts = [
                    SearchAttempt(1, SearchType.DEEP_SEARCH, "q", was_productive=True)
                ]
                collector.end_session(success=True)
            else:
                # Low performing sessions
                session.files_read = ["b.py", "c.py"]
                session.files_relevant = []
                session.ground_truth_files = ["a.py"]
                session.search_attempts = [
                    SearchAttempt(1, SearchType.SEMANTIC, "q", was_productive=False)
                ]
                collector.end_session(success=False)
        
        corr = collector.get_correlation_with_success()
        
        # Expect positive correlation for retrieval quality
        assert "retrieval_quality_score" in corr
        assert corr["retrieval_quality_score"] > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_collector_for_agent(self):
        """Should create configured collector."""
        collector = create_collector_for_agent("MyAgent", "sourcegraph")
        
        assert collector.agent_name == "MyAgent"
        assert collector.ir_tool_name == "sourcegraph"
    
    def test_compute_ir_impact_from_trace(self):
        """Should compute metrics from trace data."""
        metrics = compute_ir_impact_from_trace(
            task_id="test-001",
            agent_name="TestAgent",
            ir_tool_name="deep_search",
            files_read=["a.py", "b.py", "c.py"],
            files_relevant=["a.py", "b.py"],
            ground_truth_files=["a.py", "d.py"],
            navigation_time_sec=30.0,
            implementation_time_sec=90.0,
            navigation_tokens=400,
            implementation_tokens=800,
            search_attempts=[
                SearchAttempt(1, SearchType.DEEP_SEARCH, "query", was_productive=True)
            ],
            success=True,
        )
        
        assert metrics.task_id == "test-001"
        assert metrics.retrieval_accuracy.files_found_precision == pytest.approx(2/3, rel=0.01)
        assert metrics.actual_success is True
