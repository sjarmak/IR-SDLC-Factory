"""
Tests for Observability Module.

These tests validate the observability module for:
1. Token extraction from LLM responses (Claude, OpenAI, Gemini)
2. Tool usage tracking and statistics
3. Timing metrics and phase tracking
4. Cost calculation with model pricing
5. Run manifest generation and serialization
6. ExecutionTracer integration
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from app.ir_sdlc.observability import (
    # Token extraction
    TokenUsage,
    extract_tokens_from_claude_response,
    extract_tokens_from_openai_response,
    extract_tokens_from_gemini_response,
    extract_tokens_from_response,
    
    # Pricing
    ModelPricing,
    get_model_pricing,
    MODEL_PRICING,
    
    # Tool tracking
    ToolCategory,
    ToolInvocation,
    ToolUsageStats,
    categorize_tool,
    summarize_tool_usage,
    
    # Timing
    TimingMetrics,
    ExecutionTimer,
    
    # Cost
    CostBreakdown,
    calculate_cost,
    
    # Manifest
    RunManifest,
    
    # Tracer
    ExecutionTracer,
    create_tracer,
    compare_runs,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def claude_response() -> dict:
    """Sample Claude API response."""
    return {
        "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet-20241022",
        "usage": {
            "input_tokens": 1500,
            "output_tokens": 250,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
        },
    }


@pytest.fixture
def openai_response() -> dict:
    """Sample OpenAI API response."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "prompt_tokens_details": {
                "cached_tokens": 400,
            },
        },
    }


@pytest.fixture
def gemini_response() -> dict:
    """Sample Gemini API response."""
    return {
        "candidates": [{"content": {"parts": [{"text": "Hello!"}]}}],
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 200,
            "cachedContentTokenCount": 300,
            "totalTokenCount": 1200,
        },
    }


# =============================================================================
# Token Extraction Tests
# =============================================================================

class TestTokenUsage:
    """Tests for TokenUsage dataclass."""
    
    def test_total_tokens(self):
        """Test total token calculation."""
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150
    
    def test_effective_input_tokens(self):
        """Test effective input token calculation with cache."""
        usage = TokenUsage(input_tokens=100, cache_read_tokens=30)
        assert usage.effective_input_tokens == 70
    
    def test_addition(self):
        """Test adding two TokenUsage objects."""
        usage1 = TokenUsage(input_tokens=100, output_tokens=50)
        usage2 = TokenUsage(input_tokens=200, output_tokens=100)
        
        combined = usage1 + usage2
        
        assert combined.input_tokens == 300
        assert combined.output_tokens == 150
        assert combined.total_tokens == 450
    
    def test_to_dict(self):
        """Test serialization."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
        )
        
        result = usage.to_dict()
        
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["total_tokens"] == 150


class TestTokenExtraction:
    """Tests for token extraction from LLM responses."""
    
    def test_extract_from_claude(self, claude_response: dict):
        """Test extracting tokens from Claude response."""
        usage = extract_tokens_from_claude_response(claude_response)
        
        assert usage.input_tokens == 1500
        assert usage.output_tokens == 250
        assert usage.cache_read_tokens == 500
        assert usage.cache_write_tokens == 200
    
    def test_extract_from_openai(self, openai_response: dict):
        """Test extracting tokens from OpenAI response."""
        usage = extract_tokens_from_openai_response(openai_response)
        
        assert usage.input_tokens == 1200
        assert usage.output_tokens == 300
        assert usage.cache_read_tokens == 400
    
    def test_extract_from_gemini(self, gemini_response: dict):
        """Test extracting tokens from Gemini response."""
        usage = extract_tokens_from_gemini_response(gemini_response)
        
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 200
        assert usage.cache_read_tokens == 300
    
    def test_auto_detect_claude(self, claude_response: dict):
        """Test auto-detection of Claude response."""
        usage = extract_tokens_from_response(claude_response, "auto")
        assert usage.input_tokens == 1500
    
    def test_auto_detect_openai(self, openai_response: dict):
        """Test auto-detection of OpenAI response."""
        usage = extract_tokens_from_response(openai_response, "auto")
        assert usage.input_tokens == 1200
    
    def test_auto_detect_gemini(self, gemini_response: dict):
        """Test auto-detection of Gemini response."""
        usage = extract_tokens_from_response(gemini_response, "auto")
        assert usage.input_tokens == 1000
    
    def test_empty_response(self):
        """Test handling empty response."""
        usage = extract_tokens_from_response({}, "auto")
        assert usage.total_tokens == 0


# =============================================================================
# Model Pricing Tests
# =============================================================================

class TestModelPricing:
    """Tests for model pricing."""
    
    def test_known_model_pricing(self):
        """Test getting pricing for known models."""
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        
        assert pricing is not None
        assert pricing.input_price_per_1k == 0.003
        assert pricing.output_price_per_1k == 0.015
        assert pricing.provider == "anthropic"
    
    def test_calculate_cost(self):
        """Test cost calculation."""
        pricing = ModelPricing("test-model", 0.01, 0.03, "test")
        
        cost = pricing.calculate_cost(input_tokens=1000, output_tokens=500)
        
        # 1000 input @ $0.01/1k = $0.01
        # 500 output @ $0.03/1k = $0.015
        expected = 0.01 + 0.015
        assert abs(cost - expected) < 0.0001
    
    def test_unknown_model(self):
        """Test handling unknown model."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None
    
    def test_partial_match(self):
        """Test partial model name matching."""
        # Should match claude-3-5-sonnet-20241022
        pricing = get_model_pricing("claude-3-5-sonnet")
        assert pricing is not None


# =============================================================================
# Tool Tracking Tests
# =============================================================================

class TestToolTracking:
    """Tests for tool usage tracking."""
    
    def test_categorize_tool(self):
        """Test tool categorization."""
        assert categorize_tool("grep_search") == ToolCategory.SEARCH
        assert categorize_tool("read_file") == ToolCategory.FILE_READ
        assert categorize_tool("create_file") == ToolCategory.FILE_WRITE
        assert categorize_tool("run_in_terminal") == ToolCategory.TERMINAL
        assert categorize_tool("unknown_tool") == ToolCategory.OTHER
    
    def test_tool_invocation(self):
        """Test ToolInvocation creation."""
        invocation = ToolInvocation(
            tool_name="read_file",
            duration_ms=50.0,
            success=True,
        )
        
        assert invocation.tool_name == "read_file"
        assert invocation.category == ToolCategory.FILE_READ
        assert invocation.success is True
    
    def test_tool_usage_stats(self):
        """Test aggregating tool usage stats."""
        stats = ToolUsageStats(tool_name="read_file")
        
        inv1 = ToolInvocation(tool_name="read_file", duration_ms=10.0, success=True)
        inv2 = ToolInvocation(tool_name="read_file", duration_ms=20.0, success=True)
        inv3 = ToolInvocation(tool_name="read_file", duration_ms=30.0, success=False)
        
        stats.update(inv1)
        stats.update(inv2)
        stats.update(inv3)
        
        assert stats.call_count == 3
        assert stats.total_duration_ms == 60.0
        assert stats.avg_duration_ms == 20.0
        assert stats.min_duration_ms == 10.0
        assert stats.max_duration_ms == 30.0
        assert stats.success_count == 2
        assert stats.error_count == 1
        assert abs(stats.success_rate - 0.6667) < 0.01
    
    def test_summarize_tool_usage(self):
        """Test summarizing tool usage from invocations."""
        invocations = [
            ToolInvocation(tool_name="read_file", duration_ms=10.0, success=True),
            ToolInvocation(tool_name="read_file", duration_ms=20.0, success=True),
            ToolInvocation(tool_name="grep_search", duration_ms=50.0, success=True),
        ]
        
        summary = summarize_tool_usage(invocations)
        
        assert summary["total_calls"] == 3
        assert summary["total_duration_ms"] == 80.0
        assert summary["success_rate"] == 1.0
        assert "read_file" in summary["by_tool"]
        assert "grep_search" in summary["by_tool"]


# =============================================================================
# Timing Tests
# =============================================================================

class TestTiming:
    """Tests for timing metrics."""
    
    def test_execution_timer(self):
        """Test basic timing."""
        timer = ExecutionTimer()
        timer.start()
        
        timer.start_phase("test_phase")
        time.sleep(0.01)  # 10ms
        timer.end_phase()
        
        total = timer.stop()
        
        assert total >= 0.01
        metrics = timer.get_metrics()
        assert metrics.total_time_sec >= 0.01
    
    def test_phase_context_manager(self):
        """Test phase timing with context manager."""
        timer = ExecutionTimer()
        timer.start()
        
        with timer.phase("retrieval"):
            time.sleep(0.01)
        
        with timer.phase("reasoning"):
            time.sleep(0.01)
        
        timer.stop()
        metrics = timer.get_metrics()
        
        assert metrics.retrieval_time_sec >= 0.01
        assert metrics.reasoning_time_sec >= 0.01
    
    def test_timing_metrics_serialization(self):
        """Test TimingMetrics serialization."""
        metrics = TimingMetrics(
            total_time_sec=10.5,
            query_time_sec=2.0,
            retrieval_time_sec=3.0,
        )
        
        result = metrics.to_dict()
        
        assert result["total_time_sec"] == 10.5
        assert result["query_time_sec"] == 2.0


# =============================================================================
# Cost Calculation Tests
# =============================================================================

class TestCostCalculation:
    """Tests for cost calculation."""
    
    def test_calculate_cost_with_known_model(self):
        """Test cost calculation with known model pricing."""
        usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        
        breakdown = calculate_cost(usage, "claude-3-5-sonnet-20241022")
        
        # 10k input @ $0.003/1k = $0.03
        # 5k output @ $0.015/1k = $0.075
        assert abs(breakdown.input_token_cost - 0.03) < 0.001
        assert abs(breakdown.output_token_cost - 0.075) < 0.001
        assert abs(breakdown.total_token_cost - 0.105) < 0.001
    
    def test_calculate_cost_with_caching(self):
        """Test cost calculation with cached tokens."""
        usage = TokenUsage(
            input_tokens=10000,
            output_tokens=5000,
            cache_read_tokens=3000,
        )
        
        breakdown = calculate_cost(usage, "claude-3-5-sonnet-20241022")
        
        # Cache savings: 3k tokens @ $0.003/1k * 0.9 = $0.0081
        assert breakdown.cache_savings > 0
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model."""
        usage = TokenUsage(input_tokens=1000, output_tokens=500)
        
        breakdown = calculate_cost(usage, "unknown-model")
        
        # Should return zero costs for unknown model
        assert breakdown.total_cost == 0.0
    
    def test_cost_breakdown_serialization(self):
        """Test CostBreakdown serialization."""
        breakdown = CostBreakdown(
            model_id="test-model",
            input_token_cost=0.01,
            output_token_cost=0.02,
            total_token_cost=0.03,
            total_cost=0.03,
        )
        
        result = breakdown.to_dict()
        
        assert result["model_id"] == "test-model"
        assert result["total_cost"] == 0.03


# =============================================================================
# Run Manifest Tests
# =============================================================================

class TestRunManifest:
    """Tests for RunManifest."""
    
    def test_generate_run_id(self):
        """Test run ID generation."""
        run_id = RunManifest.generate_run_id()
        
        assert run_id.startswith("run-")
        assert len(run_id) > 20
    
    def test_manifest_creation(self):
        """Test creating a run manifest."""
        manifest = RunManifest(
            run_id="run-test-123",
            run_name="Test Run",
            model_id="claude-3-5-sonnet-20241022",
            agent_name="test-agent",
            tasks_completed=8,
            tasks_failed=2,
            tasks_total=10,
        )
        
        assert manifest.run_id == "run-test-123"
        assert manifest.tasks_completed == 8
    
    def test_manifest_serialization(self):
        """Test manifest serialization to dict."""
        manifest = RunManifest(
            run_id="run-test-123",
            run_name="Test Run",
            model_id="test-model",
            total_token_usage=TokenUsage(input_tokens=1000, output_tokens=500),
        )
        
        result = manifest.to_dict()
        
        assert result["run_id"] == "run-test-123"
        assert result["token_usage"]["input_tokens"] == 1000
    
    def test_manifest_save_and_load(self):
        """Test saving and loading manifest."""
        manifest = RunManifest(
            run_id="run-test-123",
            run_name="Test Run",
            model_id="test-model",
            total_token_usage=TokenUsage(input_tokens=1000, output_tokens=500),
            tasks_total=10,
            tasks_completed=8,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest.save(path)
            
            assert path.exists()
            
            loaded = RunManifest.load(path)
            
            assert loaded.run_id == "run-test-123"
            assert loaded.total_token_usage.input_tokens == 1000
            assert loaded.tasks_total == 10


# =============================================================================
# ExecutionTracer Tests
# =============================================================================

class TestExecutionTracer:
    """Tests for ExecutionTracer."""
    
    def test_tracer_initialization(self):
        """Test tracer initialization."""
        tracer = ExecutionTracer(
            model_id="claude-3-5-sonnet-20241022",
            agent_name="test-agent",
            benchmark_name="ir-sdlc-bench",
        )
        
        assert tracer.model_id == "claude-3-5-sonnet-20241022"
        assert tracer.agent_name == "test-agent"
    
    def test_start_and_end_run(self):
        """Test starting and ending a run."""
        tracer = create_tracer(model_id="test-model")
        
        run_id = tracer.start_run("Test Run")
        
        assert run_id.startswith("run-")
        
        tracer.end_run()
    
    def test_record_llm_response(self, claude_response: dict):
        """Test recording LLM response."""
        tracer = create_tracer(model_id="claude-3-5-sonnet-20241022")
        tracer.start_run()
        
        usage = tracer.record_llm_response(claude_response)
        
        assert usage.input_tokens == 1500
        
        # Check accumulated usage
        summary = tracer.get_summary()
        assert summary["token_usage"]["input_tokens"] == 1500
    
    def test_record_tool_call(self):
        """Test recording tool calls."""
        tracer = create_tracer(model_id="test-model")
        tracer.start_run()
        
        tracer.record_tool_call(
            tool_name="read_file",
            input_params={"path": "test.py"},
            duration_ms=25.0,
            success=True,
        )
        
        tracer.record_tool_call(
            tool_name="grep_search",
            input_params={"query": "foo"},
            duration_ms=50.0,
            success=True,
        )
        
        stats = tracer.get_tool_stats()
        
        assert "read_file" in stats
        assert "grep_search" in stats
        assert stats["read_file"].call_count == 1
    
    def test_trace_tool_call_context_manager(self):
        """Test tracing tool call with context manager."""
        tracer = create_tracer(model_id="test-model")
        tracer.start_run()
        
        with tracer.trace_tool_call("read_file", {"path": "test.py"}):
            time.sleep(0.01)  # Simulate work
        
        stats = tracer.get_tool_stats()
        assert stats["read_file"].call_count == 1
        assert stats["read_file"].total_duration_ms >= 10.0
    
    def test_tool_frequency(self):
        """Test getting tool call frequency."""
        tracer = create_tracer(model_id="test-model")
        tracer.start_run()
        
        tracer.record_tool_call("read_file", duration_ms=10.0)
        tracer.record_tool_call("read_file", duration_ms=10.0)
        tracer.record_tool_call("grep_search", duration_ms=20.0)
        
        freq = tracer.get_tool_frequency()
        
        assert freq["read_file"] == 2
        assert freq["grep_search"] == 1
    
    def test_category_stats(self):
        """Test getting category statistics."""
        tracer = create_tracer(model_id="test-model")
        tracer.start_run()
        
        tracer.record_tool_call("read_file", duration_ms=10.0)
        tracer.record_tool_call("list_dir", duration_ms=10.0)
        tracer.record_tool_call("grep_search", duration_ms=20.0)
        
        categories = tracer.get_category_stats()
        
        assert categories["file_read"] == 2
        assert categories["search"] == 1
    
    def test_iteration_tracking(self):
        """Test iteration tracking."""
        tracer = create_tracer(model_id="test-model")
        tracer.start_run()
        
        # First iteration
        iter1 = tracer.new_iteration()
        tracer.record_tool_call("read_file", duration_ms=10.0)
        tracer.record_tool_call("grep_search", duration_ms=20.0)
        
        # Second iteration
        iter2 = tracer.new_iteration()
        tracer.record_tool_call("create_file", duration_ms=30.0)
        
        assert iter1 == 1
        assert iter2 == 2
    
    def test_get_manifest(self, claude_response: dict):
        """Test generating run manifest."""
        tracer = create_tracer(
            model_id="claude-3-5-sonnet-20241022",
            agent_name="test-agent",
            benchmark_name="ir-sdlc-bench",
        )
        
        tracer.start_run("Test Run")
        tracer.record_llm_response(claude_response)
        tracer.record_tool_call("read_file", duration_ms=25.0)
        tracer.end_run()
        
        manifest = tracer.get_manifest(
            tasks_completed=8,
            tasks_failed=2,
            tasks_total=10,
        )
        
        assert manifest.run_name == "Test Run"
        assert manifest.model_id == "claude-3-5-sonnet-20241022"
        assert manifest.total_token_usage.input_tokens == 1500
        assert manifest.tasks_completed == 8
        assert manifest.cost_breakdown is not None
        assert manifest.cost_breakdown.total_cost > 0
    
    def test_save_manifest(self):
        """Test saving manifest directly from tracer."""
        tracer = create_tracer(model_id="claude-3-5-sonnet-20241022")
        tracer.start_run()
        tracer.record_tool_call("read_file", duration_ms=10.0)
        tracer.end_run()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tracer.save_manifest(
                Path(tmpdir) / "manifest.json",
                tasks_total=5,
            )
            
            assert path.exists()
            
            with open(path) as f:
                data = json.load(f)
            
            assert data["tasks_total"] == 5
    
    def test_get_summary(self, claude_response: dict):
        """Test getting execution summary."""
        tracer = create_tracer(model_id="claude-3-5-sonnet-20241022")
        tracer.start_run()
        tracer.record_llm_response(claude_response)
        tracer.record_tool_call("read_file", duration_ms=10.0)
        
        summary = tracer.get_summary()
        
        assert summary["token_usage"]["input_tokens"] == 1500
        assert summary["tool_calls"] == 1
        assert summary["estimated_cost"] > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_execution_tracing_workflow(self, claude_response: dict):
        """Test complete execution tracing workflow."""
        # Create tracer
        tracer = ExecutionTracer(
            model_id="claude-3-5-sonnet-20241022",
            agent_name="test-agent",
            ir_tool_name="sourcegraph",
            benchmark_name="ir-sdlc-bench",
            benchmark_version="1.0.0",
        )
        
        # Start run
        run_id = tracer.start_run("Full Integration Test")
        
        # Simulate agent execution
        tracer.new_iteration()
        
        # Phase 1: Retrieval
        with tracer.timer.phase("retrieval"):
            tracer.record_tool_call("grep_search", {"query": "test"}, duration_ms=100.0)
            tracer.record_tool_call("read_file", {"path": "test.py"}, duration_ms=50.0)
        
        # Record LLM response
        tracer.record_llm_response(claude_response)
        
        # Phase 2: Reasoning
        with tracer.timer.phase("reasoning"):
            time.sleep(0.01)
        
        # Second iteration
        tracer.new_iteration()
        tracer.record_tool_call("create_file", {"path": "output.py"}, duration_ms=30.0)
        tracer.record_llm_response(claude_response)  # Second response
        
        # End run
        tracer.end_run()
        
        # Generate and verify manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = tracer.save_manifest(
                Path(tmpdir) / "manifest.json",
                tasks_completed=1,
                tasks_failed=0,
                tasks_total=1,
            )
            
            # Load and verify
            loaded = RunManifest.load(manifest_path)
            
            assert loaded.run_name == "Full Integration Test"
            assert loaded.model_id == "claude-3-5-sonnet-20241022"
            assert loaded.total_token_usage.input_tokens == 3000  # 2 responses
            assert loaded.tasks_completed == 1
            assert loaded.cost_breakdown.total_cost > 0
    
    def test_compare_runs(self):
        """Test comparing multiple runs."""
        manifests = [
            RunManifest(
                run_id="run-1",
                model_id="claude-3-5-sonnet-20241022",
                total_token_usage=TokenUsage(input_tokens=1000, output_tokens=500),
                timing_metrics=TimingMetrics(total_time_sec=10.0),
                tasks_completed=8,
                tasks_total=10,
                cost_breakdown=CostBreakdown(model_id="claude-3-5-sonnet-20241022", total_cost=0.05),
            ),
            RunManifest(
                run_id="run-2",
                model_id="claude-3-5-sonnet-20241022",
                total_token_usage=TokenUsage(input_tokens=1500, output_tokens=600),
                timing_metrics=TimingMetrics(total_time_sec=15.0),
                tasks_completed=9,
                tasks_total=10,
                cost_breakdown=CostBreakdown(model_id="claude-3-5-sonnet-20241022", total_cost=0.07),
            ),
        ]
        
        comparison = compare_runs(manifests)
        
        assert comparison["run_count"] == 2
        assert comparison["token_usage"]["mean"] == 1800  # (1500 + 2100) / 2
        assert comparison["timing"]["mean"] == 12.5
        assert comparison["success_rate"] == 0.85  # 17/20
