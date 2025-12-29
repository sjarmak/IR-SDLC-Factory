"""
Observability Module for IR-SDLC-Bench Execution Tracing.

This module provides comprehensive observability for agent execution,
capturing metrics necessary for analyzing benchmark runs and costs.

Key capabilities:
1. Token count extraction from LLM outputs (Claude, GPT, etc.)
2. Tool usage tracking (MCP tools called, frequency, latency)
3. Timing metrics (query time, retrieval time, total time)
4. Cost calculation based on model pricing
5. Run manifest generation for reproducibility

Reference: CodeContextBench observability/ module pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Iterator
from contextlib import contextmanager
import threading


# =============================================================================
# Model Pricing Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelPricing:
    """Pricing information for a model."""
    
    model_id: str
    input_price_per_1k: float  # $ per 1000 input tokens
    output_price_per_1k: float  # $ per 1000 output tokens
    provider: str = "unknown"
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.input_price_per_1k
        output_cost = (output_tokens / 1000) * self.output_price_per_1k
        return input_cost + output_cost


# Common model pricing (as of December 2024)
MODEL_PRICING = {
    # Claude models
    "claude-3-opus-20240229": ModelPricing("claude-3-opus-20240229", 0.015, 0.075, "anthropic"),
    "claude-3-5-sonnet-20241022": ModelPricing("claude-3-5-sonnet-20241022", 0.003, 0.015, "anthropic"),
    "claude-3-5-haiku-20241022": ModelPricing("claude-3-5-haiku-20241022", 0.001, 0.005, "anthropic"),
    "claude-sonnet-4-20250514": ModelPricing("claude-sonnet-4-20250514", 0.003, 0.015, "anthropic"),
    
    # GPT models
    "gpt-4o": ModelPricing("gpt-4o", 0.0025, 0.01, "openai"),
    "gpt-4o-mini": ModelPricing("gpt-4o-mini", 0.00015, 0.0006, "openai"),
    "gpt-4-turbo": ModelPricing("gpt-4-turbo", 0.01, 0.03, "openai"),
    "o1": ModelPricing("o1", 0.015, 0.06, "openai"),
    "o1-mini": ModelPricing("o1-mini", 0.003, 0.012, "openai"),
    
    # Gemini models
    "gemini-1.5-pro": ModelPricing("gemini-1.5-pro", 0.00125, 0.005, "google"),
    "gemini-1.5-flash": ModelPricing("gemini-1.5-flash", 0.000075, 0.0003, "google"),
    "gemini-2.0-flash": ModelPricing("gemini-2.0-flash", 0.0001, 0.0004, "google"),
}


def get_model_pricing(model_id: str) -> Optional[ModelPricing]:
    """Get pricing for a model by ID."""
    # Try exact match first
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]
    
    # Try partial match (for model versions)
    for key, pricing in MODEL_PRICING.items():
        if key in model_id or model_id in key:
            return pricing
    
    return None


# =============================================================================
# Token Extraction
# =============================================================================

@dataclass
class TokenUsage:
    """Token usage from an LLM response."""
    
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def effective_input_tokens(self) -> int:
        """Input tokens minus cached reads (which are often cheaper)."""
        return self.input_tokens - self.cache_read_tokens
    
    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }
    
    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


def extract_tokens_from_claude_response(response: dict) -> TokenUsage:
    """Extract token counts from a Claude API response."""
    usage = TokenUsage()
    
    if "usage" in response:
        raw_usage = response["usage"]
        usage.input_tokens = raw_usage.get("input_tokens", 0)
        usage.output_tokens = raw_usage.get("output_tokens", 0)
        usage.cache_read_tokens = raw_usage.get("cache_read_input_tokens", 0)
        usage.cache_write_tokens = raw_usage.get("cache_creation_input_tokens", 0)
    
    return usage


def extract_tokens_from_openai_response(response: dict) -> TokenUsage:
    """Extract token counts from an OpenAI API response."""
    usage = TokenUsage()
    
    if "usage" in response:
        raw_usage = response["usage"]
        usage.input_tokens = raw_usage.get("prompt_tokens", 0)
        usage.output_tokens = raw_usage.get("completion_tokens", 0)
        # OpenAI caching info in a different location
        if "prompt_tokens_details" in raw_usage:
            details = raw_usage["prompt_tokens_details"]
            usage.cache_read_tokens = details.get("cached_tokens", 0)
    
    return usage


def extract_tokens_from_gemini_response(response: dict) -> TokenUsage:
    """Extract token counts from a Gemini API response."""
    usage = TokenUsage()
    
    if "usageMetadata" in response:
        raw_usage = response["usageMetadata"]
        usage.input_tokens = raw_usage.get("promptTokenCount", 0)
        usage.output_tokens = raw_usage.get("candidatesTokenCount", 0)
        usage.cache_read_tokens = raw_usage.get("cachedContentTokenCount", 0)
    
    return usage


def extract_tokens_from_response(response: dict, provider: str = "auto") -> TokenUsage:
    """Extract token counts from an LLM response, auto-detecting provider."""
    if provider == "auto":
        # Auto-detect based on response structure
        if "usage" in response and "input_tokens" in response.get("usage", {}):
            provider = "anthropic"
        elif "usage" in response and "prompt_tokens" in response.get("usage", {}):
            provider = "openai"
        elif "usageMetadata" in response:
            provider = "google"
        else:
            # Default to empty usage
            return TokenUsage()
    
    extractors = {
        "anthropic": extract_tokens_from_claude_response,
        "claude": extract_tokens_from_claude_response,
        "openai": extract_tokens_from_openai_response,
        "gpt": extract_tokens_from_openai_response,
        "google": extract_tokens_from_gemini_response,
        "gemini": extract_tokens_from_gemini_response,
    }
    
    extractor = extractors.get(provider.lower())
    if extractor:
        return extractor(response)
    
    return TokenUsage()


# =============================================================================
# Tool Usage Tracking
# =============================================================================

class ToolCategory(str, Enum):
    """Categories of MCP tools."""
    SEARCH = "search"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    TERMINAL = "terminal"
    BROWSER = "browser"
    OTHER = "other"


@dataclass
class ToolInvocation:
    """Record of a single tool invocation."""
    
    tool_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Timing
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    duration_ms: float = 0.0
    
    # Categorization
    category: ToolCategory = field(default=None)  # type: ignore
    
    # Input/Output
    input_params: dict = field(default_factory=dict)
    output_size_bytes: int = 0
    success: bool = True
    error_message: Optional[str] = None
    
    # Context
    iteration: int = 0  # Which iteration of agent loop
    sequence_number: int = 0  # Order within iteration
    
    def __post_init__(self):
        """Auto-categorize tool based on name if not provided."""
        if self.category is None:
            self.category = categorize_tool(self.tool_name)
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "category": self.category.value,
            "success": self.success,
            "error_message": self.error_message,
            "iteration": self.iteration,
            "sequence_number": self.sequence_number,
        }


@dataclass
class ToolUsageStats:
    """Aggregated statistics for tool usage."""
    
    tool_name: str
    call_count: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    max_duration_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    
    def update(self, invocation: ToolInvocation) -> None:
        """Update stats with a new invocation."""
        self.call_count += 1
        self.total_duration_ms += invocation.duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.call_count
        self.min_duration_ms = min(self.min_duration_ms, invocation.duration_ms)
        self.max_duration_ms = max(self.max_duration_ms, invocation.duration_ms)
        
        if invocation.success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.success_count / self.call_count
    
    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "call_count": self.call_count,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "min_duration_ms": self.min_duration_ms if self.min_duration_ms != float("inf") else 0.0,
            "max_duration_ms": self.max_duration_ms,
            "success_rate": self.success_rate,
        }


# Tool categorization helpers
TOOL_CATEGORIES = {
    # Search tools
    "grep_search": ToolCategory.SEARCH,
    "semantic_search": ToolCategory.SEARCH,
    "file_search": ToolCategory.SEARCH,
    "list_code_usages": ToolCategory.SEARCH,
    "codebase_search": ToolCategory.SEARCH,
    
    # File read tools
    "read_file": ToolCategory.FILE_READ,
    "list_dir": ToolCategory.FILE_READ,
    "get_file_contents": ToolCategory.FILE_READ,
    
    # File write tools
    "create_file": ToolCategory.FILE_WRITE,
    "replace_string_in_file": ToolCategory.FILE_WRITE,
    "edit_file": ToolCategory.FILE_WRITE,
    
    # Terminal tools
    "run_in_terminal": ToolCategory.TERMINAL,
    "run_command": ToolCategory.TERMINAL,
    
    # Browser tools
    "fetch_webpage": ToolCategory.BROWSER,
}


def categorize_tool(tool_name: str) -> ToolCategory:
    """Categorize a tool by name."""
    return TOOL_CATEGORIES.get(tool_name, ToolCategory.OTHER)


# =============================================================================
# Timing Metrics
# =============================================================================

@dataclass
class TimingMetrics:
    """Timing metrics for execution phases."""
    
    # Wall clock times (seconds)
    total_time_sec: float = 0.0
    query_time_sec: float = 0.0
    retrieval_time_sec: float = 0.0
    reasoning_time_sec: float = 0.0
    tool_execution_time_sec: float = 0.0
    
    # Breakdowns
    first_token_time_sec: float = 0.0  # Time to first LLM token
    streaming_time_sec: float = 0.0  # Time for streaming completion
    
    # Iteration counts
    total_iterations: int = 0
    avg_iteration_time_sec: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class ExecutionTimer:
    """Context manager for timing execution phases."""
    
    def __init__(self):
        self._phase_times: dict[str, float] = {}
        self._current_phase: Optional[str] = None
        self._phase_start: float = 0.0
        self._total_start: float = 0.0
    
    def start(self) -> None:
        """Start total timing."""
        self._total_start = time.time()
    
    def stop(self) -> float:
        """Stop total timing and return elapsed time."""
        if self._current_phase:
            self.end_phase()
        return time.time() - self._total_start
    
    def start_phase(self, phase_name: str) -> None:
        """Start timing a phase."""
        if self._current_phase:
            self.end_phase()
        self._current_phase = phase_name
        self._phase_start = time.time()
    
    def end_phase(self) -> float:
        """End current phase and return duration."""
        if not self._current_phase:
            return 0.0
        
        duration = time.time() - self._phase_start
        self._phase_times[self._current_phase] = (
            self._phase_times.get(self._current_phase, 0.0) + duration
        )
        self._current_phase = None
        return duration
    
    @contextmanager
    def phase(self, phase_name: str):
        """Context manager for timing a phase."""
        self.start_phase(phase_name)
        try:
            yield
        finally:
            self.end_phase()
    
    def get_metrics(self) -> TimingMetrics:
        """Get timing metrics."""
        total = time.time() - self._total_start if self._total_start else 0.0
        
        return TimingMetrics(
            total_time_sec=total,
            query_time_sec=self._phase_times.get("query", 0.0),
            retrieval_time_sec=self._phase_times.get("retrieval", 0.0),
            reasoning_time_sec=self._phase_times.get("reasoning", 0.0),
            tool_execution_time_sec=self._phase_times.get("tool_execution", 0.0),
            first_token_time_sec=self._phase_times.get("first_token", 0.0),
            streaming_time_sec=self._phase_times.get("streaming", 0.0),
        )


# =============================================================================
# Cost Calculator
# =============================================================================

@dataclass
class CostBreakdown:
    """Breakdown of costs for an execution."""
    
    model_id: str
    
    # Token costs
    input_token_cost: float = 0.0
    output_token_cost: float = 0.0
    total_token_cost: float = 0.0
    
    # If using cached prompts (Anthropic)
    cache_savings: float = 0.0
    
    # Tool costs (if applicable)
    tool_api_costs: float = 0.0
    
    # Total
    total_cost: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "input_token_cost": round(self.input_token_cost, 6),
            "output_token_cost": round(self.output_token_cost, 6),
            "total_token_cost": round(self.total_token_cost, 6),
            "cache_savings": round(self.cache_savings, 6),
            "total_cost": round(self.total_cost, 6),
        }


def calculate_cost(
    token_usage: TokenUsage,
    model_id: str,
    pricing: Optional[ModelPricing] = None,
) -> CostBreakdown:
    """Calculate cost for token usage."""
    if pricing is None:
        pricing = get_model_pricing(model_id)
    
    breakdown = CostBreakdown(model_id=model_id)
    
    if pricing is None:
        return breakdown
    
    # Calculate base costs
    breakdown.input_token_cost = (token_usage.input_tokens / 1000) * pricing.input_price_per_1k
    breakdown.output_token_cost = (token_usage.output_tokens / 1000) * pricing.output_price_per_1k
    breakdown.total_token_cost = breakdown.input_token_cost + breakdown.output_token_cost
    
    # Calculate cache savings (cached reads are typically free or 90% cheaper)
    if token_usage.cache_read_tokens > 0:
        full_cost = (token_usage.cache_read_tokens / 1000) * pricing.input_price_per_1k
        # Most providers offer 90% discount on cached tokens
        cached_cost = full_cost * 0.1
        breakdown.cache_savings = full_cost - cached_cost
    
    breakdown.total_cost = breakdown.total_token_cost - breakdown.cache_savings
    
    return breakdown


# =============================================================================
# Run Manifest
# =============================================================================

@dataclass
class RunManifest:
    """
    Complete manifest for a benchmark run.
    
    Contains all information needed to reproduce and analyze a run.
    """
    
    # Run identification
    run_id: str
    run_name: str = ""
    
    # Timestamps
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    
    # Configuration
    model_id: str = ""
    agent_name: str = ""
    ir_tool_name: str = ""
    benchmark_name: str = ""
    benchmark_version: str = ""
    
    # Environment
    environment: dict = field(default_factory=dict)
    
    # Token usage
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    
    # Tool usage
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    tool_stats: dict[str, ToolUsageStats] = field(default_factory=dict)
    
    # Timing
    timing_metrics: TimingMetrics = field(default_factory=TimingMetrics)
    
    # Costs
    cost_breakdown: Optional[CostBreakdown] = None
    
    # Task results
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_total: int = 0
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    
    @staticmethod
    def generate_run_id() -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        random_suffix = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
        return f"run-{timestamp}-{random_suffix}"
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "model_id": self.model_id,
            "agent_name": self.agent_name,
            "ir_tool_name": self.ir_tool_name,
            "benchmark_name": self.benchmark_name,
            "benchmark_version": self.benchmark_version,
            "environment": self.environment,
            "token_usage": self.total_token_usage.to_dict(),
            "tool_stats": {k: v.to_dict() for k, v in self.tool_stats.items()},
            "timing": self.timing_metrics.to_dict(),
            "cost": self.cost_breakdown.to_dict() if self.cost_breakdown else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "tasks_total": self.tasks_total,
            "metadata": self.metadata,
        }
    
    def save(self, output_path: Path) -> Path:
        """Save manifest to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return output_path
    
    @classmethod
    def load(cls, input_path: Path) -> "RunManifest":
        """Load manifest from JSON file."""
        with open(input_path) as f:
            data = json.load(f)
        
        manifest = cls(
            run_id=data["run_id"],
            run_name=data.get("run_name", ""),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time"),
            model_id=data.get("model_id", ""),
            agent_name=data.get("agent_name", ""),
            ir_tool_name=data.get("ir_tool_name", ""),
            benchmark_name=data.get("benchmark_name", ""),
            benchmark_version=data.get("benchmark_version", ""),
            environment=data.get("environment", {}),
            tasks_completed=data.get("tasks_completed", 0),
            tasks_failed=data.get("tasks_failed", 0),
            tasks_total=data.get("tasks_total", 0),
            metadata=data.get("metadata", {}),
        )
        
        # Parse token usage
        if "token_usage" in data:
            tu = data["token_usage"]
            manifest.total_token_usage = TokenUsage(
                input_tokens=tu.get("input_tokens", 0),
                output_tokens=tu.get("output_tokens", 0),
                cache_read_tokens=tu.get("cache_read_tokens", 0),
                cache_write_tokens=tu.get("cache_write_tokens", 0),
            )
        
        # Parse timing
        if "timing" in data:
            tm = data["timing"]
            manifest.timing_metrics = TimingMetrics(**{
                k: v for k, v in tm.items()
                if k in TimingMetrics.__dataclass_fields__
            })
        
        # Parse cost breakdown
        if "cost" in data and data["cost"] is not None:
            cb = data["cost"]
            manifest.cost_breakdown = CostBreakdown(
                model_id=cb.get("model_id", ""),
                input_token_cost=cb.get("input_token_cost", 0.0),
                output_token_cost=cb.get("output_token_cost", 0.0),
                total_token_cost=cb.get("total_token_cost", 0.0),
                cache_savings=cb.get("cache_savings", 0.0),
                total_cost=cb.get("total_cost", 0.0),
            )
        
        return manifest


# =============================================================================
# Execution Tracer
# =============================================================================

class ExecutionTracer:
    """
    Main observability class for tracing agent execution.
    
    Usage:
        tracer = ExecutionTracer(
            model_id="claude-3-5-sonnet-20241022",
            agent_name="my-agent",
            benchmark_name="ir-sdlc-bench",
        )
        
        tracer.start_run("My Benchmark Run")
        
        with tracer.timer.phase("retrieval"):
            results = do_retrieval()
        
        tracer.record_llm_response(response)
        tracer.record_tool_call("read_file", {"path": "foo.py"}, success=True)
        
        tracer.end_run()
        tracer.save_manifest("outputs/run_manifest.json")
    """
    
    def __init__(
        self,
        model_id: str,
        agent_name: str = "",
        ir_tool_name: str = "",
        benchmark_name: str = "",
        benchmark_version: str = "1.0.0",
    ):
        self.model_id = model_id
        self.agent_name = agent_name
        self.ir_tool_name = ir_tool_name
        self.benchmark_name = benchmark_name
        self.benchmark_version = benchmark_version
        
        self.timer = ExecutionTimer()
        self._token_usage = TokenUsage()
        self._tool_invocations: list[ToolInvocation] = []
        self._tool_stats: dict[str, ToolUsageStats] = defaultdict(
            lambda: ToolUsageStats(tool_name="")
        )
        
        self._run_id: Optional[str] = None
        self._run_name: str = ""
        self._current_iteration: int = 0
        self._sequence_in_iteration: int = 0
        self._lock = threading.Lock()
    
    def start_run(self, run_name: str = "") -> str:
        """Start a new run and return the run ID."""
        self._run_id = RunManifest.generate_run_id()
        self._run_name = run_name
        self.timer.start()
        self._current_iteration = 0
        self._sequence_in_iteration = 0
        return self._run_id
    
    def end_run(self) -> None:
        """End the current run."""
        self.timer.stop()
    
    def new_iteration(self) -> int:
        """Start a new agent iteration and return the iteration number."""
        self._current_iteration += 1
        self._sequence_in_iteration = 0
        return self._current_iteration
    
    def record_llm_response(
        self,
        response: dict,
        provider: str = "auto",
    ) -> TokenUsage:
        """Record token usage from an LLM response."""
        usage = extract_tokens_from_response(response, provider)
        
        with self._lock:
            self._token_usage = self._token_usage + usage
        
        return usage
    
    def record_token_usage(self, usage: TokenUsage) -> None:
        """Record token usage directly."""
        with self._lock:
            self._token_usage = self._token_usage + usage
    
    def record_tool_call(
        self,
        tool_name: str,
        input_params: Optional[dict] = None,
        duration_ms: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
        output_size_bytes: int = 0,
    ) -> ToolInvocation:
        """Record a tool invocation."""
        with self._lock:
            self._sequence_in_iteration += 1
            
            invocation = ToolInvocation(
                tool_name=tool_name,
                input_params=input_params or {},
                duration_ms=duration_ms,
                success=success,
                error_message=error_message,
                output_size_bytes=output_size_bytes,
                category=categorize_tool(tool_name),
                iteration=self._current_iteration,
                sequence_number=self._sequence_in_iteration,
            )
            
            self._tool_invocations.append(invocation)
            
            # Update stats
            if tool_name not in self._tool_stats:
                self._tool_stats[tool_name] = ToolUsageStats(tool_name=tool_name)
            self._tool_stats[tool_name].update(invocation)
        
        return invocation
    
    @contextmanager
    def trace_tool_call(self, tool_name: str, input_params: Optional[dict] = None):
        """Context manager for tracing a tool call with timing."""
        start_time = time.time()
        success = True
        error_msg = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_tool_call(
                tool_name=tool_name,
                input_params=input_params,
                duration_ms=duration_ms,
                success=success,
                error_message=error_msg,
            )
    
    def get_tool_stats(self) -> dict[str, ToolUsageStats]:
        """Get aggregated tool usage statistics."""
        return dict(self._tool_stats)
    
    def get_tool_frequency(self) -> dict[str, int]:
        """Get tool call frequency."""
        return {name: stats.call_count for name, stats in self._tool_stats.items()}
    
    def get_category_stats(self) -> dict[str, int]:
        """Get tool usage by category."""
        category_counts: dict[str, int] = defaultdict(int)
        for inv in self._tool_invocations:
            category_counts[inv.category.value] += 1
        return dict(category_counts)
    
    def get_manifest(
        self,
        tasks_completed: int = 0,
        tasks_failed: int = 0,
        tasks_total: int = 0,
        metadata: Optional[dict] = None,
    ) -> RunManifest:
        """Generate the run manifest."""
        # Calculate cost
        cost = calculate_cost(self._token_usage, self.model_id)
        
        manifest = RunManifest(
            run_id=self._run_id or RunManifest.generate_run_id(),
            run_name=self._run_name,
            end_time=datetime.now(timezone.utc).isoformat(),
            model_id=self.model_id,
            agent_name=self.agent_name,
            ir_tool_name=self.ir_tool_name,
            benchmark_name=self.benchmark_name,
            benchmark_version=self.benchmark_version,
            total_token_usage=self._token_usage,
            tool_invocations=self._tool_invocations,
            tool_stats=dict(self._tool_stats),
            timing_metrics=self.timer.get_metrics(),
            cost_breakdown=cost,
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            tasks_total=tasks_total,
            metadata=metadata or {},
            environment={
                "python_version": os.popen("python --version 2>/dev/null").read().strip(),
                "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            },
        )
        
        return manifest
    
    def save_manifest(
        self,
        output_path: Path,
        tasks_completed: int = 0,
        tasks_failed: int = 0,
        tasks_total: int = 0,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Generate and save the run manifest."""
        manifest = self.get_manifest(
            tasks_completed=tasks_completed,
            tasks_failed=tasks_failed,
            tasks_total=tasks_total,
            metadata=metadata,
        )
        return manifest.save(output_path)
    
    def get_summary(self) -> dict:
        """Get a summary of the current execution state."""
        return {
            "run_id": self._run_id,
            "token_usage": self._token_usage.to_dict(),
            "tool_calls": len(self._tool_invocations),
            "tool_frequency": self.get_tool_frequency(),
            "timing": self.timer.get_metrics().to_dict(),
            "estimated_cost": calculate_cost(self._token_usage, self.model_id).total_cost,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tracer(
    model_id: str,
    agent_name: str = "",
    ir_tool_name: str = "",
    benchmark_name: str = "ir-sdlc-bench",
) -> ExecutionTracer:
    """Create and configure an execution tracer."""
    return ExecutionTracer(
        model_id=model_id,
        agent_name=agent_name,
        ir_tool_name=ir_tool_name,
        benchmark_name=benchmark_name,
    )


def summarize_tool_usage(invocations: list[ToolInvocation]) -> dict:
    """Summarize tool usage from a list of invocations."""
    stats: dict[str, ToolUsageStats] = {}
    
    for inv in invocations:
        if inv.tool_name not in stats:
            stats[inv.tool_name] = ToolUsageStats(tool_name=inv.tool_name)
        stats[inv.tool_name].update(inv)
    
    return {
        "by_tool": {k: v.to_dict() for k, v in stats.items()},
        "total_calls": len(invocations),
        "total_duration_ms": sum(inv.duration_ms for inv in invocations),
        "success_rate": sum(1 for inv in invocations if inv.success) / len(invocations) if invocations else 0.0,
    }


def compare_runs(manifests: list[RunManifest]) -> dict:
    """Compare multiple run manifests."""
    if not manifests:
        return {}
    
    return {
        "run_count": len(manifests),
        "token_usage": {
            "total": [m.total_token_usage.total_tokens for m in manifests],
            "mean": sum(m.total_token_usage.total_tokens for m in manifests) / len(manifests),
        },
        "costs": {
            "total": [m.cost_breakdown.total_cost if m.cost_breakdown else 0.0 for m in manifests],
            "mean": sum(m.cost_breakdown.total_cost if m.cost_breakdown else 0.0 for m in manifests) / len(manifests),
        },
        "timing": {
            "total_time": [m.timing_metrics.total_time_sec for m in manifests],
            "mean": sum(m.timing_metrics.total_time_sec for m in manifests) / len(manifests),
        },
        "success_rate": sum(m.tasks_completed for m in manifests) / sum(m.tasks_total for m in manifests) if sum(m.tasks_total for m in manifests) > 0 else 0.0,
    }
