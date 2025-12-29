"""
Dashboard Integration Schema for CodeContextBench.

This module defines the data schema for benchmark results that integrates
with the CodeContextBench dashboard. It captures:

1. Task metadata (SDLC type, repo, difficulty)
2. IR-specific metrics (retrieval quality, context relevance)
3. Agent output metrics (success, token usage, time)
4. Comparative data for A/B analysis

Output is compatible with CodeContextBench's:
- .dashboard_runs/ format (run status tracking)
- jobs/ structure (detailed trial results)
- artifacts/ format (metrics comparison, LLM judge results)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict
import hashlib


# =============================================================================
# SDLC Task Type Extensions (supplements CodeContextBench TaskCategory)
# =============================================================================

class SDLCTaskType(str, Enum):
    """SDLC-specific task types for IR evaluation.
    
    These extend CodeContextBench's TaskCategory with enterprise SDLC focus.
    """
    BUG_TRIAGE = "bug_triage"
    CODE_REVIEW = "code_review"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE_UNDERSTANDING = "architecture_understanding"
    SECURITY_AUDIT = "security_audit"
    REFACTORING_ANALYSIS = "refactoring_analysis"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION_LINKING = "documentation_linking"
    FEATURE_LOCATION = "feature_location"
    CHANGE_IMPACT_ANALYSIS = "change_impact_analysis"


class IRToolType(str, Enum):
    """Types of IR tools being evaluated."""
    BASELINE = "baseline"  # No MCP, local tools only
    DEEP_SEARCH = "deep_search"  # Sourcegraph Deep Search MCP
    SOURCEGRAPH_FULL = "sourcegraph_full"  # Full Sourcegraph MCP
    KEYWORD_ONLY = "keyword_only"  # Keyword/NLS search only
    CUSTOM = "custom"  # Custom IR tool


# =============================================================================
# IR-Specific Metrics Schema
# =============================================================================

@dataclass
class IRRetrievalMetrics:
    """Metrics for evaluating IR tool retrieval quality.
    
    These measure how well the IR tool retrieved relevant context.
    """
    # Core retrieval metrics
    precision_at_1: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None  # Mean Reciprocal Rank
    ndcg_at_10: Optional[float] = None  # Normalized DCG
    
    # IR-specific signals
    first_hit_rank: Optional[int] = None  # Rank of first relevant result
    total_queries: int = 0  # Number of IR queries made
    successful_queries: int = 0  # Queries that returned relevant results
    
    # Context utilization
    context_tokens_retrieved: int = 0  # Tokens from IR results
    context_tokens_used: int = 0  # Tokens actually used by agent
    context_utilization_ratio: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AgentExecutionMetrics:
    """Metrics for agent execution performance.
    
    Compatible with CodeContextBench's enterprise_metrics_comparison.json format.
    """
    # Execution outcome
    success: bool = False
    reward: float = 0.0  # 0.0-1.0 reward from verifier
    
    # Token usage (compatible with CodeContextBench)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    
    # Tool usage breakdown
    total_steps: int = 0
    file_reads: int = 0
    grep_searches: int = 0
    glob_searches: int = 0
    bash_commands: int = 0
    
    # MCP-specific tool usage
    mcp_deep_search: int = 0
    mcp_keyword_search: int = 0
    mcp_nls_search: int = 0
    mcp_other: int = 0
    
    # Timing
    duration_sec: float = 0.0
    search_time_sec: float = 0.0  # Time spent in IR/search
    implementation_time_sec: float = 0.0  # Time spent making changes
    
    # Files
    files_mentioned: int = 0
    files_modified: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LLMJudgeScore:
    """LLM judge evaluation scores.
    
    Compatible with CodeContextBench's llm_judge_results.json format.
    """
    tests_pass: float = 0.0  # 0.0-1.0
    tests_pass_reasoning: str = ""
    tests_pass_evidence: str = ""
    
    code_changes: float = 0.0
    code_changes_reasoning: str = ""
    code_changes_evidence: str = ""
    
    architecture: float = 0.0
    architecture_reasoning: str = ""
    architecture_evidence: str = ""
    
    overall: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "tests_pass": {
                "score": self.tests_pass,
                "reasoning": self.tests_pass_reasoning,
                "evidence": self.tests_pass_evidence,
            },
            "code_changes": {
                "score": self.code_changes,
                "reasoning": self.code_changes_reasoning,
                "evidence": self.code_changes_evidence,
            },
            "architecture": {
                "score": self.architecture,
                "reasoning": self.architecture_reasoning,
                "evidence": self.architecture_evidence,
            },
            "overall": self.overall,
        }


# =============================================================================
# Main Task Result Schema
# =============================================================================

@dataclass
class IRSDLCTaskResult:
    """Complete result for an IR-SDLC benchmark task.
    
    This is the primary output schema that captures all metrics for a single
    task execution and integrates with CodeContextBench dashboard.
    """
    # Task identification
    task_id: str
    task_title: str
    sdlc_type: SDLCTaskType
    
    # Repository context
    repo_name: str
    repo_url: str
    commit_hash: str
    difficulty: str  # "easy", "medium", "hard", "expert"
    
    # IR tool configuration
    ir_tool_type: IRToolType
    ir_tool_name: str  # e.g., "DeepSearchFocusedAgent"
    agent_import_path: str  # Harbor agent import path
    model_name: str  # e.g., "anthropic/claude-haiku-4-5-20251001"
    
    # Metrics
    ir_metrics: IRRetrievalMetrics = field(default_factory=IRRetrievalMetrics)
    execution_metrics: AgentExecutionMetrics = field(default_factory=AgentExecutionMetrics)
    llm_judge: Optional[LLMJudgeScore] = None
    
    # Timestamps
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    
    # Ground truth
    ground_truth_files: List[str] = field(default_factory=list)
    retrieved_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    
    # Tags for filtering
    tags: List[str] = field(default_factory=list)
    
    # Raw data paths
    trajectory_path: Optional[str] = None
    diff_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "task_id": self.task_id,
            "task_title": self.task_title,
            "sdlc_type": self.sdlc_type.value,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "commit_hash": self.commit_hash,
            "difficulty": self.difficulty,
            "ir_tool_type": self.ir_tool_type.value,
            "ir_tool_name": self.ir_tool_name,
            "agent_import_path": self.agent_import_path,
            "model_name": self.model_name,
            "ir_metrics": self.ir_metrics.to_dict(),
            "execution_metrics": self.execution_metrics.to_dict(),
            "llm_judge": self.llm_judge.to_dict() if self.llm_judge else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "ground_truth_files": self.ground_truth_files,
            "retrieved_files": self.retrieved_files,
            "modified_files": self.modified_files,
            "tags": self.tags,
            "trajectory_path": self.trajectory_path,
            "diff_path": self.diff_path,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# Benchmark Run Schema (Compatible with .dashboard_runs/)
# =============================================================================

@dataclass
class IRSDLCBenchmarkRun:
    """A complete benchmark run with multiple task results.
    
    Compatible with CodeContextBench's .dashboard_runs/ format.
    """
    # Run identification
    run_id: str
    run_type: str = "ir_sdlc_evaluation"
    
    # Configuration
    benchmark_name: str = "IR-SDLC-Bench"
    agent_name: Optional[str] = None
    ir_tool_type: IRToolType = IRToolType.BASELINE
    model_name: str = "anthropic/claude-haiku-4-5-20251001"
    
    # Execution
    pid: Optional[int] = None
    command: Optional[str] = None
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "running"  # "running", "completed", "failed", "terminated"
    
    # Output
    output_file: Optional[str] = None
    jobs_dir: Optional[str] = None
    
    # Results
    task_results: List[IRSDLCTaskResult] = field(default_factory=list)
    
    # Aggregate metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    successful_tasks: int = 0
    mean_reward: float = 0.0
    
    def to_dashboard_format(self) -> dict:
        """Convert to CodeContextBench .dashboard_runs/ JSON format."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type,
            "pid": self.pid,
            "command": self.command,
            "profile_name": None,  # For compatibility
            "benchmark_name": self.benchmark_name,
            "agent_name": self.agent_name,
            "start_time": self.start_time,
            "status": self.status,
            "output_file": self.output_file,
            # IR-SDLC specific extensions
            "ir_tool_type": self.ir_tool_type.value,
            "model_name": self.model_name,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "successful_tasks": self.successful_tasks,
            "mean_reward": self.mean_reward,
        }
    
    def to_harbor_result_format(self) -> dict:
        """Convert to Harbor jobs/ result.json format."""
        # Group by evaluation key
        evals = {}
        for result in self.task_results:
            eval_key = f"{self.ir_tool_type.value}__{self.model_name}"
            if eval_key not in evals:
                evals[eval_key] = {
                    "n_trials": 0,
                    "n_errors": 0,
                    "metrics": [],
                    "reward_stats": {"reward": {}},
                    "exception_stats": {},
                }
            
            evals[eval_key]["n_trials"] += 1
            reward = result.execution_metrics.reward
            reward_key = str(reward)
            if reward_key not in evals[eval_key]["reward_stats"]["reward"]:
                evals[eval_key]["reward_stats"]["reward"][reward_key] = []
            evals[eval_key]["reward_stats"]["reward"][reward_key].append(result.task_id)
        
        # Calculate metrics
        for eval_key in evals:
            rewards = []
            for reward_key, tasks in evals[eval_key]["reward_stats"]["reward"].items():
                reward_val = float(reward_key)
                rewards.extend([reward_val] * len(tasks))
            if rewards:
                evals[eval_key]["metrics"].append({"mean": sum(rewards) / len(rewards)})
        
        return {
            "id": self.run_id,
            "started_at": self.start_time,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "n_total_trials": len(self.task_results),
            "stats": {
                "n_trials": len(self.task_results),
                "n_errors": sum(1 for r in self.task_results if not r.execution_metrics.success),
                "evals": evals,
            },
        }
    
    def update_aggregates(self) -> None:
        """Update aggregate metrics from task results."""
        self.total_tasks = len(self.task_results)
        self.completed_tasks = sum(1 for r in self.task_results if r.finished_at)
        self.successful_tasks = sum(1 for r in self.task_results if r.execution_metrics.success)
        if self.task_results:
            self.mean_reward = sum(r.execution_metrics.reward for r in self.task_results) / len(self.task_results)


# =============================================================================
# Comparative Analysis Schema
# =============================================================================

@dataclass 
class IRComparison:
    """Comparison between baseline and IR-enhanced agent.
    
    Compatible with CodeContextBench's enterprise_metrics_comparison.json format.
    """
    task_id: str
    sdlc_type: SDLCTaskType
    
    # Baseline metrics
    baseline_success: bool = False
    baseline_reward: float = 0.0
    baseline_tokens: int = 0
    baseline_duration_sec: float = 0.0
    baseline_ir_queries: int = 0
    
    # IR-enhanced metrics
    ir_enhanced_success: bool = False
    ir_enhanced_reward: float = 0.0
    ir_enhanced_tokens: int = 0
    ir_enhanced_duration_sec: float = 0.0
    ir_enhanced_queries: int = 0
    
    # Comparative metrics
    reward_delta: float = 0.0
    token_efficiency: float = 0.0  # (baseline_tokens - ir_tokens) / baseline_tokens
    time_efficiency: float = 0.0
    
    # LLM judge comparison
    baseline_llm_score: Optional[float] = None
    ir_enhanced_llm_score: Optional[float] = None
    llm_score_delta: Optional[float] = None
    
    # Analysis
    mcp_advantage_category: Optional[str] = None  # e.g., "Architecture Understanding"
    mcp_advantage_explanation: Optional[str] = None
    
    def compute_deltas(self) -> None:
        """Compute comparative metrics."""
        self.reward_delta = self.ir_enhanced_reward - self.baseline_reward
        
        if self.baseline_tokens > 0:
            self.token_efficiency = (self.baseline_tokens - self.ir_enhanced_tokens) / self.baseline_tokens
        
        if self.baseline_duration_sec > 0:
            self.time_efficiency = (self.baseline_duration_sec - self.ir_enhanced_duration_sec) / self.baseline_duration_sec
        
        if self.baseline_llm_score is not None and self.ir_enhanced_llm_score is not None:
            self.llm_score_delta = self.ir_enhanced_llm_score - self.baseline_llm_score
    
    def to_dict(self) -> dict:
        return {
            "task": self.task_id,
            "sdlc_type": self.sdlc_type.value,
            "baseline_tokens": self.baseline_tokens,
            "mcp_tokens": self.ir_enhanced_tokens,
            "evaluation": {
                "baseline": {
                    "success": self.baseline_success,
                    "reward": self.baseline_reward,
                    "duration_sec": self.baseline_duration_sec,
                    "overall": self.baseline_llm_score,
                },
                "mcp": {
                    "success": self.ir_enhanced_success,
                    "reward": self.ir_enhanced_reward,
                    "duration_sec": self.ir_enhanced_duration_sec,
                    "overall": self.ir_enhanced_llm_score,
                },
                "mcp_advantage": {
                    "category": self.mcp_advantage_category,
                    "explanation": self.mcp_advantage_explanation,
                },
            },
            "deltas": {
                "reward_delta": self.reward_delta,
                "token_efficiency": self.token_efficiency,
                "time_efficiency": self.time_efficiency,
                "llm_score_delta": self.llm_score_delta,
            },
        }


# =============================================================================
# Schema Validation
# =============================================================================

IR_SDLC_TASK_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "IR-SDLC Task Result Schema",
    "type": "object",
    "required": [
        "task_id",
        "task_title",
        "sdlc_type",
        "repo_name",
        "ir_tool_type",
        "ir_tool_name",
        "agent_import_path",
        "model_name",
    ],
    "properties": {
        "task_id": {"type": "string"},
        "task_title": {"type": "string"},
        "sdlc_type": {
            "type": "string",
            "enum": [t.value for t in SDLCTaskType],
        },
        "repo_name": {"type": "string"},
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard", "expert"],
        },
        "ir_tool_type": {
            "type": "string",
            "enum": [t.value for t in IRToolType],
        },
        "ir_metrics": {
            "type": "object",
            "properties": {
                "precision_at_1": {"type": ["number", "null"]},
                "precision_at_5": {"type": ["number", "null"]},
                "mrr": {"type": ["number", "null"]},
                "total_queries": {"type": "integer"},
                "context_utilization_ratio": {"type": ["number", "null"]},
            },
        },
        "execution_metrics": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "reward": {"type": "number"},
                "total_tokens": {"type": "integer"},
                "duration_sec": {"type": "number"},
            },
        },
    },
}


def validate_task_result(data: dict) -> tuple[bool, Optional[str]]:
    """Validate task result against schema."""
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=IR_SDLC_TASK_SCHEMA)
        return True, None
    except ImportError:
        # Basic validation without jsonschema
        required = ["task_id", "task_title", "sdlc_type", "repo_name", "ir_tool_type"]
        missing = [f for f in required if f not in data]
        if missing:
            return False, f"Missing required fields: {missing}"
        return True, None
    except Exception as e:
        return False, str(e)
