"""
LLM Judge for Agent Output Quality Evaluation.

This module implements an LLM-based judge to evaluate agent output quality
beyond test pass/fail. It assesses:

1. Code correctness - Does the solution address the issue?
2. Code quality - Is the code idiomatic and maintainable?
3. Completeness - Are all requirements met?
4. Efficiency - Are there unnecessary changes?

Output is compatible with CodeContextBench llm_judge_results.json format.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol

from app.ir_sdlc.dashboard_schema import LLMJudgeScore


class EvaluationDimension(str, Enum):
    """Dimensions for LLM judge evaluation."""
    TESTS_PASS = "tests_pass"
    CODE_CHANGES = "code_changes"
    ARCHITECTURE = "architecture"
    CORRECTNESS = "correctness"
    QUALITY = "quality"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"


@dataclass
class EvaluationCriterion:
    """A single evaluation criterion with rubric."""
    name: str
    dimension: EvaluationDimension
    description: str
    rubric: dict[int, str]  # score -> description
    weight: float = 1.0
    
    def get_rubric_text(self) -> str:
        """Format rubric as text for LLM prompt."""
        lines = [f"## {self.name}"]
        lines.append(self.description)
        lines.append("\nScoring rubric:")
        for score, desc in sorted(self.rubric.items()):
            lines.append(f"  {score}: {desc}")
        return "\n".join(lines)


@dataclass
class JudgeInput:
    """Input data for LLM judge evaluation."""
    # Task context
    task_id: str
    task_description: str
    
    # Ground truth (optional)
    ground_truth_diff: Optional[str] = None
    ground_truth_files: list[str] = field(default_factory=list)
    expected_behavior: Optional[str] = None
    
    # Agent output
    agent_diff: Optional[str] = None
    agent_modified_files: list[str] = field(default_factory=list)
    agent_trajectory: Optional[str] = None  # Agent's reasoning trace
    
    # Test results
    tests_passed: int = 0
    tests_total: int = 0
    test_output: Optional[str] = None
    
    # IR context
    retrieved_files: list[str] = field(default_factory=list)
    context_provided: Optional[str] = None
    
    # Metadata
    repo_name: Optional[str] = None
    commit_hash: Optional[str] = None


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation.
    
    Compatible with CodeContextBench llm_judge_results.json format.
    """
    task_id: str
    
    # Per-dimension scores (0.0 to 1.0)
    tests_pass: float = 0.0
    tests_pass_reasoning: str = ""
    tests_pass_evidence: str = ""
    
    code_changes: float = 0.0
    code_changes_reasoning: str = ""
    code_changes_evidence: str = ""
    
    architecture: float = 0.0
    architecture_reasoning: str = ""
    architecture_evidence: str = ""
    
    # Extended dimensions (IR-SDLC specific)
    correctness: float = 0.0
    correctness_reasoning: str = ""
    
    quality: float = 0.0
    quality_reasoning: str = ""
    
    completeness: float = 0.0
    completeness_reasoning: str = ""
    
    efficiency: float = 0.0
    efficiency_reasoning: str = ""
    
    # Overall score
    overall: float = 0.0
    overall_reasoning: str = ""
    
    # Metadata
    raw_response: Optional[str] = None
    model_name: Optional[str] = None
    tokens_used: int = 0
    
    def to_llm_judge_score(self) -> LLMJudgeScore:
        """Convert to LLMJudgeScore for dashboard integration."""
        return LLMJudgeScore(
            tests_pass=self.tests_pass,
            tests_pass_reasoning=self.tests_pass_reasoning,
            tests_pass_evidence=self.tests_pass_evidence,
            code_changes=self.code_changes,
            code_changes_reasoning=self.code_changes_reasoning,
            code_changes_evidence=self.code_changes_evidence,
            architecture=self.architecture,
            architecture_reasoning=self.architecture_reasoning,
            architecture_evidence=self.architecture_evidence,
            overall=self.overall,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
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
            "correctness": {
                "score": self.correctness,
                "reasoning": self.correctness_reasoning,
            },
            "quality": {
                "score": self.quality,
                "reasoning": self.quality_reasoning,
            },
            "completeness": {
                "score": self.completeness,
                "reasoning": self.completeness_reasoning,
            },
            "efficiency": {
                "score": self.efficiency,
                "reasoning": self.efficiency_reasoning,
            },
            "overall": self.overall,
            "overall_reasoning": self.overall_reasoning,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class LLMBackend(Protocol):
    """Protocol for LLM backends that can be used for judging."""
    
    def call(
        self,
        messages: list[dict],
        response_format: str = "json_object",
        **kwargs,
    ) -> tuple[str, float, int, int]:
        """
        Call the LLM with messages.
        
        Returns:
            Tuple of (content, cost, input_tokens, output_tokens)
        """
        ...


# Default evaluation criteria based on CodeContextBench and academic benchmarks
DEFAULT_CRITERIA: list[EvaluationCriterion] = [
    EvaluationCriterion(
        name="Test Pass Evaluation",
        dimension=EvaluationDimension.TESTS_PASS,
        description="Evaluate whether the agent's code changes allow tests to pass.",
        rubric={
            0: "No tests pass or code doesn't compile",
            1: "Some tests pass (<50%)",
            2: "Most tests pass (50-80%)",
            3: "Nearly all tests pass (80-99%)",
            4: "All tests pass (100%)",
        },
        weight=1.5,
    ),
    EvaluationCriterion(
        name="Code Changes Quality",
        dimension=EvaluationDimension.CODE_CHANGES,
        description="Evaluate the quality and appropriateness of code changes.",
        rubric={
            0: "Changes are incorrect, harmful, or unrelated to the task",
            1: "Changes attempt to address the task but are largely incorrect",
            2: "Changes partially address the task with some issues",
            3: "Changes correctly address the task with minor issues",
            4: "Changes are correct, minimal, and well-implemented",
        },
        weight=1.0,
    ),
    EvaluationCriterion(
        name="Architecture Preservation",
        dimension=EvaluationDimension.ARCHITECTURE,
        description="Evaluate whether changes respect existing architecture and patterns.",
        rubric={
            0: "Changes break or ignore existing architecture entirely",
            1: "Changes significantly deviate from existing patterns",
            2: "Changes partially follow existing patterns",
            3: "Changes mostly follow existing patterns with minor deviations",
            4: "Changes fully respect and integrate with existing architecture",
        },
        weight=0.8,
    ),
    EvaluationCriterion(
        name="Solution Correctness",
        dimension=EvaluationDimension.CORRECTNESS,
        description="Does the solution correctly address the stated issue or requirement?",
        rubric={
            0: "Solution does not address the issue at all",
            1: "Solution attempts to address the issue but is incorrect",
            2: "Solution partially addresses the issue",
            3: "Solution mostly addresses the issue with edge cases missed",
            4: "Solution fully and correctly addresses the issue",
        },
        weight=1.5,
    ),
    EvaluationCriterion(
        name="Code Quality",
        dimension=EvaluationDimension.QUALITY,
        description="Is the code idiomatic, readable, and maintainable?",
        rubric={
            0: "Code is unreadable or has major quality issues",
            1: "Code has significant quality issues (poor naming, no comments)",
            2: "Code is functional but has room for improvement",
            3: "Code is well-written with minor style issues",
            4: "Code is exemplary - clean, idiomatic, well-documented",
        },
        weight=0.7,
    ),
    EvaluationCriterion(
        name="Completeness",
        dimension=EvaluationDimension.COMPLETENESS,
        description="Are all requirements from the task fully addressed?",
        rubric={
            0: "No requirements are met",
            1: "Few requirements are met (<25%)",
            2: "Some requirements are met (25-75%)",
            3: "Most requirements are met (75-99%)",
            4: "All requirements are fully met",
        },
        weight=1.0,
    ),
    EvaluationCriterion(
        name="Change Efficiency",
        dimension=EvaluationDimension.EFFICIENCY,
        description="Are the changes minimal and focused, without unnecessary modifications?",
        rubric={
            0: "Changes are excessive or include unrelated modifications",
            1: "Changes include significant unnecessary code",
            2: "Changes include some unnecessary modifications",
            3: "Changes are mostly focused with minor extras",
            4: "Changes are minimal and precisely targeted",
        },
        weight=0.6,
    ),
]


def _build_evaluation_prompt(
    judge_input: JudgeInput,
    criteria: list[EvaluationCriterion],
) -> str:
    """Build the evaluation prompt for the LLM judge."""
    
    prompt_parts = [
        "You are an expert code reviewer evaluating an AI coding agent's output.",
        "Carefully analyze the agent's work and provide scores for each criterion.",
        "",
        "# TASK CONTEXT",
        f"Task ID: {judge_input.task_id}",
        f"Task Description: {judge_input.task_description}",
    ]
    
    if judge_input.repo_name:
        prompt_parts.append(f"Repository: {judge_input.repo_name}")
    
    if judge_input.expected_behavior:
        prompt_parts.extend([
            "",
            "## Expected Behavior",
            judge_input.expected_behavior,
        ])
    
    # Test results
    prompt_parts.extend([
        "",
        "# TEST RESULTS",
        f"Tests Passed: {judge_input.tests_passed} / {judge_input.tests_total}",
    ])
    
    if judge_input.test_output:
        prompt_parts.extend([
            "",
            "## Test Output",
            "```",
            judge_input.test_output[:2000],  # Truncate if too long
            "```",
        ])
    
    # Agent's diff
    if judge_input.agent_diff:
        prompt_parts.extend([
            "",
            "# AGENT'S CODE CHANGES",
            "```diff",
            judge_input.agent_diff[:5000],  # Truncate if too long
            "```",
        ])
        
        if judge_input.agent_modified_files:
            prompt_parts.extend([
                "",
                "Files modified: " + ", ".join(judge_input.agent_modified_files[:10]),
            ])
    
    # Ground truth comparison (if available)
    if judge_input.ground_truth_diff:
        prompt_parts.extend([
            "",
            "# GROUND TRUTH (Reference Solution)",
            "```diff",
            judge_input.ground_truth_diff[:5000],
            "```",
        ])
    
    # IR context
    if judge_input.retrieved_files:
        prompt_parts.extend([
            "",
            "# CONTEXT RETRIEVED BY IR TOOL",
            "Files retrieved: " + ", ".join(judge_input.retrieved_files[:10]),
        ])
    
    # Evaluation criteria
    prompt_parts.extend([
        "",
        "# EVALUATION CRITERIA",
    ])
    
    for criterion in criteria:
        prompt_parts.append("")
        prompt_parts.append(criterion.get_rubric_text())
    
    # Output format
    prompt_parts.extend([
        "",
        "# OUTPUT FORMAT",
        "Respond with a JSON object containing scores and reasoning for each criterion.",
        "Scores should be integers from 0-4 following the rubrics above.",
        "Include 'reasoning' explaining your score and 'evidence' quoting specific code.",
        "",
        "Example output format:",
        '{"tests_pass": {"score": 3, "reasoning": "...", "evidence": "..."}, ...}',
    ])
    
    return "\n".join(prompt_parts)


def _parse_judge_response(
    response: str,
    task_id: str,
    criteria: list[EvaluationCriterion],
) -> JudgeResult:
    """Parse the LLM judge's response into a JudgeResult."""
    
    result = JudgeResult(task_id=task_id, raw_response=response)
    
    # Try to extract JSON from response
    try:
        # Handle potential markdown code blocks
        json_str = response
        if "```json" in response:
            match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
            if match:
                json_str = match.group(1)
        
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # If JSON parsing fails, try to extract scores from text
        result.overall_reasoning = "Failed to parse LLM response as JSON"
        return result
    
    # Extract scores for each dimension
    dimension_mapping = {
        "tests_pass": ("tests_pass", "tests_pass_reasoning", "tests_pass_evidence"),
        "code_changes": ("code_changes", "code_changes_reasoning", "code_changes_evidence"),
        "architecture": ("architecture", "architecture_reasoning", "architecture_evidence"),
        "correctness": ("correctness", "correctness_reasoning", None),
        "quality": ("quality", "quality_reasoning", None),
        "completeness": ("completeness", "completeness_reasoning", None),
        "efficiency": ("efficiency", "efficiency_reasoning", None),
    }
    
    total_score = 0.0
    total_weight = 0.0
    
    for key, (score_attr, reasoning_attr, evidence_attr) in dimension_mapping.items():
        if key in data:
            dim_data = data[key]
            if isinstance(dim_data, dict):
                # Normalize score from 0-4 to 0.0-1.0
                raw_score = dim_data.get("score", 0)
                normalized_score = min(1.0, max(0.0, raw_score / 4.0))
                setattr(result, score_attr, normalized_score)
                
                if reasoning_attr:
                    setattr(result, reasoning_attr, dim_data.get("reasoning", ""))
                if evidence_attr:
                    setattr(result, evidence_attr, dim_data.get("evidence", ""))
                
                # Find weight for this dimension
                weight = 1.0
                for criterion in criteria:
                    if criterion.dimension.value == key:
                        weight = criterion.weight
                        break
                
                total_score += normalized_score * weight
                total_weight += weight
            elif isinstance(dim_data, (int, float)):
                # Simple numeric score
                normalized_score = min(1.0, max(0.0, dim_data / 4.0))
                setattr(result, score_attr, normalized_score)
    
    # Calculate overall score
    if total_weight > 0:
        result.overall = total_score / total_weight
    
    # Extract overall reasoning if provided
    if "overall" in data and isinstance(data["overall"], dict):
        result.overall_reasoning = data["overall"].get("reasoning", "")
    elif "overall_reasoning" in data:
        result.overall_reasoning = data["overall_reasoning"]
    
    return result


class LLMJudge:
    """
    LLM-based judge for evaluating agent output quality.
    
    Usage:
        judge = LLMJudge(model_backend)
        result = judge.evaluate(judge_input)
        score = result.to_llm_judge_score()  # For dashboard integration
    """
    
    def __init__(
        self,
        backend: Optional[LLMBackend] = None,
        criteria: Optional[list[EvaluationCriterion]] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the LLM judge.
        
        Args:
            backend: LLM backend for API calls (optional, can be set later)
            criteria: Evaluation criteria to use (defaults to DEFAULT_CRITERIA)
            model_name: Name of the model for tracking
        """
        self.backend = backend
        self.criteria = criteria or DEFAULT_CRITERIA
        self.model_name = model_name
    
    def set_backend(self, backend: LLMBackend) -> None:
        """Set the LLM backend for API calls."""
        self.backend = backend
    
    def evaluate(
        self,
        judge_input: JudgeInput,
        custom_criteria: Optional[list[EvaluationCriterion]] = None,
    ) -> JudgeResult:
        """
        Evaluate an agent's output using the LLM judge.
        
        Args:
            judge_input: Input data for evaluation
            custom_criteria: Override default criteria for this evaluation
            
        Returns:
            JudgeResult with scores for each dimension
        """
        if self.backend is None:
            raise ValueError("LLM backend not set. Call set_backend() first.")
        
        criteria = custom_criteria or self.criteria
        
        # Build the evaluation prompt
        prompt = _build_evaluation_prompt(judge_input, criteria)
        
        # Call the LLM
        messages = [
            {"role": "system", "content": "You are an expert code reviewer."},
            {"role": "user", "content": prompt},
        ]
        
        content, cost, input_tokens, output_tokens = self.backend.call(
            messages,
            response_format="json_object",
        )
        
        # Parse the response
        result = _parse_judge_response(content, judge_input.task_id, criteria)
        result.model_name = self.model_name
        result.tokens_used = input_tokens + output_tokens
        
        return result
    
    def evaluate_batch(
        self,
        inputs: list[JudgeInput],
        custom_criteria: Optional[list[EvaluationCriterion]] = None,
    ) -> list[JudgeResult]:
        """
        Evaluate multiple agent outputs.
        
        Args:
            inputs: List of judge inputs
            custom_criteria: Override default criteria
            
        Returns:
            List of JudgeResult objects
        """
        return [self.evaluate(inp, custom_criteria) for inp in inputs]


class MockLLMBackend:
    """
    Mock LLM backend for testing without API calls.
    
    Returns deterministic scores based on test pass rate.
    """
    
    def __init__(self, default_scores: Optional[dict] = None):
        self.default_scores = default_scores or {
            "tests_pass": {"score": 3, "reasoning": "Good test coverage", "evidence": ""},
            "code_changes": {"score": 3, "reasoning": "Clean changes", "evidence": ""},
            "architecture": {"score": 3, "reasoning": "Follows patterns", "evidence": ""},
            "correctness": {"score": 3, "reasoning": "Correct solution"},
            "quality": {"score": 3, "reasoning": "Good code quality"},
            "completeness": {"score": 3, "reasoning": "All requirements met"},
            "efficiency": {"score": 3, "reasoning": "Minimal changes"},
        }
    
    def call(
        self,
        messages: list[dict],
        response_format: str = "json_object",
        **kwargs,
    ) -> tuple[str, float, int, int]:
        """Return mock response."""
        response = json.dumps(self.default_scores)
        return response, 0.0, 100, 50


def create_judge_input_from_trace(
    trace: "AgentExecutionTrace",
    task_description: str,
    agent_diff: Optional[str] = None,
    ground_truth_diff: Optional[str] = None,
    test_output: Optional[str] = None,
) -> JudgeInput:
    """
    Create a JudgeInput from an AgentExecutionTrace.
    
    This is a convenience function for integrating the LLM judge
    with the existing agent metrics system.
    
    Args:
        trace: AgentExecutionTrace from agent_metrics module
        task_description: Description of the task
        agent_diff: Code diff produced by the agent
        ground_truth_diff: Reference solution diff (optional)
        test_output: Test execution output (optional)
        
    Returns:
        JudgeInput ready for LLM evaluation
    """
    # Import here to avoid circular imports
    from app.ir_sdlc.agent_metrics import AgentExecutionTrace
    
    return JudgeInput(
        task_id=trace.task_id,
        task_description=task_description,
        ground_truth_diff=ground_truth_diff,
        agent_diff=agent_diff,
        tests_passed=trace.tests_passed,
        tests_total=trace.tests_total,
        test_output=test_output,
    )


def compute_judge_aggregate_scores(results: list[JudgeResult]) -> dict:
    """
    Compute aggregate statistics from multiple judge results.
    
    Args:
        results: List of JudgeResult objects
        
    Returns:
        Dictionary with mean, std, min, max for each dimension
    """
    import statistics
    
    if not results:
        return {}
    
    dimensions = [
        "tests_pass", "code_changes", "architecture",
        "correctness", "quality", "completeness", "efficiency", "overall"
    ]
    
    aggregates = {}
    
    for dim in dimensions:
        values = [getattr(r, dim) for r in results]
        aggregates[dim] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    
    return aggregates


def export_results_to_json(
    results: list[JudgeResult],
    output_path: str,
    include_raw_response: bool = False,
) -> None:
    """
    Export judge results to JSON file (CodeContextBench compatible).
    
    Args:
        results: List of JudgeResult objects
        output_path: Path to output JSON file
        include_raw_response: Whether to include raw LLM responses
    """
    import json
    from pathlib import Path
    
    output = {
        "results": [],
        "aggregate": compute_judge_aggregate_scores(results),
        "metadata": {
            "num_results": len(results),
            "model_name": results[0].model_name if results else None,
        }
    }
    
    for result in results:
        result_dict = result.to_dict()
        if not include_raw_response:
            result_dict.pop("raw_response", None)
        output["results"].append(result_dict)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
