"""
Tests for LLM Judge for Agent Output Quality Evaluation.

Validates the llm_judge module that evaluates agent output quality
using LLM-based scoring across multiple dimensions.
"""

import json
import pytest

from app.ir_sdlc.llm_judge import (
    EvaluationDimension,
    EvaluationCriterion,
    JudgeInput,
    JudgeResult,
    LLMJudge,
    MockLLMBackend,
    DEFAULT_CRITERIA,
    _build_evaluation_prompt,
    _parse_judge_response,
    create_judge_input_from_trace,
    compute_judge_aggregate_scores,
    export_results_to_json,
)
from app.ir_sdlc.dashboard_schema import LLMJudgeScore
from app.ir_sdlc.agent_metrics import (
    AgentExecutionTrace,
    AgentMetrics,
    TaskCompletionStatus,
)


class TestJudgeInput:
    """Tests for JudgeInput dataclass."""
    
    def test_judge_input_minimal(self):
        """Test creating JudgeInput with minimal fields."""
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Fix the bug in parsing",
        )
        
        assert judge_input.task_id == "task-001"
        assert judge_input.task_description == "Fix the bug in parsing"
        assert judge_input.tests_passed == 0
        assert judge_input.tests_total == 0
        assert judge_input.agent_diff is None
    
    def test_judge_input_full(self):
        """Test creating JudgeInput with all fields."""
        judge_input = JudgeInput(
            task_id="task-002",
            task_description="Add new feature",
            ground_truth_diff="--- a/file.py\n+++ b/file.py\n+new_line",
            ground_truth_files=["file.py"],
            expected_behavior="Should add new function",
            agent_diff="--- a/file.py\n+++ b/file.py\n+my_new_line",
            agent_modified_files=["file.py"],
            agent_trajectory="1. Read file\n2. Modified",
            tests_passed=8,
            tests_total=10,
            test_output="PASSED 8/10",
            retrieved_files=["file.py", "utils.py"],
            context_provided="def existing_function()...",
            repo_name="test/repo",
            commit_hash="abc123",
        )
        
        assert judge_input.tests_passed == 8
        assert judge_input.tests_total == 10
        assert len(judge_input.retrieved_files) == 2


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""
    
    def test_judge_result_defaults(self):
        """Test JudgeResult default values."""
        result = JudgeResult(task_id="task-001")
        
        assert result.task_id == "task-001"
        assert result.tests_pass == 0.0
        assert result.code_changes == 0.0
        assert result.architecture == 0.0
        assert result.overall == 0.0
    
    def test_judge_result_to_llm_judge_score(self):
        """Test conversion to LLMJudgeScore for dashboard integration."""
        result = JudgeResult(
            task_id="task-001",
            tests_pass=0.8,
            tests_pass_reasoning="Good test coverage",
            tests_pass_evidence="8/10 tests pass",
            code_changes=0.75,
            code_changes_reasoning="Clean implementation",
            code_changes_evidence="Minimal diff",
            architecture=0.9,
            architecture_reasoning="Follows patterns",
            architecture_evidence="Uses existing utilities",
            overall=0.82,
        )
        
        score = result.to_llm_judge_score()
        
        assert isinstance(score, LLMJudgeScore)
        assert score.tests_pass == 0.8
        assert score.tests_pass_reasoning == "Good test coverage"
        assert score.code_changes == 0.75
        assert score.architecture == 0.9
        assert score.overall == 0.82
    
    def test_judge_result_to_dict(self):
        """Test serialization to dictionary."""
        result = JudgeResult(
            task_id="task-001",
            tests_pass=0.75,
            tests_pass_reasoning="Good",
            correctness=0.8,
            correctness_reasoning="Correct solution",
            overall=0.77,
            model_name="test-model",
            tokens_used=150,
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["task_id"] == "task-001"
        assert result_dict["tests_pass"]["score"] == 0.75
        assert result_dict["tests_pass"]["reasoning"] == "Good"
        assert result_dict["correctness"]["score"] == 0.8
        assert result_dict["overall"] == 0.77
        assert result_dict["model_name"] == "test-model"
        assert result_dict["tokens_used"] == 150
    
    def test_judge_result_to_json(self):
        """Test JSON serialization."""
        result = JudgeResult(
            task_id="task-001",
            tests_pass=0.5,
            overall=0.5,
        )
        
        json_str = result.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["task_id"] == "task-001"
        assert parsed["overall"] == 0.5


class TestEvaluationCriterion:
    """Tests for EvaluationCriterion."""
    
    def test_criterion_creation(self):
        """Test creating an evaluation criterion."""
        criterion = EvaluationCriterion(
            name="Test Criterion",
            dimension=EvaluationDimension.CORRECTNESS,
            description="Test description",
            rubric={
                0: "Bad",
                1: "Poor",
                2: "Fair",
                3: "Good",
                4: "Excellent",
            },
            weight=1.5,
        )
        
        assert criterion.name == "Test Criterion"
        assert criterion.dimension == EvaluationDimension.CORRECTNESS
        assert criterion.weight == 1.5
    
    def test_criterion_rubric_text(self):
        """Test rubric formatting."""
        criterion = EvaluationCriterion(
            name="Code Quality",
            dimension=EvaluationDimension.QUALITY,
            description="Evaluate code quality",
            rubric={0: "Bad", 4: "Excellent"},
        )
        
        rubric_text = criterion.get_rubric_text()
        
        assert "## Code Quality" in rubric_text
        assert "Evaluate code quality" in rubric_text
        assert "0: Bad" in rubric_text
        assert "4: Excellent" in rubric_text


class TestDefaultCriteria:
    """Tests for default evaluation criteria."""
    
    def test_default_criteria_exist(self):
        """Test that default criteria are defined."""
        assert len(DEFAULT_CRITERIA) >= 5
    
    def test_default_criteria_dimensions(self):
        """Test that all key dimensions are covered."""
        dimensions = {c.dimension for c in DEFAULT_CRITERIA}
        
        assert EvaluationDimension.TESTS_PASS in dimensions
        assert EvaluationDimension.CODE_CHANGES in dimensions
        assert EvaluationDimension.CORRECTNESS in dimensions
        assert EvaluationDimension.QUALITY in dimensions
    
    def test_default_criteria_weights(self):
        """Test that criteria have reasonable weights."""
        for criterion in DEFAULT_CRITERIA:
            assert 0.0 < criterion.weight <= 2.0


class TestBuildEvaluationPrompt:
    """Tests for prompt building."""
    
    def test_prompt_contains_task_info(self):
        """Test that prompt includes task information."""
        judge_input = JudgeInput(
            task_id="task-123",
            task_description="Fix authentication bug",
            repo_name="myorg/myrepo",
        )
        
        prompt = _build_evaluation_prompt(judge_input, DEFAULT_CRITERIA)
        
        assert "task-123" in prompt
        assert "Fix authentication bug" in prompt
        assert "myorg/myrepo" in prompt
    
    def test_prompt_contains_test_results(self):
        """Test that prompt includes test results."""
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Test task",
            tests_passed=7,
            tests_total=10,
            test_output="7 passed, 3 failed",
        )
        
        prompt = _build_evaluation_prompt(judge_input, DEFAULT_CRITERIA)
        
        assert "7 / 10" in prompt
        assert "7 passed, 3 failed" in prompt
    
    def test_prompt_contains_diff(self):
        """Test that prompt includes code diff."""
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Test task",
            agent_diff="--- a/file.py\n+++ b/file.py\n+new_code",
        )
        
        prompt = _build_evaluation_prompt(judge_input, DEFAULT_CRITERIA)
        
        assert "```diff" in prompt
        assert "+new_code" in prompt
    
    def test_prompt_contains_criteria(self):
        """Test that prompt includes evaluation criteria."""
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Test task",
        )
        
        prompt = _build_evaluation_prompt(judge_input, DEFAULT_CRITERIA)
        
        assert "EVALUATION CRITERIA" in prompt
        assert "Test Pass Evaluation" in prompt
        assert "Code Changes Quality" in prompt


class TestParseJudgeResponse:
    """Tests for parsing LLM judge responses."""
    
    def test_parse_valid_json_response(self):
        """Test parsing a valid JSON response."""
        response = json.dumps({
            "tests_pass": {"score": 3, "reasoning": "Good", "evidence": "8/10"},
            "code_changes": {"score": 4, "reasoning": "Clean"},
            "architecture": {"score": 3, "reasoning": "Follows patterns"},
            "correctness": {"score": 4, "reasoning": "Correct"},
            "quality": {"score": 3, "reasoning": "Good style"},
            "completeness": {"score": 4, "reasoning": "All done"},
            "efficiency": {"score": 3, "reasoning": "Minimal"},
        })
        
        result = _parse_judge_response(response, "task-001", DEFAULT_CRITERIA)
        
        assert result.task_id == "task-001"
        assert result.tests_pass == 0.75  # 3/4
        assert result.code_changes == 1.0  # 4/4
        assert result.architecture == 0.75
        assert result.correctness == 1.0
        assert result.overall > 0.0
    
    def test_parse_markdown_wrapped_json(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{
    "tests_pass": {"score": 2, "reasoning": "Fair"},
    "code_changes": {"score": 3, "reasoning": "Good"}
}
```"""
        
        result = _parse_judge_response(response, "task-001", DEFAULT_CRITERIA)
        
        assert result.tests_pass == 0.5  # 2/4
        assert result.code_changes == 0.75
    
    def test_parse_invalid_json_returns_empty_result(self):
        """Test that invalid JSON returns empty result with error."""
        response = "This is not valid JSON at all"
        
        result = _parse_judge_response(response, "task-001", DEFAULT_CRITERIA)
        
        assert result.task_id == "task-001"
        assert result.tests_pass == 0.0
        assert "Failed to parse" in result.overall_reasoning
    
    def test_parse_partial_response(self):
        """Test parsing response with only some dimensions."""
        response = json.dumps({
            "tests_pass": {"score": 4, "reasoning": "All pass"},
        })
        
        result = _parse_judge_response(response, "task-001", DEFAULT_CRITERIA)
        
        assert result.tests_pass == 1.0
        assert result.code_changes == 0.0  # Not provided


class TestMockLLMBackend:
    """Tests for MockLLMBackend."""
    
    def test_mock_backend_returns_valid_response(self):
        """Test that mock backend returns parseable response."""
        backend = MockLLMBackend()
        
        content, cost, input_tokens, output_tokens = backend.call(
            messages=[{"role": "user", "content": "test"}],
            response_format="json_object",
        )
        
        assert cost == 0.0
        assert input_tokens == 100
        assert output_tokens == 50
        
        # Response should be valid JSON
        data = json.loads(content)
        assert "tests_pass" in data
        assert "code_changes" in data
    
    def test_mock_backend_custom_scores(self):
        """Test mock backend with custom scores."""
        custom_scores = {
            "tests_pass": {"score": 4, "reasoning": "Perfect"},
            "code_changes": {"score": 4, "reasoning": "Perfect"},
        }
        backend = MockLLMBackend(default_scores=custom_scores)
        
        content, _, _, _ = backend.call([])
        data = json.loads(content)
        
        assert data["tests_pass"]["score"] == 4


class TestLLMJudge:
    """Tests for LLMJudge class."""
    
    def test_judge_initialization(self):
        """Test LLMJudge initialization."""
        judge = LLMJudge(
            backend=MockLLMBackend(),
            model_name="test-model",
        )
        
        assert judge.backend is not None
        assert judge.model_name == "test-model"
        assert len(judge.criteria) == len(DEFAULT_CRITERIA)
    
    def test_judge_set_backend(self):
        """Test setting backend after initialization."""
        judge = LLMJudge()
        
        assert judge.backend is None
        
        judge.set_backend(MockLLMBackend())
        
        assert judge.backend is not None
    
    def test_judge_evaluate_without_backend_raises(self):
        """Test that evaluate raises without backend."""
        judge = LLMJudge()
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Test",
        )
        
        with pytest.raises(ValueError, match="LLM backend not set"):
            judge.evaluate(judge_input)
    
    def test_judge_evaluate_basic(self):
        """Test basic evaluation with mock backend."""
        judge = LLMJudge(
            backend=MockLLMBackend(),
            model_name="mock-model",
        )
        
        judge_input = JudgeInput(
            task_id="task-001",
            task_description="Fix the authentication bug",
            tests_passed=8,
            tests_total=10,
            agent_diff="--- a/auth.py\n+++ b/auth.py\n+fixed",
        )
        
        result = judge.evaluate(judge_input)
        
        assert result.task_id == "task-001"
        assert result.model_name == "mock-model"
        assert result.tokens_used > 0
        assert result.tests_pass > 0.0
        assert result.overall > 0.0
    
    def test_judge_evaluate_batch(self):
        """Test batch evaluation."""
        judge = LLMJudge(backend=MockLLMBackend())
        
        inputs = [
            JudgeInput(task_id=f"task-{i}", task_description=f"Task {i}")
            for i in range(3)
        ]
        
        results = judge.evaluate_batch(inputs)
        
        assert len(results) == 3
        assert all(r.task_id == f"task-{i}" for i, r in enumerate(results))
    
    def test_judge_custom_criteria(self):
        """Test evaluation with custom criteria."""
        custom_criteria = [
            EvaluationCriterion(
                name="Custom Metric",
                dimension=EvaluationDimension.CORRECTNESS,
                description="Custom evaluation",
                rubric={0: "Bad", 4: "Good"},
                weight=2.0,
            ),
        ]
        
        judge = LLMJudge(
            backend=MockLLMBackend(),
            criteria=custom_criteria,
        )
        
        assert len(judge.criteria) == 1
        assert judge.criteria[0].name == "Custom Metric"


class TestCreateJudgeInputFromTrace:
    """Tests for converting AgentExecutionTrace to JudgeInput."""
    
    def test_create_from_trace_basic(self):
        """Test creating JudgeInput from basic trace."""
        trace = AgentExecutionTrace(
            task_id="task-001",
            agent_name="test-agent",
            ir_tool_name="grep",
            tests_passed=5,
            tests_total=10,
        )
        
        judge_input = create_judge_input_from_trace(
            trace=trace,
            task_description="Fix the bug",
        )
        
        assert judge_input.task_id == "task-001"
        assert judge_input.task_description == "Fix the bug"
        assert judge_input.tests_passed == 5
        assert judge_input.tests_total == 10
    
    def test_create_from_trace_with_diffs(self):
        """Test creating JudgeInput with diffs."""
        trace = AgentExecutionTrace(
            task_id="task-002",
            agent_name="agent",
            ir_tool_name="tool",
        )
        
        judge_input = create_judge_input_from_trace(
            trace=trace,
            task_description="Add feature",
            agent_diff="--- a/file.py\n+++ b/file.py",
            ground_truth_diff="--- a/file.py\n+++ b/file.py (truth)",
            test_output="All tests passed",
        )
        
        assert judge_input.agent_diff is not None
        assert judge_input.ground_truth_diff is not None
        assert judge_input.test_output == "All tests passed"


class TestComputeJudgeAggregateScores:
    """Tests for aggregating judge results."""
    
    def test_aggregate_empty_list(self):
        """Test aggregation of empty list."""
        result = compute_judge_aggregate_scores([])
        
        assert result == {}
    
    def test_aggregate_single_result(self):
        """Test aggregation of single result."""
        results = [
            JudgeResult(
                task_id="task-001",
                tests_pass=0.8,
                code_changes=0.7,
                overall=0.75,
            )
        ]
        
        aggregate = compute_judge_aggregate_scores(results)
        
        assert aggregate["tests_pass"]["mean"] == 0.8
        assert aggregate["tests_pass"]["count"] == 1
        assert aggregate["overall"]["mean"] == 0.75
    
    def test_aggregate_multiple_results(self):
        """Test aggregation of multiple results."""
        results = [
            JudgeResult(task_id="t1", tests_pass=0.6, overall=0.5),
            JudgeResult(task_id="t2", tests_pass=0.8, overall=0.7),
            JudgeResult(task_id="t3", tests_pass=1.0, overall=0.9),
        ]
        
        aggregate = compute_judge_aggregate_scores(results)
        
        assert aggregate["tests_pass"]["mean"] == pytest.approx(0.8)
        assert aggregate["tests_pass"]["min"] == 0.6
        assert aggregate["tests_pass"]["max"] == 1.0
        assert aggregate["tests_pass"]["count"] == 3
        assert aggregate["overall"]["mean"] == pytest.approx(0.7)


class TestExportResultsToJson:
    """Tests for JSON export."""
    
    def test_export_to_json_file(self, tmp_path):
        """Test exporting results to JSON file."""
        results = [
            JudgeResult(
                task_id="task-001",
                tests_pass=0.8,
                overall=0.75,
                model_name="test-model",
            ),
            JudgeResult(
                task_id="task-002",
                tests_pass=0.6,
                overall=0.65,
                model_name="test-model",
            ),
        ]
        
        output_path = tmp_path / "llm_judge_results.json"
        export_results_to_json(results, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert len(data["results"]) == 2
        assert data["metadata"]["num_results"] == 2
        assert data["metadata"]["model_name"] == "test-model"
        assert "aggregate" in data
    
    def test_export_excludes_raw_response_by_default(self, tmp_path):
        """Test that raw_response is excluded by default."""
        results = [
            JudgeResult(
                task_id="task-001",
                raw_response='{"big": "response"}',
            ),
        ]
        
        output_path = tmp_path / "results.json"
        export_results_to_json(results, str(output_path), include_raw_response=False)
        
        with open(output_path) as f:
            data = json.load(f)
        
        assert "raw_response" not in data["results"][0]


class TestAgentMetricsLLMJudgeIntegration:
    """Tests for AgentMetrics integration with LLM judge."""
    
    def test_agent_metrics_has_llm_judge_fields(self):
        """Test that AgentMetrics has LLM judge fields."""
        metrics = AgentMetrics()
        
        assert hasattr(metrics, "llm_judge_scores")
        assert hasattr(metrics, "llm_judge_aggregate")
    
    def test_add_llm_judge_results(self):
        """Test adding LLM judge results to metrics."""
        metrics = AgentMetrics(
            task_completion_rate=0.8,
            total_tasks=10,
            successful_tasks=8,
        )
        
        judge_results = [
            JudgeResult(task_id="t1", tests_pass=0.8, overall=0.75),
            JudgeResult(task_id="t2", tests_pass=0.6, overall=0.65),
        ]
        
        metrics.add_llm_judge_results(judge_results)
        
        assert len(metrics.llm_judge_scores) == 2
        assert "t1" in metrics.llm_judge_scores
        assert "t2" in metrics.llm_judge_scores
        assert "overall" in metrics.llm_judge_aggregate
    
    def test_to_summary_dict_includes_llm_judge(self):
        """Test that summary dict includes LLM judge metrics."""
        metrics = AgentMetrics(
            task_completion_rate=0.8,
            llm_judge_aggregate={
                "overall": {"mean": 0.72, "std": 0.1},
                "correctness": {"mean": 0.8, "std": 0.05},
                "quality": {"mean": 0.7, "std": 0.15},
            },
        )
        
        summary = metrics.to_summary_dict()
        
        assert summary["llm_judge_overall"] == 0.72
        assert summary["llm_judge_correctness"] == 0.8
        assert summary["llm_judge_quality"] == 0.7
    
    def test_to_dict_includes_llm_judge(self):
        """Test that full dict includes LLM judge metrics."""
        metrics = AgentMetrics()
        metrics.llm_judge_scores = {"t1": {"overall": 0.8}}
        metrics.llm_judge_aggregate = {"overall": {"mean": 0.8}}
        
        full_dict = metrics.to_dict()
        
        assert "llm_judge_scores" in full_dict
        assert "llm_judge_aggregate" in full_dict
    
    def test_get_primary_score_with_llm_judge(self):
        """Test getting LLM judge as primary score."""
        metrics = AgentMetrics(
            task_completion_rate=0.5,
            llm_judge_aggregate={
                "overall": {"mean": 0.9},
            },
        )
        
        score = metrics.get_primary_score("llm_judge_overall")
        
        assert score == 0.9


class TestCodeContextBenchCompatibility:
    """Tests for compatibility with CodeContextBench format."""
    
    def test_llm_judge_score_format_matches(self):
        """Test that LLMJudgeScore format matches CodeContextBench."""
        result = JudgeResult(
            task_id="test",
            tests_pass=0.8,
            tests_pass_reasoning="Good coverage",
            tests_pass_evidence="8/10 pass",
            code_changes=0.7,
            code_changes_reasoning="Clean",
            code_changes_evidence="Minimal diff",
            architecture=0.9,
            architecture_reasoning="Follows patterns",
            architecture_evidence="Uses utils",
            overall=0.8,
        )
        
        score = result.to_llm_judge_score()
        score_dict = score.to_dict()
        
        # Check CodeContextBench expected format
        assert "tests_pass" in score_dict
        assert "score" in score_dict["tests_pass"]
        assert "reasoning" in score_dict["tests_pass"]
        assert "evidence" in score_dict["tests_pass"]
        
        assert "code_changes" in score_dict
        assert "architecture" in score_dict
        assert "overall" in score_dict
    
    def test_export_format_matches_llm_judge_results_json(self, tmp_path):
        """Test that export format matches llm_judge_results.json."""
        results = [
            JudgeResult(
                task_id="task-001",
                tests_pass=0.8,
                tests_pass_reasoning="Good",
                code_changes=0.7,
                architecture=0.9,
                overall=0.8,
                model_name="claude-3-sonnet",
            ),
        ]
        
        output_path = tmp_path / "llm_judge_results.json"
        export_results_to_json(results, str(output_path))
        
        with open(output_path) as f:
            data = json.load(f)
        
        # Verify structure matches CodeContextBench expectations
        assert "results" in data
        assert "aggregate" in data
        assert "metadata" in data
        
        result = data["results"][0]
        assert "task_id" in result
        assert "tests_pass" in result
        assert "code_changes" in result
        assert "architecture" in result
        assert "overall" in result


class TestEndToEndEvaluation:
    """End-to-end tests for complete evaluation workflow."""
    
    def test_complete_evaluation_workflow(self):
        """Test complete workflow from trace to dashboard-ready output."""
        # 1. Create agent execution trace
        trace = AgentExecutionTrace(
            task_id="fix-auth-bug",
            agent_name="coding-agent",
            ir_tool_name="deep-search",
            completed=True,
            completion_status=TaskCompletionStatus.SUCCESS,
            compiles=True,
            tests_passed=9,
            tests_total=10,
            input_tokens=5000,
            output_tokens=1000,
            wall_clock_time_sec=45.0,
        )
        
        # 2. Create judge input from trace
        judge_input = create_judge_input_from_trace(
            trace=trace,
            task_description="Fix authentication bypass vulnerability",
            agent_diff="--- a/auth.py\n+++ b/auth.py\n@@ -10,6 +10,7 @@\n+    if not validate_token(token):\n+        raise AuthError()",
            test_output="9/10 tests passed",
        )
        
        # 3. Run LLM judge evaluation
        judge = LLMJudge(
            backend=MockLLMBackend(),
            model_name="test-judge",
        )
        
        result = judge.evaluate(judge_input)
        
        # 4. Convert to dashboard-compatible format
        score = result.to_llm_judge_score()
        
        # 5. Verify complete output
        assert result.task_id == "fix-auth-bug"
        assert result.model_name == "test-judge"
        assert result.overall > 0.0
        
        assert score.tests_pass > 0.0
        assert score.code_changes > 0.0
    
    def test_batch_evaluation_and_aggregation(self):
        """Test batch evaluation with aggregation."""
        judge = LLMJudge(backend=MockLLMBackend())
        
        inputs = [
            JudgeInput(
                task_id=f"task-{i}",
                task_description=f"Fix bug {i}",
                tests_passed=i * 2,
                tests_total=10,
            )
            for i in range(1, 6)
        ]
        
        results = judge.evaluate_batch(inputs)
        aggregate = compute_judge_aggregate_scores(results)
        
        assert len(results) == 5
        assert "overall" in aggregate
        assert aggregate["overall"]["count"] == 5
    
    def test_integration_with_agent_metrics_compute_all(self):
        """Test that LLM judge integrates with AgentMetrics.compute_all()."""
        # Create traces
        traces = [
            AgentExecutionTrace(
                task_id=f"task-{i}",
                agent_name="agent",
                ir_tool_name="tool",
                completion_status=TaskCompletionStatus.SUCCESS if i % 2 == 0 else TaskCompletionStatus.FAILURE,
                tests_passed=i,
                tests_total=5,
            )
            for i in range(5)
        ]
        
        # Compute base metrics
        metrics = AgentMetrics.compute_all(traces)
        
        # Create and add judge results
        judge = LLMJudge(backend=MockLLMBackend())
        judge_inputs = [
            create_judge_input_from_trace(t, f"Task {t.task_id}")
            for t in traces
        ]
        judge_results = judge.evaluate_batch(judge_inputs)
        
        # Add to metrics
        metrics.add_llm_judge_results(judge_results)
        
        # Verify integration
        assert metrics.total_tasks == 5
        assert len(metrics.llm_judge_scores) == 5
        assert "overall" in metrics.llm_judge_aggregate
        
        # Verify summary includes both
        summary = metrics.to_summary_dict()
        assert "task_completion_rate" in summary
        assert "llm_judge_overall" in summary
