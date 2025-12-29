"""
Tests for CodeContextBench dashboard integration.

Validates that IR-SDLC-Factory outputs are compatible with
CodeContextBench's dashboard format.
"""

import json
import tempfile
from pathlib import Path
import pytest

from app.ir_sdlc.dashboard_schema import (
    IRSDLCTaskResult,
    IRSDLCBenchmarkRun,
    IRComparison,
    IRRetrievalMetrics,
    AgentExecutionMetrics,
    LLMJudgeScore,
    IRToolType,
    SDLCTaskType,
    validate_task_result,
)
from app.ir_sdlc.dashboard_exporter import (
    CodeContextBenchExporter,
    generate_run_id,
)


class TestDashboardSchema:
    """Test dashboard schema dataclasses."""
    
    def test_ir_retrieval_metrics_to_dict(self):
        """Test IRRetrievalMetrics serialization."""
        metrics = IRRetrievalMetrics(
            precision_at_1=0.8,
            precision_at_5=0.6,
            mrr=0.75,
            total_queries=10,
            successful_queries=7,
        )
        
        data = metrics.to_dict()
        
        assert data["precision_at_1"] == 0.8
        assert data["mrr"] == 0.75
        assert data["total_queries"] == 10
        # None values should be excluded
        assert "ndcg_at_10" not in data or data.get("ndcg_at_10") is None
    
    def test_agent_execution_metrics_to_dict(self):
        """Test AgentExecutionMetrics serialization."""
        metrics = AgentExecutionMetrics(
            success=True,
            reward=1.0,
            total_prompt_tokens=50000,
            total_completion_tokens=5000,
            total_tokens=55000,
            total_steps=45,
            mcp_deep_search=5,
            duration_sec=120.5,
        )
        
        data = metrics.to_dict()
        
        assert data["success"] is True
        assert data["reward"] == 1.0
        assert data["total_tokens"] == 55000
        assert data["mcp_deep_search"] == 5
    
    def test_llm_judge_score_to_dict(self):
        """Test LLMJudgeScore serialization matches CodeContextBench format."""
        judge = LLMJudgeScore(
            tests_pass=0.9,
            tests_pass_reasoning="Tests comprehensive",
            tests_pass_evidence="Added 5 test cases",
            code_changes=0.85,
            code_changes_reasoning="Good implementation",
            code_changes_evidence="Modified 3 files",
            architecture=0.8,
            architecture_reasoning="Understood structure",
            architecture_evidence="Correct module placement",
            overall=0.85,
        )
        
        data = judge.to_dict()
        
        # Check structure matches CodeContextBench llm_judge_results.json
        assert "tests_pass" in data
        assert data["tests_pass"]["score"] == 0.9
        assert data["tests_pass"]["reasoning"] == "Tests comprehensive"
        assert data["overall"] == 0.85
    
    def test_task_result_to_dict(self):
        """Test IRSDLCTaskResult full serialization."""
        result = IRSDLCTaskResult(
            task_id="ir-sdlc-001",
            task_title="Bug triage test",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            repo_name="kubernetes/kubernetes",
            repo_url="https://github.com/kubernetes/kubernetes",
            commit_hash="abc123",
            difficulty="medium",
            ir_tool_type=IRToolType.DEEP_SEARCH,
            ir_tool_name="DeepSearchFocusedAgent",
            agent_import_path="agents.mcp_variants:DeepSearchFocusedAgent",
            model_name="anthropic/claude-haiku-4-5-20251001",
            ir_metrics=IRRetrievalMetrics(mrr=0.8, total_queries=5),
            execution_metrics=AgentExecutionMetrics(success=True, reward=1.0),
            ground_truth_files=["pkg/scheduler/scheduler.go"],
            retrieved_files=["pkg/scheduler/scheduler.go", "pkg/scheduler/factory.go"],
            tags=["scheduler", "bug-fix"],
        )
        
        data = result.to_dict()
        
        assert data["task_id"] == "ir-sdlc-001"
        assert data["sdlc_type"] == "bug_triage"
        assert data["ir_tool_type"] == "deep_search"
        assert data["ir_metrics"]["mrr"] == 0.8
        assert data["execution_metrics"]["success"] is True
    
    def test_validate_task_result(self):
        """Test schema validation."""
        valid_data = {
            "task_id": "ir-sdlc-001",
            "task_title": "Test task",
            "sdlc_type": "bug_triage",
            "repo_name": "test/repo",
            "ir_tool_type": "baseline",
            "ir_tool_name": "BaselineAgent",
            "agent_import_path": "agents:BaselineAgent",
            "model_name": "claude-haiku",
        }
        
        is_valid, error = validate_task_result(valid_data)
        assert is_valid is True
        assert error is None
    
    def test_validate_task_result_missing_field(self):
        """Test schema validation catches missing fields."""
        invalid_data = {
            "task_id": "ir-sdlc-001",
            # Missing required fields
        }
        
        is_valid, error = validate_task_result(invalid_data)
        assert is_valid is False
        assert "Missing required fields" in error or error is not None


class TestBenchmarkRun:
    """Test benchmark run schema and formatting."""
    
    def test_benchmark_run_dashboard_format(self):
        """Test IRSDLCBenchmarkRun produces CodeContextBench-compatible output."""
        run = IRSDLCBenchmarkRun(
            run_id="IR-SDLC_20251229_120000",
            benchmark_name="IR-SDLC-Bench",
            agent_name="DeepSearchFocusedAgent",
            ir_tool_type=IRToolType.DEEP_SEARCH,
            model_name="anthropic/claude-haiku-4-5-20251001",
            status="completed",
        )
        
        data = run.to_dashboard_format()
        
        # Check CodeContextBench .dashboard_runs/ compatibility
        assert "run_id" in data
        assert "run_type" in data
        assert "benchmark_name" in data
        assert "agent_name" in data
        assert "start_time" in data
        assert "status" in data
        assert data["status"] == "completed"
    
    def test_benchmark_run_harbor_format(self):
        """Test IRSDLCBenchmarkRun produces Harbor-compatible result.json."""
        run = IRSDLCBenchmarkRun(
            run_id="IR-SDLC_20251229_120000",
            ir_tool_type=IRToolType.BASELINE,
            model_name="anthropic/claude-haiku-4-5-20251001",
        )
        
        # Add a task result
        run.task_results.append(IRSDLCTaskResult(
            task_id="ir-sdlc-001",
            task_title="Test",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            repo_name="test/repo",
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
            difficulty="medium",
            ir_tool_type=IRToolType.BASELINE,
            ir_tool_name="BaselineAgent",
            agent_import_path="agents:BaselineAgent",
            model_name="anthropic/claude-haiku-4-5-20251001",
            execution_metrics=AgentExecutionMetrics(success=True, reward=1.0),
        ))
        
        data = run.to_harbor_result_format()
        
        # Check Harbor jobs/ result.json compatibility
        assert "id" in data
        assert "started_at" in data
        assert "finished_at" in data
        assert "n_total_trials" in data
        assert "stats" in data
        assert "n_trials" in data["stats"]
        assert "evals" in data["stats"]
    
    def test_update_aggregates(self):
        """Test aggregate metric calculation."""
        run = IRSDLCBenchmarkRun(run_id="test")
        
        run.task_results.append(IRSDLCTaskResult(
            task_id="task-1",
            task_title="Task 1",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            repo_name="test/repo",
            repo_url="",
            commit_hash="abc",
            difficulty="easy",
            ir_tool_type=IRToolType.BASELINE,
            ir_tool_name="Baseline",
            agent_import_path="agents:Baseline",
            model_name="claude",
            execution_metrics=AgentExecutionMetrics(success=True, reward=1.0),
            finished_at="2025-01-01T00:00:00",
        ))
        
        run.task_results.append(IRSDLCTaskResult(
            task_id="task-2",
            task_title="Task 2",
            sdlc_type=SDLCTaskType.CODE_REVIEW,
            repo_name="test/repo",
            repo_url="",
            commit_hash="abc",
            difficulty="medium",
            ir_tool_type=IRToolType.BASELINE,
            ir_tool_name="Baseline",
            agent_import_path="agents:Baseline",
            model_name="claude",
            execution_metrics=AgentExecutionMetrics(success=False, reward=0.0),
            finished_at="2025-01-01T00:01:00",
        ))
        
        run.update_aggregates()
        
        assert run.total_tasks == 2
        assert run.completed_tasks == 2
        assert run.successful_tasks == 1
        assert run.mean_reward == 0.5


class TestIRComparison:
    """Test A/B comparison schema."""
    
    def test_compute_deltas(self):
        """Test comparative metric calculation."""
        comparison = IRComparison(
            task_id="ir-sdlc-001",
            sdlc_type=SDLCTaskType.BUG_TRIAGE,
            baseline_success=False,
            baseline_reward=0.0,
            baseline_tokens=50000,
            baseline_duration_sec=120.0,
            ir_enhanced_success=True,
            ir_enhanced_reward=1.0,
            ir_enhanced_tokens=30000,
            ir_enhanced_duration_sec=80.0,
            baseline_llm_score=0.6,
            ir_enhanced_llm_score=0.9,
        )
        
        comparison.compute_deltas()
        
        assert comparison.reward_delta == 1.0
        assert comparison.token_efficiency == 0.4  # (50000-30000)/50000
        assert comparison.time_efficiency == pytest.approx(0.333, rel=0.01)
        assert comparison.llm_score_delta == pytest.approx(0.3, rel=0.01)
    
    def test_to_dict_format(self):
        """Test comparison output matches CodeContextBench format."""
        comparison = IRComparison(
            task_id="ir-sdlc-001",
            sdlc_type=SDLCTaskType.ARCHITECTURE_UNDERSTANDING,
            baseline_tokens=50000,
            ir_enhanced_tokens=30000,
            mcp_advantage_category="Architecture Understanding",
            mcp_advantage_explanation="MCP found cross-module dependencies",
        )
        
        data = comparison.to_dict()
        
        # Check CodeContextBench llm_judge_results.json compatibility
        assert data["task"] == "ir-sdlc-001"
        assert data["baseline_tokens"] == 50000
        assert data["mcp_tokens"] == 30000
        assert "evaluation" in data
        assert "baseline" in data["evaluation"]
        assert "mcp" in data["evaluation"]
        assert "mcp_advantage" in data["evaluation"]


class TestCodeContextBenchExporter:
    """Test exporter produces valid output files."""
    
    def test_export_run(self):
        """Test full run export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CodeContextBenchExporter(tmpdir)
            
            run = IRSDLCBenchmarkRun(
                run_id="IR-SDLC_test_run",
                benchmark_name="IR-SDLC-Bench",
                agent_name="TestAgent",
                ir_tool_type=IRToolType.DEEP_SEARCH,
                status="completed",
            )
            
            run.task_results.append(IRSDLCTaskResult(
                task_id="test-001",
                task_title="Test Task",
                sdlc_type=SDLCTaskType.BUG_TRIAGE,
                repo_name="test/repo",
                repo_url="https://github.com/test/repo",
                commit_hash="abc123",
                difficulty="medium",
                ir_tool_type=IRToolType.DEEP_SEARCH,
                ir_tool_name="DeepSearch",
                agent_import_path="agents:DeepSearch",
                model_name="claude",
                execution_metrics=AgentExecutionMetrics(success=True, reward=1.0),
            ))
            
            outputs = exporter.export_run(run)
            
            # Check files were created
            assert outputs["dashboard_run"].exists()
            assert outputs["job_results"].exists()
            assert outputs["task_details"].exists()
            
            # Validate dashboard run JSON
            with open(outputs["dashboard_run"]) as f:
                dashboard_data = json.load(f)
            assert dashboard_data["run_id"] == "IR-SDLC_test_run"
            assert dashboard_data["benchmark_name"] == "IR-SDLC-Bench"
            
            # Validate job results JSON
            with open(outputs["job_results"]) as f:
                job_data = json.load(f)
            assert "stats" in job_data
            assert job_data["n_total_trials"] == 1
    
    def test_export_comparison(self):
        """Test comparison export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CodeContextBenchExporter(tmpdir)
            
            comparisons = [
                IRComparison(
                    task_id="test-001",
                    sdlc_type=SDLCTaskType.BUG_TRIAGE,
                    baseline_success=False,
                    ir_enhanced_success=True,
                ),
                IRComparison(
                    task_id="test-002",
                    sdlc_type=SDLCTaskType.CODE_REVIEW,
                    baseline_success=True,
                    ir_enhanced_success=True,
                ),
            ]
            
            output_path = exporter.export_comparison(comparisons)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                data = json.load(f)
            
            assert len(data) == 2
            assert data[0]["task"] == "test-001"
    
    def test_generate_run_id(self):
        """Test run ID generation."""
        run_id = generate_run_id("IR-SDLC")
        
        assert run_id.startswith("IR-SDLC_")
        # Should contain timestamp
        assert len(run_id) > len("IR-SDLC_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
