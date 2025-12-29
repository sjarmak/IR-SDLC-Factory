"""
Tests for the ablation study runner.

Validates:
1. Task loading from JSONL
2. Simulated agent runner produces valid traces
3. Ablation study generates comparison reports
4. Summary report generation
"""

import json
import tempfile
from pathlib import Path

import pytest

from scripts.run_ablation_study import (
    AGENT_CONFIGS,
    AblationStudyConfig,
    SimulatedAgentConfig,
    SimulatedAgentRunner,
    generate_summary_report,
    load_benchmark_tasks,
    run_ablation_study,
    run_single_comparison,
)
from app.ir_sdlc.agent_metrics import TaskCompletionStatus
from app.ir_sdlc.comparative_analysis import ComparisonReport
from app.ir_sdlc.dashboard_schema import IRToolType


class TestSimulatedAgentRunner:
    """Tests for the simulated agent runner."""
    
    def test_runner_produces_valid_trace(self):
        """Runner should produce a valid AgentExecutionTrace."""
        config = SimulatedAgentConfig(
            name="TestAgent",
            ir_tool_type=IRToolType.BASELINE,
        )
        runner = SimulatedAgentRunner(config, seed=42)
        
        task = {
            "task_id": "test-001",
            "task_type": "bug_triage",
            "difficulty": "medium",
        }
        
        trace = runner.run_task(
            task_id="test-001",
            task_data=task,
            agent_import_path="test.agent.TestAgent",
            model_name="test-model",
            timeout_seconds=300,
        )
        
        assert trace.task_id == "test-001"
        assert trace.agent_name == "TestAgent"
        assert trace.ir_tool_name == "baseline"
        assert trace.total_tokens > 0
        assert trace.wall_clock_time_sec > 0
        assert trace.completion_status in TaskCompletionStatus
    
    def test_runner_respects_seed(self):
        """Same seed should produce same results."""
        config = SimulatedAgentConfig(
            name="TestAgent",
            ir_tool_type=IRToolType.BASELINE,
        )
        
        task = {"task_id": "test", "difficulty": "medium"}
        
        runner1 = SimulatedAgentRunner(config, seed=42)
        trace1 = runner1.run_task("test", task, "path", "model", 300)
        
        runner2 = SimulatedAgentRunner(config, seed=42)
        trace2 = runner2.run_task("test", task, "path", "model", 300)
        
        assert trace1.completed == trace2.completed
        assert trace1.total_tokens == trace2.total_tokens
    
    def test_runner_difficulty_affects_output(self):
        """Harder tasks should have lower success rates."""
        config = SimulatedAgentConfig(
            name="TestAgent",
            ir_tool_type=IRToolType.BASELINE,
            base_success_rate=0.7,
        )
        
        # Run many tasks and compute average success
        easy_successes = 0
        hard_successes = 0
        num_trials = 100
        
        for i in range(num_trials):
            runner = SimulatedAgentRunner(config, seed=i)
            
            easy_trace = runner.run_task(
                "easy", {"difficulty": "easy"}, "path", "model", 300
            )
            if easy_trace.completed:
                easy_successes += 1
            
            runner = SimulatedAgentRunner(config, seed=i + 1000)
            hard_trace = runner.run_task(
                "hard", {"difficulty": "expert"}, "path", "model", 300
            )
            if hard_trace.completed:
                hard_successes += 1
        
        # Easy should have higher success rate than hard
        assert easy_successes > hard_successes
    
    def test_mcp_config_improves_metrics(self):
        """MCP-enhanced config should show better metrics."""
        baseline_config = AGENT_CONFIGS["baseline"]
        treatment_config = AGENT_CONFIGS["deep_search"]
        
        task = {"task_id": "test", "difficulty": "medium"}
        
        baseline_tokens = []
        treatment_tokens = []
        
        for i in range(50):
            baseline_runner = SimulatedAgentRunner(baseline_config, seed=i)
            trace = baseline_runner.run_task("test", task, "path", "model", 300)
            baseline_tokens.append(trace.total_tokens)
            
            treatment_runner = SimulatedAgentRunner(treatment_config, seed=i)
            trace = treatment_runner.run_task("test", task, "path", "model", 300)
            treatment_tokens.append(trace.total_tokens)
        
        # Treatment should use fewer tokens on average
        assert sum(treatment_tokens) / len(treatment_tokens) < sum(baseline_tokens) / len(baseline_tokens)


class TestTaskLoading:
    """Tests for benchmark task loading."""
    
    def test_load_tasks_from_jsonl(self, tmp_path):
        """Should load tasks from JSONL file."""
        benchmark_file = tmp_path / "test.jsonl"
        tasks = [
            {"task_id": "task-1", "task_type": "bug_triage"},
            {"task_id": "task-2", "task_type": "code_review"},
            {"task_id": "task-3", "task_type": "security_audit"},
        ]
        
        with open(benchmark_file, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        
        loaded = load_benchmark_tasks(benchmark_file)
        
        assert len(loaded) == 3
        assert loaded[0]["task_id"] == "task-1"
        assert loaded[2]["task_type"] == "security_audit"
    
    def test_load_tasks_with_limit(self, tmp_path):
        """Should respect max_tasks limit."""
        benchmark_file = tmp_path / "test.jsonl"
        
        with open(benchmark_file, "w") as f:
            for i in range(10):
                f.write(json.dumps({"task_id": f"task-{i}"}) + "\n")
        
        loaded = load_benchmark_tasks(benchmark_file, max_tasks=3)
        
        assert len(loaded) == 3


class TestSingleComparison:
    """Tests for running a single comparison experiment."""
    
    def test_comparison_produces_report(self, tmp_path):
        """run_single_comparison should produce a valid ComparisonReport."""
        # Create test benchmark file
        benchmark_file = tmp_path / "test.jsonl"
        tasks = [
            {"task_id": "task-1", "task_type": "bug_triage", "difficulty": "medium"},
            {"task_id": "task-2", "task_type": "code_review", "difficulty": "hard"},
        ]
        
        with open(benchmark_file, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        
        loaded_tasks = load_benchmark_tasks(benchmark_file)
        
        report = run_single_comparison(
            config_name="test_comparison",
            baseline_config=AGENT_CONFIGS["baseline"],
            treatment_config=AGENT_CONFIGS["deep_search"],
            tasks=loaded_tasks,
            seed=42,
            output_dir=tmp_path / "output",
        )
        
        assert isinstance(report, ComparisonReport)
        assert len(report.task_comparisons) == 2
        assert report.experiment_id == "test_comparison"
        
        # Check report file was created
        report_file = tmp_path / "output" / "test_comparison_report.json"
        assert report_file.exists()


class TestAblationStudy:
    """Tests for the full ablation study."""
    
    def test_ablation_study_runs_all_experiments(self, tmp_path):
        """Ablation study should run all configured experiments."""
        # Create test benchmark file
        benchmark_file = tmp_path / "test.jsonl"
        tasks = [
            {"task_id": f"task-{i}", "task_type": "bug_triage", "difficulty": "medium"}
            for i in range(5)
        ]
        
        with open(benchmark_file, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        
        config = AblationStudyConfig(
            experiments=[
                ("baseline", "deep_search"),
                ("baseline", "sourcegraph_full"),
            ],
            benchmark_path=benchmark_file,
            max_tasks=5,
            output_dir=tmp_path / "output",
            seed=42,
        )
        
        reports = run_ablation_study(config)
        
        assert len(reports) == 2
        assert "baseline_vs_deep_search" in reports
        assert "baseline_vs_sourcegraph_full" in reports
        
        # Each report should have 5 task comparisons
        for report in reports.values():
            assert len(report.task_comparisons) == 5


class TestSummaryReport:
    """Tests for summary report generation."""
    
    def test_summary_report_generation(self, tmp_path):
        """Summary report should contain all experiments and rankings."""
        # Create test benchmark file
        benchmark_file = tmp_path / "test.jsonl"
        tasks = [
            {"task_id": f"task-{i}", "task_type": "bug_triage", "difficulty": "medium"}
            for i in range(5)
        ]
        
        with open(benchmark_file, "w") as f:
            for task in tasks:
                f.write(json.dumps(task) + "\n")
        
        config = AblationStudyConfig(
            experiments=[
                ("baseline", "deep_search"),
            ],
            benchmark_path=benchmark_file,
            max_tasks=5,
            output_dir=tmp_path / "output",
            seed=42,
        )
        
        reports = run_ablation_study(config)
        
        summary_path = tmp_path / "summary.json"
        generate_summary_report(reports, summary_path)
        
        assert summary_path.exists()
        
        with open(summary_path) as f:
            summary = json.load(f)
        
        assert "generated_at" in summary
        assert "experiments" in summary
        assert "rankings" in summary
        assert "baseline_vs_deep_search" in summary["experiments"]


class TestAgentConfigs:
    """Tests for agent configuration definitions."""
    
    def test_all_configs_have_required_fields(self):
        """All agent configs should have required fields."""
        required_fields = [
            "name",
            "ir_tool_type",
            "base_success_rate",
            "base_tokens",
            "base_time_sec",
        ]
        
        for config_name, config in AGENT_CONFIGS.items():
            for field in required_fields:
                assert hasattr(config, field), f"{config_name} missing {field}"
    
    def test_treatment_configs_have_improvements(self):
        """Treatment configs should have improvement settings."""
        # Deep search should have improvements
        deep_search = AGENT_CONFIGS["deep_search"]
        assert deep_search.success_rate_boost > 0
        assert deep_search.token_reduction_pct > 0
        assert deep_search.time_reduction_pct > 0
        
        # Full sourcegraph should have larger improvements
        full = AGENT_CONFIGS["sourcegraph_full"]
        assert full.success_rate_boost > deep_search.success_rate_boost
        assert full.token_reduction_pct > deep_search.token_reduction_pct
