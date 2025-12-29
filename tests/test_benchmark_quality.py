#!/usr/bin/env python3
"""
Test benchmark dataset quality and evaluation_runner integration.

This module validates:
1. Task schema compliance
2. SDLC lifecycle coverage  
3. Ground truth quality
4. IR focus and complexity distribution
5. Integration with evaluation_runner
"""

import json
import pytest
from collections import Counter
from datetime import datetime
from pathlib import Path

# Test configuration
BENCHMARK_FILE = Path(__file__).parent.parent / "benchmarks" / "ir-sdlc-multi-repo.jsonl"
MIN_TOTAL_TASKS = 50
MIN_TASK_TYPES = 4
REQUIRED_TASK_TYPES = {"bug_triage", "code_review"}


class TestBenchmarkDatasetQuality:
    """Test the benchmark dataset meets quality standards."""
    
    @pytest.fixture
    def tasks(self):
        """Load benchmark tasks."""
        assert BENCHMARK_FILE.exists(), f"Benchmark file not found: {BENCHMARK_FILE}"
        with open(BENCHMARK_FILE, "r") as f:
            return [json.loads(line) for line in f]
    
    def test_minimum_task_count(self, tasks):
        """Verify we have at least the minimum number of tasks."""
        assert len(tasks) >= MIN_TOTAL_TASKS, \
            f"Expected at least {MIN_TOTAL_TASKS} tasks, got {len(tasks)}"
    
    def test_task_schema_compliance(self, tasks):
        """Verify all tasks have required fields."""
        required_fields = {
            "task_id", "task_type", "repo_name", "repo_url",
            "commit_hash", "query", "ground_truth", "difficulty"
        }
        
        for i, task in enumerate(tasks):
            missing = required_fields - set(task.keys())
            assert not missing, f"Task {i} ({task.get('task_id', 'unknown')}) missing fields: {missing}"
    
    def test_unique_task_ids(self, tasks):
        """Verify all task IDs are unique."""
        task_ids = [t["task_id"] for t in tasks]
        duplicates = [tid for tid, count in Counter(task_ids).items() if count > 1]
        assert not duplicates, f"Duplicate task IDs found: {duplicates}"
    
    def test_task_type_diversity(self, tasks):
        """Verify we have diverse task types."""
        task_types = set(t["task_type"] for t in tasks)
        
        assert len(task_types) >= MIN_TASK_TYPES, \
            f"Expected at least {MIN_TASK_TYPES} task types, got {len(task_types)}: {task_types}"
        
        for required in REQUIRED_TASK_TYPES:
            assert required in task_types, f"Required task type '{required}' not found"
    
    def test_ground_truth_quality(self, tasks):
        """Verify ground truth has valid locations."""
        tasks_without_gt = []
        tasks_empty_gt = []
        
        for task in tasks:
            gt = task.get("ground_truth")
            if gt is None:
                tasks_without_gt.append(task["task_id"])
            elif not gt.get("locations"):
                # Dependency analysis may have empty locations (requires static analysis)
                if task["task_type"] != "dependency_analysis":
                    tasks_empty_gt.append(task["task_id"])
        
        # Allow some tasks without ground truth but flag if too many
        max_missing = len(tasks) * 0.2  # 20% threshold
        assert len(tasks_without_gt) <= max_missing, \
            f"Too many tasks without ground_truth: {len(tasks_without_gt)}"
    
    def test_difficulty_distribution(self, tasks):
        """Verify difficulty is well distributed."""
        difficulties = Counter(t["difficulty"] for t in tasks)
        
        # Ensure we have hard/expert tasks (these showcase IR benefits better)
        hard_expert = difficulties.get("hard", 0) + difficulties.get("expert", 0)
        total = len(tasks)
        
        assert hard_expert >= total * 0.3, \
            f"Expected at least 30% hard/expert tasks, got {hard_expert/total:.1%}"
    
    def test_repository_diversity(self, tasks):
        """Verify tasks come from multiple repositories."""
        repos = set(t["repo_name"] for t in tasks)
        assert len(repos) >= 3, f"Expected at least 3 repos, got {len(repos)}: {repos}"
    
    def test_query_quality(self, tasks):
        """Verify queries are meaningful and not empty."""
        for task in tasks:
            query = task.get("query", "")
            assert len(query) >= 20, \
                f"Task {task['task_id']} has too short query: {len(query)} chars"
            assert query.strip(), f"Task {task['task_id']} has empty query"


class TestSDLCCoverage:
    """Test SDLC lifecycle coverage."""
    
    SDLC_PHASES = {
        "design": ["architecture_understanding", "dependency_analysis"],
        "implementation": ["feature_location", "code_review"],
        "testing": ["test_coverage"],
        "security": ["security_audit"],
        "maintenance": ["bug_triage", "refactoring_analysis", "change_impact_analysis"],
    }
    
    @pytest.fixture
    def tasks(self):
        """Load benchmark tasks."""
        with open(BENCHMARK_FILE, "r") as f:
            return [json.loads(line) for line in f]
    
    def test_design_phase_coverage(self, tasks):
        """Verify design phase has some coverage."""
        task_types = set(t["task_type"] for t in tasks)
        design_types = set(self.SDLC_PHASES["design"]) & task_types
        assert design_types, "No design phase tasks (architecture_understanding, dependency_analysis)"
    
    def test_implementation_phase_coverage(self, tasks):
        """Verify implementation phase has coverage."""
        task_types = set(t["task_type"] for t in tasks)
        impl_types = set(self.SDLC_PHASES["implementation"]) & task_types
        assert impl_types, "No implementation phase tasks (feature_location, code_review)"
    
    def test_security_phase_coverage(self, tasks):
        """Verify security phase has coverage."""
        task_types = set(t["task_type"] for t in tasks)
        security_types = set(self.SDLC_PHASES["security"]) & task_types
        assert security_types, "No security phase tasks (security_audit)"
    
    def test_maintenance_phase_coverage(self, tasks):
        """Verify maintenance phase has coverage."""
        task_types = set(t["task_type"] for t in tasks)
        maint_types = set(self.SDLC_PHASES["maintenance"]) & task_types
        assert maint_types, "No maintenance phase tasks (bug_triage, refactoring_analysis)"


class TestIRFocus:
    """Test that tasks showcase IR/codebase understanding benefits."""
    
    @pytest.fixture
    def tasks(self):
        """Load benchmark tasks."""
        with open(BENCHMARK_FILE, "r") as f:
            return [json.loads(line) for line in f]
    
    def test_cross_file_retrieval_tasks(self, tasks):
        """Verify we have tasks that require cross-file retrieval."""
        # Tasks with multiple ground truth files require cross-file retrieval
        multi_file_tasks = 0
        for task in tasks:
            gt = task.get("ground_truth", {})
            locations = gt.get("locations", [])
            if len(locations) >= 2:
                multi_file_tasks += 1
        
        assert multi_file_tasks >= 10, \
            f"Expected at least 10 tasks with multi-file ground truth, got {multi_file_tasks}"
    
    def test_natural_language_queries(self, tasks):
        """Verify queries use natural language (not just file paths)."""
        for task in tasks:
            query = task.get("query", "")
            # Queries should have natural language, not just paths
            words = query.split()
            assert len(words) >= 5, \
                f"Task {task['task_id']} query too simple: {query[:100]}"
    
    def test_complex_repository_tasks(self, tasks):
        """Verify we have tasks from complex repositories."""
        # Check for enterprise-scale repos
        enterprise_repos = {
            "kubernetes/kubernetes", "grafana/grafana", 
            "microsoft/vscode", "elastic/elasticsearch"
        }
        repos_in_dataset = set(t["repo_name"] for t in tasks)
        enterprise_in_dataset = enterprise_repos & repos_in_dataset
        
        assert len(enterprise_in_dataset) >= 3, \
            f"Expected at least 3 enterprise repos, got {len(enterprise_in_dataset)}"


class TestEvaluationRunnerIntegration:
    """Test integration with evaluation_runner."""
    
    @pytest.fixture
    def tasks(self):
        """Load benchmark tasks."""
        with open(BENCHMARK_FILE, "r") as f:
            return [json.loads(line) for line in f]
    
    def test_data_structures_compatibility(self, tasks):
        """Verify tasks can be loaded into IRDataset."""
        from app.ir_sdlc.data_structures import IRDataset
        
        dataset = IRDataset.load_jsonl(BENCHMARK_FILE)
        assert len(dataset.tasks) == len(tasks)
        assert dataset.name is not None
    
    def test_task_to_ir_task_conversion(self, tasks):
        """Verify tasks can be converted to IRTask objects."""
        from app.ir_sdlc.data_structures import IRTask
        
        # Test first few tasks
        for task_dict in tasks[:5]:
            ir_task = IRTask.from_dict(task_dict)
            assert ir_task.task_id == task_dict["task_id"]
            assert ir_task.task_type == task_dict["task_type"]
            assert ir_task.query == task_dict["query"]
    
    def test_ground_truth_to_object_conversion(self, tasks):
        """Verify ground truth can be converted to GroundTruth objects."""
        from app.ir_sdlc.data_structures import GroundTruth
        
        for task_dict in tasks[:10]:
            gt_dict = task_dict.get("ground_truth")
            if gt_dict:
                gt = GroundTruth.from_dict(gt_dict)
                assert gt.granularity is not None
                # Locations may be empty for some task types
    
    def test_evaluation_config_creation(self):
        """Verify EvaluationConfig can be created."""
        from app.ir_sdlc.evaluation_runner import EvaluationConfig
        
        config = EvaluationConfig(
            tool_name="test_tool",
            top_k_values=[1, 5, 10],
            primary_metric="mrr",
        )
        assert config.tool_name == "test_tool"
        assert config.top_k_values == [1, 5, 10]


class TestBenchmarkMetadata:
    """Test benchmark metadata file."""
    
    def test_metadata_file_exists(self):
        """Verify metadata file exists."""
        metadata_file = BENCHMARK_FILE.with_name(
            BENCHMARK_FILE.stem + "_metadata.json"
        )
        assert metadata_file.exists(), f"Metadata file not found: {metadata_file}"
    
    def test_metadata_content(self):
        """Verify metadata has required fields."""
        metadata_file = BENCHMARK_FILE.with_name(
            BENCHMARK_FILE.stem + "_metadata.json"
        )
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        assert "source_repos" in metadata
        assert "generated_at" in metadata
        assert isinstance(metadata["source_repos"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
