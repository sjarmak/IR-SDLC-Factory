"""
Tests for SWE-bench Integration Adapter.

These tests validate the SWE-bench adapter for:
1. SWE-bench data structure parsing
2. Task conversion from SWE-bench to IR format
3. Environment configuration handling
4. Agent configuration
5. Harbor registry generation
6. Full adapter workflow
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.ir_sdlc.data_structures import RetrievalGranularity
from app.ir_sdlc.swebench_adapter import (
    # Data structures
    SWEBenchInstance,
    SWEBenchDataset,
    SWEBenchSplit,
    
    # Configuration
    IRToolConfig,
    AgentConfig,
    SWEBenchEnvironment,
    STANDARD_ENVIRONMENTS,
    
    # Conversion
    SWEBenchTaskConverter,
    
    # Adapter
    SWEBenchAdapterConfig,
    SWEBenchAdapter,
    
    # Convenience functions
    convert_swebench_to_ir,
    create_agent_config,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_swebench_instance() -> dict:
    """Sample SWE-bench instance data."""
    return {
        "instance_id": "django__django-12345",
        "repo": "django/django",
        "version": "3.2",
        "base_commit": "abc123def456789",
        "problem_statement": "Bug: QuerySet.values() raises TypeError when using annotated fields.\n\nWhen using .values() with annotated fields, the following error occurs:\nTypeError: 'int' object is not subscriptable",
        "hints_text": "The issue is likely in the Values class implementation.",
        "patch": """diff --git a/django/db/models/query.py b/django/db/models/query.py
index abc123..def456 100644
--- a/django/db/models/query.py
+++ b/django/db/models/query.py
@@ -100,7 +100,8 @@ class QuerySet:
-        return self._values
+        if hasattr(self, '_cached_values'):
+            return self._cached_values
+        return self._values
diff --git a/django/db/models/sql/compiler.py b/django/db/models/sql/compiler.py
index 111222..333444 100644
--- a/django/db/models/sql/compiler.py
+++ b/django/db/models/sql/compiler.py
@@ -50,6 +50,7 @@ class SQLCompiler:
+        # Fix for annotated fields
         pass
""",
        "test_patch": "test patch content",
        "FAIL_TO_PASS": "tests.test_queryset.TestValues.test_annotated_values",
        "PASS_TO_PASS": "tests.test_queryset.TestValues.test_basic_values",
        "created_at": "2023-01-15T10:00:00Z",
    }


@pytest.fixture
def sample_swebench_instance_obj(sample_swebench_instance: dict) -> SWEBenchInstance:
    """Sample SWE-bench instance object."""
    return SWEBenchInstance.from_dict(sample_swebench_instance)


@pytest.fixture
def sample_swebench_dataset(sample_swebench_instance: dict) -> SWEBenchDataset:
    """Sample SWE-bench dataset with multiple instances."""
    instance2 = {
        "instance_id": "requests__requests-5678",
        "repo": "psf/requests",
        "version": "2.28",
        "base_commit": "xyz789abc",
        "problem_statement": "Session.request() not handling timeout correctly",
        "hints_text": "",
        "patch": """diff --git a/requests/sessions.py b/requests/sessions.py
index aaa111..bbb222 100644
--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -200,3 +200,4 @@ class Session:
+        # Handle timeout
         pass
""",
    }
    
    return SWEBenchDataset(
        name="test-dataset",
        split=SWEBenchSplit.VERIFIED,
        instances=[
            SWEBenchInstance.from_dict(sample_swebench_instance),
            SWEBenchInstance.from_dict(instance2),
        ],
    )


# =============================================================================
# SWEBenchInstance Tests
# =============================================================================

class TestSWEBenchInstance:
    """Tests for SWEBenchInstance dataclass."""
    
    def test_from_dict(self, sample_swebench_instance: dict):
        """Test creating instance from dict."""
        instance = SWEBenchInstance.from_dict(sample_swebench_instance)
        
        assert instance.instance_id == "django__django-12345"
        assert instance.repo == "django/django"
        assert instance.version == "3.2"
        assert instance.base_commit == "abc123def456789"
        assert "QuerySet.values()" in instance.problem_statement
    
    def test_to_dict(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test serialization to dict."""
        result = sample_swebench_instance_obj.to_dict()
        
        assert result["instance_id"] == "django__django-12345"
        assert result["repo"] == "django/django"
    
    def test_repo_url(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test repo URL generation."""
        assert sample_swebench_instance_obj.repo_url == "https://github.com/django/django.git"
    
    def test_files_changed(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test extracting files changed from patch."""
        files = sample_swebench_instance_obj.files_changed
        
        assert len(files) == 2
        assert "django/db/models/query.py" in files
        assert "django/db/models/sql/compiler.py" in files
    
    def test_files_changed_empty_patch(self):
        """Test files_changed with empty patch."""
        instance = SWEBenchInstance(
            instance_id="test",
            repo="test/test",
            version="1.0",
            base_commit="abc",
            problem_statement="test",
        )
        
        assert instance.files_changed == []


# =============================================================================
# SWEBenchDataset Tests
# =============================================================================

class TestSWEBenchDataset:
    """Tests for SWEBenchDataset."""
    
    def test_load_jsonl(self, sample_swebench_instance: dict):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            dataset = SWEBenchDataset.load_jsonl(path)
            
            assert len(dataset) == 1
            assert dataset.instances[0].instance_id == "django__django-12345"
    
    def test_load_json(self, sample_swebench_instance: dict):
        """Test loading from JSON array file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            with open(path, "w") as f:
                json.dump([sample_swebench_instance], f)
            
            dataset = SWEBenchDataset.load_json(path)
            
            assert len(dataset) == 1
    
    def test_save_jsonl(self, sample_swebench_dataset: SWEBenchDataset):
        """Test saving to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "output.jsonl"
            sample_swebench_dataset.save_jsonl(path)
            
            assert path.exists()
            
            # Verify content
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
    
    def test_filter_by_repo(self, sample_swebench_dataset: SWEBenchDataset):
        """Test filtering by repository."""
        filtered = sample_swebench_dataset.filter_by_repo("django/django")
        
        assert len(filtered) == 1
        assert filtered.instances[0].repo == "django/django"
    
    def test_iteration(self, sample_swebench_dataset: SWEBenchDataset):
        """Test iterating over dataset."""
        repos = [i.repo for i in sample_swebench_dataset]
        
        assert "django/django" in repos
        assert "psf/requests" in repos


# =============================================================================
# Configuration Tests
# =============================================================================

class TestAgentConfig:
    """Tests for AgentConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AgentConfig(
            name="test-agent",
            import_path="agents.test:TestAgent",
        )
        
        assert config.model_id == "claude-3-5-sonnet-20241022"
        assert config.ir_tool == IRToolConfig.BASELINE
        assert config.agent_timeout == 1800
    
    def test_to_dict(self):
        """Test serialization."""
        config = AgentConfig(
            name="test-agent",
            import_path="agents.test:TestAgent",
            ir_tool=IRToolConfig.SOURCEGRAPH_MCP,
            ir_tool_endpoint="http://localhost:8080",
        )
        
        result = config.to_dict()
        
        assert result["name"] == "test-agent"
        assert result["ir_tool"] == "sourcegraph_mcp"
        assert result["ir_tool_endpoint"] == "http://localhost:8080"


class TestSWEBenchEnvironment:
    """Tests for SWEBenchEnvironment."""
    
    def test_default_values(self):
        """Test default environment values."""
        env = SWEBenchEnvironment(name="test-env")
        
        assert env.base_image == "python:3.11-slim"
        assert env.cpus == 4
        assert env.memory_mb == 8192
        assert env.network_access is True
    
    def test_to_environment_config(self):
        """Test conversion to EnvironmentConfig."""
        env = SWEBenchEnvironment(
            name="custom-env",
            cpus=8,
            memory_mb=16384,
        )
        
        config = env.to_environment_config()
        
        assert config.cpus == 8
        assert config.memory_mb == 16384
    
    def test_standard_environments(self):
        """Test standard environment configurations."""
        assert "baseline" in STANDARD_ENVIRONMENTS
        assert "deepsearch" in STANDARD_ENVIRONMENTS
        assert "sourcegraph" in STANDARD_ENVIRONMENTS


# =============================================================================
# Task Converter Tests
# =============================================================================

class TestSWEBenchTaskConverter:
    """Tests for SWEBenchTaskConverter."""
    
    def test_convert_instance(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test converting a single instance."""
        converter = SWEBenchTaskConverter()
        
        task = converter.convert_instance(sample_swebench_instance_obj)
        
        assert task.task_type == "bug_fix"
        assert task.repo_name == "django/django"
        assert task.commit_hash == "abc123def456789"
        assert "QuerySet.values()" in task.query
        assert task.ground_truth is not None
        assert len(task.ground_truth.locations) == 2
    
    def test_convert_dataset(self, sample_swebench_dataset: SWEBenchDataset):
        """Test converting entire dataset."""
        converter = SWEBenchTaskConverter()
        
        ir_dataset = converter.convert_dataset(sample_swebench_dataset)
        
        assert ir_dataset.name == "ir-test-dataset"
        assert len(ir_dataset.tasks) == 2
        assert ir_dataset.metadata["source"] == "swe-bench"
    
    def test_ground_truth_extraction(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test ground truth extraction from patch."""
        converter = SWEBenchTaskConverter()
        
        task = converter.convert_instance(sample_swebench_instance_obj)
        gt = task.ground_truth
        
        assert gt.source == "swe-bench-patch"
        assert gt.confidence == 1.0
        
        file_paths = [loc.file_path for loc in gt.locations]
        assert "django/db/models/query.py" in file_paths
    
    def test_difficulty_estimation(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test difficulty estimation."""
        converter = SWEBenchTaskConverter()
        
        task = converter.convert_instance(sample_swebench_instance_obj)
        
        # 2 files, moderate patch size -> should be medium or easy
        assert task.difficulty in ["easy", "medium"]
    
    def test_tags_generation(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test tag generation."""
        converter = SWEBenchTaskConverter()
        
        task = converter.convert_instance(sample_swebench_instance_obj)
        
        assert "swe-bench" in task.tags
        assert "bug-fix" in task.tags
        assert "django-django" in task.tags
        assert "python" in task.tags
    
    def test_function_granularity(self, sample_swebench_instance_obj: SWEBenchInstance):
        """Test function-level granularity extraction."""
        converter = SWEBenchTaskConverter(
            granularity=RetrievalGranularity.FUNCTION,
            extract_functions=True,
        )
        
        task = converter.convert_instance(sample_swebench_instance_obj)
        
        assert task.ground_truth.granularity == RetrievalGranularity.FUNCTION


# =============================================================================
# SWEBenchAdapter Tests
# =============================================================================

class TestSWEBenchAdapter:
    """Tests for SWEBenchAdapter."""
    
    def test_load_dataset(self, sample_swebench_instance: dict):
        """Test loading dataset through adapter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            config = SWEBenchAdapterConfig(dataset_path=path)
            adapter = SWEBenchAdapter(config)
            
            dataset = adapter.load_dataset()
            
            assert len(dataset) == 1
    
    def test_load_and_convert(self, sample_swebench_instance: dict):
        """Test full load and convert workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            config = SWEBenchAdapterConfig(dataset_path=path)
            adapter = SWEBenchAdapter(config)
            
            ir_dataset = adapter.load_and_convert()
            
            assert len(ir_dataset.tasks) == 1
            assert ir_dataset.tasks[0].task_type == "bug_fix"
    
    def test_generate_registry(self, sample_swebench_instance: dict):
        """Test Harbor registry generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            config = SWEBenchAdapterConfig(
                dataset_path=path,
                git_url="https://github.com/test/repo.git",
            )
            adapter = SWEBenchAdapter(config)
            adapter.load_and_convert()
            
            registry = adapter.generate_registry()
            
            assert "datasets" in registry
            assert len(registry["datasets"]) == 1
    
    def test_save_ir_dataset(self, sample_swebench_instance: dict):
        """Test saving converted IR dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            with open(input_path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            output_path = Path(tmpdir) / "output.jsonl"
            
            config = SWEBenchAdapterConfig(dataset_path=input_path)
            adapter = SWEBenchAdapter(config)
            adapter.load_and_convert()
            
            saved_path = adapter.save_ir_dataset(output_path)
            
            assert saved_path.exists()
    
    def test_get_repos(self, sample_swebench_instance: dict):
        """Test getting repository list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            config = SWEBenchAdapterConfig(dataset_path=path)
            adapter = SWEBenchAdapter(config)
            adapter.load_dataset()
            
            repos = adapter.get_repos()
            
            assert "django/django" in repos
    
    def test_filter_by_repo(self, sample_swebench_instance: dict):
        """Test filtering adapter by repository."""
        instance2 = sample_swebench_instance.copy()
        instance2["instance_id"] = "requests__requests-5678"
        instance2["repo"] = "psf/requests"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
                f.write(json.dumps(instance2) + "\n")
            
            config = SWEBenchAdapterConfig(dataset_path=path)
            adapter = SWEBenchAdapter(config)
            adapter.load_dataset()
            
            filtered = adapter.filter_by_repo("django/django")
            
            assert len(filtered._swe_dataset) == 1
    
    def test_create_evaluation_config(self, sample_swebench_instance: dict):
        """Test creating evaluation configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            with open(path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            config = SWEBenchAdapterConfig(dataset_path=path)
            adapter = SWEBenchAdapter(config)
            adapter.load_and_convert()
            
            agent_config = AgentConfig(
                name="test-agent",
                import_path="agents.test:Agent",
                ir_tool=IRToolConfig.SOURCEGRAPH_MCP,
            )
            
            eval_config = adapter.create_evaluation_config(agent_config)
            
            assert eval_config["agent"]["name"] == "test-agent"
            assert eval_config["agent"]["ir_tool"] == "sourcegraph_mcp"
            assert eval_config["dataset"]["task_count"] == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_convert_swebench_to_ir(self, sample_swebench_dataset: SWEBenchDataset):
        """Test convert_swebench_to_ir function."""
        ir_dataset = convert_swebench_to_ir(sample_swebench_dataset)
        
        assert len(ir_dataset.tasks) == 2
        assert all(t.task_type == "bug_fix" for t in ir_dataset.tasks)
    
    def test_create_agent_config(self):
        """Test create_agent_config function."""
        config = create_agent_config(
            name="my-agent",
            import_path="agents.my:Agent",
            ir_tool=IRToolConfig.DEEPSEARCH_MCP,
        )
        
        assert config.name == "my-agent"
        assert config.ir_tool == IRToolConfig.DEEPSEARCH_MCP


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self, sample_swebench_instance: dict):
        """Test complete adapter workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input file
            input_path = tmpdir / "swebench.jsonl"
            with open(input_path, "w") as f:
                f.write(json.dumps(sample_swebench_instance) + "\n")
            
            # Configure adapter
            config = SWEBenchAdapterConfig(
                dataset_path=input_path,
                dataset_split=SWEBenchSplit.VERIFIED,
                granularity=RetrievalGranularity.FILE,
                output_dir=tmpdir,
                git_url="https://github.com/test/benchmarks.git",
            )
            
            adapter = SWEBenchAdapter(config)
            
            # Load and convert
            ir_dataset = adapter.load_and_convert()
            
            # Verify conversion
            assert len(ir_dataset.tasks) == 1
            task = ir_dataset.tasks[0]
            assert task.repo_name == "django/django"
            assert task.task_type == "bug_fix"
            assert len(task.ground_truth.locations) == 2
            
            # Save IR dataset
            ir_path = adapter.save_ir_dataset(tmpdir / "ir-dataset.jsonl")
            assert ir_path.exists()
            
            # Generate registry
            registry = adapter.generate_registry()
            assert len(registry["datasets"]) == 1
            
            # Create evaluation config
            agent = AgentConfig(
                name="eval-agent",
                import_path="agents.eval:EvalAgent",
                ir_tool=IRToolConfig.BASELINE,
            )
            
            eval_config = adapter.create_evaluation_config(agent)
            
            assert eval_config["agent"]["name"] == "eval-agent"
            assert eval_config["dataset"]["task_count"] == 1
    
    def test_multi_repo_workflow(self):
        """Test workflow with multiple repositories."""
        instances = [
            {
                "instance_id": "django__django-1",
                "repo": "django/django",
                "version": "3.2",
                "base_commit": "abc123",
                "problem_statement": "Django issue 1",
                "patch": "diff --git a/file1.py b/file1.py\n+test",
            },
            {
                "instance_id": "django__django-2",
                "repo": "django/django",
                "version": "3.2",
                "base_commit": "def456",
                "problem_statement": "Django issue 2",
                "patch": "diff --git a/file2.py b/file2.py\n+test",
            },
            {
                "instance_id": "requests__requests-1",
                "repo": "psf/requests",
                "version": "2.28",
                "base_commit": "ghi789",
                "problem_statement": "Requests issue 1",
                "patch": "diff --git a/requests/api.py b/requests/api.py\n+test",
            },
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create input file
            input_path = tmpdir / "swebench.jsonl"
            with open(input_path, "w") as f:
                for inst in instances:
                    f.write(json.dumps(inst) + "\n")
            
            config = SWEBenchAdapterConfig(
                dataset_path=input_path,
                git_url="https://github.com/test/bench.git",
            )
            
            adapter = SWEBenchAdapter(config)
            adapter.load_dataset()
            
            # Check repos
            repos = adapter.get_repos()
            assert len(repos) == 2
            assert "django/django" in repos
            assert "psf/requests" in repos
            
            # Filter to django
            django_adapter = adapter.filter_by_repo("django/django")
            assert len(django_adapter._swe_dataset) == 2
            
            # Convert filtered
            ir_dataset = django_adapter.convert_to_ir_dataset()
            assert len(ir_dataset.tasks) == 2
            assert all(t.repo_name == "django/django" for t in ir_dataset.tasks)
