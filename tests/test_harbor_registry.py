"""
Tests for Harbor Registry Generator.

These tests validate the HarborRegistryGenerator implementation for:
1. EnvironmentConfig, MetricConfig, TaskEntry, DatasetEntry structures
2. Registry generation from IRDataset objects
3. Registry generation from JSONL files
4. Registry validation
5. Task directory generation integration
"""

import json
import tempfile
from pathlib import Path

import pytest

from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
)
from app.ir_sdlc.harbor_registry import (
    EnvironmentConfig,
    MetricConfig,
    TaskEntry,
    DatasetEntry,
    HarborRegistryConfig,
    HarborRegistryGenerator,
    generate_registry_from_jsonl,
    generate_registry_from_datasets,
    load_registry,
    validate_registry,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ir_task() -> IRTask:
    """Create a sample IR task for testing."""
    return IRTask(
        task_id="test-repo-bug_triage-abc123",
        task_type="bug_triage",
        repo_name="test-org/test-repo",
        repo_url="https://github.com/test-org/test-repo.git",
        commit_hash="abc123def456",
        query="Fix null pointer exception in UserService.getUser()",
        context={"stack_trace": "NullPointerException at line 42"},
        ground_truth=GroundTruth(
            locations=[
                CodeLocation(file_path="src/UserService.java", start_line=40, end_line=50),
                CodeLocation(file_path="src/User.java", start_line=10, end_line=20),
            ],
            granularity=RetrievalGranularity.FUNCTION,
        ),
        difficulty="medium",
        tags=["bug", "null-pointer"],
    )


@pytest.fixture
def sample_ir_dataset(sample_ir_task: IRTask) -> IRDataset:
    """Create a sample IR dataset for testing."""
    task2 = IRTask(
        task_id="test-repo-code_review-def456",
        task_type="code_review",
        repo_name="test-org/test-repo",
        repo_url="https://github.com/test-org/test-repo.git",
        commit_hash="def456ghi789",
        query="Review authentication changes in PR #123",
        difficulty="hard",
        tags=["security", "review"],
    )
    
    return IRDataset(
        name="test-benchmark",
        version="1.0.0",
        description="Test benchmark for unit tests",
        tasks=[sample_ir_task, task2],
        metadata={"tags": ["test", "unit-test"]},
    )


@pytest.fixture
def sample_registry_config() -> HarborRegistryConfig:
    """Create a sample registry config for testing."""
    return HarborRegistryConfig(
        git_url="https://github.com/test-org/ir-sdlc-bench.git",
        git_commit_id="main",
        base_path="datasets",
        registry_name="test-registry",
        registry_version="1.0.0",
    )


# =============================================================================
# EnvironmentConfig Tests
# =============================================================================

class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""
    
    def test_default_values(self):
        """Test that EnvironmentConfig has sensible defaults."""
        config = EnvironmentConfig()
        
        assert config.environment_type == "docker"
        assert config.base_image == "python:3.11-slim"
        assert config.cpus == 2
        assert config.memory_mb == 4096
        assert config.build_timeout == 600
        assert config.network_access is False
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = EnvironmentConfig(
            environment_type="daytona",
            cpus=4,
            memory_mb=8192,
            gpu_required=True,
        )
        
        result = config.to_dict()
        
        assert result["type"] == "daytona"
        assert result["resources"]["cpus"] == 4
        assert result["resources"]["memory_mb"] == 8192
        assert result["resources"]["gpu"] is True
        assert result["timeouts"]["build"] == 600
    
    def test_custom_timeouts(self):
        """Test custom timeout configuration."""
        config = EnvironmentConfig(
            build_timeout=1200,
            agent_timeout=3600,
            verifier_timeout=600,
        )
        
        result = config.to_dict()
        
        assert result["timeouts"]["build"] == 1200
        assert result["timeouts"]["agent"] == 3600
        assert result["timeouts"]["verifier"] == 600


# =============================================================================
# MetricConfig Tests
# =============================================================================

class TestMetricConfig:
    """Tests for MetricConfig dataclass."""
    
    def test_default_values(self):
        """Test that MetricConfig has sensible defaults."""
        config = MetricConfig()
        
        assert config.metric_type == "mean"
        assert config.primary is False
        assert config.weight == 1.0
        assert config.threshold is None
    
    def test_to_dict_minimal(self):
        """Test minimal serialization."""
        config = MetricConfig(metric_type="max")
        
        result = config.to_dict()
        
        assert result == {"type": "max"}
        assert "primary" not in result
        assert "weight" not in result
    
    def test_to_dict_full(self):
        """Test full serialization with all fields."""
        config = MetricConfig(
            metric_type="weighted_mean",
            primary=True,
            weight=0.5,
            threshold=0.8,
        )
        
        result = config.to_dict()
        
        assert result["type"] == "weighted_mean"
        assert result["primary"] is True
        assert result["weight"] == 0.5
        assert result["threshold"] == 0.8


# =============================================================================
# TaskEntry Tests
# =============================================================================

class TestTaskEntry:
    """Tests for TaskEntry dataclass."""
    
    def test_minimal_task_entry(self):
        """Test creating a minimal task entry."""
        entry = TaskEntry(
            name="test-task",
            git_url="https://github.com/test/repo.git",
            git_commit_id="abc123",
            path="datasets/test/test-task",
        )
        
        result = entry.to_dict()
        
        assert result["name"] == "test-task"
        assert result["git_url"] == "https://github.com/test/repo.git"
        assert result["git_commit_id"] == "abc123"
        assert result["path"] == "datasets/test/test-task"
        assert "difficulty" not in result
    
    def test_full_task_entry(self):
        """Test creating a full task entry with all fields."""
        entry = TaskEntry(
            name="test-task",
            git_url="https://github.com/test/repo.git",
            git_commit_id="abc123",
            path="datasets/test/test-task",
            difficulty="hard",
            task_type="bug_triage",
            tags=["bug", "critical"],
            repo_name="test/repo",
            agent_timeout=3600,
        )
        
        result = entry.to_dict()
        
        assert result["difficulty"] == "hard"
        assert result["task_type"] == "bug_triage"
        assert result["tags"] == ["bug", "critical"]
        assert result["repo_name"] == "test/repo"
        assert result["agent_timeout"] == 3600
    
    def test_from_ir_task(self, sample_ir_task: IRTask):
        """Test creating TaskEntry from IRTask."""
        entry = TaskEntry.from_ir_task(
            task=sample_ir_task,
            git_url="https://github.com/benchmark/repo.git",
            git_commit_id="main",
            base_path="datasets",
            dataset_name="test-benchmark",
        )
        
        assert entry.name == sample_ir_task.task_id
        assert entry.difficulty == "medium"
        assert entry.task_type == "bug_triage"
        assert entry.repo_name == "test-org/test-repo"
        assert "datasets/test-benchmark/" in entry.path


# =============================================================================
# DatasetEntry Tests
# =============================================================================

class TestDatasetEntry:
    """Tests for DatasetEntry dataclass."""
    
    def test_from_ir_dataset(self, sample_ir_dataset: IRDataset):
        """Test creating DatasetEntry from IRDataset."""
        entry = DatasetEntry.from_ir_dataset(
            dataset=sample_ir_dataset,
            git_url="https://github.com/benchmark/repo.git",
            git_commit_id="v1.0.0",
        )
        
        assert entry.name == "test-benchmark"
        assert entry.version == "1.0.0"
        assert entry.description == "Test benchmark for unit tests"
        assert len(entry.tasks) == 2
        assert entry.category == "information_retrieval"
        assert entry.tags == ["test", "unit-test"]
    
    def test_to_dict(self, sample_ir_dataset: IRDataset):
        """Test serialization to dictionary."""
        entry = DatasetEntry.from_ir_dataset(
            dataset=sample_ir_dataset,
            git_url="https://github.com/benchmark/repo.git",
            git_commit_id="v1.0.0",
        )
        
        result = entry.to_dict()
        
        assert result["name"] == "test-benchmark"
        assert result["version"] == "1.0.0"
        assert len(result["tasks"]) == 2
        assert len(result["metrics"]) == 2  # Default mean + max
        assert result["author"] == "IR-SDLC-Bench"
    
    def test_with_environment(self, sample_ir_dataset: IRDataset):
        """Test DatasetEntry with custom environment."""
        env = EnvironmentConfig(cpus=4, memory_mb=8192)
        entry = DatasetEntry.from_ir_dataset(
            dataset=sample_ir_dataset,
            git_url="https://github.com/benchmark/repo.git",
            git_commit_id="v1.0.0",
            environment=env,
        )
        
        result = entry.to_dict()
        
        assert "environment" in result
        assert result["environment"]["resources"]["cpus"] == 4


# =============================================================================
# HarborRegistryGenerator Tests
# =============================================================================

class TestHarborRegistryGenerator:
    """Tests for HarborRegistryGenerator class."""
    
    def test_init(self, sample_registry_config: HarborRegistryConfig):
        """Test generator initialization."""
        generator = HarborRegistryGenerator(sample_registry_config)
        
        assert generator.config == sample_registry_config
        assert len(generator.datasets) == 0
    
    def test_add_dataset(
        self,
        sample_registry_config: HarborRegistryConfig,
        sample_ir_dataset: IRDataset,
    ):
        """Test adding a dataset to the generator."""
        generator = HarborRegistryGenerator(sample_registry_config)
        
        entry = generator.add_dataset(sample_ir_dataset)
        
        assert len(generator.datasets) == 1
        assert entry.name == "test-benchmark"
        assert len(entry.tasks) == 2
    
    def test_generate(
        self,
        sample_registry_config: HarborRegistryConfig,
        sample_ir_dataset: IRDataset,
    ):
        """Test generating the full registry."""
        generator = HarborRegistryGenerator(sample_registry_config)
        generator.add_dataset(sample_ir_dataset)
        
        registry = generator.generate()
        
        assert registry["name"] == "test-registry"
        assert registry["version"] == "1.0.0"
        assert "generated_at" in registry
        assert len(registry["datasets"]) == 1
        assert registry["datasets"][0]["name"] == "test-benchmark"
    
    def test_save_registry(
        self,
        sample_registry_config: HarborRegistryConfig,
        sample_ir_dataset: IRDataset,
    ):
        """Test saving registry to file."""
        generator = HarborRegistryGenerator(sample_registry_config)
        generator.add_dataset(sample_ir_dataset)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "registry.json"
            saved_path = generator.save(output_path)
            
            assert saved_path.exists()
            
            with open(saved_path) as f:
                loaded = json.load(f)
            
            assert loaded["name"] == "test-registry"
            assert len(loaded["datasets"]) == 1
    
    def test_add_dataset_from_jsonl(self, sample_registry_config: HarborRegistryConfig):
        """Test adding dataset from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test JSONL file
            jsonl_path = Path(tmpdir) / "test.jsonl"
            task_data = {
                "task_id": "test-task-1",
                "task_type": "bug_triage",
                "repo_name": "test/repo",
                "repo_url": "https://github.com/test/repo.git",
                "commit_hash": "abc123",
                "query": "Test query",
                "difficulty": "easy",
                "tags": ["test"],
            }
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(task_data) + "\n")
            
            generator = HarborRegistryGenerator(sample_registry_config)
            entry = generator.add_dataset_from_jsonl(
                jsonl_path=jsonl_path,
                name="jsonl-dataset",
                version="2.0.0",
                description="Dataset from JSONL",
            )
            
            assert entry.name == "jsonl-dataset"
            assert entry.version == "2.0.0"
            assert len(entry.tasks) == 1


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_generate_registry_from_datasets(self, sample_ir_dataset: IRDataset):
        """Test generate_registry_from_datasets function."""
        registry = generate_registry_from_datasets(
            datasets=[sample_ir_dataset],
            git_url="https://github.com/test/repo.git",
            git_commit_id="main",
            registry_name="test-registry",
        )
        
        assert registry["name"] == "test-registry"
        assert len(registry["datasets"]) == 1
    
    def test_generate_registry_from_jsonl(self):
        """Test generate_registry_from_jsonl function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSONL
            jsonl_path = Path(tmpdir) / "benchmark.jsonl"
            task = {
                "task_id": "task-1",
                "task_type": "code_review",
                "repo_name": "org/repo",
                "repo_url": "https://github.com/org/repo.git",
                "commit_hash": "xyz789",
                "query": "Review this code",
            }
            with open(jsonl_path, "w") as f:
                f.write(json.dumps(task) + "\n")
            
            registry = generate_registry_from_jsonl(
                jsonl_path=jsonl_path,
                git_url="https://github.com/test/bench.git",
                name="my-benchmark",
            )
            
            assert len(registry["datasets"]) == 1
            assert registry["datasets"][0]["name"] == "my-benchmark"
    
    def test_load_and_validate_registry(self, sample_ir_dataset: IRDataset):
        """Test loading and validating a registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and save registry
            registry = generate_registry_from_datasets(
                datasets=[sample_ir_dataset],
                git_url="https://github.com/test/repo.git",
                output_path=Path(tmpdir) / "registry.json",
            )
            
            # Load it back
            loaded = load_registry(Path(tmpdir) / "registry.json")
            
            # Validate
            errors = validate_registry(loaded)
            
            assert len(errors) == 0


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for registry validation."""
    
    def test_validate_valid_registry(self):
        """Test validation passes for valid registry."""
        registry = {
            "name": "test",
            "version": "1.0.0",
            "datasets": [
                {
                    "name": "dataset-1",
                    "version": "1.0.0",
                    "tasks": [
                        {
                            "name": "task-1",
                            "git_url": "https://github.com/test/repo.git",
                            "git_commit_id": "abc123",
                            "path": "datasets/task-1",
                        }
                    ],
                }
            ],
        }
        
        errors = validate_registry(registry)
        
        assert len(errors) == 0
    
    def test_validate_missing_required_fields(self):
        """Test validation catches missing required fields."""
        registry = {"version": "1.0.0"}
        
        errors = validate_registry(registry)
        
        assert any("name" in e for e in errors)
        assert any("datasets" in e for e in errors)
    
    def test_validate_missing_task_fields(self):
        """Test validation catches missing task fields."""
        registry = {
            "name": "test",
            "version": "1.0.0",
            "datasets": [
                {
                    "name": "dataset-1",
                    "version": "1.0.0",
                    "tasks": [
                        {"name": "task-1"}  # Missing git_url, git_commit_id, path
                    ],
                }
            ],
        }
        
        errors = validate_registry(registry)
        
        assert any("git_url" in e for e in errors)
        assert any("git_commit_id" in e for e in errors)
        assert any("path" in e for e in errors)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_full_workflow(self, sample_ir_dataset: IRDataset):
        """Test complete workflow: create, generate, save, load, validate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 1. Create config
            config = HarborRegistryConfig(
                git_url="https://github.com/ir-sdlc/benchmark.git",
                git_commit_id="v1.0.0",
                registry_name="ir-sdlc-bench",
                registry_version="1.0.0",
            )
            
            # 2. Create generator and add dataset
            generator = HarborRegistryGenerator(config)
            generator.add_dataset(sample_ir_dataset)
            
            # 3. Generate and save
            registry_path = generator.save(tmpdir / "registry.json")
            
            # 4. Load and validate
            loaded = load_registry(registry_path)
            errors = validate_registry(loaded)
            
            # Assertions
            assert registry_path.exists()
            assert len(errors) == 0
            assert loaded["name"] == "ir-sdlc-bench"
            assert loaded["git_url"] == "https://github.com/ir-sdlc/benchmark.git"
            assert len(loaded["datasets"]) == 1
            
            dataset = loaded["datasets"][0]
            assert dataset["name"] == "test-benchmark"
            assert len(dataset["tasks"]) == 2
            
            task = dataset["tasks"][0]
            assert "git_url" in task
            assert "git_commit_id" in task
            assert "path" in task
    
    def test_multiple_datasets(self):
        """Test generating registry with multiple datasets."""
        dataset1 = IRDataset(
            name="dataset-1",
            version="1.0.0",
            description="First dataset",
            tasks=[
                IRTask(
                    task_id="d1-task-1",
                    task_type="bug_triage",
                    repo_name="org/repo1",
                    repo_url="https://github.com/org/repo1.git",
                    commit_hash="aaa111",
                    query="Find bug",
                )
            ],
        )
        
        dataset2 = IRDataset(
            name="dataset-2",
            version="2.0.0",
            description="Second dataset",
            tasks=[
                IRTask(
                    task_id="d2-task-1",
                    task_type="code_review",
                    repo_name="org/repo2",
                    repo_url="https://github.com/org/repo2.git",
                    commit_hash="bbb222",
                    query="Review code",
                )
            ],
        )
        
        registry = generate_registry_from_datasets(
            datasets=[dataset1, dataset2],
            git_url="https://github.com/bench/repo.git",
        )
        
        assert len(registry["datasets"]) == 2
        assert registry["datasets"][0]["name"] == "dataset-1"
        assert registry["datasets"][1]["name"] == "dataset-2"
