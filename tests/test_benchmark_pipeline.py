"""
Tests for SDLC Benchmark Task Generation Pipeline.

Tests validate:
1. GitHubIssue and GitHubPullRequest data structures
2. GroundTruthExtractor parsing logic
3. DifficultyEstimator scoring
4. SDLCBenchmarkPipeline task generation
5. IRTask structure compatibility with dashboard_exporter
"""

import json
import pytest
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.ir_sdlc.benchmark_pipeline import (
    GitHubIssue,
    GitHubPullRequest,
    GitHubIssueMiner,
    GroundTruthExtractor,
    DifficultyEstimator,
    RepoComplexityStats,
    SDLCBenchmarkPipeline,
    CommitDiff,
)
from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
)
from app.ir_sdlc.task_types import SDLCTaskType
from app.ir_sdlc.dashboard_schema import (
    IRSDLCTaskResult,
    SDLCTaskType as DashboardSDLCTaskType,
    IRToolType,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_github_issue():
    """Sample GitHub issue data."""
    return GitHubIssue(
        issue_number=123,
        title="NullPointerException in UserService.getUser()",
        body="""When calling getUser with null id, the service crashes.

Traceback (most recent call last):
  File "user_service.py", line 45, in get_user
    return self.repository.find(user_id)
TypeError: NoneType object is not subscriptable

Expected: Should return None or raise InvalidIdError
Actual: Crashes with TypeError
""",
        labels=["bug", "high-priority"],
        state="closed",
        created_at="2024-01-15T10:00:00Z",
        closed_at="2024-01-20T15:30:00Z",
        issue_url="https://github.com/test/repo/issues/123",
        user="testuser",
        linked_pr_number=456,
        fix_commit="abc123def456",
        has_stack_trace=True,
        stack_trace="Traceback (most recent call last):\n  File ...",
        error_type="TypeError",
    )


@pytest.fixture
def sample_github_pr():
    """Sample GitHub pull request data."""
    return GitHubPullRequest(
        pr_number=456,
        title="Fix NullPointerException in UserService",
        body="Fixes #123\n\nAdded null check before repository call.",
        labels=["bug-fix"],
        state="closed",
        merged=True,
        created_at="2024-01-18T09:00:00Z",
        merged_at="2024-01-20T15:30:00Z",
        pr_url="https://github.com/test/repo/pull/456",
        user="developer",
        changed_files=["src/services/user_service.py", "tests/test_user_service.py"],
        additions=15,
        deletions=3,
        reviewed_files=["src/services/user_service.py"],
        review_comments=[
            {"path": "src/services/user_service.py", "body": "Good fix!"},
        ],
        linked_issue_number=123,
        merge_commit_sha="abc123def456",
    )


@pytest.fixture
def sample_repo_stats():
    """Sample repository complexity stats."""
    return RepoComplexityStats(
        full_name="kubernetes/kubernetes",
        file_count=15000,
        directory_count=2500,
        lines_of_code=5000000,
        contributor_count=3000,
        commit_count=100000,
        language_count=5,
        size_kb=500000,
        stars=100000,
        forks=35000,
        complexity_score=75.0,
    )


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth with multiple locations."""
    return GroundTruth(
        locations=[
            CodeLocation(file_path="src/services/user_service.py", start_line=45, end_line=50),
            CodeLocation(file_path="src/models/user.py", start_line=10, end_line=15),
        ],
        granularity=RetrievalGranularity.FUNCTION,
        source="automatic",
        confidence=0.9,
    )


# =============================================================================
# GitHubIssue Tests
# =============================================================================

class TestGitHubIssue:
    """Tests for GitHubIssue data structure."""
    
    def test_to_source_data_includes_all_fields(self, sample_github_issue):
        """Verify to_source_data includes all required fields for task generators."""
        source_data = sample_github_issue.to_source_data()
        
        assert "title" in source_data
        assert "body" in source_data
        assert "labels" in source_data
        assert "issue_url" in source_data
        assert "fix_commit" in source_data
        assert "stack_trace" in source_data
        assert "error_type" in source_data
    
    def test_to_source_data_values_match(self, sample_github_issue):
        """Verify to_source_data values match issue fields."""
        source_data = sample_github_issue.to_source_data()
        
        assert source_data["title"] == sample_github_issue.title
        assert source_data["body"] == sample_github_issue.body
        assert source_data["labels"] == sample_github_issue.labels
        assert source_data["fix_commit"] == sample_github_issue.fix_commit


class TestGitHubPullRequest:
    """Tests for GitHubPullRequest data structure."""
    
    def test_to_source_data_includes_pr_fields(self, sample_github_pr):
        """Verify to_source_data includes PR-specific fields."""
        source_data = sample_github_pr.to_source_data()
        
        assert "pr_number" in source_data
        assert "changed_files" in source_data
        assert "additions" in source_data
        assert "deletions" in source_data
        assert "reviewed_files" in source_data
    
    def test_extract_mentioned_files_from_comments(self, sample_github_pr):
        """Verify file paths are extracted from review comments."""
        source_data = sample_github_pr.to_source_data()
        
        assert "src/services/user_service.py" in source_data["comment_mentioned_files"]


# =============================================================================
# GroundTruthExtractor Tests
# =============================================================================

class TestGroundTruthExtractor:
    """Tests for GroundTruthExtractor."""
    
    def test_is_relevant_file_excludes_lock_files(self):
        """Verify lock files are filtered out."""
        extractor = GroundTruthExtractor(github_token="fake-token")
        
        assert not extractor._is_relevant_file("package-lock.json")
        assert not extractor._is_relevant_file("yarn.lock")
        assert not extractor._is_relevant_file("Gemfile.lock")
    
    def test_is_relevant_file_excludes_generated_files(self):
        """Verify generated files are filtered out."""
        extractor = GroundTruthExtractor(github_token="fake-token")
        
        assert not extractor._is_relevant_file("bundle.min.js")
        assert not extractor._is_relevant_file("styles.min.css")
        assert not extractor._is_relevant_file("api.generated.go")
    
    def test_is_relevant_file_excludes_vendor_dirs(self):
        """Verify vendor directories are filtered out."""
        extractor = GroundTruthExtractor(github_token="fake-token")
        
        assert not extractor._is_relevant_file("vendor/some/lib.go")
        assert not extractor._is_relevant_file("node_modules/package/index.js")
    
    def test_is_relevant_file_accepts_source_files(self):
        """Verify source code files are accepted."""
        extractor = GroundTruthExtractor(github_token="fake-token")
        
        assert extractor._is_relevant_file("src/main.py")
        assert extractor._is_relevant_file("lib/utils.js")
        assert extractor._is_relevant_file("pkg/handler/api.go")
        assert extractor._is_relevant_file("src/services/user_service.py")
    
    def test_parse_patch_locations_extracts_line_numbers(self):
        """Verify patch parsing extracts correct line numbers."""
        extractor = GroundTruthExtractor(github_token="fake-token")
        
        patch = """--- a/src/main.py
+++ b/src/main.py
@@ -10,5 +10,7 @@ def main():
     print("hello")
+    print("added line")
     return 0
"""
        
        locations = extractor._parse_patch_locations(patch, "src/main.py")
        
        assert len(locations) == 1
        assert locations[0].file_path == "src/main.py"
        assert locations[0].start_line == 10


# =============================================================================
# DifficultyEstimator Tests
# =============================================================================

class TestDifficultyEstimator:
    """Tests for DifficultyEstimator."""
    
    def test_easy_difficulty_for_single_file(self, sample_repo_stats):
        """Verify single-file tasks are rated as easier."""
        estimator = DifficultyEstimator()
        
        # Reduce repo complexity for this test
        small_repo = RepoComplexityStats(
            full_name="small/repo",
            complexity_score=10.0,
        )
        
        single_file_gt = GroundTruth(
            locations=[CodeLocation(file_path="main.py")],
            granularity=RetrievalGranularity.FILE,
        )
        
        difficulty = estimator.estimate_difficulty(small_repo, single_file_gt)
        
        assert difficulty == "easy"
    
    def test_hard_difficulty_for_complex_repo_multi_file(self, sample_repo_stats, sample_ground_truth):
        """Verify complex repos with multiple files are rated harder."""
        estimator = DifficultyEstimator()
        
        # Add more files to ground truth
        complex_gt = GroundTruth(
            locations=[
                CodeLocation(file_path="src/a.py"),
                CodeLocation(file_path="src/b.py"),
                CodeLocation(file_path="lib/c.py"),
                CodeLocation(file_path="lib/d.py"),
                CodeLocation(file_path="pkg/e.py"),
            ],
            granularity=RetrievalGranularity.FILE,
        )
        
        difficulty = estimator.estimate_difficulty(sample_repo_stats, complex_gt)
        
        assert difficulty in ["hard", "expert"]
    
    def test_complexity_score_calculation(self):
        """Verify complexity score is calculated correctly."""
        estimator = DifficultyEstimator()
        
        stats = RepoComplexityStats(
            full_name="test/repo",
            size_kb=100000,  # ~100MB
            lines_of_code=1000000,
            contributor_count=200,
            language_count=3,
            stars=50000,
        )
        
        score = estimator._calculate_complexity_score(stats)
        
        # Should be positive and reasonable
        assert 0 < score <= 100
        assert score > 20  # Non-trivial repo


# =============================================================================
# SDLCBenchmarkPipeline Integration Tests
# =============================================================================

class TestSDLCBenchmarkPipeline:
    """Integration tests for SDLCBenchmarkPipeline."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Verify pipeline initializes with correct components."""
        pipeline = SDLCBenchmarkPipeline(
            github_token="fake-token",
            output_dir=str(tmp_path),
        )
        
        assert pipeline.miner is not None
        assert pipeline.extractor is not None
        assert pipeline.estimator is not None
        assert SDLCTaskType.BUG_TRIAGE in pipeline.generators
        assert SDLCTaskType.CODE_REVIEW in pipeline.generators
    
    def test_infer_vulnerability_type_from_labels(self, tmp_path):
        """Verify vulnerability type inference from labels."""
        pipeline = SDLCBenchmarkPipeline(
            github_token="fake-token",
            output_dir=str(tmp_path),
        )
        
        assert pipeline._infer_vulnerability_type(["xss", "security"]) == "Cross-Site Scripting"
        assert pipeline._infer_vulnerability_type(["sqli"]) == "SQL Injection"  # Uses sqli keyword
        assert pipeline._infer_vulnerability_type(["csrf"]) == "Cross-Site Request Forgery"
        # 'sql-injection' matches 'injection' keyword first
        assert pipeline._infer_vulnerability_type(["sql-injection"]) == "Injection"
        assert pipeline._infer_vulnerability_type(["unrelated"]) is None
    
    def test_infer_severity_from_labels(self, tmp_path):
        """Verify severity inference from labels."""
        pipeline = SDLCBenchmarkPipeline(
            github_token="fake-token",
            output_dir=str(tmp_path),
        )
        
        assert pipeline._infer_severity(["critical"]) == "critical"
        assert pipeline._infer_severity(["high-priority"]) == "high"
        assert pipeline._infer_severity(["medium", "security"]) == "medium"
        assert pipeline._infer_severity(["bug"]) == "unknown"


# =============================================================================
# IRTask Compatibility Tests
# =============================================================================

class TestIRTaskCompatibility:
    """Tests for IRTask compatibility with dashboard exporter."""
    
    def test_irtask_serialization(self, sample_ground_truth):
        """Verify IRTask can be serialized to JSON."""
        task = IRTask(
            task_id="test-task-001",
            task_type="bug_triage",
            repo_name="test/repo",
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
            query="Find the bug in user service",
            context={"error_type": "NullPointer"},
            ground_truth=sample_ground_truth,
            difficulty="medium",
            tags=["bug", "high-priority"],
        )
        
        # Should serialize without error
        task_dict = task.to_dict()
        json_str = json.dumps(task_dict)
        
        # Should deserialize back
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "test-task-001"
        assert parsed["task_type"] == "bug_triage"
        assert parsed["difficulty"] == "medium"
    
    def test_irtask_ground_truth_serialization(self, sample_ground_truth):
        """Verify ground truth serializes correctly within IRTask."""
        task = IRTask(
            task_id="test-task-002",
            task_type="code_review",
            repo_name="test/repo",
            repo_url="https://github.com/test/repo",
            commit_hash="def456",
            query="Review this PR",
            ground_truth=sample_ground_truth,
        )
        
        task_dict = task.to_dict()
        
        assert task_dict["ground_truth"] is not None
        assert len(task_dict["ground_truth"]["locations"]) == 2
        assert task_dict["ground_truth"]["granularity"] == "function"
    
    def test_irtask_can_convert_to_dashboard_result(self, sample_ground_truth):
        """Verify IRTask data can populate IRSDLCTaskResult."""
        task = IRTask(
            task_id="test-task-003",
            task_type="bug_triage",
            repo_name="kubernetes/kubernetes",
            repo_url="https://github.com/kubernetes/kubernetes",
            commit_hash="abc123",
            query="Find the null pointer bug",
            ground_truth=sample_ground_truth,
            difficulty="hard",
            tags=["bug"],
        )
        
        # Create dashboard result from task
        result = IRSDLCTaskResult(
            task_id=task.task_id,
            task_title=task.query[:50],
            sdlc_type=DashboardSDLCTaskType.BUG_TRIAGE,
            repo_name=task.repo_name,
            repo_url=task.repo_url,
            commit_hash=task.commit_hash,
            difficulty=task.difficulty,
            ir_tool_type=IRToolType.BASELINE,
            ir_tool_name="BaselineAgent",
            agent_import_path="agents.baseline",
            model_name="gpt-4",
            ground_truth_files=[loc.file_path for loc in task.ground_truth.locations],
        )
        
        # Should serialize without error
        result_dict = result.to_dict()
        
        assert result_dict["task_id"] == "test-task-003"
        assert result_dict["sdlc_type"] == "bug_triage"
        assert result_dict["difficulty"] == "hard"
        assert len(result_dict["ground_truth_files"]) == 2


# =============================================================================
# IRDataset Tests
# =============================================================================

class TestIRDataset:
    """Tests for IRDataset serialization."""
    
    def test_dataset_save_jsonl(self, tmp_path, sample_ground_truth):
        """Verify dataset saves to JSONL correctly."""
        tasks = [
            IRTask(
                task_id=f"task-{i}",
                task_type="bug_triage",
                repo_name="test/repo",
                repo_url="https://github.com/test/repo",
                commit_hash=f"commit{i}",
                query=f"Task {i}",
                ground_truth=sample_ground_truth,
            )
            for i in range(3)
        ]
        
        dataset = IRDataset(
            name="test-dataset",
            version="1.0.0",
            description="Test dataset",
            tasks=tasks,
        )
        
        output_path = tmp_path / "test.jsonl"
        dataset.save_jsonl(output_path)
        
        # Read and verify
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3
        
        for i, line in enumerate(lines):
            task_data = json.loads(line)
            assert task_data["task_id"] == f"task-{i}"
    
    def test_dataset_load_jsonl(self, tmp_path, sample_ground_truth):
        """Verify dataset loads from JSONL correctly."""
        # Create JSONL file
        tasks_data = [
            {
                "task_id": f"task-{i}",
                "task_type": "bug_triage",
                "repo_name": "test/repo",
                "repo_url": "https://github.com/test/repo",
                "commit_hash": f"commit{i}",
                "query": f"Task {i}",
                "ground_truth": sample_ground_truth.to_dict(),
            }
            for i in range(3)
        ]
        
        jsonl_path = tmp_path / "input.jsonl"
        with open(jsonl_path, "w") as f:
            for task in tasks_data:
                f.write(json.dumps(task) + "\n")
        
        # Load dataset
        dataset = IRDataset.load_jsonl(jsonl_path, name="loaded-dataset")
        
        assert len(dataset.tasks) == 3
        assert dataset.tasks[0].task_id == "task-0"
        assert dataset.tasks[0].ground_truth is not None


# =============================================================================
# GitHub Miner Mock Tests
# =============================================================================

class TestGitHubIssueMinerParsing:
    """Tests for GitHubIssueMiner parsing methods."""
    
    def test_extract_stack_trace_python(self):
        """Verify Python stack trace extraction."""
        miner = GitHubIssueMiner(github_token="fake-token")
        
        body = """The app crashed with:

Traceback (most recent call last):
  File "main.py", line 10, in <module>
    result = func()
  File "utils.py", line 5, in func
    raise ValueError("oops")
ValueError: oops

Please fix this."""
        
        trace = miner._extract_stack_trace(body)
        
        assert trace is not None
        assert "Traceback" in trace
        assert "ValueError" in trace
    
    def test_extract_stack_trace_java(self):
        """Verify Java stack trace extraction."""
        miner = GitHubIssueMiner(github_token="fake-token")
        
        body = """Got this error:

Exception in thread "main" java.lang.NullPointerException
    at com.example.Main.process(Main.java:15)
    at com.example.Main.main(Main.java:8)

Steps to reproduce..."""
        
        trace = miner._extract_stack_trace(body)
        
        assert trace is not None
        assert "NullPointerException" in trace
    
    def test_extract_error_type(self):
        """Verify error type extraction."""
        miner = GitHubIssueMiner(github_token="fake-token")
        
        assert miner._extract_error_type("TypeError: cannot read property") == "TypeError"
        assert miner._extract_error_type("NullPointerException: null") == "NullPointerException"
        assert miner._extract_error_type("panic: runtime error") == "runtime"
        assert miner._extract_error_type("CVE-2024-1234 vulnerability") == "CVE-2024-1234"
    
    def test_parse_issue_handles_missing_fields(self):
        """Verify issue parsing handles missing optional fields."""
        miner = GitHubIssueMiner(github_token="fake-token")
        
        minimal_data = {
            "number": 1,
            "html_url": "https://github.com/test/repo/issues/1",
        }
        
        issue = miner._parse_issue(minimal_data, "test/repo")
        
        assert issue is not None
        assert issue.issue_number == 1
        assert issue.title == ""
        assert issue.body == ""
        assert issue.labels == []


# =============================================================================
# End-to-End Mock Test
# =============================================================================

class TestPipelineEndToEnd:
    """End-to-end test with mocked GitHub API."""
    
    def test_generate_bug_triage_tasks_e2e(self, tmp_path, sample_ground_truth):
        """Test full task generation flow with mocked miner."""
        # Create a mock issue
        mock_issue = GitHubIssue(
            issue_number=100,
            title="Bug: NullPointer in handler",
            body="TypeError: cannot read property 'x' of undefined",
            labels=["bug"],
            state="closed",
            created_at="2024-01-01T00:00:00Z",
            closed_at="2024-01-05T00:00:00Z",
            issue_url="https://github.com/test/repo/issues/100",
            user="user1",
            linked_pr_number=101,
            fix_commit="abc123",
        )
        
        # Create pipeline
        pipeline = SDLCBenchmarkPipeline(
            github_token="fake-token",
            output_dir=str(tmp_path),
        )
        
        # Mock the miner to return our test issue
        def mock_mine_bugs(repo, max_issues, require_linked_pr=True):
            yield mock_issue
        
        pipeline.miner.mine_bug_issues = mock_mine_bugs
        
        # Mock ground truth extraction
        def mock_extract(repo, commit_sha, granularity):
            return GroundTruth(
                locations=[CodeLocation(file_path="src/handler.js", start_line=10)],
                granularity=granularity,
                source="automatic",
                confidence=0.9,
            )
        
        pipeline.extractor.extract_from_commit = mock_extract
        
        # Mock repo stats
        pipeline.estimator.get_repo_stats = lambda repo: RepoComplexityStats(
            full_name=repo,
            complexity_score=25.0,
        )
        
        # Run pipeline
        dataset = pipeline.generate_from_repo(
            repo="test/repo",
            task_types=[SDLCTaskType.BUG_TRIAGE],
            max_tasks_per_type=1,
        )
        
        # Verify result
        assert len(dataset.tasks) == 1
        task = dataset.tasks[0]
        assert task.task_type == "bug_triage"
        assert task.repo_name == "test/repo"
        assert task.ground_truth is not None
        assert len(task.ground_truth.locations) >= 1
        assert task.ground_truth.locations[0].file_path == "src/handler.js"
    
    def test_dataset_save_and_reload(self, tmp_path, sample_ground_truth):
        """Test that generated dataset can be saved and reloaded."""
        # Create a task directly
        task = IRTask(
            task_id="test-save-001",
            task_type="bug_triage",
            repo_name="test/repo",
            repo_url="https://github.com/test/repo",
            commit_hash="abc123",
            query="Find the bug",
            ground_truth=sample_ground_truth,
            difficulty="medium",
        )
        
        dataset = IRDataset(
            name="save-test",
            version="1.0.0",
            description="Test save/reload",
            tasks=[task],
        )
        
        # Save
        output_path = tmp_path / "test.jsonl"
        dataset.save_jsonl(output_path)
        
        # Reload
        reloaded = IRDataset.load_jsonl(output_path, name="reloaded")
        
        assert len(reloaded.tasks) == 1
        assert reloaded.tasks[0].task_id == "test-save-001"
        assert reloaded.tasks[0].ground_truth is not None
        assert len(reloaded.tasks[0].ground_truth.locations) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
