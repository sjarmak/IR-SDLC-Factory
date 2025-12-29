"""
Tests for enterprise ground truth extraction pipeline.

Validates:
1. Individual extraction strategies
2. Pipeline orchestration
3. Location deduplication and confidence boosting
4. Convenience functions
"""

import pytest
from typing import Dict, Any, List

from app.ir_sdlc.ground_truth_extraction import (
    GroundTruthSource,
    SourcedLocation,
    EnterpriseGroundTruth,
    GitCommitExtractionStrategy,
    PRFilesExtractionStrategy,
    PRReviewExtractionStrategy,
    IssueTextExtractionStrategy,
    StackTraceExtractionStrategy,
    TestFilesExtractionStrategy,
    EnterpriseGroundTruthPipeline,
    extract_ground_truth_from_issue,
)
from app.ir_sdlc.data_structures import CodeLocation, RetrievalGranularity


class TestGitCommitExtractionStrategy:
    """Tests for git commit extraction."""
    
    def test_extracts_files_from_commit(self):
        """Should extract files from commit data."""
        strategy = GitCommitExtractionStrategy()
        
        context = {
            "commit_sha": "abc123",
            "commit_message": "Fix authentication bug",
            "commit_files": [
                {"filename": "src/auth.py", "status": "modified"},
                {"filename": "src/utils.py", "status": "modified"},
            ],
        }
        
        locations = strategy.extract(context)
        
        assert len(locations) == 2
        assert all(loc.source == GroundTruthSource.GIT_COMMIT for loc in locations)
        assert locations[0].location.file_path == "src/auth.py"
        assert locations[0].confidence >= 0.9
    
    def test_filters_irrelevant_files(self):
        """Should filter out lock files and generated files."""
        strategy = GitCommitExtractionStrategy()
        
        context = {
            "commit_sha": "abc123",
            "commit_message": "Update deps",
            "commit_files": [
                "src/auth.py",
                "package-lock.json",
                "node_modules/foo/bar.js",
                "README.md",
                "src/handler.py",
            ],
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/auth.py" in file_paths
        assert "src/handler.py" in file_paths
        assert "package-lock.json" not in file_paths
        assert "README.md" not in file_paths
    
    def test_adjusts_confidence_for_deleted_files(self):
        """Should lower confidence for deleted files."""
        strategy = GitCommitExtractionStrategy()
        
        context = {
            "commit_sha": "abc123",
            "commit_message": "Cleanup",
            "commit_files": [
                {"filename": "src/old.py", "status": "deleted"},
                {"filename": "src/new.py", "status": "added"},
            ],
        }
        
        locations = strategy.extract(context)
        
        deleted = next(l for l in locations if l.location.file_path == "src/old.py")
        added = next(l for l in locations if l.location.file_path == "src/new.py")
        
        assert deleted.confidence < added.confidence


class TestPRFilesExtractionStrategy:
    """Tests for PR files extraction."""
    
    def test_extracts_pr_files(self):
        """Should extract files from PR data."""
        strategy = PRFilesExtractionStrategy()
        
        context = {
            "pr_number": 123,
            "pr_files": [
                {"filename": "src/feature.py"},
                {"filename": "tests/test_feature.py"},
            ],
        }
        
        locations = strategy.extract(context)
        
        assert len(locations) == 2
        assert all(loc.source == GroundTruthSource.PR_FILES for loc in locations)


class TestPRReviewExtractionStrategy:
    """Tests for PR review extraction."""
    
    def test_extracts_review_file_mentions(self):
        """Should extract files from review comments."""
        strategy = PRReviewExtractionStrategy()
        
        context = {
            "pr_number": 123,
            "review_comments": [
                {"path": "src/auth.py", "line": 42, "body": "Consider adding error handling"},
                {"path": "src/db.py", "line": 100, "body": "This query could be optimized with a longer explanation here that exceeds 100 characters for testing the boost"},
            ],
        }
        
        locations = strategy.extract(context)
        
        assert len(locations) == 2
        assert all(loc.source == GroundTruthSource.PR_REVIEW for loc in locations)
        
        auth_loc = next(l for l in locations if l.location.file_path == "src/auth.py")
        assert auth_loc.location.start_line == 42
    
    def test_boosts_confidence_for_substantive_comments(self):
        """Should boost confidence for long comments."""
        strategy = PRReviewExtractionStrategy()
        
        short_comment = {
            "pr_number": 1,
            "review_comments": [
                {"path": "a.py", "body": "LGTM"},
            ],
        }
        
        long_comment = {
            "pr_number": 1,
            "review_comments": [
                {"path": "b.py", "body": "This is a very detailed comment about the implementation that provides substantive feedback on the code changes and suggests improvements that would make the code more maintainable and performant in the long run."},
            ],
        }
        
        short_locs = strategy.extract(short_comment)
        long_locs = strategy.extract(long_comment)
        
        assert long_locs[0].confidence > short_locs[0].confidence


class TestIssueTextExtractionStrategy:
    """Tests for issue text extraction."""
    
    def test_extracts_file_paths_from_issue_body(self):
        """Should extract file paths mentioned in issue body."""
        strategy = IssueTextExtractionStrategy()
        
        context = {
            "issue_title": "Bug in src/auth.py",
            "issue_body": """
I found a bug in the authentication module.

The error occurs in `src/auth/handler.py` when calling the login function.
Also related: tests/test_auth.py

```python
# From src/auth/utils.py
def validate():
    pass
```
            """,
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/auth.py" in file_paths
        assert "src/auth/handler.py" in file_paths
        assert "tests/test_auth.py" in file_paths
    
    def test_extracts_from_comments(self):
        """Should extract file paths from comments."""
        strategy = IssueTextExtractionStrategy()
        
        context = {
            "issue_title": "Bug report",
            "issue_body": "Something is broken",
            "issue_comments": [
                {"body": "I traced it to src/db/connection.py"},
                {"body": "Also check src/cache.py"},
            ],
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/db/connection.py" in file_paths
        assert "src/cache.py" in file_paths
    
    def test_filters_invalid_paths(self):
        """Should filter out invalid paths."""
        strategy = IssueTextExtractionStrategy()
        
        context = {
            "issue_body": """
Check http://example.com/path.js
Also user@email.com
Valid: src/foo.py
node_modules/bar.js
            """,
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/foo.py" in file_paths
        assert not any("http" in p for p in file_paths)
        assert not any("@" in p for p in file_paths)


class TestStackTraceExtractionStrategy:
    """Tests for stack trace extraction."""
    
    def test_extracts_python_stack_trace(self):
        """Should extract files from Python stack traces."""
        strategy = StackTraceExtractionStrategy()
        
        context = {
            "stack_trace": '''
Traceback (most recent call last):
  File "src/main.py", line 42, in <module>
    run()
  File "src/runner.py", line 100, in run
    process()
  File "src/processor.py", line 55, in process
    raise ValueError("error")
ValueError: error
            '''
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/main.py" in file_paths
        assert "src/runner.py" in file_paths
        assert "src/processor.py" in file_paths
    
    def test_extracts_javascript_stack_trace(self):
        """Should extract files from JavaScript stack traces."""
        strategy = StackTraceExtractionStrategy()
        
        context = {
            "stack_trace": '''
Error: Something went wrong
    at processData (src/handler.js:42:15)
    at Object.run (src/runner.js:100:10)
            '''
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/handler.js" in file_paths
        assert "src/runner.js" in file_paths
    
    def test_filters_library_files(self):
        """Should filter out library/vendor files."""
        strategy = StackTraceExtractionStrategy()
        
        context = {
            "stack_trace": '''
  File "/usr/lib/python3.9/site-packages/django/core/handlers/base.py", line 181
  File "src/views.py", line 42
  File "/home/user/.venv/lib/python3.9/site-packages/requests/api.py", line 75
            '''
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "src/views.py" in file_paths
        assert not any("site-packages" in p for p in file_paths)


class TestTestFilesExtractionStrategy:
    """Tests for test file extraction."""
    
    def test_identifies_test_files(self):
        """Should identify test file patterns."""
        strategy = TestFilesExtractionStrategy()
        
        context = {
            "commit_files": [
                {"filename": "src/auth.py", "status": "modified"},
                {"filename": "tests/test_auth.py", "status": "added"},
                {"filename": "src/handler_test.go", "status": "modified"},
                {"filename": "src/utils.spec.js", "status": "added"},
            ],
        }
        
        locations = strategy.extract(context)
        
        file_paths = [loc.location.file_path for loc in locations]
        assert "tests/test_auth.py" in file_paths
        assert "src/handler_test.go" in file_paths
        assert "src/utils.spec.js" in file_paths
        assert "src/auth.py" not in file_paths
    
    def test_boosts_confidence_for_new_tests(self):
        """Should boost confidence for newly added tests."""
        strategy = TestFilesExtractionStrategy()
        
        context = {
            "commit_files": [
                {"filename": "tests/test_new.py", "status": "added"},
                {"filename": "tests/test_old.py", "status": "modified"},
            ],
        }
        
        locations = strategy.extract(context)
        
        new_test = next(l for l in locations if "test_new" in l.location.file_path)
        old_test = next(l for l in locations if "test_old" in l.location.file_path)
        
        assert new_test.confidence > old_test.confidence


class TestEnterpriseGroundTruthPipeline:
    """Tests for the full pipeline."""
    
    def test_combines_multiple_sources(self):
        """Should combine results from multiple strategies."""
        pipeline = EnterpriseGroundTruthPipeline()
        
        context = {
            "commit_sha": "abc123",
            "commit_message": "Fix bug",
            "commit_files": ["src/auth.py"],
            "issue_body": "Bug in src/handler.py",
            "pr_files": ["src/db.py"],
        }
        
        result = pipeline.extract(context, task_id="test-001")
        
        assert isinstance(result, EnterpriseGroundTruth)
        assert len(result.sources_used) >= 2
        assert len(result.locations) >= 2
    
    def test_deduplicates_locations(self):
        """Should deduplicate locations from multiple sources."""
        pipeline = EnterpriseGroundTruthPipeline()
        
        context = {
            "commit_files": ["src/auth.py"],
            "pr_files": ["src/auth.py"],  # Same file
            "issue_body": "Check src/auth.py",  # Same file
        }
        
        result = pipeline.extract(context)
        
        # Should be deduplicated to single location
        auth_locs = [l for l in result.locations if l.file_path == "src/auth.py"]
        assert len(auth_locs) == 1
    
    def test_boosts_confidence_for_multiple_sources(self):
        """Should boost confidence when multiple sources agree."""
        pipeline = EnterpriseGroundTruthPipeline()
        
        # Single source
        single_context = {
            "commit_files": ["src/auth.py"],
        }
        
        # Multiple sources
        multi_context = {
            "commit_files": ["src/auth.py"],
            "pr_files": ["src/auth.py"],
            "issue_body": "Check src/auth.py",
        }
        
        single_result = pipeline.extract(single_context)
        multi_result = pipeline.extract(multi_context)
        
        # Find auth.py location in each
        single_auth = next(
            sl for sl in single_result.sourced_locations 
            if sl.location.file_path == "src/auth.py"
        )
        multi_auth = next(
            sl for sl in multi_result.sourced_locations 
            if sl.location.file_path == "src/auth.py"
        )
        
        # Multi-source should have higher confidence
        assert multi_auth.confidence > single_auth.confidence
    
    def test_filters_by_confidence_threshold(self):
        """Should filter locations below confidence threshold."""
        pipeline = EnterpriseGroundTruthPipeline(min_confidence=0.8)
        
        context = {
            # Git commit (high confidence)
            "commit_files": ["src/auth.py"],
            # Issue text (lower confidence)
            "issue_body": "Maybe check src/maybe.py",
        }
        
        result = pipeline.extract(context)
        
        # High confidence location should be included
        file_paths = [l.file_path for l in result.locations]
        assert "src/auth.py" in file_paths
    
    def test_converts_to_ground_truth(self):
        """Should convert to base GroundTruth format."""
        pipeline = EnterpriseGroundTruthPipeline()
        
        context = {
            "commit_files": ["src/auth.py", "src/db.py"],
        }
        
        result = pipeline.extract(context)
        base_gt = result.to_ground_truth()
        
        assert len(base_gt.locations) == 2
        assert base_gt.source == "automatic"
        assert base_gt.confidence > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_extract_from_issue(self):
        """Should extract ground truth from issue data."""
        issue_data = {
            "title": "Bug in authentication",
            "body": "Error in src/auth.py line 42",
            "issue_url": "https://github.com/org/repo/issues/1",
            "comments": [],
        }
        
        commit_data = {
            "sha": "abc123",
            "message": "Fix auth bug",
            "files": [{"filename": "src/auth.py"}],
        }
        
        result = extract_ground_truth_from_issue(
            issue_data=issue_data,
            commit_data=commit_data,
        )
        
        assert isinstance(result, EnterpriseGroundTruth)
        assert len(result.locations) >= 1
        assert "src/auth.py" in [l.file_path for l in result.locations]


class TestEnterpriseGroundTruth:
    """Tests for EnterpriseGroundTruth dataclass."""
    
    def test_to_dict(self):
        """Should serialize to dictionary."""
        gt = EnterpriseGroundTruth(
            locations=[CodeLocation(file_path="src/auth.py")],
            sources_used={GroundTruthSource.GIT_COMMIT},
            overall_confidence=0.9,
            task_id="test-001",
        )
        
        d = gt.to_dict()
        
        assert "locations" in d
        assert "sources_used" in d
        assert "overall_confidence" in d
        assert d["overall_confidence"] == 0.9
    
    def test_to_ground_truth_conversion(self):
        """Should convert to base GroundTruth."""
        gt = EnterpriseGroundTruth(
            locations=[
                CodeLocation(file_path="src/a.py"),
                CodeLocation(file_path="src/b.py"),
            ],
            sources_used={GroundTruthSource.GIT_COMMIT, GroundTruthSource.PR_FILES},
            overall_confidence=0.85,
        )
        
        base = gt.to_ground_truth()
        
        assert len(base.locations) == 2
        assert base.confidence == 0.85
        assert "sources_used" in base.metadata
