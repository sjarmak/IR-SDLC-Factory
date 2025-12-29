"""
Enterprise Ground Truth Extraction Pipeline for IR-SDLC-Bench.

This module provides comprehensive ground truth extraction from enterprise
codebases. It extracts relevant file locations from multiple sources:

1. Git History - Commits that fixed issues
2. PR File Changes - Reviewer-validated relevant files  
3. Issue-to-Code Links - Files mentioned in issue resolution
4. Test Coverage Changes - New tests added for bug fixes

Output: GroundTruth objects with CodeLocation lists per task.

The extraction prioritizes accuracy by cross-referencing multiple sources
and computing confidence scores based on signal quality.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from app.ir_sdlc.data_structures import (
    CodeLocation,
    GroundTruth,
    RetrievalGranularity,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Ground Truth Sources
# =============================================================================

class GroundTruthSource(str, Enum):
    """Sources of ground truth data."""
    GIT_COMMIT = "git_commit"  # From fix commits
    PR_FILES = "pr_files"  # PR changed files
    PR_REVIEW = "pr_review"  # Files mentioned in reviews
    ISSUE_BODY = "issue_body"  # Files mentioned in issue body
    ISSUE_COMMENT = "issue_comment"  # Files mentioned in comments
    TEST_ADDITION = "test_addition"  # New test files added
    TEST_MODIFICATION = "test_modification"  # Modified test files
    STACK_TRACE = "stack_trace"  # Files from stack traces
    CODE_REFERENCE = "code_reference"  # Files referenced in code blocks
    MANUAL = "manual"  # Manually annotated


@dataclass
class SourcedLocation:
    """A code location with provenance information."""
    location: CodeLocation
    source: GroundTruthSource
    confidence: float  # 0.0 to 1.0
    evidence: str = ""  # What led to this extraction
    
    def to_dict(self) -> dict:
        return {
            "location": self.location.to_dict(),
            "source": self.source.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }


@dataclass
class EnterpriseGroundTruth:
    """Enhanced ground truth with multi-source evidence.
    
    Extends the base GroundTruth with detailed provenance and
    confidence scores from multiple extraction sources.
    """
    # Core locations (deduplicated, ranked by confidence)
    locations: List[CodeLocation] = field(default_factory=list)
    
    # Source breakdown
    sourced_locations: List[SourcedLocation] = field(default_factory=list)
    sources_used: Set[GroundTruthSource] = field(default_factory=set)
    
    # Scoring
    granularity: RetrievalGranularity = RetrievalGranularity.FILE
    overall_confidence: float = 0.0
    confidence_by_source: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    task_id: str = ""
    extraction_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_ground_truth(self) -> GroundTruth:
        """Convert to base GroundTruth format."""
        return GroundTruth(
            locations=self.locations,
            granularity=self.granularity,
            source="automatic",
            confidence=self.overall_confidence,
            metadata={
                "sources_used": [s.value for s in self.sources_used],
                "location_count": len(self.locations),
                **self.metadata,
            },
        )
    
    def to_dict(self) -> dict:
        return {
            "locations": [loc.to_dict() for loc in self.locations],
            "sourced_locations": [sl.to_dict() for sl in self.sourced_locations],
            "sources_used": [s.value for s in self.sources_used],
            "granularity": self.granularity.value,
            "overall_confidence": self.overall_confidence,
            "confidence_by_source": self.confidence_by_source,
            "task_id": self.task_id,
            "extraction_time": self.extraction_time,
            "metadata": self.metadata,
        }


# =============================================================================
# Extraction Strategies
# =============================================================================

class ExtractionStrategy(ABC):
    """Abstract base for ground truth extraction strategies."""
    
    @property
    @abstractmethod
    def source_type(self) -> GroundTruthSource:
        """Return the source type for this strategy."""
        pass
    
    @property
    @abstractmethod
    def base_confidence(self) -> float:
        """Return the base confidence for this source."""
        pass
    
    @abstractmethod
    def extract(
        self,
        context: Dict[str, Any],
    ) -> List[SourcedLocation]:
        """Extract ground truth locations.
        
        Args:
            context: Extraction context with all available data
            
        Returns:
            List of sourced locations
        """
        pass


class GitCommitExtractionStrategy(ExtractionStrategy):
    """Extract ground truth from git commit diffs."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.GIT_COMMIT
    
    @property
    def base_confidence(self) -> float:
        return 0.95  # High confidence - fix commits are authoritative
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        commit_files = context.get("commit_files", [])
        commit_sha = context.get("commit_sha", "")
        commit_message = context.get("commit_message", "")
        
        for file_info in commit_files:
            if isinstance(file_info, str):
                file_path = file_info
                status = "modified"
            else:
                file_path = file_info.get("filename", file_info.get("path", ""))
                status = file_info.get("status", "modified")
            
            if not file_path or not self._is_relevant_file(file_path):
                continue
            
            # Adjust confidence based on change type
            confidence = self.base_confidence
            if status == "added":
                confidence *= 0.9  # New files slightly less certain
            elif status == "deleted":
                confidence *= 0.7  # Deleted files less relevant
            
            loc = CodeLocation(file_path=file_path)
            
            locations.append(SourcedLocation(
                location=loc,
                source=self.source_type,
                confidence=confidence,
                evidence=f"Changed in commit {commit_sha[:8]}: {commit_message[:50]}",
            ))
        
        return locations
    
    def _is_relevant_file(self, file_path: str) -> bool:
        """Check if file is relevant for benchmarking."""
        skip_patterns = [
            r"\.lock$", r"package-lock\.json$", r"yarn\.lock$",
            r"\.min\.(js|css)$", r"\.generated\.", r"^vendor/",
            r"^node_modules/", r"^\.github/", r"\.md$", r"\.txt$",
            r"^docs/", r"CHANGELOG", r"LICENSE", r"AUTHORS",
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False
        return True


class PRFilesExtractionStrategy(ExtractionStrategy):
    """Extract ground truth from PR changed files."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.PR_FILES
    
    @property
    def base_confidence(self) -> float:
        return 0.90  # High confidence - PR files are validated
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        pr_files = context.get("pr_files", [])
        pr_number = context.get("pr_number", "")
        
        for file_info in pr_files:
            if isinstance(file_info, str):
                file_path = file_info
            else:
                file_path = file_info.get("filename", file_info.get("path", ""))
            
            if not file_path or not self._is_relevant_file(file_path):
                continue
            
            loc = CodeLocation(file_path=file_path)
            
            locations.append(SourcedLocation(
                location=loc,
                source=self.source_type,
                confidence=self.base_confidence,
                evidence=f"Changed in PR #{pr_number}",
            ))
        
        return locations
    
    def _is_relevant_file(self, file_path: str) -> bool:
        skip_patterns = [
            r"\.lock$", r"package-lock\.json$", r"\.min\.",
            r"^vendor/", r"^node_modules/",
        ]
        for pattern in skip_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False
        return True


class PRReviewExtractionStrategy(ExtractionStrategy):
    """Extract ground truth from PR review comments."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.PR_REVIEW
    
    @property
    def base_confidence(self) -> float:
        return 0.85  # High confidence - reviewer attention
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        review_comments = context.get("review_comments", [])
        pr_number = context.get("pr_number", "")
        
        for comment in review_comments:
            file_path = comment.get("path", "")
            if not file_path:
                continue
            
            # Boost confidence if comment has substantive content
            confidence = self.base_confidence
            body = comment.get("body", "")
            if len(body) > 100:
                confidence = min(0.95, confidence + 0.05)
            
            line = comment.get("line") or comment.get("original_line")
            
            loc = CodeLocation(
                file_path=file_path,
                start_line=line,
                end_line=line,
            )
            
            locations.append(SourcedLocation(
                location=loc,
                source=self.source_type,
                confidence=confidence,
                evidence=f"Reviewed in PR #{pr_number}: {body[:50]}...",
            ))
        
        return locations


class IssueTextExtractionStrategy(ExtractionStrategy):
    """Extract file references from issue body and comments."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.ISSUE_BODY
    
    @property
    def base_confidence(self) -> float:
        return 0.70  # Medium confidence - mentioned but not validated
    
    # Common file path patterns
    FILE_PATTERNS = [
        # Explicit paths
        r'(?:^|[\s`"\'])([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})(?:[\s`"\':]|$)',
        # src/foo/bar.py style
        r'(?:src|lib|pkg|app|test|tests)/[a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10}',
        # In code blocks
        r'```[a-zA-Z]*\s*\n[^`]*?([a-zA-Z0-9_/.-]+\.[a-zA-Z]{1,10})',
    ]
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        issue_body = context.get("issue_body", "")
        issue_title = context.get("issue_title", "")
        issue_comments = context.get("issue_comments", [])
        
        # Extract from issue body
        text_sources = [
            ("body", issue_body, self.base_confidence),
            ("title", issue_title, self.base_confidence * 0.9),
        ]
        
        # Add comments
        for i, comment in enumerate(issue_comments):
            comment_body = comment.get("body", "") if isinstance(comment, dict) else str(comment)
            text_sources.append((f"comment_{i}", comment_body, self.base_confidence * 0.85))
        
        seen_files = set()
        
        for source_name, text, confidence in text_sources:
            for file_path in self._extract_file_paths(text):
                if file_path in seen_files:
                    continue
                seen_files.add(file_path)
                
                if not self._is_valid_source_file(file_path):
                    continue
                
                loc = CodeLocation(file_path=file_path)
                
                locations.append(SourcedLocation(
                    location=loc,
                    source=self.source_type,
                    confidence=confidence,
                    evidence=f"Mentioned in issue {source_name}",
                ))
        
        return locations
    
    def _extract_file_paths(self, text: str) -> List[str]:
        """Extract file paths from text."""
        if not text:
            return []
        
        paths = set()
        
        for pattern in self.FILE_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                # Clean up match
                path = match.strip('`"\'').strip()
                if self._looks_like_file_path(path):
                    paths.add(path)
        
        return list(paths)
    
    def _looks_like_file_path(self, path: str) -> bool:
        """Check if string looks like a valid file path."""
        if len(path) < 3 or len(path) > 200:
            return False
        
        # Must have extension
        if "." not in path:
            return False
        
        # Check extension is reasonable
        ext = path.rsplit(".", 1)[-1].lower()
        valid_extensions = {
            "py", "js", "ts", "tsx", "jsx", "go", "rs", "java", "kt",
            "cpp", "c", "h", "hpp", "cs", "rb", "php", "scala", "swift",
            "yaml", "yml", "json", "xml", "toml", "cfg", "ini",
        }
        
        return ext in valid_extensions
    
    def _is_valid_source_file(self, file_path: str) -> bool:
        """Check if extracted path is a valid source file."""
        skip_patterns = [
            r"^http", r"^www\.", r"@", r"^node_modules",
            r"^vendor/", r"\.lock$", r"\.min\.",
        ]
        for pattern in skip_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False
        return True


class StackTraceExtractionStrategy(ExtractionStrategy):
    """Extract file references from stack traces."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.STACK_TRACE
    
    @property
    def base_confidence(self) -> float:
        return 0.85  # High confidence - stack traces point to bugs
    
    # Stack trace patterns by language
    STACK_PATTERNS = [
        # Python: File "path/file.py", line 123
        (r'File "([^"]+\.py)", line (\d+)', "python"),
        # JavaScript/Node: at function (path/file.js:123:45)
        (r'at\s+.*?\(([^:]+\.[jt]sx?):\d+:\d+\)', "javascript"),
        # Go: /path/file.go:123
        (r'([^\s:]+\.go):(\d+)', "go"),
        # Java: at package.Class.method(File.java:123)
        (r'at\s+[^\(]+\(([^:]+\.java):(\d+)\)', "java"),
        # Rust: at path/file.rs:123
        (r'([^\s:]+\.rs):(\d+)', "rust"),
        # Generic: file:line pattern
        (r'([^\s:]+\.[a-zA-Z]{1,4}):(\d+)', "generic"),
    ]
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        stack_trace = context.get("stack_trace", "")
        error_text = context.get("error_text", "")
        
        # Combine sources
        full_text = f"{stack_trace}\n{error_text}"
        
        for file_path, line_num in self._extract_stack_files(full_text):
            # Focus on files that look like project files
            if not self._is_project_file(file_path):
                continue
            
            loc = CodeLocation(
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
            )
            
            # Higher confidence for first frame (likely the bug)
            confidence = self.base_confidence
            
            locations.append(SourcedLocation(
                location=loc,
                source=self.source_type,
                confidence=confidence,
                evidence=f"In stack trace at line {line_num}",
            ))
        
        return locations
    
    def _extract_stack_files(self, text: str) -> List[Tuple[str, int]]:
        """Extract file paths and line numbers from stack trace."""
        if not text:
            return []
        
        results = []
        seen = set()
        
        for pattern, lang in self.STACK_PATTERNS:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    file_path = match[0]
                    line_num = int(match[1]) if len(match) > 1 else None
                else:
                    file_path = match
                    line_num = None
                
                key = (file_path, line_num)
                if key not in seen:
                    seen.add(key)
                    results.append((file_path, line_num))
        
        return results
    
    def _is_project_file(self, file_path: str) -> bool:
        """Check if file is likely part of the project, not a library."""
        skip_patterns = [
            r"^/usr/", r"^/lib/", r"site-packages",
            r"node_modules", r"vendor/", r"\.venv",
            r"^<", r"^/opt/", r"^/home/.*?/\.",
        ]
        for pattern in skip_patterns:
            if re.search(pattern, file_path):
                return False
        return True


class TestFilesExtractionStrategy(ExtractionStrategy):
    """Extract ground truth from test file additions/modifications."""
    
    @property
    def source_type(self) -> GroundTruthSource:
        return GroundTruthSource.TEST_ADDITION
    
    @property
    def base_confidence(self) -> float:
        return 0.80  # High confidence - tests validate the fix
    
    TEST_FILE_PATTERNS = [
        r"test_[a-zA-Z0-9_]+\.[a-zA-Z]+$",  # Python: test_foo.py
        r"[a-zA-Z0-9_]+_test\.[a-zA-Z]+$",  # Go: foo_test.go
        r"[a-zA-Z0-9_]+\.test\.[a-zA-Z]+$",  # JS: foo.test.js
        r"[a-zA-Z0-9_]+\.spec\.[a-zA-Z]+$",  # JS: foo.spec.js
        r"tests?/",  # In tests directory
        r"__tests__/",  # Jest style
        r"spec/",  # RSpec style
    ]
    
    def extract(self, context: Dict[str, Any]) -> List[SourcedLocation]:
        locations = []
        
        # Check commit files for test additions
        commit_files = context.get("commit_files", [])
        pr_files = context.get("pr_files", [])
        
        all_files = []
        for f in commit_files:
            if isinstance(f, str):
                all_files.append({"filename": f, "status": "modified"})
            else:
                all_files.append(f)
        
        for f in pr_files:
            if isinstance(f, str):
                all_files.append({"filename": f, "status": "modified"})
            else:
                all_files.append(f)
        
        for file_info in all_files:
            file_path = file_info.get("filename", file_info.get("path", ""))
            status = file_info.get("status", "modified")
            
            if not self._is_test_file(file_path):
                continue
            
            # Boost confidence for new tests
            confidence = self.base_confidence
            if status == "added":
                confidence = min(0.95, confidence + 0.1)
            
            loc = CodeLocation(file_path=file_path)
            
            source = (
                GroundTruthSource.TEST_ADDITION if status == "added"
                else GroundTruthSource.TEST_MODIFICATION
            )
            
            locations.append(SourcedLocation(
                location=loc,
                source=source,
                confidence=confidence,
                evidence=f"Test file {status}",
            ))
        
        return locations
    
    def _is_test_file(self, file_path: str) -> bool:
        """Check if file is a test file."""
        for pattern in self.TEST_FILE_PATTERNS:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        return False


# =============================================================================
# Ground Truth Pipeline
# =============================================================================

class EnterpriseGroundTruthPipeline:
    """Pipeline for extracting ground truth from enterprise codebases.
    
    Orchestrates multiple extraction strategies and combines results
    with confidence-weighted deduplication.
    """
    
    def __init__(
        self,
        strategies: Optional[List[ExtractionStrategy]] = None,
        min_confidence: float = 0.5,
    ):
        """Initialize the pipeline.
        
        Args:
            strategies: List of extraction strategies to use
            min_confidence: Minimum confidence threshold for inclusion
        """
        self.strategies = strategies or [
            GitCommitExtractionStrategy(),
            PRFilesExtractionStrategy(),
            PRReviewExtractionStrategy(),
            IssueTextExtractionStrategy(),
            StackTraceExtractionStrategy(),
            TestFilesExtractionStrategy(),
        ]
        self.min_confidence = min_confidence
    
    def extract(
        self,
        context: Dict[str, Any],
        task_id: str = "",
        granularity: RetrievalGranularity = RetrievalGranularity.FILE,
    ) -> EnterpriseGroundTruth:
        """Extract ground truth from all sources.
        
        Args:
            context: Extraction context with all available data
            task_id: Task identifier
            granularity: Desired retrieval granularity
            
        Returns:
            EnterpriseGroundTruth with combined results
        """
        all_sourced_locations: List[SourcedLocation] = []
        sources_used: Set[GroundTruthSource] = set()
        confidence_by_source: Dict[str, List[float]] = {}
        
        # Run all strategies
        for strategy in self.strategies:
            try:
                locations = strategy.extract(context)
                if locations:
                    all_sourced_locations.extend(locations)
                    sources_used.add(strategy.source_type)
                    
                    # Track confidence by source
                    source_name = strategy.source_type.value
                    if source_name not in confidence_by_source:
                        confidence_by_source[source_name] = []
                    for loc in locations:
                        confidence_by_source[source_name].append(loc.confidence)
            except Exception as e:
                logger.warning(f"Strategy {strategy.source_type} failed: {e}")
                continue
        
        # Deduplicate and merge locations
        merged_locations, merged_sourced = self._deduplicate_locations(
            all_sourced_locations,
            granularity,
        )
        
        # Filter by confidence threshold
        final_locations = [
            loc for loc in merged_locations
            if self._get_location_confidence(loc, merged_sourced) >= self.min_confidence
        ]
        
        # Compute overall confidence
        overall_confidence = self._compute_overall_confidence(merged_sourced)
        
        # Average confidence by source
        avg_confidence_by_source = {
            source: sum(confs) / len(confs)
            for source, confs in confidence_by_source.items()
            if confs
        }
        
        return EnterpriseGroundTruth(
            locations=final_locations,
            sourced_locations=merged_sourced,
            sources_used=sources_used,
            granularity=granularity,
            overall_confidence=overall_confidence,
            confidence_by_source=avg_confidence_by_source,
            task_id=task_id,
            metadata={
                "strategy_count": len(self.strategies),
                "total_sourced_locations": len(all_sourced_locations),
                "merged_locations": len(merged_locations),
                "final_locations": len(final_locations),
            },
        )
    
    def _deduplicate_locations(
        self,
        sourced_locations: List[SourcedLocation],
        granularity: RetrievalGranularity,
    ) -> Tuple[List[CodeLocation], List[SourcedLocation]]:
        """Deduplicate locations, keeping highest confidence for each."""
        # Group by file path (or file+line for finer granularity)
        location_map: Dict[str, List[SourcedLocation]] = {}
        
        for sl in sourced_locations:
            if granularity == RetrievalGranularity.FILE:
                key = sl.location.file_path
            else:
                key = f"{sl.location.file_path}:{sl.location.start_line or 0}"
            
            if key not in location_map:
                location_map[key] = []
            location_map[key].append(sl)
        
        # Merge duplicates
        merged_locations = []
        merged_sourced = []
        
        for key, sls in location_map.items():
            # Take the location from highest confidence source
            best_sl = max(sls, key=lambda x: x.confidence)
            
            # Boost confidence if multiple sources agree
            if len(sls) > 1:
                boost = min(0.1 * (len(sls) - 1), 0.2)
                best_sl = SourcedLocation(
                    location=best_sl.location,
                    source=best_sl.source,
                    confidence=min(1.0, best_sl.confidence + boost),
                    evidence=f"{best_sl.evidence} (confirmed by {len(sls)} sources)",
                )
            
            merged_locations.append(best_sl.location)
            merged_sourced.append(best_sl)
        
        # Sort by confidence
        merged_sourced.sort(key=lambda x: x.confidence, reverse=True)
        merged_locations = [sl.location for sl in merged_sourced]
        
        return merged_locations, merged_sourced
    
    def _get_location_confidence(
        self,
        location: CodeLocation,
        sourced_locations: List[SourcedLocation],
    ) -> float:
        """Get confidence for a location."""
        for sl in sourced_locations:
            if sl.location.file_path == location.file_path:
                return sl.confidence
        return 0.0
    
    def _compute_overall_confidence(
        self,
        sourced_locations: List[SourcedLocation],
    ) -> float:
        """Compute overall confidence from sourced locations."""
        if not sourced_locations:
            return 0.0
        
        # Weight by position and confidence
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i, sl in enumerate(sourced_locations):
            # Earlier (higher confidence) locations get more weight
            position_weight = 1.0 / (i + 1)
            weighted_sum += sl.confidence * position_weight
            total_weight += position_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def extract_ground_truth_from_issue(
    issue_data: Dict[str, Any],
    commit_data: Optional[Dict[str, Any]] = None,
    pr_data: Optional[Dict[str, Any]] = None,
) -> EnterpriseGroundTruth:
    """Extract ground truth from an issue with optional commit/PR data.
    
    Convenience function for common use case of issue-based extraction.
    
    Args:
        issue_data: Issue data with title, body, comments
        commit_data: Optional fix commit data
        pr_data: Optional linked PR data
        
    Returns:
        Extracted EnterpriseGroundTruth
    """
    context = {
        "issue_title": issue_data.get("title", ""),
        "issue_body": issue_data.get("body", ""),
        "issue_comments": issue_data.get("comments", []),
        "stack_trace": issue_data.get("stack_trace", ""),
    }
    
    if commit_data:
        context.update({
            "commit_sha": commit_data.get("sha", ""),
            "commit_message": commit_data.get("message", ""),
            "commit_files": commit_data.get("files", []),
        })
    
    if pr_data:
        context.update({
            "pr_number": pr_data.get("number", ""),
            "pr_files": pr_data.get("files", []),
            "review_comments": pr_data.get("review_comments", []),
        })
    
    pipeline = EnterpriseGroundTruthPipeline()
    return pipeline.extract(
        context,
        task_id=issue_data.get("issue_url", ""),
    )


def extract_ground_truth_from_commit(
    repo_path: str,
    commit_sha: str,
) -> EnterpriseGroundTruth:
    """Extract ground truth from a local git commit.
    
    Args:
        repo_path: Path to local git repository
        commit_sha: Commit SHA to analyze
        
    Returns:
        Extracted EnterpriseGroundTruth
    """
    try:
        # Get commit info
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s", commit_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        commit_message = result.stdout.strip()
        
        # Get changed files
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "--name-status", "-r", commit_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        
        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    status, path = parts[0], parts[1]
                    files.append({"filename": path, "status": status})
        
        context = {
            "commit_sha": commit_sha,
            "commit_message": commit_message,
            "commit_files": files,
        }
        
        pipeline = EnterpriseGroundTruthPipeline()
        return pipeline.extract(context, task_id=f"commit:{commit_sha}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git command failed: {e}")
        return EnterpriseGroundTruth(task_id=f"commit:{commit_sha}")


def batch_extract_from_commits(
    repo_path: str,
    commit_shas: List[str],
) -> List[EnterpriseGroundTruth]:
    """Extract ground truth from multiple commits.
    
    Args:
        repo_path: Path to local git repository
        commit_shas: List of commit SHAs to analyze
        
    Returns:
        List of extracted EnterpriseGroundTruth objects
    """
    results = []
    
    for sha in commit_shas:
        gt = extract_ground_truth_from_commit(repo_path, sha)
        results.append(gt)
    
    return results
