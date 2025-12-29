"""
SDLC Benchmark Task Generation Pipeline.

This module generates IR benchmark tasks from enterprise GitHub repositories.
It mines issues, PRs, and commits to create realistic SDLC-focused evaluation
tasks with ground truth derived from actual developer activity.

Pipeline flow:
1. GitHubIssueMiner - Mines issues and linked PRs
2. GroundTruthExtractor - Extracts file locations from fix commits
3. DifficultyEstimator - Scores task difficulty based on repo complexity
4. SDLCBenchmarkPipeline - Orchestrates task generation

Output: JSONL dataset of IRTask objects compatible with Harbor and dashboard_exporter.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator

import requests

from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
)
from app.ir_sdlc.task_types import (
    SDLCTaskType,
    SDLCTaskGenerator,
    BugTriageTask,
    CodeReviewTask,
    SecurityAuditTask,
    RefactoringAnalysisTask,
    TestCoverageTask,
    TASK_GENERATORS,
    get_task_generator,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# GitHub Issue Mining
# =============================================================================

@dataclass
class GitHubIssue:
    """Represents a mined GitHub issue."""
    issue_number: int
    title: str
    body: str
    labels: list[str]
    state: str
    created_at: str
    closed_at: Optional[str]
    issue_url: str
    user: str
    
    # Linked PR/commit info
    linked_pr_number: Optional[int] = None
    linked_pr_url: Optional[str] = None
    fix_commit: Optional[str] = None
    
    # Parsed data
    has_stack_trace: bool = False
    stack_trace: Optional[str] = None
    error_type: Optional[str] = None
    component: Optional[str] = None
    
    def to_source_data(self) -> dict:
        """Convert to format expected by task generators."""
        return {
            "title": self.title,
            "body": self.body,
            "labels": self.labels,
            "issue_url": self.issue_url,
            "fix_commit": self.fix_commit,
            "stack_trace": self.stack_trace,
            "error_type": self.error_type,
            "component": self.component,
        }


@dataclass
class GitHubPullRequest:
    """Represents a mined GitHub pull request."""
    pr_number: int
    title: str
    body: str
    labels: list[str]
    state: str
    merged: bool
    created_at: str
    merged_at: Optional[str]
    pr_url: str
    user: str
    
    # Change info
    changed_files: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    
    # Review info
    reviewed_files: list[str] = field(default_factory=list)
    review_comments: list[dict] = field(default_factory=list)
    
    # Linked issue
    linked_issue_number: Optional[int] = None
    merge_commit_sha: Optional[str] = None
    
    def to_source_data(self) -> dict:
        """Convert to format expected by task generators."""
        return {
            "title": self.title,
            "body": self.body,
            "labels": self.labels,
            "pr_url": self.pr_url,
            "pr_number": self.pr_number,
            "changed_files": self.changed_files,
            "additions": self.additions,
            "deletions": self.deletions,
            "reviewed_files": self.reviewed_files,
            "comment_mentioned_files": self._extract_mentioned_files(),
            "fix_commit": self.merge_commit_sha,
        }
    
    def _extract_mentioned_files(self) -> list[str]:
        """Extract file paths mentioned in review comments."""
        files = []
        for comment in self.review_comments:
            path = comment.get("path")
            if path:
                files.append(path)
        return list(set(files))


class GitHubIssueMiner:
    """
    Mines GitHub issues and PRs for benchmark task generation.
    
    Focuses on:
    - Bug reports with linked fix commits
    - PRs with review activity
    - Security advisories with patches
    - Refactoring PRs with clear scope
    """
    
    def __init__(self, github_token: str, rate_limit_delay: float = 0.5):
        """
        Initialize the miner.
        
        Args:
            github_token: GitHub API token
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.token = github_token
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })
    
    def mine_bug_issues(
        self,
        repo: str,
        max_issues: int = 50,
        require_linked_pr: bool = True,
    ) -> Iterator[GitHubIssue]:
        """
        Mine closed bug issues with linked fix PRs.
        
        Args:
            repo: Repository in "owner/repo" format
            max_issues: Maximum number of issues to return
            require_linked_pr: Only return issues with linked PRs
            
        Yields:
            GitHubIssue objects
        """
        logger.info(f"Mining bug issues from {repo}")
        
        # Search for closed issues with bug label
        query = f"repo:{repo} is:issue is:closed label:bug"
        
        count = 0
        for issue_data in self._search_issues(query, max_issues * 2):
            if count >= max_issues:
                break
            
            issue = self._parse_issue(issue_data, repo)
            if issue is None:
                continue
            
            # Try to find linked PR
            if require_linked_pr:
                linked_pr = self._find_linked_pr(repo, issue.issue_number)
                if linked_pr is None:
                    continue
                issue.linked_pr_number = linked_pr["number"]
                issue.linked_pr_url = linked_pr["html_url"]
                issue.fix_commit = linked_pr.get("merge_commit_sha")
            
            # Parse stack trace from body
            issue.stack_trace = self._extract_stack_trace(issue.body)
            issue.has_stack_trace = issue.stack_trace is not None
            issue.error_type = self._extract_error_type(issue.body)
            
            count += 1
            yield issue
    
    def mine_prs_for_review(
        self,
        repo: str,
        max_prs: int = 50,
        min_review_comments: int = 1,
    ) -> Iterator[GitHubPullRequest]:
        """
        Mine merged PRs with review activity.
        
        Args:
            repo: Repository in "owner/repo" format
            max_prs: Maximum number of PRs to return
            min_review_comments: Minimum review comments required
            
        Yields:
            GitHubPullRequest objects
        """
        logger.info(f"Mining reviewed PRs from {repo}")
        
        query = f"repo:{repo} is:pr is:merged review:approved"
        
        count = 0
        for pr_data in self._search_issues(query, max_prs * 2, issue_type="pr"):
            if count >= max_prs:
                break
            
            pr = self._fetch_pr_details(repo, pr_data["number"])
            if pr is None:
                continue
            
            # Get review comments
            pr.review_comments = self._fetch_review_comments(repo, pr.pr_number)
            
            if len(pr.review_comments) < min_review_comments:
                continue
            
            # Get changed files
            pr.changed_files = self._fetch_pr_files(repo, pr.pr_number)
            pr.reviewed_files = list(set(
                c.get("path") for c in pr.review_comments if c.get("path")
            ))
            
            count += 1
            yield pr
    
    def mine_security_issues(
        self,
        repo: str,
        max_issues: int = 20,
    ) -> Iterator[GitHubIssue]:
        """
        Mine security-related issues.
        
        Args:
            repo: Repository in "owner/repo" format
            max_issues: Maximum number of issues to return
            
        Yields:
            GitHubIssue objects with security context
        """
        logger.info(f"Mining security issues from {repo}")
        
        # Try different security-related labels
        labels = ["security", "vulnerability", "cve", "security-issue"]
        
        for label in labels:
            query = f"repo:{repo} is:issue is:closed label:{label}"
            
            for issue_data in self._search_issues(query, max_issues // len(labels)):
                issue = self._parse_issue(issue_data, repo)
                if issue is None:
                    continue
                
                # Look for CVE references
                cve_match = re.search(r"CVE-\d{4}-\d{4,}", issue.body or "")
                if cve_match:
                    issue.error_type = cve_match.group(0)
                
                # Try to find linked PR
                linked_pr = self._find_linked_pr(repo, issue.issue_number)
                if linked_pr:
                    issue.linked_pr_number = linked_pr["number"]
                    issue.fix_commit = linked_pr.get("merge_commit_sha")
                
                yield issue
    
    def _search_issues(
        self,
        query: str,
        max_results: int,
        issue_type: str = "issue",
    ) -> Iterator[dict]:
        """Execute GitHub search API query."""
        page = 1
        per_page = min(100, max_results)
        count = 0
        
        while count < max_results:
            url = "https://api.github.com/search/issues"
            params = {
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            
            try:
                response = self.session.get(url, params=params)
                self._handle_rate_limit(response)
                response.raise_for_status()
                data = response.json()
                
                items = data.get("items", [])
                if not items:
                    break
                
                for item in items:
                    if count >= max_results:
                        break
                    yield item
                    count += 1
                
                page += 1
                time.sleep(self.rate_limit_delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Search error: {e}")
                break
    
    def _parse_issue(self, issue_data: dict, repo: str) -> Optional[GitHubIssue]:
        """Parse issue data into GitHubIssue."""
        try:
            return GitHubIssue(
                issue_number=issue_data["number"],
                title=issue_data.get("title", ""),
                body=issue_data.get("body") or "",
                labels=[l["name"] for l in issue_data.get("labels", [])],
                state=issue_data.get("state", ""),
                created_at=issue_data.get("created_at", ""),
                closed_at=issue_data.get("closed_at"),
                issue_url=issue_data.get("html_url", ""),
                user=issue_data.get("user", {}).get("login", ""),
            )
        except Exception as e:
            logger.warning(f"Failed to parse issue: {e}")
            return None
    
    def _find_linked_pr(self, repo: str, issue_number: int) -> Optional[dict]:
        """Find PR that closes this issue."""
        # Check timeline for linking events
        url = f"https://api.github.com/repos/{repo}/issues/{issue_number}/timeline"
        
        try:
            response = self.session.get(url, headers={"Accept": "application/vnd.github.mockingbird-preview+json"})
            self._handle_rate_limit(response)
            response.raise_for_status()
            
            for event in response.json():
                if event.get("event") == "cross-referenced":
                    source = event.get("source", {}).get("issue", {})
                    if source.get("pull_request"):
                        # This is a PR that references the issue
                        pr_number = source.get("number")
                        return self._get_pr(repo, pr_number)
                        
        except requests.exceptions.RequestException as e:
            logger.debug(f"Timeline fetch failed: {e}")
        
        return None
    
    def _get_pr(self, repo: str, pr_number: int) -> Optional[dict]:
        """Get PR details."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
        
        try:
            response = self.session.get(url)
            self._handle_rate_limit(response)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None
    
    def _fetch_pr_details(self, repo: str, pr_number: int) -> Optional[GitHubPullRequest]:
        """Fetch full PR details."""
        pr_data = self._get_pr(repo, pr_number)
        if pr_data is None:
            return None
        
        try:
            return GitHubPullRequest(
                pr_number=pr_data["number"],
                title=pr_data.get("title", ""),
                body=pr_data.get("body") or "",
                labels=[l["name"] for l in pr_data.get("labels", [])],
                state=pr_data.get("state", ""),
                merged=pr_data.get("merged", False),
                created_at=pr_data.get("created_at", ""),
                merged_at=pr_data.get("merged_at"),
                pr_url=pr_data.get("html_url", ""),
                user=pr_data.get("user", {}).get("login", ""),
                additions=pr_data.get("additions", 0),
                deletions=pr_data.get("deletions", 0),
                merge_commit_sha=pr_data.get("merge_commit_sha"),
            )
        except Exception as e:
            logger.warning(f"Failed to parse PR: {e}")
            return None
    
    def _fetch_pr_files(self, repo: str, pr_number: int) -> list[str]:
        """Fetch list of files changed in PR."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        
        try:
            response = self.session.get(url)
            self._handle_rate_limit(response)
            response.raise_for_status()
            return [f["filename"] for f in response.json()]
        except requests.exceptions.RequestException:
            return []
    
    def _fetch_review_comments(self, repo: str, pr_number: int) -> list[dict]:
        """Fetch review comments on PR."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
        
        try:
            response = self.session.get(url)
            self._handle_rate_limit(response)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return []
    
    def _extract_stack_trace(self, body: str) -> Optional[str]:
        """Extract stack trace from issue body."""
        if not body:
            return None
        
        # Common stack trace patterns
        patterns = [
            # Python tracebacks
            r"(Traceback \(most recent call last\):[\s\S]*?(?:\n\n|\Z))",
            # Java/Kotlin exceptions
            r"((?:Exception|Error|Throwable)[^\n]*\n(?:\s+at\s+[^\n]+\n)+)",
            # JavaScript errors
            r"((?:Error|TypeError|ReferenceError)[^\n]*\n(?:\s+at\s+[^\n]+\n)+)",
            # Go panics
            r"(panic:[\s\S]*?(?:\n\n|\Z))",
            # Rust panics
            r"(thread '[^']+' panicked[\s\S]*?(?:\n\n|\Z))",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1)[:2000]  # Truncate long traces
        
        return None
    
    def _extract_error_type(self, body: str) -> Optional[str]:
        """Extract error type from issue body."""
        if not body:
            return None
        
        # Common error type patterns
        patterns = [
            r"(\w+Error):",
            r"(\w+Exception):",
            r"panic:\s+(\w+)",
            r"CVE-\d{4}-\d+",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, body)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        return None
    
    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle GitHub rate limiting."""
        if "X-RateLimit-Remaining" in response.headers:
            remaining = int(response.headers["X-RateLimit-Remaining"])
            if remaining < 10:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(0, reset_time - time.time()) + 1
                logger.warning(f"Rate limit low ({remaining}), waiting {wait_time}s")
                time.sleep(min(wait_time, 60))


# =============================================================================
# Ground Truth Extraction
# =============================================================================

@dataclass
class CommitDiff:
    """Parsed commit diff with file locations."""
    commit_sha: str
    message: str
    files: list[str]
    additions: int
    deletions: int
    patch: str
    locations: list[CodeLocation] = field(default_factory=list)


class GroundTruthExtractor:
    """
    Extracts ground truth file locations from commit diffs.
    
    Uses git history and PR changes to determine which files
    are relevant for each task.
    """
    
    def __init__(self, github_token: str):
        """
        Initialize the extractor.
        
        Args:
            github_token: GitHub API token
        """
        self.token = github_token
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })
    
    def extract_from_commit(
        self,
        repo: str,
        commit_sha: str,
        granularity: RetrievalGranularity = RetrievalGranularity.FILE,
    ) -> GroundTruth:
        """
        Extract ground truth from a specific commit.
        
        Args:
            repo: Repository in "owner/repo" format
            commit_sha: Commit SHA to analyze
            granularity: Desired retrieval granularity
            
        Returns:
            GroundTruth with extracted locations
        """
        diff = self._fetch_commit_diff(repo, commit_sha)
        if diff is None:
            return GroundTruth(
                locations=[],
                granularity=granularity,
                source="automatic",
                confidence=0.0,
            )
        
        locations = self._parse_diff_to_locations(diff, granularity)
        
        return GroundTruth(
            locations=locations,
            granularity=granularity,
            source="automatic",
            confidence=0.9,  # High confidence for fix commits
            metadata={
                "commit_sha": commit_sha,
                "commit_message": diff.message,
                "total_files": len(diff.files),
                "additions": diff.additions,
                "deletions": diff.deletions,
            },
        )
    
    def extract_from_pr(
        self,
        repo: str,
        pr_number: int,
        include_review_files: bool = True,
        granularity: RetrievalGranularity = RetrievalGranularity.FILE,
    ) -> GroundTruth:
        """
        Extract ground truth from a pull request.
        
        Args:
            repo: Repository in "owner/repo" format
            pr_number: PR number
            include_review_files: Include files mentioned in reviews
            granularity: Desired retrieval granularity
            
        Returns:
            GroundTruth with extracted locations
        """
        locations = []
        
        # Get changed files
        changed_files = self._fetch_pr_files(repo, pr_number)
        for file_path in changed_files:
            if self._is_relevant_file(file_path):
                locations.append(CodeLocation(file_path=file_path))
        
        # Get review file mentions
        if include_review_files:
            review_files = self._fetch_review_file_mentions(repo, pr_number)
            for file_path in review_files:
                if file_path not in [l.file_path for l in locations]:
                    locations.append(CodeLocation(file_path=file_path))
        
        return GroundTruth(
            locations=locations,
            granularity=granularity,
            source="automatic",
            confidence=0.85,
            metadata={
                "pr_number": pr_number,
                "changed_files_count": len(changed_files),
            },
        )
    
    def _fetch_commit_diff(self, repo: str, commit_sha: str) -> Optional[CommitDiff]:
        """Fetch commit diff from GitHub."""
        url = f"https://api.github.com/repos/{repo}/commits/{commit_sha}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            files = [f["filename"] for f in data.get("files", [])]
            additions = sum(f.get("additions", 0) for f in data.get("files", []))
            deletions = sum(f.get("deletions", 0) for f in data.get("files", []))
            
            # Build patch from file patches
            patches = []
            for f in data.get("files", []):
                if f.get("patch"):
                    patches.append(f"--- a/{f['filename']}\n+++ b/{f['filename']}\n{f['patch']}")
            
            return CommitDiff(
                commit_sha=commit_sha,
                message=data.get("commit", {}).get("message", ""),
                files=files,
                additions=additions,
                deletions=deletions,
                patch="\n".join(patches),
            )
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch commit {commit_sha}: {e}")
            return None
    
    def _fetch_pr_files(self, repo: str, pr_number: int) -> list[str]:
        """Fetch list of files changed in PR."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return [f["filename"] for f in response.json()]
        except requests.exceptions.RequestException:
            return []
    
    def _fetch_review_file_mentions(self, repo: str, pr_number: int) -> list[str]:
        """Fetch files mentioned in PR review comments."""
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return list(set(c.get("path") for c in response.json() if c.get("path")))
        except requests.exceptions.RequestException:
            return []
    
    def _parse_diff_to_locations(
        self,
        diff: CommitDiff,
        granularity: RetrievalGranularity,
    ) -> list[CodeLocation]:
        """Parse diff into code locations at specified granularity."""
        locations = []
        
        for file_path in diff.files:
            if not self._is_relevant_file(file_path):
                continue
            
            if granularity == RetrievalGranularity.FILE:
                locations.append(CodeLocation(file_path=file_path))
            else:
                # For finer granularity, parse the patch
                file_locations = self._parse_patch_locations(diff.patch, file_path)
                locations.extend(file_locations)
        
        return locations
    
    def _parse_patch_locations(self, patch: str, target_file: str) -> list[CodeLocation]:
        """Parse git patch to extract line-level locations for a file."""
        locations = []
        current_file = None
        
        for line in patch.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
            elif line.startswith("@@") and current_file == target_file:
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
                if match:
                    start_line = int(match.group(2))
                    count = int(match.group(3)) if match.group(3) else 1
                    locations.append(CodeLocation(
                        file_path=target_file,
                        start_line=start_line,
                        end_line=start_line + count - 1,
                    ))
        
        return locations
    
    def _is_relevant_file(self, file_path: str) -> bool:
        """Check if file is relevant for benchmarking (not config/generated)."""
        # Skip common non-relevant files
        skip_patterns = [
            r"\.lock$",
            r"package-lock\.json$",
            r"yarn\.lock$",
            r"\.min\.(js|css)$",
            r"\.generated\.",
            r"^vendor/",
            r"^node_modules/",
            r"^\.github/",
            r"\.md$",  # Documentation
            r"\.txt$",
            r"^docs/",
            r"CHANGELOG",
            r"LICENSE",
            r"AUTHORS",
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False
        
        return True


# =============================================================================
# Difficulty Estimation
# =============================================================================

@dataclass
class RepoComplexityStats:
    """Repository complexity statistics."""
    full_name: str
    file_count: int = 0
    directory_count: int = 0
    lines_of_code: int = 0
    contributor_count: int = 0
    commit_count: int = 0
    language_count: int = 0
    size_kb: int = 0
    stars: int = 0
    forks: int = 0
    
    # Complexity score (0-100)
    complexity_score: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


class DifficultyEstimator:
    """
    Estimates task difficulty based on repository and task characteristics.
    
    Uses repo complexity, issue scope, and ground truth spread to
    determine difficulty levels: easy, medium, hard, expert.
    """
    
    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize the estimator.
        
        Args:
            github_token: GitHub API token for fetching repo stats
        """
        self.token = github_token
        self.session = None
        if github_token:
            self.session = requests.Session()
            self.session.headers.update({
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            })
    
    def get_repo_stats(self, repo: str) -> RepoComplexityStats:
        """
        Fetch repository complexity statistics.
        
        Args:
            repo: Repository in "owner/repo" format
            
        Returns:
            RepoComplexityStats with complexity metrics
        """
        stats = RepoComplexityStats(full_name=repo)
        
        if not self.session:
            return stats
        
        # Fetch repo info
        try:
            response = self.session.get(f"https://api.github.com/repos/{repo}")
            response.raise_for_status()
            data = response.json()
            
            stats.size_kb = data.get("size", 0)
            stats.stars = data.get("stargazers_count", 0)
            stats.forks = data.get("forks_count", 0)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch repo stats: {e}")
        
        # Fetch languages
        try:
            response = self.session.get(f"https://api.github.com/repos/{repo}/languages")
            response.raise_for_status()
            languages = response.json()
            stats.language_count = len(languages)
            stats.lines_of_code = sum(languages.values())  # Approximate
            
        except requests.exceptions.RequestException:
            pass
        
        # Fetch contributors (limited)
        try:
            response = self.session.get(
                f"https://api.github.com/repos/{repo}/contributors",
                params={"per_page": 1, "anon": "1"},
            )
            response.raise_for_status()
            # Use link header to get count
            if "Link" in response.headers:
                # Parse last page from link header
                link = response.headers["Link"]
                last_page_match = re.search(r'page=(\d+)>; rel="last"', link)
                if last_page_match:
                    stats.contributor_count = int(last_page_match.group(1))
            else:
                stats.contributor_count = len(response.json())
                
        except requests.exceptions.RequestException:
            pass
        
        # Calculate complexity score
        stats.complexity_score = self._calculate_complexity_score(stats)
        
        return stats
    
    def _calculate_complexity_score(self, stats: RepoComplexityStats) -> float:
        """Calculate normalized complexity score (0-100)."""
        score = 0.0
        
        # Size contribution (0-30 points)
        if stats.size_kb > 0:
            size_score = min(30, stats.size_kb / 10000)  # 300MB = max
            score += size_score
        
        # LOC contribution (0-30 points)
        if stats.lines_of_code > 0:
            loc_score = min(30, stats.lines_of_code / 3333333)  # 100M lines = max
            score += loc_score
        
        # Contributors (0-20 points)
        if stats.contributor_count > 0:
            contrib_score = min(20, stats.contributor_count / 50)  # 1000 = max
            score += contrib_score
        
        # Languages (0-10 points)
        score += min(10, stats.language_count * 2)
        
        # Stars as popularity indicator (0-10 points)
        if stats.stars > 0:
            star_score = min(10, stats.stars / 10000)  # 100k = max
            score += star_score
        
        return min(100, score)
    
    def estimate_difficulty(
        self,
        repo_stats: RepoComplexityStats,
        ground_truth: GroundTruth,
        source_data: Optional[dict] = None,
    ) -> str:
        """
        Estimate task difficulty.
        
        Args:
            repo_stats: Repository complexity stats
            ground_truth: Ground truth for the task
            source_data: Optional source issue/PR data
            
        Returns:
            Difficulty level: "easy", "medium", "hard", or "expert"
        """
        score = 0.0
        
        # Repository complexity (0-40 points)
        score += repo_stats.complexity_score * 0.4
        
        # Ground truth spread (0-30 points)
        file_count = len(ground_truth.locations)
        if file_count == 1:
            score += 0
        elif file_count <= 3:
            score += 10
        elif file_count <= 7:
            score += 20
        else:
            score += 30
        
        # Check for cross-directory spread
        if file_count > 1:
            dirs = set(str(Path(loc.file_path).parent) for loc in ground_truth.locations)
            if len(dirs) > 3:
                score += 10
        
        # Issue complexity (0-20 points)
        if source_data:
            body = source_data.get("body", "")
            
            # Long descriptions indicate complexity
            if len(body) > 2000:
                score += 10
            elif len(body) > 500:
                score += 5
            
            # Stack traces indicate debugging complexity
            if source_data.get("stack_trace"):
                score += 5
            
            # Multiple labels indicate cross-cutting concerns
            labels = source_data.get("labels", [])
            if len(labels) > 3:
                score += 5
        
        # Map score to difficulty
        if score < 25:
            return "easy"
        elif score < 50:
            return "medium"
        elif score < 75:
            return "hard"
        else:
            return "expert"


# =============================================================================
# Main Pipeline
# =============================================================================

class SDLCBenchmarkPipeline:
    """
    Main pipeline for generating SDLC benchmark tasks.
    
    Orchestrates:
    1. Issue/PR mining
    2. Ground truth extraction
    3. Difficulty estimation
    4. Task generation and serialization
    """
    
    def __init__(
        self,
        github_token: str,
        output_dir: Optional[str] = None,
        rate_limit_delay: float = 0.5,
    ):
        """
        Initialize the pipeline.
        
        Args:
            github_token: GitHub API token
            output_dir: Directory for output files (default: ./benchmarks)
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.github_token = github_token
        self.output_dir = Path(output_dir) if output_dir else Path("./benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.miner = GitHubIssueMiner(github_token, rate_limit_delay)
        self.extractor = GroundTruthExtractor(github_token)
        self.estimator = DifficultyEstimator(github_token)
        
        # Task generators
        self.generators = {
            SDLCTaskType.BUG_TRIAGE: BugTriageTask(),
            SDLCTaskType.CODE_REVIEW: CodeReviewTask(),
            SDLCTaskType.SECURITY_AUDIT: SecurityAuditTask(),
            SDLCTaskType.REFACTORING_ANALYSIS: RefactoringAnalysisTask(),
            SDLCTaskType.TEST_COVERAGE: TestCoverageTask(),
        }
    
    def generate_from_repo(
        self,
        repo: str,
        task_types: Optional[list[SDLCTaskType]] = None,
        max_tasks_per_type: int = 10,
    ) -> IRDataset:
        """
        Generate benchmark tasks from a repository.
        
        Args:
            repo: Repository in "owner/repo" format
            task_types: List of task types to generate (default: all)
            max_tasks_per_type: Maximum tasks per type
            
        Returns:
            IRDataset containing generated tasks
        """
        if task_types is None:
            task_types = [
                SDLCTaskType.BUG_TRIAGE,
                SDLCTaskType.CODE_REVIEW,
                SDLCTaskType.SECURITY_AUDIT,
            ]
        
        logger.info(f"Generating benchmark tasks from {repo}")
        
        # Get repo stats
        repo_stats = self.estimator.get_repo_stats(repo)
        logger.info(f"Repository complexity score: {repo_stats.complexity_score:.1f}")
        
        # Get repo info for URL
        repo_url = f"https://github.com/{repo}"
        
        tasks = []
        
        for task_type in task_types:
            logger.info(f"Generating {task_type.value} tasks...")
            
            type_tasks = self._generate_tasks_for_type(
                repo=repo,
                repo_url=repo_url,
                task_type=task_type,
                repo_stats=repo_stats,
                max_tasks=max_tasks_per_type,
            )
            
            tasks.extend(type_tasks)
            logger.info(f"  Generated {len(type_tasks)} {task_type.value} tasks")
        
        # Create dataset
        dataset = IRDataset(
            name=f"ir-sdlc-{repo.replace('/', '-')}",
            version="1.0.0",
            description=f"IR-SDLC benchmark tasks from {repo}",
            tasks=tasks,
            metadata={
                "source_repo": repo,
                "repo_stats": repo_stats.to_dict(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "task_types": [t.value for t in task_types],
            },
        )
        
        return dataset
    
    def generate_from_repos(
        self,
        repos: list[str],
        task_types: Optional[list[SDLCTaskType]] = None,
        max_tasks_per_repo: int = 25,
    ) -> IRDataset:
        """
        Generate benchmark tasks from multiple repositories.
        
        Args:
            repos: List of repositories in "owner/repo" format
            task_types: List of task types to generate
            max_tasks_per_repo: Maximum tasks per repository
            
        Returns:
            Combined IRDataset
        """
        all_tasks = []
        
        for repo in repos:
            try:
                dataset = self.generate_from_repo(
                    repo=repo,
                    task_types=task_types,
                    max_tasks_per_type=max_tasks_per_repo // (len(task_types) if task_types else 3),
                )
                all_tasks.extend(dataset.tasks)
                logger.info(f"Completed {repo}: {len(dataset.tasks)} tasks")
            except Exception as e:
                logger.error(f"Failed to process {repo}: {e}")
        
        return IRDataset(
            name="ir-sdlc-multi-repo",
            version="1.0.0",
            description=f"IR-SDLC benchmark tasks from {len(repos)} repositories",
            tasks=all_tasks,
            metadata={
                "source_repos": repos,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    
    def _generate_tasks_for_type(
        self,
        repo: str,
        repo_url: str,
        task_type: SDLCTaskType,
        repo_stats: RepoComplexityStats,
        max_tasks: int,
    ) -> list[IRTask]:
        """Generate tasks of a specific type from a repository."""
        tasks = []
        generator = self.generators.get(task_type)
        
        if generator is None:
            logger.warning(f"No generator for {task_type}")
            return tasks
        
        if task_type == SDLCTaskType.BUG_TRIAGE:
            tasks = self._generate_bug_triage_tasks(
                repo, repo_url, generator, repo_stats, max_tasks
            )
        elif task_type == SDLCTaskType.CODE_REVIEW:
            tasks = self._generate_code_review_tasks(
                repo, repo_url, generator, repo_stats, max_tasks
            )
        elif task_type == SDLCTaskType.SECURITY_AUDIT:
            tasks = self._generate_security_tasks(
                repo, repo_url, generator, repo_stats, max_tasks
            )
        
        return tasks
    
    def _generate_bug_triage_tasks(
        self,
        repo: str,
        repo_url: str,
        generator: SDLCTaskGenerator,
        repo_stats: RepoComplexityStats,
        max_tasks: int,
    ) -> list[IRTask]:
        """Generate bug triage tasks from issues."""
        tasks = []
        
        for issue in self.miner.mine_bug_issues(repo, max_issues=max_tasks * 2):
            if len(tasks) >= max_tasks:
                break
            
            # Get ground truth from fix commit
            if issue.fix_commit:
                ground_truth = self.extractor.extract_from_commit(
                    repo, issue.fix_commit, generator.default_granularity
                )
            else:
                continue  # Skip issues without fix commits
            
            if not ground_truth.locations:
                continue  # Skip if no ground truth
            
            # Prepare source data
            source_data = issue.to_source_data()
            source_data["patch"] = ""  # We don't have the full patch in source_data
            source_data["ground_truth_files"] = [l.file_path for l in ground_truth.locations]
            
            # Estimate difficulty
            difficulty = self.estimator.estimate_difficulty(
                repo_stats, ground_truth, source_data
            )
            
            # Create task
            task = generator.create_task(
                source_data=source_data,
                repo_name=repo,
                repo_url=repo_url,
                commit_hash=issue.fix_commit,
                repo_stats=repo_stats.to_dict(),
            )
            
            # Override with our ground truth and difficulty
            task.ground_truth = ground_truth
            task.difficulty = difficulty
            
            tasks.append(task)
        
        return tasks
    
    def _generate_code_review_tasks(
        self,
        repo: str,
        repo_url: str,
        generator: SDLCTaskGenerator,
        repo_stats: RepoComplexityStats,
        max_tasks: int,
    ) -> list[IRTask]:
        """Generate code review tasks from PRs."""
        tasks = []
        
        for pr in self.miner.mine_prs_for_review(repo, max_prs=max_tasks * 2):
            if len(tasks) >= max_tasks:
                break
            
            # Get ground truth from PR
            ground_truth = self.extractor.extract_from_pr(
                repo, pr.pr_number, include_review_files=True
            )
            
            if not ground_truth.locations:
                continue
            
            # Prepare source data
            source_data = pr.to_source_data()
            
            # Estimate difficulty
            difficulty = self.estimator.estimate_difficulty(
                repo_stats, ground_truth, source_data
            )
            
            # Create task
            task = generator.create_task(
                source_data=source_data,
                repo_name=repo,
                repo_url=repo_url,
                commit_hash=pr.merge_commit_sha or "",
                repo_stats=repo_stats.to_dict(),
            )
            
            task.ground_truth = ground_truth
            task.difficulty = difficulty
            
            tasks.append(task)
        
        return tasks
    
    def _generate_security_tasks(
        self,
        repo: str,
        repo_url: str,
        generator: SDLCTaskGenerator,
        repo_stats: RepoComplexityStats,
        max_tasks: int,
    ) -> list[IRTask]:
        """Generate security audit tasks from security issues."""
        tasks = []
        
        for issue in self.miner.mine_security_issues(repo, max_issues=max_tasks * 2):
            if len(tasks) >= max_tasks:
                break
            
            # Get ground truth from fix commit
            if issue.fix_commit:
                ground_truth = self.extractor.extract_from_commit(
                    repo, issue.fix_commit, generator.default_granularity
                )
            else:
                continue
            
            if not ground_truth.locations:
                continue
            
            # Prepare source data for security task
            source_data = {
                "title": issue.title,
                "description": issue.body,
                "cve_id": issue.error_type if issue.error_type and issue.error_type.startswith("CVE") else None,
                "vulnerability_type": self._infer_vulnerability_type(issue.labels),
                "vulnerable_files": [{"file_path": l.file_path} for l in ground_truth.locations],
                "severity": self._infer_severity(issue.labels),
                "issue_url": issue.issue_url,
                "fix_commit": issue.fix_commit,
            }
            
            # Estimate difficulty
            difficulty = self.estimator.estimate_difficulty(
                repo_stats, ground_truth, source_data
            )
            
            # Create task
            task = generator.create_task(
                source_data=source_data,
                repo_name=repo,
                repo_url=repo_url,
                commit_hash=issue.fix_commit,
                repo_stats=repo_stats.to_dict(),
            )
            
            task.ground_truth = ground_truth
            task.difficulty = difficulty
            
            tasks.append(task)
        
        return tasks
    
    def _infer_vulnerability_type(self, labels: list[str]) -> Optional[str]:
        """Infer vulnerability type from issue labels."""
        vuln_keywords = {
            "xss": "Cross-Site Scripting",
            "sqli": "SQL Injection",
            "injection": "Injection",
            "csrf": "Cross-Site Request Forgery",
            "auth": "Authentication",
            "dos": "Denial of Service",
            "overflow": "Buffer Overflow",
            "rce": "Remote Code Execution",
            "path-traversal": "Path Traversal",
        }
        
        for label in labels:
            label_lower = label.lower()
            for keyword, vuln_type in vuln_keywords.items():
                if keyword in label_lower:
                    return vuln_type
        
        return None
    
    def _infer_severity(self, labels: list[str]) -> str:
        """Infer security severity from labels."""
        label_str = " ".join(labels).lower()
        
        if "critical" in label_str:
            return "critical"
        elif "high" in label_str:
            return "high"
        elif "medium" in label_str or "moderate" in label_str:
            return "medium"
        elif "low" in label_str or "minor" in label_str:
            return "low"
        
        return "unknown"
    
    def save_dataset(
        self,
        dataset: IRDataset,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save dataset to JSONL file.
        
        Args:
            dataset: Dataset to save
            filename: Output filename (default: {dataset.name}.jsonl)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"{dataset.name}.jsonl"
        
        output_path = self.output_dir / filename
        dataset.save_jsonl(output_path)
        
        # Also save metadata
        meta_path = self.output_dir / f"{dataset.name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(dataset.metadata, f, indent=2)
        
        logger.info(f"Saved {len(dataset.tasks)} tasks to {output_path}")
        
        return output_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for benchmark generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate SDLC benchmark tasks from GitHub repositories"
    )
    parser.add_argument(
        "repos",
        nargs="*",
        help="Repository names in owner/repo format",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmarks",
        help="Output directory for benchmark files",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=25,
        help="Maximum tasks per repository",
    )
    parser.add_argument(
        "--task-types",
        nargs="+",
        choices=["bug_triage", "code_review", "security_audit"],
        default=["bug_triage", "code_review"],
        help="Task types to generate",
    )
    parser.add_argument(
        "--use-selector",
        action="store_true",
        help="Use enterprise repo selector to find repos automatically",
    )
    parser.add_argument(
        "--selector-orgs",
        nargs="+",
        help="Organizations to scan with selector",
    )
    parser.add_argument(
        "--min-loc",
        type=int,
        default=500_000,
        help="Minimum LOC for repo selection",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=10,
        help="Maximum repos to select",
    )
    
    args = parser.parse_args()
    
    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable required")
        return 1
    
    # Determine repos to use
    repos = args.repos
    
    if args.use_selector or not repos:
        # Use repo selector to find enterprise repos
        from app.ir_sdlc.repo_selector import (
            EnterpriseRepoSelector,
            SelectionCriteria,
        )
        
        print("Using enterprise repo selector...")
        selector = EnterpriseRepoSelector(
            github_token=token,
            compute_loc=False,  # Use estimates for speed
        )
        
        candidates = selector.discover_candidates(
            orgs=args.selector_orgs,
            repos=repos if repos else None,
        )
        
        criteria = SelectionCriteria(min_loc=args.min_loc)
        selected = selector.select_repos(
            candidates,
            criteria=criteria,
            max_total=args.max_repos,
        )
        
        repos = [c.full_name for c in selected]
        print(f"Selected {len(repos)} repos: {repos}")
    
    if not repos:
        print("No repositories to process. Use --use-selector or provide repo names.")
        return 1
    
    # Initialize pipeline
    pipeline = SDLCBenchmarkPipeline(
        github_token=token,
        output_dir=args.output_dir,
    )
    
    # Parse task types
    task_types = [SDLCTaskType(t) for t in args.task_types]
    
    # Generate tasks
    if len(repos) == 1:
        dataset = pipeline.generate_from_repo(
            repo=repos[0],
            task_types=task_types,
            max_tasks_per_type=args.max_tasks,
        )
    else:
        dataset = pipeline.generate_from_repos(
            repos=repos,
            task_types=task_types,
            max_tasks_per_repo=args.max_tasks,
        )
    
    # Save
    output_path = pipeline.save_dataset(dataset)
    print(f"Generated {len(dataset.tasks)} tasks -> {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
