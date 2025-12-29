"""
SDLC Task Types for IR-SDLC-Bench.

Each task type represents a different information retrieval scenario
that occurs during the software development lifecycle.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from app.ir_sdlc.data_structures import (
    IRTask,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
)


class SDLCTaskType(Enum):
    """Enumeration of SDLC task types for IR evaluation."""
    BUG_TRIAGE = "bug_triage"
    CODE_REVIEW = "code_review"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    ARCHITECTURE_UNDERSTANDING = "architecture_understanding"
    SECURITY_AUDIT = "security_audit"
    REFACTORING_ANALYSIS = "refactoring_analysis"
    TEST_COVERAGE = "test_coverage"
    DOCUMENTATION_LINKING = "documentation_linking"
    FEATURE_LOCATION = "feature_location"
    CHANGE_IMPACT_ANALYSIS = "change_impact_analysis"


@dataclass
class SDLCTaskGenerator(ABC):
    """Base class for generating SDLC-specific IR tasks."""

    task_type: SDLCTaskType
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    @abstractmethod
    def generate_query(self, source_data: dict) -> str:
        """Generate the natural language query from source data."""
        pass

    @abstractmethod
    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        """Extract ground truth retrieval targets from source data."""
        pass

    @abstractmethod
    def extract_context(self, source_data: dict) -> dict:
        """Extract additional context for the task."""
        pass

    def estimate_difficulty(self, source_data: dict, repo_stats: dict) -> str:
        """Estimate task difficulty based on various factors."""
        # Base difficulty on repository size and task complexity
        file_count = repo_stats.get("file_count", 0)
        loc = repo_stats.get("lines_of_code", 0)

        if file_count < 100 or loc < 10000:
            base_difficulty = "easy"
        elif file_count < 1000 or loc < 100000:
            base_difficulty = "medium"
        elif file_count < 5000 or loc < 500000:
            base_difficulty = "hard"
        else:
            base_difficulty = "expert"

        return base_difficulty

    def create_task(
        self,
        source_data: dict,
        repo_name: str,
        repo_url: str,
        commit_hash: str,
        repo_stats: Optional[dict] = None,
    ) -> IRTask:
        """Create an IR task from source data."""
        repo_stats = repo_stats or {}

        return IRTask(
            task_id="",  # Will be auto-generated
            task_type=self.task_type.value,
            repo_name=repo_name,
            repo_url=repo_url,
            commit_hash=commit_hash,
            query=self.generate_query(source_data),
            context=self.extract_context(source_data),
            ground_truth=self.extract_ground_truth(source_data),
            difficulty=self.estimate_difficulty(source_data, repo_stats),
            tags=self.generate_tags(source_data),
            source_issue=source_data.get("issue_url"),
            source_commit=source_data.get("fix_commit"),
            repo_stats=repo_stats,
        )

    def generate_tags(self, source_data: dict) -> list[str]:
        """Generate tags for the task."""
        return [self.task_type.value]


@dataclass
class BugTriageTask(SDLCTaskGenerator):
    """
    Bug Triage & Localization Task.

    Given a bug report, the IR tool must retrieve the relevant code locations
    that need to be examined or modified to fix the bug.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.BUG_TRIAGE, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FUNCTION

    def generate_query(self, source_data: dict) -> str:
        """Generate query from bug report."""
        title = source_data.get("title", "")
        body = source_data.get("body", "")

        # Include stack trace if available
        stack_trace = source_data.get("stack_trace", "")

        query_parts = []
        if title:
            query_parts.append(f"Bug Report: {title}")
        if body:
            # Truncate very long bodies
            body_truncated = body[:2000] if len(body) > 2000 else body
            query_parts.append(f"\nDescription:\n{body_truncated}")
        if stack_trace:
            query_parts.append(f"\nStack Trace:\n{stack_trace[:1000]}")

        return "\n".join(query_parts)

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        """Extract ground truth from the bug fix commit."""
        locations = []

        # Parse the fix patch to get modified files/functions
        patch = source_data.get("patch", "")
        if patch:
            locations.extend(self._parse_patch_locations(patch))

        # Also include explicitly marked locations
        if source_data.get("ground_truth_files"):
            for file_path in source_data["ground_truth_files"]:
                locations.append(CodeLocation(file_path=file_path))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={"fix_commit": source_data.get("fix_commit")},
        )

    def extract_context(self, source_data: dict) -> dict:
        """Extract additional context."""
        return {
            "issue_url": source_data.get("issue_url"),
            "labels": source_data.get("labels", []),
            "component": source_data.get("component"),
            "error_type": source_data.get("error_type"),
        }

    def _parse_patch_locations(self, patch: str) -> list[CodeLocation]:
        """Parse a git diff patch to extract file locations."""
        locations = []
        current_file = None

        for line in patch.split("\n"):
            # Match file paths in diff header
            if line.startswith("--- a/"):
                current_file = line[6:]
            elif line.startswith("+++ b/"):
                file_path = line[6:]
                if file_path != "/dev/null":
                    current_file = file_path
            elif line.startswith("@@") and current_file:
                # Parse hunk header for line numbers
                match = re.search(r"@@ -(\d+)", line)
                if match:
                    start_line = int(match.group(1))
                    locations.append(CodeLocation(
                        file_path=current_file,
                        start_line=start_line,
                    ))

        # Deduplicate by file
        seen_files = set()
        unique_locations = []
        for loc in locations:
            if loc.file_path not in seen_files:
                seen_files.add(loc.file_path)
                unique_locations.append(loc)

        return unique_locations

    def generate_tags(self, source_data: dict) -> list[str]:
        tags = [self.task_type.value]
        labels = source_data.get("labels", [])

        # Add relevant labels as tags
        for label in labels:
            if isinstance(label, str):
                tags.append(label.lower().replace(" ", "_"))

        return tags


@dataclass
class CodeReviewTask(SDLCTaskGenerator):
    """
    Code Review Assistance Task.

    Given a pull request, the IR tool must retrieve related code
    that provides context for the review.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.CODE_REVIEW, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    def generate_query(self, source_data: dict) -> str:
        """Generate query from PR information."""
        title = source_data.get("title", "")
        body = source_data.get("body", "")
        changed_files = source_data.get("changed_files", [])

        query_parts = [f"Pull Request: {title}"]

        if body:
            query_parts.append(f"\nDescription:\n{body[:1500]}")

        if changed_files:
            query_parts.append(f"\nChanged files: {', '.join(changed_files[:10])}")

        query_parts.append("\nFind related code that provides context for reviewing this change.")

        return "\n".join(query_parts)

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        """Extract ground truth from review activity."""
        locations = []

        # Files that reviewers commented on or examined
        reviewed_files = source_data.get("reviewed_files", [])
        for file_path in reviewed_files:
            locations.append(CodeLocation(file_path=file_path))

        # Files mentioned in review comments
        comment_files = source_data.get("comment_mentioned_files", [])
        for file_path in comment_files:
            locations.append(CodeLocation(file_path=file_path))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={"pr_number": source_data.get("pr_number")},
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "pr_url": source_data.get("pr_url"),
            "changed_files": source_data.get("changed_files", []),
            "additions": source_data.get("additions"),
            "deletions": source_data.get("deletions"),
        }


@dataclass
class DependencyAnalysisTask(SDLCTaskGenerator):
    """
    Dependency Analysis Task.

    Given a component, retrieve its dependencies and dependents.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.DEPENDENCY_ANALYSIS, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    def generate_query(self, source_data: dict) -> str:
        target_file = source_data.get("target_file", "")
        target_function = source_data.get("target_function", "")
        target_class = source_data.get("target_class", "")

        if target_function:
            return f"Find all code that depends on or is depended upon by the function `{target_function}` in `{target_file}`"
        elif target_class:
            return f"Find all code that depends on or is depended upon by the class `{target_class}` in `{target_file}`"
        else:
            return f"Find all code that depends on or is depended upon by `{target_file}`"

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Dependencies (files/modules this component imports)
        dependencies = source_data.get("dependencies", [])
        for dep in dependencies:
            locations.append(CodeLocation(file_path=dep))

        # Dependents (files/modules that import this component)
        dependents = source_data.get("dependents", [])
        for dep in dependents:
            locations.append(CodeLocation(file_path=dep))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={
                "dependency_count": len(dependencies),
                "dependent_count": len(dependents),
            },
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "target_file": source_data.get("target_file"),
            "target_function": source_data.get("target_function"),
            "target_class": source_data.get("target_class"),
            "analysis_type": source_data.get("analysis_type", "both"),  # imports, exports, both
        }


@dataclass
class ArchitectureUnderstandingTask(SDLCTaskGenerator):
    """
    Architecture Understanding Task.

    Given a feature or component query, retrieve the relevant
    architectural components.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.ARCHITECTURE_UNDERSTANDING, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    def generate_query(self, source_data: dict) -> str:
        feature = source_data.get("feature", "")
        query_type = source_data.get("query_type", "implementation")

        if query_type == "implementation":
            return f"Find the code that implements the {feature} feature"
        elif query_type == "entry_point":
            return f"Find the entry point(s) for the {feature} functionality"
        elif query_type == "data_flow":
            return f"Find the code involved in the data flow for {feature}"
        else:
            return f"Find all code related to {feature}"

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Expert-annotated or documentation-derived locations
        component_files = source_data.get("component_files", [])
        for file_path in component_files:
            locations.append(CodeLocation(file_path=file_path))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source=source_data.get("ground_truth_source", "expert"),
            metadata={"feature": source_data.get("feature")},
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "feature": source_data.get("feature"),
            "component": source_data.get("component"),
            "layer": source_data.get("layer"),  # e.g., "api", "service", "data"
            "documentation_links": source_data.get("documentation_links", []),
        }


@dataclass
class SecurityAuditTask(SDLCTaskGenerator):
    """
    Security Audit Task.

    Given a CVE or vulnerability pattern, retrieve potentially affected code.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.SECURITY_AUDIT, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FUNCTION

    def generate_query(self, source_data: dict) -> str:
        cve_id = source_data.get("cve_id", "")
        vulnerability_type = source_data.get("vulnerability_type", "")
        description = source_data.get("description", "")

        query_parts = []
        if cve_id:
            query_parts.append(f"Security vulnerability {cve_id}")
        if vulnerability_type:
            query_parts.append(f"Type: {vulnerability_type}")
        if description:
            query_parts.append(f"\n{description[:1000]}")

        query_parts.append("\nFind all code locations that may be affected by or contribute to this vulnerability.")

        return "\n".join(query_parts)

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Known vulnerable locations from CVE fix
        vulnerable_files = source_data.get("vulnerable_files", [])
        for file_info in vulnerable_files:
            if isinstance(file_info, str):
                locations.append(CodeLocation(file_path=file_info))
            elif isinstance(file_info, dict):
                locations.append(CodeLocation(
                    file_path=file_info.get("file_path", ""),
                    function_name=file_info.get("function"),
                    start_line=file_info.get("line"),
                ))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={
                "cve_id": source_data.get("cve_id"),
                "severity": source_data.get("severity"),
            },
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "cve_id": source_data.get("cve_id"),
            "vulnerability_type": source_data.get("vulnerability_type"),
            "severity": source_data.get("severity"),
            "affected_versions": source_data.get("affected_versions", []),
            "cwe_ids": source_data.get("cwe_ids", []),
        }


@dataclass
class RefactoringAnalysisTask(SDLCTaskGenerator):
    """
    Refactoring Analysis Task.

    Given a refactoring intent, retrieve code that should be modified.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.REFACTORING_ANALYSIS, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FUNCTION

    def generate_query(self, source_data: dict) -> str:
        refactoring_type = source_data.get("refactoring_type", "")
        target = source_data.get("target", "")
        description = source_data.get("description", "")

        if refactoring_type and target:
            return f"Perform {refactoring_type} refactoring on {target}. {description}"
        else:
            return f"Refactoring task: {description}"

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Files modified in the refactoring commit
        refactored_files = source_data.get("refactored_files", [])
        for file_path in refactored_files:
            locations.append(CodeLocation(file_path=file_path))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={"refactoring_type": source_data.get("refactoring_type")},
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "refactoring_type": source_data.get("refactoring_type"),
            "target": source_data.get("target"),
            "motivation": source_data.get("motivation"),
        }


@dataclass
class TestCoverageTask(SDLCTaskGenerator):
    """
    Test Coverage Analysis Task.

    Given code changes, retrieve relevant tests.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.TEST_COVERAGE, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    def generate_query(self, source_data: dict) -> str:
        changed_files = source_data.get("changed_files", [])
        change_description = source_data.get("change_description", "")

        query_parts = ["Find tests that cover or are relevant to the following changes:"]

        if changed_files:
            query_parts.append(f"\nModified files: {', '.join(changed_files[:10])}")

        if change_description:
            query_parts.append(f"\nChange description: {change_description}")

        return "\n".join(query_parts)

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Tests that actually cover the changed code
        relevant_tests = source_data.get("relevant_tests", [])
        for test_file in relevant_tests:
            locations.append(CodeLocation(file_path=test_file))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source="automatic",
            metadata={"coverage_analysis": source_data.get("coverage_analysis")},
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "changed_files": source_data.get("changed_files", []),
            "change_type": source_data.get("change_type"),  # bug_fix, feature, refactor
        }


@dataclass
class DocumentationLinkingTask(SDLCTaskGenerator):
    """
    Documentation Linking Task.

    Given code or a query, retrieve relevant documentation.
    """

    task_type: SDLCTaskType = field(default=SDLCTaskType.DOCUMENTATION_LINKING, init=False)
    default_granularity: RetrievalGranularity = RetrievalGranularity.FILE

    def generate_query(self, source_data: dict) -> str:
        code_file = source_data.get("code_file", "")
        code_function = source_data.get("code_function", "")
        topic = source_data.get("topic", "")

        if code_file and code_function:
            return f"Find documentation for the function `{code_function}` in `{code_file}`"
        elif code_file:
            return f"Find documentation for `{code_file}`"
        elif topic:
            return f"Find documentation about {topic}"
        else:
            return "Find relevant documentation"

    def extract_ground_truth(self, source_data: dict) -> GroundTruth:
        locations = []

        # Documentation files linked to the code
        doc_files = source_data.get("documentation_files", [])
        for doc_file in doc_files:
            locations.append(CodeLocation(file_path=doc_file))

        return GroundTruth(
            locations=locations,
            granularity=self.default_granularity,
            source=source_data.get("ground_truth_source", "automatic"),
            metadata={},
        )

    def extract_context(self, source_data: dict) -> dict:
        return {
            "code_file": source_data.get("code_file"),
            "code_function": source_data.get("code_function"),
            "topic": source_data.get("topic"),
        }


# Registry of task generators
TASK_GENERATORS: dict[SDLCTaskType, type[SDLCTaskGenerator]] = {
    SDLCTaskType.BUG_TRIAGE: BugTriageTask,
    SDLCTaskType.CODE_REVIEW: CodeReviewTask,
    SDLCTaskType.DEPENDENCY_ANALYSIS: DependencyAnalysisTask,
    SDLCTaskType.ARCHITECTURE_UNDERSTANDING: ArchitectureUnderstandingTask,
    SDLCTaskType.SECURITY_AUDIT: SecurityAuditTask,
    SDLCTaskType.REFACTORING_ANALYSIS: RefactoringAnalysisTask,
    SDLCTaskType.TEST_COVERAGE: TestCoverageTask,
    SDLCTaskType.DOCUMENTATION_LINKING: DocumentationLinkingTask,
}


def get_task_generator(task_type: SDLCTaskType | str) -> SDLCTaskGenerator:
    """Get a task generator instance for the given task type."""
    if isinstance(task_type, str):
        task_type = SDLCTaskType(task_type)

    generator_class = TASK_GENERATORS.get(task_type)
    if generator_class is None:
        raise ValueError(f"Unknown task type: {task_type}")

    return generator_class(task_type)
