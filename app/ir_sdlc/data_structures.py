"""
Data structures for IR-SDLC-Bench tasks and evaluation results.

These structures are designed to be serializable to JSON and compatible
with the Harbor framework's task format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import hashlib


class RetrievalGranularity(Enum):
    """Granularity levels for retrieval evaluation."""
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    LINE = "line"
    SNIPPET = "snippet"
    SEMANTIC = "semantic"


@dataclass
class CodeLocation:
    """Represents a location in the codebase."""
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    symbol_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> "CodeLocation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def matches(self, other: "CodeLocation", granularity: RetrievalGranularity) -> bool:
        """Check if this location matches another at the given granularity."""
        if granularity == RetrievalGranularity.FILE:
            return self.file_path == other.file_path
        elif granularity == RetrievalGranularity.FUNCTION:
            return (self.file_path == other.file_path and
                    self.function_name == other.function_name)
        elif granularity == RetrievalGranularity.CLASS:
            return (self.file_path == other.file_path and
                    self.class_name == other.class_name)
        elif granularity == RetrievalGranularity.LINE:
            if self.file_path != other.file_path:
                return False
            if self.start_line is None or other.start_line is None:
                return False
            # Check for line overlap
            self_end = self.end_line or self.start_line
            other_end = other.end_line or other.start_line
            return not (self_end < other.start_line or other_end < self.start_line)
        else:
            # For snippet/semantic, use file + line overlap
            return self.matches(other, RetrievalGranularity.LINE)


@dataclass
class GroundTruth:
    """Ground truth for an IR task - the correct retrieval targets."""
    locations: list[CodeLocation] = field(default_factory=list)
    relevance_scores: Optional[list[float]] = None  # For graded relevance
    granularity: RetrievalGranularity = RetrievalGranularity.FILE
    source: str = "automatic"  # "automatic", "expert", "hybrid"
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "locations": [loc.to_dict() for loc in self.locations],
            "relevance_scores": self.relevance_scores,
            "granularity": self.granularity.value,
            "source": self.source,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundTruth":
        return cls(
            locations=[CodeLocation.from_dict(loc) for loc in data.get("locations", [])],
            relevance_scores=data.get("relevance_scores"),
            granularity=RetrievalGranularity(data.get("granularity", "file")),
            source=data.get("source", "automatic"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RetrievalResult:
    """A single retrieval result from an IR tool."""
    location: CodeLocation
    score: float
    snippet: Optional[str] = None
    explanation: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "location": self.location.to_dict(),
            "score": self.score,
            "snippet": self.snippet,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetrievalResult":
        return cls(
            location=CodeLocation.from_dict(data["location"]),
            score=data["score"],
            snippet=data.get("snippet"),
            explanation=data.get("explanation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IRTask:
    """
    An Information Retrieval task for SDLC evaluation.

    This is the core task structure that can be converted to Harbor format.
    """
    # Task identification
    task_id: str
    task_type: str  # One of the SDLC task types

    # Repository information
    repo_name: str  # e.g., "kubernetes/kubernetes"
    repo_url: str
    commit_hash: str

    # Task content
    query: str  # The natural language query/problem statement
    context: dict = field(default_factory=dict)  # Additional context (e.g., stack trace, PR info)

    # Ground truth
    ground_truth: Optional[GroundTruth] = None

    # Metadata
    difficulty: str = "medium"  # "easy", "medium", "hard", "expert"
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_issue: Optional[str] = None  # Original issue/PR URL if applicable
    source_commit: Optional[str] = None  # Commit that resolved the issue

    # Repository stats (for complexity estimation)
    repo_stats: dict = field(default_factory=dict)  # files, loc, contributors, etc.

    # Harbor compatibility
    harbor_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.task_id:
            self.task_id = self._generate_task_id()

    def _generate_task_id(self) -> str:
        """Generate a unique task ID based on content hash."""
        content = f"{self.repo_name}:{self.task_type}:{self.query}:{self.commit_hash}"
        hash_val = hashlib.sha256(content.encode()).hexdigest()[:12]
        repo_slug = self.repo_name.replace("/", "__")
        return f"{repo_slug}-{self.task_type}-{hash_val}"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "commit_hash": self.commit_hash,
            "query": self.query,
            "context": self.context,
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "created_at": self.created_at,
            "source_issue": self.source_issue,
            "source_commit": self.source_commit,
            "repo_stats": self.repo_stats,
            "harbor_metadata": self.harbor_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IRTask":
        ground_truth = None
        if data.get("ground_truth"):
            ground_truth = GroundTruth.from_dict(data["ground_truth"])

        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            repo_name=data["repo_name"],
            repo_url=data["repo_url"],
            commit_hash=data["commit_hash"],
            query=data["query"],
            context=data.get("context", {}),
            ground_truth=ground_truth,
            difficulty=data.get("difficulty", "medium"),
            tags=data.get("tags", []),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            source_issue=data.get("source_issue"),
            source_commit=data.get("source_commit"),
            repo_stats=data.get("repo_stats", {}),
            harbor_metadata=data.get("harbor_metadata", {}),
        )

    def to_json(self, path: Optional[Path] = None) -> str:
        """Serialize to JSON string, optionally saving to file."""
        json_str = json.dumps(self.to_dict(), indent=2)
        if path:
            path.write_text(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str: str) -> "IRTask":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: Path) -> "IRTask":
        """Load from JSON file."""
        return cls.from_json(path.read_text())


@dataclass
class IREvaluationResult:
    """Results from evaluating an IR tool on a single task."""
    task_id: str
    tool_name: str

    # Retrieval results
    retrieved_results: list[RetrievalResult] = field(default_factory=list)

    # Computed metrics
    metrics: dict = field(default_factory=dict)  # precision@k, recall@k, mrr, ndcg, etc.

    # Timing
    retrieval_time_ms: float = 0.0
    indexing_time_ms: Optional[float] = None

    # Metadata
    tool_config: dict = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "retrieved_results": [r.to_dict() for r in self.retrieved_results],
            "metrics": self.metrics,
            "retrieval_time_ms": self.retrieval_time_ms,
            "indexing_time_ms": self.indexing_time_ms,
            "tool_config": self.tool_config,
            "evaluated_at": self.evaluated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IREvaluationResult":
        return cls(
            task_id=data["task_id"],
            tool_name=data["tool_name"],
            retrieved_results=[RetrievalResult.from_dict(r) for r in data.get("retrieved_results", [])],
            metrics=data.get("metrics", {}),
            retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
            indexing_time_ms=data.get("indexing_time_ms"),
            tool_config=data.get("tool_config", {}),
            evaluated_at=data.get("evaluated_at", datetime.utcnow().isoformat()),
        )

    def to_harbor_reward(self) -> dict:
        """Convert to Harbor reward format for /logs/verifier/reward.json."""
        # Use the primary metric as the main reward, include others as additional metrics
        reward = {}

        # Add all metrics to the reward
        for metric_name, value in self.metrics.items():
            if isinstance(value, (int, float)):
                reward[metric_name] = float(value)

        return reward


@dataclass
class IRDataset:
    """A collection of IR tasks forming a benchmark dataset."""
    name: str
    version: str
    description: str
    tasks: list[IRTask] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
            "metadata": self.metadata,
        }

    def save_jsonl(self, path: Path) -> None:
        """Save tasks to JSONL format."""
        with open(path, "w") as f:
            for task in self.tasks:
                f.write(json.dumps(task.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: Path, name: str = "unnamed", version: str = "1.0") -> "IRDataset":
        """Load tasks from JSONL format."""
        tasks = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    tasks.append(IRTask.from_dict(json.loads(line)))
        return cls(name=name, version=version, description="", tasks=tasks)

    def to_harbor_registry(self, git_url: str, git_commit_id: str, base_path: str = "datasets") -> dict:
        """Generate Harbor registry.json entry for this dataset."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metrics": [
                {"type": "mean"},
                {"type": "max"},
            ],
            "tasks": [
                {
                    "name": task.task_id,
                    "git_url": git_url,
                    "git_commit_id": git_commit_id,
                    "path": f"{base_path}/{self.name}/{task.task_id}",
                }
                for task in self.tasks
            ],
        }
