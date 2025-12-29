#!/usr/bin/env python3
"""
IR Tool Interface for IR-SDLC-Bench.

This module defines the abstract interface that all IR tools must implement
to be evaluated on the benchmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import time

from app.ir_sdlc.data_structures import RetrievalResult, CodeLocation


@dataclass
class IRToolConfig:
    """Configuration for an IR tool."""
    name: str
    version: str = "1.0"
    parameters: dict = field(default_factory=dict)

    # Resource limits
    max_index_time_sec: float = 3600.0  # 1 hour
    max_query_time_sec: float = 60.0
    max_memory_mb: int = 16384  # 16GB

    # Retrieval settings
    default_top_k: int = 20


class IRToolInterface(ABC):
    """
    Abstract interface for information retrieval tools.

    All IR tools being evaluated must implement this interface.
    """

    def __init__(self, config: IRToolConfig):
        self.config = config
        self._indexed_repo: Optional[str] = None
        self._index_time_ms: float = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_indexed(self) -> bool:
        return self._indexed_repo is not None

    @abstractmethod
    def index_repository(self, repo_path: str, **kwargs) -> None:
        """
        Index a repository for search.

        This method should build any necessary indexes or embeddings
        for the repository to enable fast retrieval.

        Args:
            repo_path: Absolute path to the repository root
            **kwargs: Additional indexing options
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant code given a query.

        Args:
            query: Natural language query
            top_k: Number of results to return
            **kwargs: Additional retrieval options

        Returns:
            List of RetrievalResult ordered by relevance (highest first)
        """
        pass

    def retrieve_with_context(
        self,
        query: str,
        context: dict,
        top_k: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        """
        Retrieve with additional context.

        Default implementation ignores context and calls retrieve().
        Override to use context (e.g., current file, stack trace).

        Args:
            query: Natural language query
            context: Additional context dict (e.g., current_file, stack_trace)
            top_k: Number of results to return
            **kwargs: Additional retrieval options

        Returns:
            List of RetrievalResult ordered by relevance
        """
        return self.retrieve(query, top_k, **kwargs)

    def get_info(self) -> dict:
        """Get information about the IR tool."""
        return {
            "name": self.config.name,
            "version": self.config.version,
            "parameters": self.config.parameters,
            "indexed_repo": self._indexed_repo,
            "index_time_ms": self._index_time_ms,
        }

    def clear_index(self) -> None:
        """Clear the current index."""
        self._indexed_repo = None
        self._index_time_ms = 0.0


class TimedIRTool(IRToolInterface):
    """
    Wrapper that adds timing to any IR tool implementation.
    """

    def __init__(self, tool: IRToolInterface):
        super().__init__(tool.config)
        self._tool = tool
        self._last_query_time_ms: float = 0.0

    def index_repository(self, repo_path: str, **kwargs) -> None:
        start = time.perf_counter()
        self._tool.index_repository(repo_path, **kwargs)
        self._index_time_ms = (time.perf_counter() - start) * 1000
        self._indexed_repo = repo_path

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[RetrievalResult]:
        start = time.perf_counter()
        results = self._tool.retrieve(query, top_k, **kwargs)
        self._last_query_time_ms = (time.perf_counter() - start) * 1000
        return results

    def retrieve_with_context(
        self,
        query: str,
        context: dict,
        top_k: int = 10,
        **kwargs,
    ) -> list[RetrievalResult]:
        start = time.perf_counter()
        results = self._tool.retrieve_with_context(query, context, top_k, **kwargs)
        self._last_query_time_ms = (time.perf_counter() - start) * 1000
        return results

    @property
    def last_query_time_ms(self) -> float:
        return self._last_query_time_ms


# ============================================================================
# Example IR Tool Implementations
# ============================================================================

class GrepBasedIRTool(IRToolInterface):
    """
    Simple grep-based IR tool for baseline comparison.

    Uses keyword matching to find relevant files.
    """

    def __init__(self, config: Optional[IRToolConfig] = None):
        if config is None:
            config = IRToolConfig(name="grep-baseline", version="1.0")
        super().__init__(config)
        self._repo_path: Optional[Path] = None
        self._file_index: list[Path] = []

    def index_repository(self, repo_path: str, **kwargs) -> None:
        """Index by collecting all code files."""
        self._repo_path = Path(repo_path)
        self._file_index = []

        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
            '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
            '.kt', '.scala', '.md', '.txt', '.yaml', '.yml', '.json',
        }

        for file_path in self._repo_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in code_extensions:
                # Skip common non-code directories
                parts = file_path.parts
                if any(p in parts for p in ['node_modules', 'vendor', 'venv', '__pycache__', '.git']):
                    continue
                self._file_index.append(file_path)

        self._indexed_repo = repo_path

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[RetrievalResult]:
        """Retrieve files by keyword matching."""
        if not self._repo_path:
            return []

        # Extract keywords from query
        keywords = self._extract_keywords(query)

        # Score each file
        scored_files = []
        for file_path in self._file_index:
            try:
                content = file_path.read_text(errors='ignore').lower()
                score = sum(content.count(kw.lower()) for kw in keywords)
                if score > 0:
                    rel_path = str(file_path.relative_to(self._repo_path))
                    scored_files.append((rel_path, score, content[:500]))
            except:
                continue

        # Sort by score
        scored_files.sort(key=lambda x: x[1], reverse=True)

        # Convert to results
        results = []
        for rel_path, score, snippet in scored_files[:top_k]:
            results.append(RetrievalResult(
                location=CodeLocation(file_path=rel_path),
                score=float(score),
                snippet=snippet,
            ))

        return results

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from a query."""
        import re
        # Remove common stop words and extract alphanumeric tokens
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                      'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'through', 'during', 'before', 'after', 'above', 'below',
                      'between', 'under', 'again', 'further', 'then', 'once',
                      'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'each', 'few', 'more', 'most', 'other', 'some', 'such',
                      'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
                      'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
                      'until', 'while', 'this', 'that', 'these', 'those', 'find',
                      'code', 'file', 'function', 'class', 'method'}

        tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', query)
        keywords = [t for t in tokens if t.lower() not in stop_words and len(t) > 2]

        return keywords


class FilePathIRTool(IRToolInterface):
    """
    IR tool that matches based on file paths and names.

    Useful for tasks where the query contains specific file or function names.
    """

    def __init__(self, config: Optional[IRToolConfig] = None):
        if config is None:
            config = IRToolConfig(name="filepath-matcher", version="1.0")
        super().__init__(config)
        self._repo_path: Optional[Path] = None
        self._file_index: dict[str, list[str]] = {}  # filename -> list of paths

    def index_repository(self, repo_path: str, **kwargs) -> None:
        """Index by building a filename lookup."""
        self._repo_path = Path(repo_path)
        self._file_index = {}

        for file_path in self._repo_path.rglob("*"):
            if file_path.is_file():
                parts = file_path.parts
                if any(p in parts for p in ['node_modules', 'vendor', 'venv', '__pycache__', '.git']):
                    continue

                filename = file_path.name.lower()
                rel_path = str(file_path.relative_to(self._repo_path))

                if filename not in self._file_index:
                    self._file_index[filename] = []
                self._file_index[filename].append(rel_path)

        self._indexed_repo = repo_path

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[RetrievalResult]:
        """Retrieve files by filename matching."""
        if not self._repo_path:
            return []

        import re

        # Extract potential filenames from query
        patterns = [
            r'`([^`]+\.[a-zA-Z]+)`',  # backtick-wrapped filenames
            r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z]+)',  # file.ext pattern
            r'([a-zA-Z_][a-zA-Z0-9_]+)',  # identifiers
        ]

        candidates = []
        for pattern in patterns:
            matches = re.findall(pattern, query)
            candidates.extend(matches)

        # Score files
        scored_files = []
        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Exact match
            if candidate_lower in self._file_index:
                for path in self._file_index[candidate_lower]:
                    scored_files.append((path, 100.0))

            # Partial match
            for filename, paths in self._file_index.items():
                if candidate_lower in filename:
                    for path in paths:
                        scored_files.append((path, 50.0))
                elif filename in candidate_lower:
                    for path in paths:
                        scored_files.append((path, 30.0))

        # Deduplicate and sort
        seen = set()
        unique_scored = []
        for path, score in sorted(scored_files, key=lambda x: -x[1]):
            if path not in seen:
                seen.add(path)
                unique_scored.append((path, score))

        # Convert to results
        results = []
        for rel_path, score in unique_scored[:top_k]:
            results.append(RetrievalResult(
                location=CodeLocation(file_path=rel_path),
                score=score,
            ))

        return results


# Registry of available IR tools
IR_TOOL_REGISTRY: dict[str, type[IRToolInterface]] = {
    "grep-baseline": GrepBasedIRTool,
    "filepath-matcher": FilePathIRTool,
}


def get_ir_tool(name: str, config: Optional[IRToolConfig] = None) -> IRToolInterface:
    """Get an IR tool instance by name."""
    if name not in IR_TOOL_REGISTRY:
        raise ValueError(f"Unknown IR tool: {name}. Available: {list(IR_TOOL_REGISTRY.keys())}")

    tool_class = IR_TOOL_REGISTRY[name]
    if config is None:
        config = IRToolConfig(name=name, version="1.0")
    return tool_class(config)


def register_ir_tool(name: str, tool_class: type[IRToolInterface]) -> None:
    """Register a new IR tool implementation."""
    IR_TOOL_REGISTRY[name] = tool_class
