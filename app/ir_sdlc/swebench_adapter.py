"""
SWE-bench Integration Adapter for IR-SDLC-Bench.

This module provides integration between SWE-bench dataset format and
the IR-SDLC-Bench framework, enabling evaluation of IR tools on
SWE-bench tasks.

Key features:
1. Task conversion from SWE-bench to Harbor format
2. Environment configuration (baseline, deepsearch_mcp, sourcegraph_mcp)
3. Integration with swebench-verified dataset
4. Agent import path configuration

Reference: CodeContextBench swe_bench_configs/ and benchmarks/swebench_pro/
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional, Literal

from app.ir_sdlc.data_structures import (
    IRTask,
    IRDataset,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
)
from app.ir_sdlc.harbor_registry import (
    HarborRegistryConfig,
    HarborRegistryGenerator,
    EnvironmentConfig,
    DatasetEntry,
)


# =============================================================================
# SWE-bench Data Structures
# =============================================================================

class SWEBenchSplit(str, Enum):
    """SWE-bench dataset splits."""
    LITE = "lite"
    VERIFIED = "verified"
    FULL = "full"
    PRO = "pro"


@dataclass
class SWEBenchInstance:
    """
    A single SWE-bench task instance.
    
    Based on the SWE-bench dataset format:
    https://github.com/princeton-nlp/SWE-bench
    """
    
    # Instance identification
    instance_id: str
    
    # Repository information
    repo: str  # e.g., "django/django"
    version: str  # e.g., "3.0"
    base_commit: str
    
    # Problem specification
    problem_statement: str
    hints_text: str = ""
    
    # Solution information
    patch: str = ""
    test_patch: str = ""
    
    # Environment requirements
    environment_setup_commit: str = ""
    FAIL_TO_PASS: str = ""  # Tests that should pass after fix
    PASS_TO_PASS: str = ""  # Tests that should still pass
    
    # Metadata
    created_at: str = ""
    
    @classmethod
    def from_dict(cls, data: dict) -> "SWEBenchInstance":
        """Create instance from SWE-bench JSON data."""
        return cls(
            instance_id=data.get("instance_id", ""),
            repo=data.get("repo", ""),
            version=data.get("version", ""),
            base_commit=data.get("base_commit", ""),
            problem_statement=data.get("problem_statement", ""),
            hints_text=data.get("hints_text", ""),
            patch=data.get("patch", ""),
            test_patch=data.get("test_patch", ""),
            environment_setup_commit=data.get("environment_setup_commit", ""),
            FAIL_TO_PASS=data.get("FAIL_TO_PASS", ""),
            PASS_TO_PASS=data.get("PASS_TO_PASS", ""),
            created_at=data.get("created_at", ""),
        )
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def repo_url(self) -> str:
        """Get the GitHub URL for the repository."""
        return f"https://github.com/{self.repo}.git"
    
    @property
    def files_changed(self) -> list[str]:
        """Extract files changed from the patch."""
        if not self.patch:
            return []
        
        # Parse diff headers to extract file paths
        # Format: diff --git a/path/to/file b/path/to/file
        pattern = r'diff --git a/(.+?) b/\1'
        matches = re.findall(pattern, self.patch)
        return list(set(matches))


@dataclass
class SWEBenchDataset:
    """Collection of SWE-bench instances."""
    
    name: str
    split: SWEBenchSplit
    instances: list[SWEBenchInstance] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def load_jsonl(cls, path: Path, name: Optional[str] = None, split: SWEBenchSplit = SWEBenchSplit.VERIFIED) -> "SWEBenchDataset":
        """Load dataset from JSONL file."""
        instances = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instances.append(SWEBenchInstance.from_dict(data))
        
        return cls(
            name=name or path.stem,
            split=split,
            instances=instances,
        )
    
    @classmethod
    def load_json(cls, path: Path, name: Optional[str] = None, split: SWEBenchSplit = SWEBenchSplit.VERIFIED) -> "SWEBenchDataset":
        """Load dataset from JSON file (array format)."""
        with open(path) as f:
            data = json.load(f)
        
        instances = [SWEBenchInstance.from_dict(item) for item in data]
        
        return cls(
            name=name or path.stem,
            split=split,
            instances=instances,
        )
    
    def save_jsonl(self, path: Path) -> None:
        """Save dataset to JSONL file."""
        with open(path, "w") as f:
            for instance in self.instances:
                f.write(json.dumps(instance.to_dict()) + "\n")
    
    def filter_by_repo(self, repo: str) -> "SWEBenchDataset":
        """Filter instances by repository."""
        filtered = [i for i in self.instances if i.repo == repo]
        return SWEBenchDataset(
            name=f"{self.name}-{repo.replace('/', '-')}",
            split=self.split,
            instances=filtered,
        )
    
    def __len__(self) -> int:
        return len(self.instances)
    
    def __iter__(self) -> Iterator[SWEBenchInstance]:
        return iter(self.instances)


# =============================================================================
# Environment Configuration
# =============================================================================

class IRToolConfig(str, Enum):
    """IR tool configurations for SWE-bench evaluation."""
    BASELINE = "baseline"  # No IR tool, just agent reasoning
    DEEPSEARCH_MCP = "deepsearch_mcp"  # DeepSearch MCP server
    SOURCEGRAPH_MCP = "sourcegraph_mcp"  # Sourcegraph MCP server
    GREPTILE_MCP = "greptile_mcp"  # Greptile MCP server
    CUSTOM = "custom"  # Custom MCP configuration


@dataclass
class AgentConfig:
    """Configuration for the agent running SWE-bench tasks."""
    
    name: str
    import_path: str  # Python import path for the agent
    model_id: str = "claude-3-5-sonnet-20241022"
    
    # IR tool configuration
    ir_tool: IRToolConfig = IRToolConfig.BASELINE
    ir_tool_endpoint: Optional[str] = None
    
    # Timeouts
    agent_timeout: int = 1800
    per_task_timeout: int = 300
    
    # Resource limits
    max_tokens: int = 100000
    max_iterations: int = 50
    
    # Environment variables
    env_vars: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "import_path": self.import_path,
            "model_id": self.model_id,
            "ir_tool": self.ir_tool.value,
            "timeouts": {
                "agent": self.agent_timeout,
                "per_task": self.per_task_timeout,
            },
            "limits": {
                "max_tokens": self.max_tokens,
                "max_iterations": self.max_iterations,
            },
        }
        
        if self.ir_tool_endpoint:
            result["ir_tool_endpoint"] = self.ir_tool_endpoint
        if self.env_vars:
            result["env_vars"] = self.env_vars
            
        return result


@dataclass
class SWEBenchEnvironment:
    """Environment configuration for running SWE-bench evaluations."""
    
    name: str
    description: str = ""
    
    # Docker configuration
    base_image: str = "python:3.11-slim"
    dockerfile_path: Optional[str] = None
    
    # Resources
    cpus: int = 4
    memory_mb: int = 8192
    storage_mb: int = 50000
    
    # Network
    network_access: bool = True  # SWE-bench often needs network for pip
    
    # Python environment
    python_version: str = "3.11"
    pip_packages: list[str] = field(default_factory=list)
    
    # Repository setup
    clone_depth: int = 0  # 0 = full clone
    
    def to_environment_config(self) -> EnvironmentConfig:
        """Convert to HarborRegistry EnvironmentConfig."""
        return EnvironmentConfig(
            environment_type="docker",
            base_image=self.base_image,
            dockerfile_path=self.dockerfile_path,
            cpus=self.cpus,
            memory_mb=self.memory_mb,
            storage_mb=self.storage_mb,
            network_access=self.network_access,
        )
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "docker": {
                "base_image": self.base_image,
                "dockerfile": self.dockerfile_path,
            },
            "resources": {
                "cpus": self.cpus,
                "memory_mb": self.memory_mb,
                "storage_mb": self.storage_mb,
            },
            "python": {
                "version": self.python_version,
                "packages": self.pip_packages,
            },
            "network_access": self.network_access,
            "clone_depth": self.clone_depth,
        }


# Standard environment configurations
STANDARD_ENVIRONMENTS = {
    "baseline": SWEBenchEnvironment(
        name="baseline",
        description="Baseline environment without IR tools",
    ),
    "deepsearch": SWEBenchEnvironment(
        name="deepsearch",
        description="Environment with DeepSearch MCP server",
        pip_packages=["deepsearch-mcp"],
    ),
    "sourcegraph": SWEBenchEnvironment(
        name="sourcegraph",
        description="Environment with Sourcegraph MCP server",
        pip_packages=["sourcegraph-mcp"],
    ),
}


# =============================================================================
# SWE-bench to IR Task Conversion
# =============================================================================

class SWEBenchTaskConverter:
    """
    Converts SWE-bench instances to IR-SDLC-Bench tasks.
    
    This enables running IR evaluation on SWE-bench problems by
    treating the file localization step as an IR task.
    """
    
    def __init__(
        self,
        granularity: RetrievalGranularity = RetrievalGranularity.FILE,
        extract_functions: bool = False,
    ):
        self.granularity = granularity
        self.extract_functions = extract_functions
    
    def convert_instance(self, instance: SWEBenchInstance) -> IRTask:
        """Convert a SWE-bench instance to an IR task."""
        # Extract ground truth from patch
        ground_truth = self._extract_ground_truth(instance)
        
        # Build the query from problem statement
        query = self._build_query(instance)
        
        # Extract context
        context = self._extract_context(instance)
        
        # Estimate difficulty based on patch complexity
        difficulty = self._estimate_difficulty(instance)
        
        # Create the IR task
        task = IRTask(
            task_id=self._generate_task_id(instance),
            task_type="bug_fix",  # SWE-bench tasks are primarily bug fixes
            repo_name=instance.repo,
            repo_url=instance.repo_url,
            commit_hash=instance.base_commit,
            query=query,
            context=context,
            ground_truth=ground_truth,
            difficulty=difficulty,
            tags=self._generate_tags(instance),
            source_issue=instance.instance_id,
            harbor_metadata={
                "swe_bench_instance_id": instance.instance_id,
                "swe_bench_version": instance.version,
            },
        )
        
        return task
    
    def convert_dataset(self, dataset: SWEBenchDataset) -> IRDataset:
        """Convert an entire SWE-bench dataset to IR dataset."""
        tasks = [self.convert_instance(i) for i in dataset.instances]
        
        return IRDataset(
            name=f"ir-{dataset.name}",
            version="1.0.0",
            description=f"IR tasks derived from SWE-bench {dataset.split.value} split",
            tasks=tasks,
            metadata={
                "source": "swe-bench",
                "split": dataset.split.value,
                "task_count": len(tasks),
            },
        )
    
    def _generate_task_id(self, instance: SWEBenchInstance) -> str:
        """Generate a unique task ID."""
        # Use SWE-bench instance_id as base
        # Format: django__django-12345 -> django__django-bug_fix-12345
        parts = instance.instance_id.split("-")
        if len(parts) >= 2:
            repo_part = "-".join(parts[:-1])
            issue_part = parts[-1]
            return f"{repo_part}-bug_fix-{issue_part}"
        return f"{instance.instance_id}-bug_fix"
    
    def _build_query(self, instance: SWEBenchInstance) -> str:
        """Build the IR query from the problem statement."""
        query_parts = [
            f"Bug Fix Task: {instance.instance_id}",
            "",
            "Problem Description:",
            instance.problem_statement[:2000],  # Limit length
        ]
        
        if instance.hints_text:
            query_parts.extend([
                "",
                "Hints:",
                instance.hints_text[:500],
            ])
        
        query_parts.extend([
            "",
            "Find the files that need to be modified to fix this issue.",
        ])
        
        return "\n".join(query_parts)
    
    def _extract_ground_truth(self, instance: SWEBenchInstance) -> GroundTruth:
        """Extract ground truth file locations from the patch."""
        files = instance.files_changed
        
        locations = [
            CodeLocation(file_path=f)
            for f in files
        ]
        
        # If we want function-level granularity, parse the patch
        if self.extract_functions and self.granularity == RetrievalGranularity.FUNCTION:
            locations = self._extract_function_locations(instance)
        
        return GroundTruth(
            locations=locations,
            granularity=self.granularity,
            source="swe-bench-patch",
            confidence=1.0,  # Patches are ground truth
            metadata={
                "patch_files_count": len(files),
                "instance_id": instance.instance_id,
            },
        )
    
    def _extract_function_locations(self, instance: SWEBenchInstance) -> list[CodeLocation]:
        """Extract function-level locations from the patch."""
        locations = []
        
        # Parse diff hunks to find function names
        # This is a simplified heuristic - full parsing would need language-specific tools
        current_file = ""
        
        for line in instance.patch.split("\n"):
            # Track current file
            if line.startswith("diff --git"):
                match = re.search(r'b/(.+)$', line)
                if match:
                    current_file = match.group(1)
            
            # Look for function headers in context
            # Python: def function_name(
            # Java/JS: function_name(
            elif line.startswith("@@") and current_file:
                # Extract function context if present
                func_match = re.search(r'def\s+(\w+)\s*\(|class\s+(\w+)', line)
                if func_match:
                    func_name = func_match.group(1) or func_match.group(2)
                    locations.append(CodeLocation(
                        file_path=current_file,
                        function_name=func_name,
                    ))
        
        # Deduplicate and return
        seen = set()
        unique = []
        for loc in locations:
            key = (loc.file_path, loc.function_name)
            if key not in seen:
                seen.add(key)
                unique.append(loc)
        
        # Fall back to file-level if no functions found
        if not unique:
            return [CodeLocation(file_path=f) for f in instance.files_changed]
        
        return unique
    
    def _extract_context(self, instance: SWEBenchInstance) -> dict:
        """Extract additional context for the task."""
        context = {
            "swe_bench_instance_id": instance.instance_id,
            "version": instance.version,
            "files_changed": instance.files_changed,
        }
        
        # Add test information if available
        if instance.FAIL_TO_PASS:
            context["failing_tests"] = instance.FAIL_TO_PASS
        
        return context
    
    def _estimate_difficulty(self, instance: SWEBenchInstance) -> str:
        """Estimate task difficulty based on patch complexity."""
        files_count = len(instance.files_changed)
        patch_lines = len(instance.patch.split("\n"))
        
        if files_count == 1 and patch_lines < 50:
            return "easy"
        elif files_count <= 3 and patch_lines < 150:
            return "medium"
        elif files_count <= 5 and patch_lines < 300:
            return "hard"
        else:
            return "expert"
    
    def _generate_tags(self, instance: SWEBenchInstance) -> list[str]:
        """Generate tags for the task."""
        tags = ["swe-bench", "bug-fix"]
        
        # Add repository name
        repo_tag = instance.repo.replace("/", "-")
        tags.append(repo_tag)
        
        # Add version
        if instance.version:
            tags.append(f"v{instance.version}")
        
        # Detect language from file extensions
        extensions = set()
        for f in instance.files_changed:
            ext = Path(f).suffix.lower()
            if ext:
                extensions.add(ext[1:])  # Remove leading dot
        
        lang_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "java": "java",
            "go": "go",
            "rs": "rust",
            "rb": "ruby",
        }
        
        for ext in extensions:
            if ext in lang_map:
                tags.append(lang_map[ext])
        
        return tags


# =============================================================================
# SWE-bench Adapter (Main Interface)
# =============================================================================

@dataclass
class SWEBenchAdapterConfig:
    """Configuration for the SWE-bench adapter."""
    
    # Dataset source
    dataset_path: Optional[Path] = None
    dataset_split: SWEBenchSplit = SWEBenchSplit.VERIFIED
    
    # Conversion settings
    granularity: RetrievalGranularity = RetrievalGranularity.FILE
    extract_functions: bool = False
    
    # Output settings
    output_dir: Optional[Path] = None
    
    # Harbor registry settings
    git_url: str = ""
    git_commit_id: str = "HEAD"


class SWEBenchAdapter:
    """
    Main adapter for SWE-bench integration with IR-SDLC-Bench.
    
    Provides:
    - Loading SWE-bench datasets
    - Converting to IR task format
    - Generating Harbor registry entries
    - Running evaluations with different IR tool configurations
    
    Example usage:
        adapter = SWEBenchAdapter(
            SWEBenchAdapterConfig(
                dataset_path=Path("swe-bench-verified.jsonl"),
                git_url="https://github.com/org/benchmarks.git",
            )
        )
        
        # Load and convert dataset
        ir_dataset = adapter.load_and_convert()
        
        # Generate Harbor registry
        registry = adapter.generate_registry(ir_dataset)
    """
    
    def __init__(self, config: SWEBenchAdapterConfig):
        self.config = config
        self.converter = SWEBenchTaskConverter(
            granularity=config.granularity,
            extract_functions=config.extract_functions,
        )
        self._swe_dataset: Optional[SWEBenchDataset] = None
        self._ir_dataset: Optional[IRDataset] = None
    
    def load_dataset(self, path: Optional[Path] = None) -> SWEBenchDataset:
        """Load SWE-bench dataset from file."""
        path = path or self.config.dataset_path
        if path is None:
            raise ValueError("Dataset path not specified")
        
        if path.suffix == ".jsonl":
            self._swe_dataset = SWEBenchDataset.load_jsonl(
                path,
                split=self.config.dataset_split,
            )
        else:
            self._swe_dataset = SWEBenchDataset.load_json(
                path,
                split=self.config.dataset_split,
            )
        
        return self._swe_dataset
    
    def convert_to_ir_dataset(
        self,
        swe_dataset: Optional[SWEBenchDataset] = None,
    ) -> IRDataset:
        """Convert SWE-bench dataset to IR dataset."""
        dataset = swe_dataset or self._swe_dataset
        if dataset is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")
        
        self._ir_dataset = self.converter.convert_dataset(dataset)
        return self._ir_dataset
    
    def load_and_convert(self, path: Optional[Path] = None) -> IRDataset:
        """Load and convert in one step."""
        self.load_dataset(path)
        return self.convert_to_ir_dataset()
    
    def generate_registry(
        self,
        ir_dataset: Optional[IRDataset] = None,
        environment: Optional[SWEBenchEnvironment] = None,
    ) -> dict:
        """Generate Harbor registry for the converted dataset."""
        dataset = ir_dataset or self._ir_dataset
        if dataset is None:
            raise ValueError("No IR dataset available. Call load_and_convert first.")
        
        env = environment or STANDARD_ENVIRONMENTS["baseline"]
        
        config = HarborRegistryConfig(
            git_url=self.config.git_url,
            git_commit_id=self.config.git_commit_id,
            registry_name=f"swe-bench-{self.config.dataset_split.value}",
        )
        
        generator = HarborRegistryGenerator(config)
        generator.add_dataset(dataset, env.to_environment_config())
        
        return generator.generate()
    
    def save_ir_dataset(
        self,
        output_path: Optional[Path] = None,
        ir_dataset: Optional[IRDataset] = None,
    ) -> Path:
        """Save converted IR dataset to JSONL."""
        dataset = ir_dataset or self._ir_dataset
        if dataset is None:
            raise ValueError("No IR dataset available")
        
        if output_path is None:
            output_dir = self.config.output_dir or Path.cwd()
            output_path = output_dir / f"ir-swebench-{self.config.dataset_split.value}.jsonl"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_jsonl(output_path)
        
        return output_path
    
    def create_evaluation_config(
        self,
        agent_config: AgentConfig,
        environment: Optional[SWEBenchEnvironment] = None,
    ) -> dict:
        """Create a complete evaluation configuration."""
        env = environment or STANDARD_ENVIRONMENTS.get(
            agent_config.ir_tool.value,
            STANDARD_ENVIRONMENTS["baseline"],
        )
        
        return {
            "agent": agent_config.to_dict(),
            "environment": env.to_dict(),
            "dataset": {
                "name": self._ir_dataset.name if self._ir_dataset else "",
                "split": self.config.dataset_split.value,
                "task_count": len(self._ir_dataset.tasks) if self._ir_dataset else 0,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def get_repos(self) -> list[str]:
        """Get list of unique repositories in the dataset."""
        if self._swe_dataset is None:
            return []
        return list(set(i.repo for i in self._swe_dataset.instances))
    
    def filter_by_repo(self, repo: str) -> "SWEBenchAdapter":
        """Create a new adapter filtered to a specific repository."""
        if self._swe_dataset is None:
            raise ValueError("No dataset loaded")
        
        new_config = SWEBenchAdapterConfig(
            dataset_split=self.config.dataset_split,
            granularity=self.config.granularity,
            extract_functions=self.config.extract_functions,
            output_dir=self.config.output_dir,
            git_url=self.config.git_url,
            git_commit_id=self.config.git_commit_id,
        )
        
        new_adapter = SWEBenchAdapter(new_config)
        new_adapter._swe_dataset = self._swe_dataset.filter_by_repo(repo)
        
        return new_adapter


# =============================================================================
# Convenience Functions
# =============================================================================

def load_swebench_verified(path: Path) -> SWEBenchDataset:
    """Load SWE-bench verified dataset."""
    return SWEBenchDataset.load_jsonl(path, split=SWEBenchSplit.VERIFIED)


def load_swebench_lite(path: Path) -> SWEBenchDataset:
    """Load SWE-bench lite dataset."""
    return SWEBenchDataset.load_jsonl(path, split=SWEBenchSplit.LITE)


def convert_swebench_to_ir(
    swe_dataset: SWEBenchDataset,
    granularity: RetrievalGranularity = RetrievalGranularity.FILE,
) -> IRDataset:
    """Convert SWE-bench dataset to IR dataset."""
    converter = SWEBenchTaskConverter(granularity=granularity)
    return converter.convert_dataset(swe_dataset)


def create_agent_config(
    name: str,
    import_path: str,
    ir_tool: IRToolConfig = IRToolConfig.BASELINE,
    model_id: str = "claude-3-5-sonnet-20241022",
) -> AgentConfig:
    """Create an agent configuration."""
    return AgentConfig(
        name=name,
        import_path=import_path,
        ir_tool=ir_tool,
        model_id=model_id,
    )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for SWE-bench adapter."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench dataset to IR-SDLC-Bench format"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to SWE-bench JSONL or JSON file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output path for IR dataset JSONL",
    )
    parser.add_argument(
        "--split",
        choices=["lite", "verified", "full", "pro"],
        default="verified",
        help="SWE-bench split type",
    )
    parser.add_argument(
        "--granularity",
        choices=["file", "function"],
        default="file",
        help="Ground truth granularity",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Also generate Harbor registry.json",
    )
    parser.add_argument(
        "--git-url",
        help="Git URL for registry generation",
    )
    parser.add_argument(
        "--filter-repo",
        help="Filter to specific repository (e.g., django/django)",
    )
    parser.add_argument(
        "--list-repos",
        action="store_true",
        help="List repositories in the dataset and exit",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    split_map = {
        "lite": SWEBenchSplit.LITE,
        "verified": SWEBenchSplit.VERIFIED,
        "full": SWEBenchSplit.FULL,
        "pro": SWEBenchSplit.PRO,
    }
    
    granularity_map = {
        "file": RetrievalGranularity.FILE,
        "function": RetrievalGranularity.FUNCTION,
    }
    
    config = SWEBenchAdapterConfig(
        dataset_path=args.input,
        dataset_split=split_map[args.split],
        granularity=granularity_map[args.granularity],
        output_dir=args.output.parent if args.output else None,
        git_url=args.git_url or "",
    )
    
    adapter = SWEBenchAdapter(config)
    adapter.load_dataset()
    
    # List repos if requested
    if args.list_repos:
        repos = adapter.get_repos()
        print(f"Found {len(repos)} repositories:")
        for repo in sorted(repos):
            count = sum(1 for i in adapter._swe_dataset.instances if i.repo == repo)
            print(f"  {repo}: {count} instances")
        return
    
    # Filter if requested
    if args.filter_repo:
        adapter = adapter.filter_by_repo(args.filter_repo)
        print(f"Filtered to {len(adapter._swe_dataset)} instances from {args.filter_repo}")
    
    # Convert
    ir_dataset = adapter.convert_to_ir_dataset()
    print(f"Converted {len(ir_dataset.tasks)} tasks")
    
    # Save IR dataset
    if args.output:
        output_path = adapter.save_ir_dataset(args.output)
        print(f"Saved IR dataset to: {output_path}")
    
    # Generate registry if requested
    if args.registry:
        if not args.git_url:
            print("Warning: --git-url not provided, using placeholder")
            adapter.config.git_url = "https://github.com/placeholder/repo.git"
        
        registry = adapter.generate_registry()
        with open(args.registry, "w") as f:
            json.dump(registry, f, indent=2)
        print(f"Saved registry to: {args.registry}")


if __name__ == "__main__":
    main()
