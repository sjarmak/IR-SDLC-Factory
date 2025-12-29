"""
Harbor Registry Generator for IR-SDLC-Bench.

This module generates Harbor-compatible registry.json files for benchmark tasks,
enabling seamless integration with the Harbor evaluation framework.

Registry Format Reference: CodeContextBench/configs/harbor/registry.json

Key features:
- Task definitions with git_url and git_commit_id
- Dataset versioning and metadata
- Environment configuration (docker/daytona)
- Integration with HarborTaskGenerator for task directory generation
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Iterator, Literal

from app.ir_sdlc.data_structures import IRTask, IRDataset
from app.ir_sdlc.harbor_adapter import HarborTaskGenerator, HarborConfig


# =============================================================================
# Configuration Types
# =============================================================================

@dataclass
class EnvironmentConfig:
    """Configuration for task execution environment."""
    
    environment_type: Literal["docker", "daytona"] = "docker"
    
    # Docker-specific settings
    dockerfile_path: Optional[str] = None
    base_image: str = "python:3.11-slim"
    
    # Resource limits
    cpus: int = 2
    memory_mb: int = 4096
    storage_mb: int = 20480
    gpu_required: bool = False
    
    # Timeout settings (seconds)
    build_timeout: int = 600
    agent_timeout: int = 1800
    verifier_timeout: int = 300
    
    # Network settings
    network_access: bool = False
    
    def to_dict(self) -> dict:
        return {
            "type": self.environment_type,
            "dockerfile": self.dockerfile_path,
            "base_image": self.base_image,
            "resources": {
                "cpus": self.cpus,
                "memory_mb": self.memory_mb,
                "storage_mb": self.storage_mb,
                "gpu": self.gpu_required,
            },
            "timeouts": {
                "build": self.build_timeout,
                "agent": self.agent_timeout,
                "verifier": self.verifier_timeout,
            },
            "network_access": self.network_access,
        }


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics."""
    
    metric_type: Literal["mean", "max", "min", "sum", "weighted_mean"] = "mean"
    primary: bool = False
    weight: float = 1.0
    threshold: Optional[float] = None
    
    def to_dict(self) -> dict:
        result = {"type": self.metric_type}
        if self.primary:
            result["primary"] = True
        if self.weight != 1.0:
            result["weight"] = self.weight
        if self.threshold is not None:
            result["threshold"] = self.threshold
        return result


@dataclass 
class TaskEntry:
    """A single task entry in the registry."""
    
    name: str
    git_url: str
    git_commit_id: str
    path: str
    
    # Optional metadata
    difficulty: Optional[str] = None
    task_type: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    repo_name: Optional[str] = None
    
    # Override settings
    agent_timeout: Optional[int] = None
    verifier_timeout: Optional[int] = None
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "git_url": self.git_url,
            "git_commit_id": self.git_commit_id,
            "path": self.path,
        }
        
        # Add optional fields
        if self.difficulty:
            result["difficulty"] = self.difficulty
        if self.task_type:
            result["task_type"] = self.task_type
        if self.tags:
            result["tags"] = self.tags
        if self.repo_name:
            result["repo_name"] = self.repo_name
        if self.agent_timeout:
            result["agent_timeout"] = self.agent_timeout
        if self.verifier_timeout:
            result["verifier_timeout"] = self.verifier_timeout
            
        return result

    @classmethod
    def from_ir_task(
        cls,
        task: IRTask,
        git_url: str,
        git_commit_id: str,
        base_path: str = "datasets",
        dataset_name: Optional[str] = None,
    ) -> "TaskEntry":
        """Create a TaskEntry from an IRTask."""
        if dataset_name:
            path = f"{base_path}/{dataset_name}/{task.task_id}"
        else:
            path = f"{base_path}/{task.task_id}"
            
        return cls(
            name=task.task_id,
            git_url=git_url,
            git_commit_id=git_commit_id,
            path=path,
            difficulty=task.difficulty,
            task_type=task.task_type,
            tags=task.tags,
            repo_name=task.repo_name,
        )


@dataclass
class DatasetEntry:
    """A dataset entry in the registry."""
    
    name: str
    version: str
    description: str
    tasks: list[TaskEntry] = field(default_factory=list)
    
    # Metrics configuration
    metrics: list[MetricConfig] = field(default_factory=lambda: [
        MetricConfig(metric_type="mean", primary=True),
        MetricConfig(metric_type="max"),
    ])
    
    # Environment configuration
    environment: Optional[EnvironmentConfig] = None
    
    # Metadata
    author: str = "IR-SDLC-Bench"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)
    category: str = "information_retrieval"
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metrics": [m.to_dict() for m in self.metrics],
            "tasks": [t.to_dict() for t in self.tasks],
            "author": self.author,
            "created_at": self.created_at,
            "category": self.category,
        }
        
        if self.tags:
            result["tags"] = self.tags
        if self.environment:
            result["environment"] = self.environment.to_dict()
            
        return result
    
    @classmethod
    def from_ir_dataset(
        cls,
        dataset: IRDataset,
        git_url: str,
        git_commit_id: str = "HEAD",
        base_path: str = "datasets",
        environment: Optional[EnvironmentConfig] = None,
    ) -> "DatasetEntry":
        """Create a DatasetEntry from an IRDataset."""
        task_entries = [
            TaskEntry.from_ir_task(
                task=task,
                git_url=git_url,
                git_commit_id=git_commit_id,
                base_path=base_path,
                dataset_name=dataset.name,
            )
            for task in dataset.tasks
        ]
        
        # Extract tags from metadata
        tags = dataset.metadata.get("tags", [])
        
        return cls(
            name=dataset.name,
            version=dataset.version,
            description=dataset.description,
            tasks=task_entries,
            environment=environment,
            tags=tags,
        )


# =============================================================================
# Registry Generator
# =============================================================================

@dataclass
class HarborRegistryConfig:
    """Configuration for Harbor registry generation."""
    
    # Git repository settings
    git_url: str
    git_commit_id: str = "HEAD"
    
    # Paths
    base_path: str = "datasets"
    output_dir: Optional[str] = None
    
    # Environment defaults
    default_environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    
    # Registry metadata
    registry_name: str = "ir-sdlc-bench"
    registry_version: str = "1.0.0"
    registry_description: str = "IR-SDLC-Bench: Information Retrieval for SDLC Tasks"


class HarborRegistryGenerator:
    """
    Generates Harbor-compatible registry.json files for IR benchmark tasks.
    
    The registry format follows the Harbor specification and supports:
    - Multiple datasets with versioning
    - Task definitions with git URLs and commit IDs
    - Environment configuration (docker/daytona)
    - Metric configuration for evaluation
    
    Example usage:
        >>> config = HarborRegistryConfig(
        ...     git_url="https://github.com/org/ir-sdlc-bench.git",
        ...     git_commit_id="abc123",
        ... )
        >>> generator = HarborRegistryGenerator(config)
        >>> generator.add_dataset(dataset)
        >>> registry = generator.generate()
    """
    
    def __init__(self, config: HarborRegistryConfig):
        self.config = config
        self.datasets: list[DatasetEntry] = []
        self._task_generator = HarborTaskGenerator(
            HarborConfig(
                verifier_timeout=config.default_environment.verifier_timeout,
                agent_timeout=config.default_environment.agent_timeout,
                build_timeout=config.default_environment.build_timeout,
                cpus=config.default_environment.cpus,
                memory_mb=config.default_environment.memory_mb,
                storage_mb=config.default_environment.storage_mb,
            )
        )
    
    def add_dataset(
        self,
        dataset: IRDataset,
        environment: Optional[EnvironmentConfig] = None,
    ) -> DatasetEntry:
        """
        Add a dataset to the registry.
        
        Args:
            dataset: The IRDataset to add
            environment: Optional environment configuration override
            
        Returns:
            The created DatasetEntry
        """
        env = environment or self.config.default_environment
        
        entry = DatasetEntry.from_ir_dataset(
            dataset=dataset,
            git_url=self.config.git_url,
            git_commit_id=self.config.git_commit_id,
            base_path=self.config.base_path,
            environment=env,
        )
        
        self.datasets.append(entry)
        return entry
    
    def add_dataset_from_jsonl(
        self,
        jsonl_path: Path,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        environment: Optional[EnvironmentConfig] = None,
    ) -> DatasetEntry:
        """
        Add a dataset from a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file containing tasks
            name: Dataset name
            version: Dataset version
            description: Dataset description
            environment: Optional environment configuration
            
        Returns:
            The created DatasetEntry
        """
        dataset = IRDataset.load_jsonl(
            path=jsonl_path,
            name=name,
            version=version,
        )
        dataset.description = description
        
        return self.add_dataset(dataset, environment)
    
    def generate(self) -> dict:
        """
        Generate the complete registry.json structure.
        
        Returns:
            Dictionary suitable for serialization to registry.json
        """
        return {
            "$schema": "https://harbor.ai/registry/v1/schema.json",
            "name": self.config.registry_name,
            "version": self.config.registry_version,
            "description": self.config.registry_description,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_url": self.config.git_url,
            "git_commit_id": self.config.git_commit_id,
            "datasets": [d.to_dict() for d in self.datasets],
        }
    
    def save(self, output_path: Optional[Path] = None) -> Path:
        """
        Save the registry to a JSON file.
        
        Args:
            output_path: Path to save the registry.json file
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_dir = Path(self.config.output_dir) if self.config.output_dir else Path.cwd()
            output_path = output_dir / "registry.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        registry = self.generate()
        with open(output_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        return output_path
    
    def generate_task_directories(
        self,
        output_dir: Path,
        dataset_name: Optional[str] = None,
    ) -> list[Path]:
        """
        Generate Harbor task directories for all datasets.
        
        Args:
            output_dir: Base directory for task output
            dataset_name: Optional filter to generate only for a specific dataset
            
        Returns:
            List of paths to generated task directories
        """
        generated_paths = []
        
        for dataset_entry in self.datasets:
            if dataset_name and dataset_entry.name != dataset_name:
                continue
            
            # Reconstruct IRDataset from entries
            tasks = []
            for task_entry in dataset_entry.tasks:
                # Load task from JSONL if available, or create minimal task
                task = IRTask(
                    task_id=task_entry.name,
                    task_type=task_entry.task_type or "generic",
                    repo_name=task_entry.repo_name or "",
                    repo_url=task_entry.git_url,
                    commit_hash=task_entry.git_commit_id,
                    query="",  # Would need to be loaded from source
                    difficulty=task_entry.difficulty or "medium",
                    tags=task_entry.tags,
                )
                tasks.append(task)
            
            dataset = IRDataset(
                name=dataset_entry.name,
                version=dataset_entry.version,
                description=dataset_entry.description,
                tasks=tasks,
            )
            
            paths = self._task_generator.generate_dataset(dataset, output_dir)
            generated_paths.extend(paths)
        
        return generated_paths


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_registry_from_jsonl(
    jsonl_path: Path,
    git_url: str,
    git_commit_id: str = "HEAD",
    name: Optional[str] = None,
    version: str = "1.0.0",
    description: str = "",
    output_path: Optional[Path] = None,
) -> dict:
    """
    Generate a Harbor registry.json from a JSONL benchmark file.
    
    Args:
        jsonl_path: Path to the JSONL file
        git_url: Git URL for the benchmark repository
        git_commit_id: Git commit ID
        name: Dataset name (defaults to filename without extension)
        version: Dataset version
        description: Dataset description
        output_path: Optional path to save the registry
        
    Returns:
        The registry dictionary
    """
    if name is None:
        name = jsonl_path.stem
    
    config = HarborRegistryConfig(
        git_url=git_url,
        git_commit_id=git_commit_id,
    )
    
    generator = HarborRegistryGenerator(config)
    generator.add_dataset_from_jsonl(
        jsonl_path=jsonl_path,
        name=name,
        version=version,
        description=description,
    )
    
    registry = generator.generate()
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    return registry


def generate_registry_from_datasets(
    datasets: list[IRDataset],
    git_url: str,
    git_commit_id: str = "HEAD",
    registry_name: str = "ir-sdlc-bench",
    output_path: Optional[Path] = None,
) -> dict:
    """
    Generate a Harbor registry.json from multiple IRDataset objects.
    
    Args:
        datasets: List of IRDataset objects
        git_url: Git URL for the benchmark repository
        git_commit_id: Git commit ID
        registry_name: Name for the registry
        output_path: Optional path to save the registry
        
    Returns:
        The registry dictionary
    """
    config = HarborRegistryConfig(
        git_url=git_url,
        git_commit_id=git_commit_id,
        registry_name=registry_name,
    )
    
    generator = HarborRegistryGenerator(config)
    for dataset in datasets:
        generator.add_dataset(dataset)
    
    registry = generator.generate()
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(registry, f, indent=2)
    
    return registry


def load_registry(registry_path: Path) -> dict:
    """Load a registry.json file."""
    with open(registry_path) as f:
        return json.load(f)


def validate_registry(registry: dict) -> list[str]:
    """
    Validate a registry structure.
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required top-level fields
    required_fields = ["name", "version", "datasets"]
    for field in required_fields:
        if field not in registry:
            errors.append(f"Missing required field: {field}")
    
    # Validate datasets
    datasets = registry.get("datasets", [])
    if not isinstance(datasets, list):
        errors.append("'datasets' must be a list")
    else:
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                errors.append(f"Dataset {i} must be a dictionary")
                continue
            
            dataset_required = ["name", "version", "tasks"]
            for field in dataset_required:
                if field not in dataset:
                    errors.append(f"Dataset {i} missing required field: {field}")
            
            # Validate tasks
            tasks = dataset.get("tasks", [])
            for j, task in enumerate(tasks):
                task_required = ["name", "git_url", "git_commit_id", "path"]
                for field in task_required:
                    if field not in task:
                        errors.append(f"Dataset {i}, Task {j} missing required field: {field}")
    
    return errors


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for harbor registry generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Harbor registry.json for IR-SDLC benchmarks"
    )
    parser.add_argument(
        "jsonl_files",
        nargs="+",
        type=Path,
        help="JSONL benchmark file(s) to include in registry",
    )
    parser.add_argument(
        "--git-url",
        required=True,
        help="Git URL for the benchmark repository",
    )
    parser.add_argument(
        "--git-commit",
        default="HEAD",
        help="Git commit ID (default: HEAD)",
    )
    parser.add_argument(
        "--name",
        default="ir-sdlc-bench",
        help="Registry name",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Registry version",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("registry.json"),
        help="Output path for registry.json",
    )
    parser.add_argument(
        "--generate-tasks",
        type=Path,
        help="Also generate task directories to this path",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate an existing registry file",
    )
    
    args = parser.parse_args()
    
    if args.validate:
        # Validate mode
        for path in args.jsonl_files:
            if path.suffix == ".json":
                registry = load_registry(path)
                errors = validate_registry(registry)
                if errors:
                    print(f"Validation errors in {path}:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print(f"Registry {path} is valid")
        return
    
    # Generate mode
    config = HarborRegistryConfig(
        git_url=args.git_url,
        git_commit_id=args.git_commit,
        registry_name=args.name,
        registry_version=args.version,
    )
    
    generator = HarborRegistryGenerator(config)
    
    for jsonl_path in args.jsonl_files:
        dataset_name = jsonl_path.stem
        print(f"Adding dataset: {dataset_name}")
        generator.add_dataset_from_jsonl(
            jsonl_path=jsonl_path,
            name=dataset_name,
            version=args.version,
        )
    
    output_path = generator.save(args.output)
    print(f"Registry saved to: {output_path}")
    
    if args.generate_tasks:
        print(f"Generating task directories to: {args.generate_tasks}")
        paths = generator.generate_task_directories(args.generate_tasks)
        print(f"Generated {len(paths)} task directories")


if __name__ == "__main__":
    main()
