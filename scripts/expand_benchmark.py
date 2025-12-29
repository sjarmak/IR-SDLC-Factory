#!/usr/bin/env python3
"""
Expand the IR-SDLC benchmark dataset.

This script:
1. Adds security_audit tasks from existing repos
2. Adds dependency_analysis and feature_location tasks
3. Expands to 50+ tasks total
4. Validates SDLC coverage
5. Generates task quality metrics
"""

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ir_sdlc.benchmark_pipeline import SDLCBenchmarkPipeline, SDLCTaskType
from app.ir_sdlc.data_structures import IRDataset, IRTask, GroundTruth, CodeLocation, RetrievalGranularity


# SDLC lifecycle phases mapped to task types
SDLC_PHASES = {
    "requirements": [],  # Future: requirement tracing
    "design": ["architecture_understanding", "dependency_analysis"],
    "implementation": ["feature_location", "code_review"],
    "testing": ["test_coverage"],
    "security": ["security_audit"],
    "maintenance": ["bug_triage", "refactoring_analysis", "change_impact_analysis"],
    "documentation": ["documentation_linking"],
}

# Task types that specifically showcase IR/codebase understanding benefits
IR_FOCUSED_TASK_TYPES = {
    "bug_triage": {
        "ir_benefit": "Find relevant code from natural language bug descriptions and stack traces",
        "complexity_factors": ["large codebase", "cross-module bugs", "indirect causation"],
    },
    "code_review": {
        "ir_benefit": "Discover related code patterns, similar implementations, and potential side effects",
        "complexity_factors": ["scattered changes", "API usage patterns", "historical context"],
    },
    "security_audit": {
        "ir_benefit": "Identify potentially vulnerable code patterns across entire codebase",
        "complexity_factors": ["attack surface analysis", "data flow tracking", "dependency chains"],
    },
    "feature_location": {
        "ir_benefit": "Map high-level feature names to scattered implementation files",
        "complexity_factors": ["modular architecture", "layered design", "plugin systems"],
    },
    "dependency_analysis": {
        "ir_benefit": "Trace import chains and identify all affected code from changes",
        "complexity_factors": ["deep dependency trees", "circular dependencies", "runtime dependencies"],
    },
    "architecture_understanding": {
        "ir_benefit": "Navigate complex architectural patterns and understand component roles",
        "complexity_factors": ["microservices", "event-driven", "large module count"],
    },
}


def analyze_sdlc_coverage(tasks: list[dict]) -> dict:
    """Analyze SDLC lifecycle coverage of current tasks."""
    task_types = Counter(t["task_type"] for t in tasks)
    
    coverage = {}
    for phase, types in SDLC_PHASES.items():
        phase_tasks = sum(task_types.get(tt, 0) for tt in types)
        coverage[phase] = {
            "task_count": phase_tasks,
            "types_present": [tt for tt in types if task_types.get(tt, 0) > 0],
            "types_missing": [tt for tt in types if task_types.get(tt, 0) == 0],
        }
    
    return coverage


def analyze_ir_focus(tasks: list[dict]) -> dict:
    """Analyze how well tasks showcase IR/codebase understanding benefits."""
    task_types = Counter(t["task_type"] for t in tasks)
    difficulties = Counter(t["difficulty"] for t in tasks)
    
    ir_analysis = {
        "task_type_distribution": dict(task_types),
        "difficulty_distribution": dict(difficulties),
        "ir_showcasing_coverage": {},
        "recommendations": [],
    }
    
    for task_type, info in IR_FOCUSED_TASK_TYPES.items():
        count = task_types.get(task_type, 0)
        ir_analysis["ir_showcasing_coverage"][task_type] = {
            "count": count,
            "ir_benefit": info["ir_benefit"],
            "has_coverage": count > 0,
        }
        
        if count == 0:
            ir_analysis["recommendations"].append(
                f"Add {task_type} tasks to showcase: {info['ir_benefit']}"
            )
        elif count < 5:
            ir_analysis["recommendations"].append(
                f"Expand {task_type} tasks (currently {count}) for better coverage"
            )
    
    # Check difficulty balance for IR relevance
    expert_hard = difficulties.get("expert", 0) + difficulties.get("hard", 0)
    total = sum(difficulties.values())
    if total > 0:
        hard_ratio = expert_hard / total
        if hard_ratio < 0.5:
            ir_analysis["recommendations"].append(
                f"Increase hard/expert tasks (currently {hard_ratio:.0%}) - "
                "complex tasks better showcase IR benefits"
            )
    
    return ir_analysis


def generate_feature_location_tasks(repos: list[str], existing_tasks: list[dict]) -> list[IRTask]:
    """
    Generate feature_location tasks from known features in enterprise repos.
    
    These tasks test the ability to find scattered implementation code from
    high-level feature descriptions - a key IR capability.
    """
    # Known features in enterprise repos with their implementations
    KNOWN_FEATURES = {
        "kubernetes/kubernetes": [
            {
                "feature": "pod scheduling",
                "description": "Find all code involved in scheduling pods to nodes",
                "component_files": [
                    "pkg/scheduler/schedule_one.go",
                    "pkg/scheduler/framework/interface.go",
                    "pkg/scheduler/framework/runtime/framework.go",
                ],
            },
            {
                "feature": "service discovery",
                "description": "Find code that implements service discovery and load balancing",
                "component_files": [
                    "pkg/proxy/iptables/proxier.go",
                    "pkg/controller/endpoint/endpoints_controller.go",
                ],
            },
        ],
        "grafana/grafana": [
            {
                "feature": "dashboard provisioning",
                "description": "Find code that handles dashboard provisioning from files",
                "component_files": [
                    "pkg/services/provisioning/dashboards/file_reader.go",
                    "pkg/services/provisioning/dashboards/dashboard.go",
                ],
            },
            {
                "feature": "alerting rules evaluation",
                "description": "Find the code that evaluates alerting rules",
                "component_files": [
                    "pkg/services/ngalert/eval/eval.go",
                    "pkg/services/ngalert/schedule/schedule.go",
                ],
            },
        ],
        "microsoft/vscode": [
            {
                "feature": "extension host",
                "description": "Find code that manages VS Code extension host process",
                "component_files": [
                    "src/vs/workbench/api/node/extensionHostProcess.ts",
                    "src/vs/workbench/services/extensions/common/extensionHostManager.ts",
                ],
            },
            {
                "feature": "code completion",
                "description": "Find the code that implements inline code completions",
                "component_files": [
                    "src/vs/editor/contrib/inlineCompletions/browser/inlineCompletionsController.ts",
                    "src/vs/editor/contrib/inlineCompletions/browser/suggestWidgetInlineCompletionProvider.ts",
                ],
            },
        ],
        "elastic/elasticsearch": [
            {
                "feature": "index shard allocation",
                "description": "Find code that handles shard allocation decisions",
                "component_files": [
                    "server/src/main/java/org/elasticsearch/cluster/routing/allocation/AllocationService.java",
                    "server/src/main/java/org/elasticsearch/cluster/routing/allocation/decider/AllocationDeciders.java",
                ],
            },
        ],
    }
    
    tasks = []
    for repo, features in KNOWN_FEATURES.items():
        for feature_info in features:
            # Check if similar task already exists
            existing_queries = [t.get("query", "").lower() for t in existing_tasks]
            if any(feature_info["feature"].lower() in q for q in existing_queries):
                continue
            
            task = IRTask(
                task_id="",  # Will be auto-generated
                task_type="feature_location",
                repo_name=repo,
                repo_url=f"https://github.com/{repo}",
                commit_hash="HEAD",
                query=f"Find the code that implements {feature_info['feature']}: {feature_info['description']}",
                context={
                    "feature": feature_info["feature"],
                    "search_scope": "full_repository",
                },
                ground_truth=GroundTruth(
                    locations=[CodeLocation(file_path=f) for f in feature_info["component_files"]],
                    granularity=RetrievalGranularity.FILE,
                    source="expert",
                    confidence=0.9,
                    metadata={"feature": feature_info["feature"]},
                ),
                difficulty="hard",
                tags=["feature_location", "architecture"],
            )
            tasks.append(task)
    
    return tasks


def generate_dependency_analysis_tasks(repos: list[str], existing_tasks: list[dict]) -> list[IRTask]:
    """
    Generate dependency_analysis tasks from core components.
    
    These tasks test the ability to trace dependencies and dependents
    of key modules - critical for change impact analysis.
    """
    CORE_COMPONENTS = {
        "kubernetes/kubernetes": [
            {
                "target_file": "pkg/apis/core/types.go",
                "target_class": "Pod",
                "description": "Core Pod type definition",
            },
            {
                "target_file": "staging/src/k8s.io/client-go/kubernetes/clientset.go",
                "target_class": "Clientset",
                "description": "Main Kubernetes client interface",
            },
        ],
        "grafana/grafana": [
            {
                "target_file": "pkg/services/sqlstore/sqlstore.go",
                "target_class": "SQLStore",
                "description": "Core database access layer",
            },
        ],
        "elastic/elasticsearch": [
            {
                "target_file": "server/src/main/java/org/elasticsearch/action/ActionModule.java",
                "target_class": "ActionModule",
                "description": "Action registration and routing",
            },
        ],
    }
    
    tasks = []
    for repo, components in CORE_COMPONENTS.items():
        for component in components:
            # Skip if similar exists
            existing_files = [t.get("context", {}).get("target_file", "") for t in existing_tasks]
            if component["target_file"] in existing_files:
                continue
            
            query = (
                f"Find all code that depends on or is depended upon by "
                f"the {component.get('target_class', 'module')} in `{component['target_file']}`"
            )
            
            task = IRTask(
                task_id="",
                task_type="dependency_analysis",
                repo_name=repo,
                repo_url=f"https://github.com/{repo}",
                commit_hash="HEAD",
                query=query,
                context={
                    "target_file": component["target_file"],
                    "target_class": component.get("target_class"),
                    "analysis_type": "both",
                },
                ground_truth=GroundTruth(
                    locations=[],  # Dependency tasks need tool-specific extraction
                    granularity=RetrievalGranularity.FILE,
                    source="automatic",
                    confidence=0.7,
                    metadata={"requires_static_analysis": True},
                ),
                difficulty="expert",
                tags=["dependency_analysis", "change_impact"],
            )
            tasks.append(task)
    
    return tasks


def expand_benchmark(
    input_file: Path,
    output_file: Path,
    github_token: str | None = None,
    target_count: int = 50,
    add_security: bool = True,
) -> dict:
    """
    Expand the benchmark dataset.
    
    Args:
        input_file: Path to existing benchmark JSONL
        output_file: Path for expanded benchmark
        github_token: GitHub token for API calls (optional)
        target_count: Target number of tasks
        add_security: Whether to add security_audit tasks
        
    Returns:
        Summary of expansion results
    """
    # Load existing tasks
    with open(input_file, "r") as f:
        existing_tasks = [json.loads(line) for line in f]
    
    print(f"Loaded {len(existing_tasks)} existing tasks")
    
    # Analyze current coverage
    coverage = analyze_sdlc_coverage(existing_tasks)
    ir_focus = analyze_ir_focus(existing_tasks)
    
    print("\n=== Current SDLC Coverage ===")
    for phase, info in coverage.items():
        print(f"  {phase}: {info['task_count']} tasks")
        if info["types_missing"]:
            print(f"    Missing: {', '.join(info['types_missing'])}")
    
    print("\n=== IR Focus Analysis ===")
    for rec in ir_focus["recommendations"]:
        print(f"  • {rec}")
    
    # Collect new tasks
    new_tasks = []
    repos = list(set(t["repo_name"] for t in existing_tasks))
    
    # Add feature_location tasks
    print("\nGenerating feature_location tasks...")
    feature_tasks = generate_feature_location_tasks(repos, existing_tasks)
    new_tasks.extend(feature_tasks)
    print(f"  Added {len(feature_tasks)} feature_location tasks")
    
    # Add dependency_analysis tasks
    print("Generating dependency_analysis tasks...")
    dep_tasks = generate_dependency_analysis_tasks(repos, existing_tasks)
    new_tasks.extend(dep_tasks)
    print(f"  Added {len(dep_tasks)} dependency_analysis tasks")
    
    # Add security_audit tasks using pipeline if token available
    if add_security and github_token:
        print("Generating security_audit tasks...")
        try:
            pipeline = SDLCBenchmarkPipeline(github_token)
            for repo in repos:
                sec_tasks = pipeline._generate_security_tasks(
                    repo=repo,
                    repo_url=f"https://github.com/{repo}",
                    generator=pipeline.generators[SDLCTaskType.SECURITY_AUDIT],
                    repo_stats=pipeline.estimator.get_repo_stats(repo),
                    max_tasks=3,
                )
                new_tasks.extend(sec_tasks)
                print(f"  Added {len(sec_tasks)} security tasks from {repo}")
        except Exception as e:
            print(f"  Warning: Could not generate security tasks: {e}")
    
    # Convert new IRTask objects to dicts and assign IDs
    all_tasks = existing_tasks.copy()
    for task in new_tasks:
        if hasattr(task, "to_dict"):
            task_dict = task.to_dict()
        else:
            task_dict = task.__dict__ if hasattr(task, "__dict__") else task
        
        # Generate task ID if needed
        if not task_dict.get("task_id"):
            import hashlib
            hash_input = f"{task_dict['repo_name']}-{task_dict['task_type']}-{task_dict.get('query', '')[:100]}"
            task_dict["task_id"] = f"{task_dict['repo_name'].replace('/', '__')}-{task_dict['task_type']}-{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"
        
        # Ensure created_at
        if not task_dict.get("created_at"):
            task_dict["created_at"] = datetime.now(timezone.utc).isoformat()
        
        all_tasks.append(task_dict)
    
    # Check if we need more tasks
    if len(all_tasks) < target_count:
        shortfall = target_count - len(all_tasks)
        print(f"\nNeed {shortfall} more tasks to reach target of {target_count}")
        print("Consider running with GitHub token for additional security_audit tasks")
    
    # Write expanded dataset
    with open(output_file, "w") as f:
        for task in all_tasks:
            f.write(json.dumps(task) + "\n")
    
    print(f"\nWrote {len(all_tasks)} tasks to {output_file}")
    
    # Update metadata
    metadata_file = output_file.with_suffix("").with_name(output_file.stem + "_metadata.json")
    metadata = {
        "source_repos": repos,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_count": len(all_tasks),
        "task_types": dict(Counter(t["task_type"] for t in all_tasks)),
        "sdlc_coverage": analyze_sdlc_coverage(all_tasks),
        "ir_focus": analyze_ir_focus(all_tasks),
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Updated metadata: {metadata_file}")
    
    # Final summary
    final_coverage = analyze_sdlc_coverage(all_tasks)
    final_ir = analyze_ir_focus(all_tasks)
    
    print("\n=== Final Summary ===")
    print(f"Total tasks: {len(all_tasks)}")
    print(f"Task types: {dict(Counter(t['task_type'] for t in all_tasks))}")
    print(f"Repos: {len(repos)}")
    
    return {
        "total_tasks": len(all_tasks),
        "added_tasks": len(new_tasks),
        "task_types": dict(Counter(t["task_type"] for t in all_tasks)),
        "sdlc_coverage": final_coverage,
        "ir_focus": final_ir,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Expand IR-SDLC benchmark dataset")
    parser.add_argument(
        "--input",
        default="benchmarks/ir-sdlc-multi-repo.jsonl",
        help="Input benchmark file",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/ir-sdlc-multi-repo.jsonl",
        help="Output benchmark file (can be same as input)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target number of tasks",
    )
    parser.add_argument(
        "--no-security",
        action="store_true",
        help="Skip security_audit task generation",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze coverage, don't expand",
    )
    
    args = parser.parse_args()
    
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("Note: GITHUB_TOKEN not set - security_audit generation will be skipped")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.analyze_only:
        with open(input_path, "r") as f:
            tasks = [json.loads(line) for line in f]
        
        print("\n=== SDLC Coverage Analysis ===")
        coverage = analyze_sdlc_coverage(tasks)
        for phase, info in coverage.items():
            status = "✓" if info["task_count"] > 0 else "✗"
            print(f"  {status} {phase}: {info['task_count']} tasks")
            if info["types_missing"]:
                print(f"      Missing: {', '.join(info['types_missing'])}")
        
        print("\n=== IR Focus Analysis ===")
        ir_focus = analyze_ir_focus(tasks)
        for task_type, info in ir_focus["ir_showcasing_coverage"].items():
            status = "✓" if info["has_coverage"] else "✗"
            print(f"  {status} {task_type}: {info['count']} tasks")
            print(f"      IR benefit: {info['ir_benefit']}")
        
        print("\n=== Recommendations ===")
        for rec in ir_focus["recommendations"]:
            print(f"  • {rec}")
    else:
        result = expand_benchmark(
            input_file=input_path,
            output_file=output_path,
            github_token=github_token,
            target_count=args.target,
            add_security=not args.no_security,
        )
