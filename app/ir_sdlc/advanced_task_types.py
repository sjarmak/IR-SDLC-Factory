#!/usr/bin/env python3
"""
Advanced SDLC Task Types for Agent Reasoning Evaluation.

This module extends the basic IR tasks to evaluate deeper agent capabilities:

A. Architecture Understanding & Mental Model Building
   - Can the agent build a correct mental model of a large system?
   - Requirement disambiguation, abstraction identification, minimal deltas

B. Testing & Correctness
   - Can the agent make changes safe?
   - Test discovery, edge-case identification, regression test creation

C. Cross-Repo / Multi-Service Reasoning
   - Can the agent reason beyond a single repo checkout?
   - API contracts, downstream breakage, version skew

D. Maintenance & Evolution
   - Can the agent safely evolve an existing system?
   - Large refactors, dependency upgrades, dead-code elimination
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class AdvancedTaskType(Enum):
    """Advanced task types for agent reasoning evaluation."""
    
    # A. Architecture Understanding & Mental Model
    REQUIREMENT_DISAMBIGUATION = "requirement_disambiguation"
    ABSTRACTION_IDENTIFICATION = "abstraction_identification"
    ARCHITECTURE_DELTA = "architecture_delta"
    INVARIANT_DISCOVERY = "invariant_discovery"
    
    # B. Testing & Correctness
    TEST_DISCOVERY_AUGMENTATION = "test_discovery_augmentation"
    EDGE_CASE_IDENTIFICATION = "edge_case_identification"
    BUG_TO_REGRESSION_TEST = "bug_to_regression_test"
    TEST_LAYER_SELECTION = "test_layer_selection"
    
    # C. Cross-Repo / Multi-Service Reasoning
    API_CONTRACT_ANALYSIS = "api_contract_analysis"
    DOWNSTREAM_IMPACT = "downstream_impact"
    VERSION_SKEW_ANALYSIS = "version_skew_analysis"
    ROLLOUT_CONSTRAINT = "rollout_constraint"
    
    # D. Maintenance & Evolution
    SAFE_REFACTOR = "safe_refactor"
    DEPENDENCY_UPGRADE = "dependency_upgrade"
    DEAD_CODE_ELIMINATION = "dead_code_elimination"
    CONTRACT_PRESERVATION = "contract_preservation"


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating agent responses."""
    
    # What the agent should demonstrate
    success_signals: list[str]
    
    # What would indicate failure
    failure_signals: list[str]
    
    # Required artifacts in response
    required_artifacts: list[str]
    
    # Scoring rubric (0-1 scale)
    scoring_rubric: dict[str, float]


@dataclass
class AdvancedTask:
    """An advanced reasoning task for agent evaluation."""
    
    task_id: str
    task_type: AdvancedTaskType
    category: str  # A, B, C, or D
    
    # Repository context
    repo_name: str
    repo_url: str
    commit_hash: str
    
    # The challenge
    scenario: str  # Detailed scenario description
    vague_prompt: str  # What a user might actually ask
    
    # Context the agent has access to
    available_context: dict
    
    # Ground truth for evaluation
    ground_truth: dict
    evaluation_criteria: EvaluationCriteria
    
    # Metadata
    difficulty: str
    tags: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "category": self.category,
            "repo_name": self.repo_name,
            "repo_url": self.repo_url,
            "commit_hash": self.commit_hash,
            "scenario": self.scenario,
            "vague_prompt": self.vague_prompt,
            "available_context": self.available_context,
            "ground_truth": self.ground_truth,
            "evaluation_criteria": {
                "success_signals": self.evaluation_criteria.success_signals,
                "failure_signals": self.evaluation_criteria.failure_signals,
                "required_artifacts": self.evaluation_criteria.required_artifacts,
                "scoring_rubric": self.evaluation_criteria.scoring_rubric,
            },
            "difficulty": self.difficulty,
            "tags": self.tags,
            "created_at": self.created_at,
        }


# =============================================================================
# A. Architecture Understanding & Mental Model Tasks
# =============================================================================

ARCHITECTURE_MENTAL_MODEL_TASKS = [
    # A1. Requirement Disambiguation
    {
        "task_type": AdvancedTaskType.REQUIREMENT_DISAMBIGUATION,
        "category": "A",
        "repo_name": "kubernetes/kubernetes",
        "scenario": """
A product manager says: "We need to add support for GPU scheduling."

The agent must:
1. Identify that GPU scheduling already exists (device plugins, extended resources)
2. Clarify what's actually being asked (new GPU type? different scheduling policy?)
3. Reference the existing abstractions (ResourceName, DevicePlugin interface)
4. Propose clarifying questions based on repo conventions
""",
        "vague_prompt": "Add GPU scheduling support to Kubernetes",
        "ground_truth": {
            "existing_abstractions": [
                "pkg/kubelet/cm/devicemanager/manager.go",
                "staging/src/k8s.io/api/core/v1/types.go:ResourceName",
                "pkg/scheduler/framework/plugins/noderesources/",
            ],
            "clarifying_questions": [
                "Which GPU vendor/type (NVIDIA, AMD, Intel)?",
                "Scheduling policy: bin-packing, spread, topology-aware?",
                "MIG (Multi-Instance GPU) support needed?",
            ],
            "repo_conventions": [
                "Device plugins register via gRPC",
                "Extended resources use 'nvidia.com/gpu' naming",
                "Scheduler plugins implement Score/Filter interfaces",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "References existing device plugin system",
                "Identifies extended resources mechanism",
                "Asks clarifying questions before proposing changes",
                "Mentions topology manager for NUMA-aware scheduling",
            ],
            failure_signals=[
                "Proposes new resource type without checking existing",
                "Invents new scheduling mechanism instead of using plugins",
                "Starts implementing without disambiguation",
            ],
            required_artifacts=[
                "list of existing relevant abstractions",
                "clarifying questions for PM",
                "links to existing implementations",
            ],
            scoring_rubric={
                "identifies_existing_abstractions": 0.3,
                "asks_clarifying_questions": 0.3,
                "references_repo_conventions": 0.2,
                "proposes_minimal_delta": 0.2,
            },
        ),
        "difficulty": "expert",
        "tags": ["disambiguation", "existing-abstractions", "scheduler"],
    },
    
    # A2. Architecture Delta Analysis
    {
        "task_type": AdvancedTaskType.ARCHITECTURE_DELTA,
        "category": "A",
        "repo_name": "grafana/grafana",
        "scenario": """
Request: "Add support for alerting on Loki log patterns."

The agent must determine:
1. What already exists (Loki data source, alerting framework, log queries)
2. What's the minimal delta (likely just a new alert condition type)
3. What invariants must be preserved (alert evaluation loop, state machine)
4. What repo conventions to follow (how other data sources integrate)
""",
        "vague_prompt": "I want alerts when certain log patterns appear in Loki",
        "ground_truth": {
            "already_exists": [
                "pkg/services/ngalert/ - unified alerting framework",
                "pkg/tsdb/loki/ - Loki data source",
                "pkg/services/ngalert/eval/ - alert evaluation",
            ],
            "minimal_delta": [
                "New log-based condition type in ngalert",
                "Loki query adapter for alert evaluation",
            ],
            "invariants": [
                "Alert state machine (Normal -> Pending -> Firing)",
                "Evaluation interval constraints",
                "Notification routing unchanged",
            ],
            "conventions": [
                "Data sources implement QueryData interface",
                "Alert conditions are pluggable",
                "State stored in annotation database",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Identifies ngalert as the alerting system",
                "Recognizes Loki data source exists",
                "Proposes extending existing condition types",
                "Preserves alert state machine invariants",
            ],
            failure_signals=[
                "Proposes separate alerting system for logs",
                "Ignores existing ngalert framework",
                "Breaks alert state transitions",
            ],
            required_artifacts=[
                "list of what exists",
                "minimal changes needed",
                "invariants that must not change",
            ],
            scoring_rubric={
                "correct_existing_system_analysis": 0.3,
                "minimal_delta_proposal": 0.3,
                "invariant_preservation": 0.25,
                "convention_adherence": 0.15,
            },
        ),
        "difficulty": "expert",
        "tags": ["architecture-delta", "alerting", "minimal-change"],
    },
    
    # A3. Invariant Discovery
    {
        "task_type": AdvancedTaskType.INVARIANT_DISCOVERY,
        "category": "A",
        "repo_name": "elastic/elasticsearch",
        "scenario": """
Task: "Before modifying the cluster state update mechanism, identify all invariants."

The agent must discover:
1. Consistency invariants (single master, monotonic version)
2. Safety invariants (no data loss during rebalance)
3. Liveness invariants (cluster must eventually converge)
4. Ordering invariants (applied in version order)
""",
        "vague_prompt": "What invariants must I preserve when modifying cluster state updates?",
        "ground_truth": {
            "consistency_invariants": [
                "Single elected master at any time",
                "Cluster state version is monotonically increasing",
                "All nodes eventually see same cluster state",
            ],
            "safety_invariants": [
                "No acknowledged write is lost",
                "Shard allocation respects replica count",
                "Primary promotion only when primary fails",
            ],
            "liveness_invariants": [
                "Cluster converges to stable state",
                "Failed nodes are eventually removed",
                "Pending tasks are eventually processed",
            ],
            "relevant_code": [
                "server/src/main/java/org/elasticsearch/cluster/ClusterState.java",
                "server/src/main/java/org/elasticsearch/cluster/coordination/",
                "server/src/main/java/org/elasticsearch/cluster/service/MasterService.java",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Identifies master election invariants",
                "Recognizes cluster state versioning",
                "Mentions shard allocation safety",
                "References Raft-like consensus",
            ],
            failure_signals=[
                "Misses consistency requirements",
                "Ignores distributed systems invariants",
                "Proposes changes that could lose data",
            ],
            required_artifacts=[
                "list of invariants by category",
                "code locations enforcing each invariant",
                "test cases validating invariants",
            ],
            scoring_rubric={
                "consistency_invariants": 0.25,
                "safety_invariants": 0.25,
                "liveness_invariants": 0.25,
                "code_references": 0.25,
            },
        ),
        "difficulty": "expert",
        "tags": ["invariants", "distributed-systems", "cluster-state"],
    },
]


# =============================================================================
# B. Testing & Correctness Tasks
# =============================================================================

TESTING_CORRECTNESS_TASKS = [
    # B1. Test Discovery & Augmentation
    {
        "task_type": AdvancedTaskType.TEST_DISCOVERY_AUGMENTATION,
        "category": "B",
        "repo_name": "microsoft/vscode",
        "scenario": """
A change is made to the extension host communication protocol.

The agent must:
1. Discover existing tests for extension host
2. Identify which tests cover the modified code path
3. Determine if tests are in the correct layer (unit vs integration)
4. Propose augmentations for missing coverage
""",
        "vague_prompt": "I changed the extension host protocol. What tests should I update?",
        "ground_truth": {
            "existing_tests": [
                "src/vs/workbench/api/test/",
                "src/vs/workbench/services/extensions/test/",
            ],
            "test_layers": {
                "unit": "Protocol message serialization",
                "integration": "Extension activation",
                "e2e": "Extension actually works in VS Code",
            },
            "coverage_gaps": [
                "Error handling for malformed messages",
                "Backward compatibility with old extensions",
                "Performance under message flood",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Finds existing extension host tests",
                "Identifies correct test layer for change",
                "Proposes tests at multiple layers",
                "Considers backward compatibility testing",
            ],
            failure_signals=[
                "Only generates new tests, ignores existing",
                "Tests in wrong layer (e2e for unit-testable code)",
                "Misses error handling coverage",
            ],
            required_artifacts=[
                "list of relevant existing tests",
                "layer classification",
                "proposed test augmentations",
            ],
            scoring_rubric={
                "test_discovery": 0.25,
                "layer_classification": 0.25,
                "gap_identification": 0.25,
                "augmentation_quality": 0.25,
            },
        ),
        "difficulty": "hard",
        "tags": ["testing", "test-discovery", "extension-host"],
    },
    
    # B2. Bug to Regression Test
    {
        "task_type": AdvancedTaskType.BUG_TO_REGRESSION_TEST,
        "category": "B",
        "repo_name": "grafana/grafana",
        "scenario": """
Bug report: "Dashboard fails to load when panel has null title and time range is 'now-5m to now'."

The agent must:
1. Reproduce the exact conditions
2. Create a regression test that fails before fix, passes after
3. Ensure test is in correct layer (not too high, not too low)
4. Avoid "test passes but bug remains" anti-pattern
""",
        "vague_prompt": "Create a regression test for this dashboard loading bug",
        "ground_truth": {
            "bug_conditions": [
                "panel.title === null (not empty string)",
                "time range uses relative 'now' syntax",
                "dashboard has more than one panel",
            ],
            "correct_test_layer": "integration test in pkg/services/dashboards/",
            "anti_patterns_to_avoid": [
                "Testing only the fix, not the original condition",
                "Mocking away the actual failure",
                "E2E test that's flaky due to timing",
            ],
            "regression_test_structure": {
                "setup": "Dashboard with null title panel + relative time",
                "action": "Load dashboard",
                "assertion": "Dashboard loads without error",
                "teardown": "Clean up test dashboard",
            },
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Test reproduces exact bug conditions",
                "Test fails before fix is applied",
                "Test is at appropriate layer (not e2e for logic bug)",
                "Test assertions are specific to the bug",
            ],
            failure_signals=[
                "Test uses mocks that hide the bug",
                "Test only checks 'no error' without validating behavior",
                "Test is flaky or timing-dependent",
            ],
            required_artifacts=[
                "failing test code",
                "explanation of bug conditions",
                "verification that test catches the bug",
            ],
            scoring_rubric={
                "exact_condition_reproduction": 0.3,
                "correct_layer": 0.2,
                "assertion_specificity": 0.25,
                "no_anti_patterns": 0.25,
            },
        ),
        "difficulty": "hard",
        "tags": ["regression-test", "bug-reproduction", "testing"],
    },
]


# =============================================================================
# C. Cross-Repo / Multi-Service Reasoning Tasks
# =============================================================================

CROSS_REPO_TASKS = [
    # C1. API Contract Analysis
    {
        "task_type": AdvancedTaskType.API_CONTRACT_ANALYSIS,
        "category": "C",
        "repo_name": "kubernetes/kubernetes",
        "scenario": """
Proposed change: Add a new field to PodSpec.

The agent must reason about:
1. API compatibility (adding vs changing vs removing fields)
2. Downstream consumers (kubectl, client-go, operators)
3. Versioning (v1 vs v1beta1 implications)
4. Rollout constraints (mixed version clusters)
""",
        "vague_prompt": "I want to add a new field to PodSpec. What should I consider?",
        "ground_truth": {
            "compatibility_rules": [
                "Adding optional fields is safe in v1",
                "New required fields need new API version",
                "Removing fields requires deprecation cycle",
            ],
            "downstream_consumers": [
                "kubectl - needs updated schema",
                "client-go - regenerate clients",
                "operators - may depend on field presence",
                "cloud providers - GKE, EKS, AKS",
            ],
            "rollout_constraints": [
                "Mixed version clusters during upgrade",
                "Old kubelets won't understand new fields",
                "Feature gates for gradual rollout",
            ],
            "required_changes": [
                "staging/src/k8s.io/api/core/v1/types.go",
                "api/openapi-spec/swagger.json",
                "hack/update-codegen.sh",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Mentions API compatibility rules",
                "Identifies downstream client impacts",
                "Considers mixed version clusters",
                "Proposes feature gate for rollout",
            ],
            failure_signals=[
                "Ignores versioning implications",
                "Misses downstream consumer impacts",
                "No consideration of rollback",
            ],
            required_artifacts=[
                "compatibility analysis",
                "list of affected consumers",
                "rollout plan with feature gate",
            ],
            scoring_rubric={
                "compatibility_analysis": 0.25,
                "downstream_impact": 0.25,
                "rollout_plan": 0.25,
                "version_considerations": 0.25,
            },
        ),
        "difficulty": "expert",
        "tags": ["api-contract", "cross-repo", "versioning"],
    },
    
    # C2. Downstream Impact Analysis
    {
        "task_type": AdvancedTaskType.DOWNSTREAM_IMPACT,
        "category": "C",
        "repo_name": "elastic/elasticsearch",
        "scenario": """
Change: Modify the shard allocation response format.

The agent must identify:
1. All consumers of this API (Kibana, Logstash, Beats, clients)
2. Behavioral changes implied for consumers
3. Breaking vs non-breaking classification
4. Migration path for each consumer
""",
        "vague_prompt": "I'm changing the _cluster/allocation/explain API response. What breaks?",
        "ground_truth": {
            "consumers": [
                "Kibana - Cluster management UI",
                "Curator - Index lifecycle management",
                "Elastic Agent - Monitoring",
                "Java/Python/Go clients",
            ],
            "behavioral_changes": {
                "Kibana": "UI may show wrong allocation reason",
                "Curator": "Allocation checks may fail",
                "Clients": "Deserialization may break",
            },
            "breaking_classification": {
                "adding_field": "non-breaking",
                "removing_field": "breaking",
                "changing_type": "breaking",
                "changing_enum_values": "potentially breaking",
            },
            "migration_path": [
                "Version response with 'version' field",
                "Support old format for N releases",
                "Document migration in CHANGELOG",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Identifies Kibana as primary consumer",
                "Mentions client library impacts",
                "Proposes backward compatible approach",
                "Suggests versioning strategy",
            ],
            failure_signals=[
                "Only considers Elasticsearch internal",
                "Ignores external consumers",
                "Proposes breaking change without migration",
            ],
            required_artifacts=[
                "consumer impact matrix",
                "breaking change classification",
                "migration guide outline",
            ],
            scoring_rubric={
                "consumer_identification": 0.25,
                "behavioral_analysis": 0.25,
                "breaking_classification": 0.25,
                "migration_path": 0.25,
            },
        ),
        "difficulty": "expert",
        "tags": ["downstream-impact", "api-change", "multi-service"],
    },
]


# =============================================================================
# D. Maintenance & Evolution Tasks
# =============================================================================

MAINTENANCE_EVOLUTION_TASKS = [
    # D1. Safe Refactor
    {
        "task_type": AdvancedTaskType.SAFE_REFACTOR,
        "category": "D",
        "repo_name": "microsoft/vscode",
        "scenario": """
Refactor: Extract the command palette into a standalone module.

The agent must:
1. Identify all usages of current implementation
2. Define the public contract to preserve
3. Plan incremental extraction (not big bang)
4. Update tests, docs, and migration notes together
""",
        "vague_prompt": "Refactor the command palette to be more modular",
        "ground_truth": {
            "current_usages": [
                "Keyboard shortcuts (Ctrl+Shift+P)",
                "Extensions contributing commands",
                "Menu integration",
                "Telemetry hooks",
            ],
            "public_contract": [
                "ICommandPalette interface",
                "Command registration API",
                "Keyboard binding integration",
            ],
            "incremental_plan": [
                "1. Extract interfaces without changing impl",
                "2. Move impl behind interface",
                "3. Update consumers one by one",
                "4. Remove old locations",
            ],
            "artifacts_to_update": [
                "Unit tests for new module",
                "Integration tests for consumers",
                "API documentation",
                "Extension authoring guide",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Maps all current usages",
                "Defines preserved public interface",
                "Proposes incremental extraction",
                "Updates tests and docs together",
            ],
            failure_signals=[
                "Big bang refactor proposal",
                "Breaks public extension API",
                "Ignores test updates",
                "No migration notes",
            ],
            required_artifacts=[
                "usage map",
                "interface definition",
                "incremental PR plan",
                "test update plan",
            ],
            scoring_rubric={
                "usage_mapping": 0.2,
                "contract_preservation": 0.3,
                "incremental_approach": 0.3,
                "artifact_updates": 0.2,
            },
        ),
        "difficulty": "expert",
        "tags": ["refactor", "modularization", "safe-evolution"],
    },
    
    # D2. Dependency Upgrade
    {
        "task_type": AdvancedTaskType.DEPENDENCY_UPGRADE,
        "category": "D",
        "repo_name": "grafana/grafana",
        "scenario": """
Upgrade: Go from 1.21 to 1.22.

The agent must:
1. Identify breaking changes in Go 1.22
2. Find code patterns affected
3. Assess test coverage of affected areas
4. Plan staged upgrade with validation
""",
        "vague_prompt": "Upgrade Grafana to Go 1.22",
        "ground_truth": {
            "breaking_changes": [
                "Loop variable semantics change",
                "Range over int",
                "Enhanced HTTP routing patterns",
            ],
            "affected_code_patterns": [
                "Closures capturing loop variables",
                "for i := 0; i < n; i++ patterns",
                "http.HandleFunc usage",
            ],
            "validation_plan": [
                "Run full test suite",
                "Check for loop variable warnings",
                "Benchmark performance-critical paths",
                "Test HTTP handlers specifically",
            ],
            "staged_rollout": [
                "1. Update go.mod in separate PR",
                "2. Fix any immediate build breaks",
                "3. Run vet for new warnings",
                "4. Full regression test",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Identifies loop variable change",
                "Searches for affected patterns",
                "Plans staged upgrade",
                "Includes validation steps",
            ],
            failure_signals=[
                "Just updates go.mod",
                "Ignores breaking changes",
                "No validation plan",
            ],
            required_artifacts=[
                "breaking change list",
                "affected code locations",
                "upgrade plan with PRs",
                "validation checklist",
            ],
            scoring_rubric={
                "breaking_change_analysis": 0.3,
                "code_impact_assessment": 0.25,
                "staged_plan": 0.25,
                "validation_coverage": 0.2,
            },
        ),
        "difficulty": "hard",
        "tags": ["dependency-upgrade", "go", "maintenance"],
    },
    
    # D3. Dead Code Elimination
    {
        "task_type": AdvancedTaskType.DEAD_CODE_ELIMINATION,
        "category": "D",
        "repo_name": "kubernetes/kubernetes",
        "scenario": """
Task: Remove deprecated PodSecurityPolicy support.

The agent must:
1. Find all PSP-related code (not just the obvious)
2. Identify downstream dependencies on PSP
3. Ensure no runtime code paths still reference PSP
4. Update docs, tests, and feature gates together
""",
        "vague_prompt": "Remove PodSecurityPolicy code now that it's been deprecated",
        "ground_truth": {
            "obvious_locations": [
                "pkg/apis/policy/v1beta1/podsecuritypolicy*",
                "plugin/pkg/admission/podsecurity/",
            ],
            "hidden_dependencies": [
                "RBAC roles granting PSP permissions",
                "Controller manager flags",
                "API server admission config",
                "e2e tests referencing PSP",
            ],
            "runtime_references": [
                "Admission webhook configs",
                "Controller watch lists",
                "Audit log policies",
            ],
            "artifacts_to_update": [
                "API deprecation docs",
                "Migration guide to Pod Security Admission",
                "Release notes",
                "Feature gate removal",
            ],
        },
        "evaluation_criteria": EvaluationCriteria(
            success_signals=[
                "Finds non-obvious PSP references",
                "Identifies RBAC implications",
                "Plans removal with migration docs",
                "Updates feature gates",
            ],
            failure_signals=[
                "Only removes obvious files",
                "Leaves broken RBAC roles",
                "No migration documentation",
            ],
            required_artifacts=[
                "complete PSP reference list",
                "dependency analysis",
                "removal PR sequence",
                "migration guide",
            ],
            scoring_rubric={
                "completeness_of_discovery": 0.3,
                "hidden_dependency_finding": 0.25,
                "safe_removal_plan": 0.25,
                "documentation_updates": 0.2,
            },
        ),
        "difficulty": "expert",
        "tags": ["dead-code", "deprecation", "safe-removal"],
    },
]


def generate_task_id(task_dict: dict) -> str:
    """Generate a unique task ID."""
    hash_input = f"{task_dict['repo_name']}-{task_dict['task_type'].value}-{task_dict['scenario'][:100]}"
    hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:12]
    return f"{task_dict['repo_name'].replace('/', '__')}-{task_dict['task_type'].value}-{hash_suffix}"


def create_advanced_tasks() -> list[AdvancedTask]:
    """Create all advanced evaluation tasks."""
    all_tasks = []
    
    task_collections = [
        ARCHITECTURE_MENTAL_MODEL_TASKS,
        TESTING_CORRECTNESS_TASKS,
        CROSS_REPO_TASKS,
        MAINTENANCE_EVOLUTION_TASKS,
    ]
    
    for collection in task_collections:
        for task_dict in collection:
            task = AdvancedTask(
                task_id=generate_task_id(task_dict),
                task_type=task_dict["task_type"],
                category=task_dict["category"],
                repo_name=task_dict["repo_name"],
                repo_url=f"https://github.com/{task_dict['repo_name']}",
                commit_hash="HEAD",
                scenario=task_dict["scenario"].strip(),
                vague_prompt=task_dict["vague_prompt"],
                available_context={},
                ground_truth=task_dict["ground_truth"],
                evaluation_criteria=task_dict["evaluation_criteria"],
                difficulty=task_dict["difficulty"],
                tags=task_dict["tags"],
            )
            all_tasks.append(task)
    
    return all_tasks


def export_advanced_benchmark(output_file: Path) -> None:
    """Export advanced tasks to JSONL file."""
    tasks = create_advanced_tasks()
    
    with open(output_file, "w") as f:
        for task in tasks:
            f.write(json.dumps(task.to_dict()) + "\n")
    
    print(f"Exported {len(tasks)} advanced tasks to {output_file}")
    
    # Summary
    by_category = {}
    for task in tasks:
        by_category.setdefault(task.category, []).append(task)
    
    print("\nTasks by category:")
    for cat, cat_tasks in sorted(by_category.items()):
        print(f"  {cat}: {len(cat_tasks)} tasks")
        for t in cat_tasks:
            print(f"    - {t.task_type.value}")


if __name__ == "__main__":
    output = Path("benchmarks/ir-sdlc-advanced-reasoning.jsonl")
    export_advanced_benchmark(output)
