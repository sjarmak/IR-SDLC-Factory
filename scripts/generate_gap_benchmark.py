#!/usr/bin/env python3
"""
Generate benchmark tasks for CACM-identified gaps.

This script creates tasks for the missing SDLC phases:
- Debugging / Root Cause Analysis
- Code Transformation / Migration
- CI/CD Diagnosis
- Test Evolution
- Cross-System Reasoning
- Documentation

Each task includes human-aligned scoring criteria.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


def generate_task_id(repo: str, task_type: str, title: str) -> str:
    """Generate unique task ID."""
    hash_input = f"{repo}-{task_type}-{title}"
    return f"{repo.replace('/', '__')}-{task_type}-{hashlib.md5(hash_input.encode()).hexdigest()[:12]}"


# =============================================================================
# DEBUGGING / ROOT CAUSE ANALYSIS TASKS
# =============================================================================

DEBUGGING_TASKS = [
    {
        "task_type": "stack_trace_to_fix",
        "repo_name": "grafana/grafana",
        "title": "Nil pointer panic in alerting notification",
        "scenario": """
Production alert: Grafana crashes intermittently when sending alert notifications.

Stack trace:
```
panic: runtime error: invalid memory address or nil pointer dereference
[signal SIGSEGV: segmentation violation code=0x1]

goroutine 847 [running]:
pkg/services/ngalert/notifier.(*Alertmanager).SendNotification(...)
    /app/pkg/services/ngalert/notifier/alertmanager.go:234
pkg/services/ngalert/schedule.(*schedule).notify(...)
    /app/pkg/services/ngalert/schedule/schedule.go:445
```

This happens ~2% of the time, always during high-load periods.
""",
        "vague_prompt": "Grafana keeps crashing when sending alerts. Fix it.",
        "ground_truth": {
            "root_cause": "Contact point can be nil when notification policy references deleted receiver",
            "root_cause_location": "pkg/services/ngalert/notifier/alertmanager.go",
            "fix_approach": "Add nil check before dereferencing contact point, log warning for missing receiver",
            "related_code": [
                "pkg/services/ngalert/notifier/alertmanager.go",
                "pkg/services/ngalert/api/api_receiver.go",
                "pkg/services/ngalert/models/contact_point.go",
            ],
        },
        "scoring_criteria": {
            "correct_root_cause": {
                "weight": 0.4,
                "levels": {
                    "exact": "Identifies nil contact point from deleted receiver",
                    "partial": "Identifies nil dereference but wrong cause",
                    "wrong": "Blames unrelated code",
                },
            },
            "fix_correctness": {
                "weight": 0.4,
                "levels": {
                    "correct": "Adds nil check + handles gracefully",
                    "partial": "Adds check but silent failure",
                    "wrong": "Doesn't fix the issue",
                },
            },
            "explanation": {
                "weight": 0.2,
                "levels": {
                    "educational": "Explains race condition clearly",
                    "adequate": "Explains fix without context",
                },
            },
        },
        "difficulty": "hard",
        "tags": ["debugging", "panic", "nil-pointer", "concurrency"],
    },
    {
        "task_type": "flaky_test_diagnosis",
        "repo_name": "kubernetes/kubernetes",
        "title": "Pod scheduling test flakes on CI",
        "scenario": """
This test fails approximately 5-10% of CI runs:

```
=== FAIL: TestScheduler_SchedulePod (0.23s)
    scheduler_test.go:445: expected pod to be scheduled within 5s, got timeout
    scheduler_test.go:446: pod status: Pending
```

The test passes consistently on developer machines.
CI runs on smaller instances with 2 CPUs.

Test code summary:
- Creates a scheduler with default config
- Submits a pod
- Waits 5 seconds for scheduling
- Asserts pod is scheduled
""",
        "vague_prompt": "This scheduling test keeps failing on CI. It works on my machine.",
        "ground_truth": {
            "root_cause": "Race condition: test doesn't wait for informer cache sync before scheduling",
            "contributing_factors": [
                "CI has slower I/O than dev machines",
                "5s timeout too short for cold cache",
                "Missing WaitForCacheSync call",
            ],
            "fix_approach": "Add informer cache sync wait before scheduling, increase timeout with CI detection",
            "related_code": [
                "pkg/scheduler/scheduler_test.go",
                "pkg/scheduler/scheduler.go",
                "staging/src/k8s.io/client-go/tools/cache/shared_informer.go",
            ],
        },
        "scoring_criteria": {
            "identifies_race_condition": {
                "weight": 0.4,
                "levels": {
                    "exact": "Identifies informer cache sync issue",
                    "partial": "Identifies timing issue but wrong component",
                    "wrong": "Blames unrelated factor",
                },
            },
            "fix_quality": {
                "weight": 0.4,
                "levels": {
                    "correct": "Adds cache sync + appropriate timeout",
                    "partial": "Just increases timeout",
                    "wrong": "Doesn't address the race",
                },
            },
        },
        "difficulty": "expert",
        "tags": ["testing", "flaky", "race-condition", "kubernetes"],
    },
    {
        "task_type": "log_analysis",
        "repo_name": "elastic/elasticsearch",
        "title": "Cluster yellow status investigation",
        "scenario": """
Production cluster showing yellow status. Relevant log snippets:

```
[2024-01-15T14:32:01] [WARN] [cluster.routing.allocation.decider] 
  [node-3] high disk watermark [90%] exceeded on [node-3], 
  shards will be relocated away from this node

[2024-01-15T14:32:45] [INFO] [cluster.routing.allocation] 
  Rerouting 24 shards from [node-3] to other nodes

[2024-01-15T14:33:12] [WARN] [cluster.routing.allocation]
  unable to allocate shard [logs-2024.01.15][2]: no eligible nodes

[2024-01-15T14:33:15] [WARN] [cluster.health]
  Cluster status changed from [GREEN] to [YELLOW]
```

The ops team wants to know: What happened? Will it recover? What should we do?
""",
        "vague_prompt": "Elasticsearch cluster went yellow. Why?",
        "ground_truth": {
            "root_cause": "Disk watermark exceeded on node-3, triggering shard relocation, but other nodes also near capacity",
            "chain_of_events": [
                "node-3 exceeded 90% disk",
                "ES started relocating 24 shards",
                "No other node had capacity for some shards",
                "Unallocated shards caused yellow status",
            ],
            "will_recover": "Maybe - depends on disk space on other nodes",
            "recommended_actions": [
                "Add disk capacity to cluster",
                "Delete old indices to free space",
                "Adjust watermark thresholds temporarily",
            ],
        },
        "scoring_criteria": {
            "correct_diagnosis": {
                "weight": 0.4,
                "levels": {
                    "complete": "Identifies full chain of events",
                    "partial": "Identifies disk issue but not cascade",
                },
            },
            "actionable_advice": {
                "weight": 0.4,
                "levels": {
                    "complete": "Multiple options with trade-offs",
                    "partial": "Single recommendation",
                },
            },
            "recovery_prediction": {
                "weight": 0.2,
                "levels": {
                    "correct": "Conditional based on cluster state",
                    "partial": "Generic answer",
                },
            },
        },
        "difficulty": "hard",
        "tags": ["debugging", "log-analysis", "elasticsearch", "ops"],
    },
]


# =============================================================================
# CODE TRANSFORMATION / MIGRATION TASKS
# =============================================================================

CODE_TRANSFORMATION_TASKS = [
    {
        "task_type": "api_migration",
        "repo_name": "grafana/grafana",
        "title": "Migrate from deprecated context API",
        "scenario": """
Grafana's codebase uses a deprecated context pattern in many places:

Old pattern (deprecated):
```go
func (s *Service) DoSomething(ctx *models.ReqContext) error {
    user := ctx.SignedInUser
    orgID := ctx.OrgId
    // ...
}
```

New pattern (required):
```go
func (s *Service) DoSomething(ctx context.Context) error {
    user := appcontext.User(ctx)
    orgID := identity.OrgIDFromContext(ctx)
    // ...
}
```

Migrate the alerting service (pkg/services/ngalert/) to use the new pattern.
There are approximately 45 usages across 12 files.
""",
        "vague_prompt": "Update the alerting service to use the new context API",
        "ground_truth": {
            "files_to_change": 12,
            "usages_to_migrate": 45,
            "key_changes": [
                "Replace *models.ReqContext with context.Context",
                "Use appcontext.User(ctx) for user access",
                "Use identity.OrgIDFromContext(ctx) for org ID",
                "Update callers to construct proper context",
            ],
            "pitfalls": [
                "Some code paths need both old and new context during transition",
                "Test fixtures need updating",
                "HTTP handlers need middleware changes",
            ],
        },
        "scoring_criteria": {
            "completeness": {
                "weight": 0.4,
                "levels": {
                    "all": "All 45 usages migrated correctly",
                    "most": ">80% migrated",
                    "some": ">50% migrated",
                    "few": "<50% migrated",
                },
            },
            "correctness": {
                "weight": 0.4,
                "levels": {
                    "correct": "All tests pass, behavior unchanged",
                    "partial": "Minor issues in edge cases",
                    "broken": "Behavior changed or tests fail",
                },
            },
            "idiomatic": {
                "weight": 0.2,
                "levels": {
                    "clean": "Follows Go context best practices",
                    "functional": "Works but not idiomatic",
                },
            },
        },
        "difficulty": "hard",
        "tags": ["migration", "api-change", "go", "context"],
    },
    {
        "task_type": "merge_conflict_resolution",
        "repo_name": "microsoft/vscode",
        "title": "Resolve extension activation conflicts",
        "scenario": """
Two branches have conflicting changes to extension activation:

Branch `main`:
- Refactored extension host to use async activation
- Changed ExtensionHostManager.activate() signature
- Added activation timing telemetry

Branch `feature-eager-activation`:  
- Added eager activation for workspace-trust-sensitive extensions
- Modified activation order logic
- Added new activation event types

Merge conflicts in:
- src/vs/workbench/services/extensions/common/extensionHostManager.ts
- src/vs/workbench/services/extensions/common/extensionActivation.ts

Both changes are valuable and should be preserved.
""",
        "vague_prompt": "Merge the eager-activation feature into main",
        "ground_truth": {
            "conflict_resolution": [
                "Keep async activation from main",
                "Keep eager activation logic from feature branch",
                "Combine activation event types",
                "Ensure timing telemetry covers eager activations",
            ],
            "semantic_merges": [
                "ExtensionHostManager.activate() needs both async + eager handling",
                "Activation order must respect both async and eager constraints",
            ],
            "test_requirements": [
                "Existing async activation tests pass",
                "New eager activation behavior works",
                "No activation order regressions",
            ],
        },
        "scoring_criteria": {
            "both_features_work": {
                "weight": 0.5,
                "levels": {
                    "yes": "Async and eager activation both work",
                    "partial": "One feature compromised",
                    "no": "One or both features broken",
                },
            },
            "no_regressions": {
                "weight": 0.3,
                "levels": {
                    "clean": "All existing tests pass",
                    "minor": "Some test adjustments needed",
                    "major": "Significant test failures",
                },
            },
            "code_quality": {
                "weight": 0.2,
                "levels": {
                    "clean": "Merged code is maintainable",
                    "messy": "Works but needs cleanup",
                },
            },
        },
        "difficulty": "expert",
        "tags": ["merge-conflict", "refactoring", "typescript"],
    },
]


# =============================================================================
# CI/CD DIAGNOSIS TASKS
# =============================================================================

CI_CD_TASKS = [
    {
        "task_type": "build_failure_diagnosis",
        "repo_name": "kubernetes/kubernetes",
        "title": "Cross-compilation failure for arm64",
        "scenario": """
CI build for arm64 is failing:

```
# Building for linux/arm64
go build -o _output/bin/kube-apiserver ./cmd/kube-apiserver
# k8s.io/kubernetes/vendor/go.etcd.io/bbolt
vendor/go.etcd.io/bbolt/bolt_arm64.go:8:2: undefined: maxMapSize
vendor/go.etcd.io/bbolt/bolt_arm64.go:9:2: undefined: maxAllocSize
FAIL
```

This started failing after a recent dependency update.
The amd64 build works fine.
""",
        "vague_prompt": "ARM64 build is broken. Fix it.",
        "ground_truth": {
            "root_cause": "bbolt version incompatible with current Go version for arm64",
            "diagnosis_steps": [
                "Check recent changes to go.mod",
                "Identify bbolt version change",
                "Check bbolt release notes for arm64 issues",
            ],
            "fix_options": [
                "Pin bbolt to compatible version",
                "Upgrade Go to version with fix",
                "Apply patch for arm64 constants",
            ],
        },
        "scoring_criteria": {
            "correct_diagnosis": {
                "weight": 0.4,
                "levels": {
                    "exact": "Identifies bbolt + Go version incompatibility",
                    "partial": "Identifies dependency issue",
                    "wrong": "Blames unrelated component",
                },
            },
            "fix_works": {
                "weight": 0.4,
                "levels": {
                    "yes": "Build passes on arm64",
                    "partial": "Build passes but other issues",
                    "no": "Still fails",
                },
            },
            "minimal_change": {
                "weight": 0.2,
                "levels": {
                    "minimal": "Only changes necessary deps",
                    "acceptable": "Some extra changes",
                    "excessive": "Too many unrelated changes",
                },
            },
        },
        "difficulty": "hard",
        "tags": ["ci", "build-failure", "cross-compilation", "arm64"],
    },
    {
        "task_type": "dependency_conflict_resolution",
        "repo_name": "grafana/grafana",
        "title": "React peer dependency conflicts",
        "scenario": """
After updating to a new charting library, npm install fails:

```
npm ERR! ERESOLVE unable to resolve dependency tree
npm ERR! 
npm ERR! peer react@"^17.0.0" from @visx/visx@2.18.0
npm ERR! node_modules/@visx/visx
npm ERR!   @visx/visx@"^2.18.0" from the root project
npm ERR! 
npm ERR! Could not resolve dependency:
npm ERR! peer react@"^17.0.0 || ^18.0.0" from react-dom@18.2.0
npm ERR! node_modules/react-dom
npm ERR!   react-dom@"^18.2.0" from the root project
npm ERR!
npm ERR! Fix the upstream dependency conflict
```

Grafana is on React 18. The @visx/visx library needs to work for data visualization.
""",
        "vague_prompt": "Can't install dependencies anymore. npm is complaining about React versions.",
        "ground_truth": {
            "root_cause": "@visx/visx 2.18.0 peer dependency too strict (only ^17.0.0)",
            "resolution_options": [
                "Upgrade to @visx/visx 3.x (supports React 18)",
                "Use npm overrides to force React 18",
                "Find alternative visx-compatible React 18 version",
            ],
            "recommended": "Upgrade @visx/visx to 3.x, update breaking API usages",
            "migration_effort": "~15 files need API updates for visx 3.x",
        },
        "scoring_criteria": {
            "correct_diagnosis": {
                "weight": 0.3,
                "levels": {
                    "exact": "Identifies @visx version incompatibility",
                    "partial": "Identifies peer dep issue",
                },
            },
            "working_resolution": {
                "weight": 0.5,
                "levels": {
                    "clean": "Upgrade to compatible version",
                    "workaround": "Use overrides (may cause issues)",
                    "broken": "Doesn't resolve",
                },
            },
            "handles_migration": {
                "weight": 0.2,
                "levels": {
                    "complete": "Updates breaking API usages",
                    "partial": "Some usages updated",
                },
            },
        },
        "difficulty": "medium",
        "tags": ["ci", "dependencies", "npm", "react"],
    },
]


# =============================================================================
# TEST EVOLUTION TASKS
# =============================================================================

TEST_EVOLUTION_TASKS = [
    {
        "task_type": "test_gap_identification",
        "repo_name": "elastic/elasticsearch",
        "title": "Identify test gaps in snapshot restore",
        "scenario": """
The snapshot restore functionality has had several production bugs recently:
1. Partial restore fails silently
2. Cross-cluster restore corrupts index mappings
3. Restore from cold storage times out

Current test coverage appears adequate (85% line coverage), but bugs keep escaping.

Review the test suite for snapshot restore and identify what's missing.

Key files:
- server/src/test/java/org/elasticsearch/snapshots/SnapshotRestoreTests.java
- server/src/test/java/org/elasticsearch/snapshots/RestoreServiceTests.java
""",
        "vague_prompt": "Our snapshot tests have good coverage but bugs keep escaping. What are we missing?",
        "ground_truth": {
            "identified_gaps": [
                "No tests for partial restore with some shards unavailable",
                "Cross-cluster tests don't verify mapping compatibility",
                "No slow storage simulation in tests",
                "Missing concurrent restore + write tests",
                "No tests for restore interruption recovery",
            ],
            "false_coverage": [
                "Tests cover happy path extensively",
                "Error paths have mocks that hide real behavior",
                "Integration tests run with minimal data",
            ],
            "recommended_tests": [
                "Partial restore with simulated failures",
                "Cross-cluster with incompatible mappings",
                "Timeout behavior with throttled I/O",
                "Concurrent operations during restore",
            ],
        },
        "scoring_criteria": {
            "real_gaps_found": {
                "weight": 0.5,
                "levels": {
                    "critical": "Finds gaps that match production bugs",
                    "useful": "Finds real gaps but not production-relevant",
                    "trivial": "Only finds minor coverage gaps",
                },
            },
            "no_false_positives": {
                "weight": 0.2,
                "levels": {
                    "clean": "All identified gaps are real",
                    "some_noise": "Some false positives",
                    "noisy": "Many false positives",
                },
            },
            "actionable": {
                "weight": 0.3,
                "levels": {
                    "complete": "Clear test specifications for each gap",
                    "partial": "Some actionable recommendations",
                    "vague": "Observations without clear actions",
                },
            },
        },
        "difficulty": "expert",
        "tags": ["testing", "test-gaps", "coverage-analysis", "elasticsearch"],
    },
    {
        "task_type": "test_maintenance",
        "repo_name": "microsoft/vscode",
        "title": "Update tests after Settings UI refactor",
        "scenario": """
The Settings UI was refactored from class-based to functional React components.

20 tests are now failing with:
```
TypeError: Cannot read property 'state' of undefined
    at SettingsEditor.test.ts:45
```

The tests were written for the old class-based API:
```typescript
// Old test pattern
const editor = new SettingsEditor();
editor.setState({ query: 'font' });
expect(editor.getFilteredSettings()).toHaveLength(5);
```

Update the tests to work with the new functional component API while still validating the same behaviors.
""",
        "vague_prompt": "The Settings UI tests are broken after the refactor. Fix them.",
        "ground_truth": {
            "required_changes": [
                "Replace class instantiation with render()",
                "Use testing-library queries instead of state access",
                "Test through user interactions not internal state",
                "Mock hooks instead of class methods",
            ],
            "behavior_to_preserve": [
                "Filtering settings by query",
                "Category navigation",
                "Setting modification",
                "Search highlighting",
            ],
            "new_test_pattern": """
// New test pattern
render(<SettingsEditor />);
await userEvent.type(screen.getByRole('searchbox'), 'font');
expect(screen.getAllByTestId('setting-item')).toHaveLength(5);
""",
        },
        "scoring_criteria": {
            "tests_pass": {
                "weight": 0.4,
                "levels": {
                    "all": "All 20 tests pass",
                    "most": ">15 tests pass",
                    "some": ">10 tests pass",
                    "few": "<10 tests pass",
                },
            },
            "same_coverage": {
                "weight": 0.3,
                "levels": {
                    "equivalent": "Same behaviors tested",
                    "reduced": "Some behaviors lost",
                    "expanded": "More behaviors tested",
                },
            },
            "idiomatic": {
                "weight": 0.3,
                "levels": {
                    "modern": "Uses testing-library best practices",
                    "functional": "Works but not idiomatic",
                    "legacy": "Still testing implementation details",
                },
            },
        },
        "difficulty": "medium",
        "tags": ["testing", "test-maintenance", "refactoring", "react"],
    },
]


# =============================================================================
# DOCUMENTATION TASKS
# =============================================================================

DOCUMENTATION_TASKS = [
    {
        "task_type": "doc_staleness_detection",
        "repo_name": "kubernetes/kubernetes",
        "title": "Audit kubectl documentation accuracy",
        "scenario": """
Users report that kubectl documentation is often out of date with actual behavior.

Review the kubectl documentation against the current implementation for:
- `kubectl apply` - particularly the strategic merge patch behavior
- `kubectl rollout` - especially the new `rollout history` format
- `kubectl debug` - which has had several recent additions

Identify any discrepancies between docs and actual behavior.
""",
        "vague_prompt": "Check if our kubectl docs are accurate",
        "ground_truth": {
            "discrepancies_found": [
                "kubectl apply --server-side not documented for namespace conflicts",
                "kubectl rollout history shows different columns than documented",
                "kubectl debug --copy-to not in stable docs despite being GA",
                "kubectl apply --prune behavior differs from docs for CRDs",
            ],
            "docs_to_update": [
                "docs/reference/generated/kubectl/kubectl-commands.md",
                "docs/concepts/workloads/controllers/deployment.md",
            ],
            "implementation_locations": [
                "staging/src/k8s.io/kubectl/pkg/cmd/apply/",
                "staging/src/k8s.io/kubectl/pkg/cmd/rollout/",
            ],
        },
        "scoring_criteria": {
            "real_discrepancies": {
                "weight": 0.5,
                "levels": {
                    "significant": "Finds user-impacting discrepancies",
                    "minor": "Finds cosmetic discrepancies",
                    "none": "Misses real discrepancies",
                },
            },
            "no_false_positives": {
                "weight": 0.2,
                "levels": {
                    "accurate": "All flagged items are real issues",
                    "mostly": "Some false positives",
                },
            },
            "actionable_fixes": {
                "weight": 0.3,
                "levels": {
                    "complete": "Specific doc updates proposed",
                    "partial": "General areas identified",
                },
            },
        },
        "difficulty": "hard",
        "tags": ["documentation", "kubectl", "accuracy-audit"],
    },
]


def generate_all_tasks():
    """Generate all gap-filling tasks."""
    all_tasks = []
    
    task_collections = [
        ("debugging", DEBUGGING_TASKS),
        ("code_transformation", CODE_TRANSFORMATION_TASKS),
        ("ci_cd", CI_CD_TASKS),
        ("test_evolution", TEST_EVOLUTION_TASKS),
        ("documentation", DOCUMENTATION_TASKS),
    ]
    
    for category, tasks in task_collections:
        for task in tasks:
            task["task_id"] = generate_task_id(
                task["repo_name"],
                task["task_type"],
                task["title"],
            )
            task["category"] = category
            task["created_at"] = datetime.now(timezone.utc).isoformat()
            all_tasks.append(task)
    
    return all_tasks


def export_gap_benchmark(output_file: Path):
    """Export the gap-filling benchmark."""
    tasks = generate_all_tasks()
    
    with open(output_file, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    
    print(f"Exported {len(tasks)} gap-filling tasks to {output_file}")
    
    # Summary by category
    by_category = {}
    for task in tasks:
        cat = task["category"]
        by_category.setdefault(cat, []).append(task)
    
    print("\nTasks by CACM gap:")
    for cat, cat_tasks in sorted(by_category.items()):
        print(f"  {cat}: {len(cat_tasks)} tasks")
        for t in cat_tasks:
            print(f"    - {t['task_type']}: {t['title']}")


if __name__ == "__main__":
    output = Path("benchmarks/ir-sdlc-gap-filling.jsonl")
    export_gap_benchmark(output)
