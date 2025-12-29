"""
IR-SDLC-Bench: Comprehensive SDLC Benchmark Taxonomy

This module defines a portfolio-based benchmark framework that addresses the gaps
identified in Satish Chandra's CACM blog post "Benchmarks for AI in Software Engineering".

Key Design Principles:
1. SDLC Coverage - Tasks across ALL phases, not just "write the patch"
2. Enterprise Realism - Large repos, dependency chains, regulated industries
3. Human-Aligned Scoring - Partial credit, "good enough to ship" criteria
4. Living Pipeline - Refresh mechanisms, contamination resistance
5. Dual Purpose - Works for both ML training and product evaluation

Gap Analysis from CACM Blog:
- ✓ Representativeness (enterprise-scale repos)
- ✓ Diversity (multiple languages, industries, task types)
- ✓ Headroom (graduated difficulty, expert-level tasks)
- ✓ Contamination resistance (refresh pipeline, version-pinned tasks)
- ✓ Robust scoring (human-aligned rubrics, not just test pass/fail)
- ✓ SDLC breadth (all phases covered)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# =============================================================================
# SDLC Phase Taxonomy
# =============================================================================

class SDLCPhase(Enum):
    """Complete software development lifecycle phases."""
    
    # Planning & Requirements
    REQUIREMENTS = "requirements"
    
    # Design & Architecture  
    DESIGN = "design"
    
    # Implementation
    IMPLEMENTATION = "implementation"
    
    # Quality Assurance
    TESTING = "testing"
    CODE_REVIEW = "code_review"
    
    # Security
    SECURITY = "security"
    
    # Deployment & Operations
    CI_CD = "ci_cd"
    DEPLOYMENT = "deployment"
    
    # Maintenance & Evolution
    MAINTENANCE = "maintenance"
    DEBUGGING = "debugging"
    
    # Documentation
    DOCUMENTATION = "documentation"


# =============================================================================
# Task Type Taxonomy (Addressing CACM Gaps)
# =============================================================================

class TaskCategory(Enum):
    """High-level task categories mapped to CACM gaps."""
    
    # Currently well-covered
    CODE_GENERATION = "code_generation"  # HumanEval territory
    BUG_FIXING = "bug_fixing"           # SWE-bench territory
    CODE_REVIEW = "code_review"         # Our strength
    
    # CACM-identified gaps (our focus)
    DEBUGGING = "debugging"             # Root cause analysis
    CODE_TRANSFORMATION = "code_transformation"  # Refactoring, migration
    REASONING = "reasoning"             # Understanding unfamiliar systems
    TEST_EVOLUTION = "test_evolution"   # Test adequacy, not just generation
    CI_DIAGNOSIS = "ci_diagnosis"       # Build failures, pipeline issues
    CROSS_SYSTEM = "cross_system"       # Multi-repo, API contracts


@dataclass
class TaskType:
    """A specific task type within a category."""
    
    name: str
    category: TaskCategory
    sdlc_phase: SDLCPhase
    description: str
    
    # Evaluation characteristics
    requires_execution: bool  # Needs actual code execution to score
    requires_human_judgment: bool  # Some aspects need human eval
    can_be_automated: bool  # Scoring can be fully automated
    
    # CACM criteria alignment
    enterprise_relevant: bool  # Matters for real enterprise teams
    ml_trainable: bool  # Can be used for model training
    
    # Example prompts
    example_vague_prompt: str
    example_precise_prompt: str


# Complete task type taxonomy
TASK_TYPE_TAXONOMY = {
    # ==========================================================================
    # GAP 1: DEBUGGING / ROOT CAUSE ANALYSIS (Not covered by existing benchmarks)
    # ==========================================================================
    
    "stack_trace_to_fix": TaskType(
        name="stack_trace_to_fix",
        category=TaskCategory.DEBUGGING,
        sdlc_phase=SDLCPhase.DEBUGGING,
        description="Given a stack trace, identify the root cause and fix location",
        requires_execution=False,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="The app is crashing, here's the stack trace",
        example_precise_prompt="Analyze this NullPointerException stack trace and identify the root cause in the authentication module",
    ),
    
    "log_analysis": TaskType(
        name="log_analysis",
        category=TaskCategory.DEBUGGING,
        sdlc_phase=SDLCPhase.DEBUGGING,
        description="Analyze logs to identify failure patterns and root causes",
        requires_execution=False,
        requires_human_judgment=True,  # Causation vs correlation
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Production is slow, check the logs",
        example_precise_prompt="Analyze these application logs to identify why request latency spiked at 3:42 PM",
    ),
    
    "flaky_test_diagnosis": TaskType(
        name="flaky_test_diagnosis",
        category=TaskCategory.DEBUGGING,
        sdlc_phase=SDLCPhase.TESTING,
        description="Identify why a test is flaky and how to fix it",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="This test sometimes fails, fix it",
        example_precise_prompt="This integration test fails ~10% of runs. Identify the race condition and fix it.",
    ),
    
    "performance_regression": TaskType(
        name="performance_regression",
        category=TaskCategory.DEBUGGING,
        sdlc_phase=SDLCPhase.DEBUGGING,
        description="Identify the cause of a performance regression between commits",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,  # Hard to get ground truth at scale
        example_vague_prompt="The app got slower after the last release",
        example_precise_prompt="Response time increased 40% between commits A and B. Identify the regressing change.",
    ),
    
    # ==========================================================================
    # GAP 2: CODE TRANSFORMATION (Migration, Refactoring)
    # ==========================================================================
    
    "api_migration": TaskType(
        name="api_migration",
        category=TaskCategory.CODE_TRANSFORMATION,
        sdlc_phase=SDLCPhase.MAINTENANCE,
        description="Migrate code from deprecated API to new API",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Update the code to use the new SDK",
        example_precise_prompt="Migrate from AWS SDK v2 to v3 in this module, preserving existing behavior",
    ),
    
    "framework_upgrade": TaskType(
        name="framework_upgrade",
        category=TaskCategory.CODE_TRANSFORMATION,
        sdlc_phase=SDLCPhase.MAINTENANCE,
        description="Upgrade framework version with breaking changes",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Upgrade to React 19",
        example_precise_prompt="Upgrade from React 18 to 19, handling the new use() hook and compiler changes",
    ),
    
    "language_version_upgrade": TaskType(
        name="language_version_upgrade",
        category=TaskCategory.CODE_TRANSFORMATION,
        sdlc_phase=SDLCPhase.MAINTENANCE,
        description="Upgrade language version (e.g., Python 3.9 to 3.12)",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Update to Python 3.12",
        example_precise_prompt="Upgrade from Python 3.9 to 3.12, using new features where appropriate (match, tomllib)",
    ),
    
    "extract_module": TaskType(
        name="extract_module",
        category=TaskCategory.CODE_TRANSFORMATION,
        sdlc_phase=SDLCPhase.MAINTENANCE,
        description="Extract code into a reusable module/package",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="Make this code reusable",
        example_precise_prompt="Extract the authentication logic into a standalone package that can be shared across services",
    ),
    
    "merge_conflict_resolution": TaskType(
        name="merge_conflict_resolution",
        category=TaskCategory.CODE_TRANSFORMATION,
        sdlc_phase=SDLCPhase.IMPLEMENTATION,
        description="Resolve merge conflicts preserving intent from both branches",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Fix the merge conflicts",
        example_precise_prompt="Resolve these merge conflicts between feature-auth and main, preserving the rate limiting from main and the OAuth from feature-auth",
    ),
    
    # ==========================================================================
    # GAP 3: TEST EVOLUTION (Not just generation)
    # ==========================================================================
    
    "test_gap_identification": TaskType(
        name="test_gap_identification",
        category=TaskCategory.TEST_EVOLUTION,
        sdlc_phase=SDLCPhase.TESTING,
        description="Identify missing test coverage for a component",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Are our tests good enough?",
        example_precise_prompt="Identify edge cases not covered by existing tests for the payment processing module",
    ),
    
    "test_maintenance": TaskType(
        name="test_maintenance",
        category=TaskCategory.TEST_EVOLUTION,
        sdlc_phase=SDLCPhase.TESTING,
        description="Update tests after implementation changes",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Tests are failing after my change",
        example_precise_prompt="Update tests to match the new API signature while still validating the same behaviors",
    ),
    
    "mutation_testing_analysis": TaskType(
        name="mutation_testing_analysis",
        category=TaskCategory.TEST_EVOLUTION,
        sdlc_phase=SDLCPhase.TESTING,
        description="Analyze mutation testing results to improve test quality",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="Improve our test quality",
        example_precise_prompt="These 15 mutations survived. Add assertions that would kill them.",
    ),
    
    # ==========================================================================
    # GAP 4: CI/CD DIAGNOSIS
    # ==========================================================================
    
    "build_failure_diagnosis": TaskType(
        name="build_failure_diagnosis",
        category=TaskCategory.CI_DIAGNOSIS,
        sdlc_phase=SDLCPhase.CI_CD,
        description="Diagnose and fix CI build failures",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="CI is red",
        example_precise_prompt="The build failed with 'Module not found'. Identify the missing dependency and fix it.",
    ),
    
    "pipeline_optimization": TaskType(
        name="pipeline_optimization",
        category=TaskCategory.CI_DIAGNOSIS,
        sdlc_phase=SDLCPhase.CI_CD,
        description="Optimize CI/CD pipeline for speed or reliability",
        requires_execution=True,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="CI is too slow",
        example_precise_prompt="This CI pipeline takes 45 minutes. Identify parallelization opportunities and caching improvements.",
    ),
    
    "dependency_conflict_resolution": TaskType(
        name="dependency_conflict_resolution",
        category=TaskCategory.CI_DIAGNOSIS,
        sdlc_phase=SDLCPhase.CI_CD,
        description="Resolve conflicting dependency versions",
        requires_execution=True,
        requires_human_judgment=False,
        can_be_automated=True,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="npm install fails",
        example_precise_prompt="Resolve this peer dependency conflict between react-router v6 and our react version",
    ),
    
    # ==========================================================================
    # GAP 5: CROSS-SYSTEM REASONING
    # ==========================================================================
    
    "api_versioning_strategy": TaskType(
        name="api_versioning_strategy",
        category=TaskCategory.CROSS_SYSTEM,
        sdlc_phase=SDLCPhase.DESIGN,
        description="Design backward-compatible API changes",
        requires_execution=False,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="We need to change this API",
        example_precise_prompt="Add a 'priority' field to the Task API without breaking existing clients",
    ),
    
    "service_dependency_mapping": TaskType(
        name="service_dependency_mapping",
        category=TaskCategory.CROSS_SYSTEM,
        sdlc_phase=SDLCPhase.DESIGN,
        description="Map dependencies between microservices",
        requires_execution=False,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="What services does this call?",
        example_precise_prompt="Map all downstream service dependencies of the order-service, including async and event-driven",
    ),
    
    "rollback_safety_analysis": TaskType(
        name="rollback_safety_analysis",
        category=TaskCategory.CROSS_SYSTEM,
        sdlc_phase=SDLCPhase.DEPLOYMENT,
        description="Assess whether a deployment can be safely rolled back",
        requires_execution=False,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=False,
        example_vague_prompt="Can we roll this back?",
        example_precise_prompt="Analyze whether this database migration is reversible and what the rollback procedure would be",
    ),
    
    # ==========================================================================
    # GAP 6: DOCUMENTATION (Often ignored in benchmarks)
    # ==========================================================================
    
    "doc_staleness_detection": TaskType(
        name="doc_staleness_detection",
        category=TaskCategory.REASONING,
        sdlc_phase=SDLCPhase.DOCUMENTATION,
        description="Identify documentation that's out of sync with code",
        requires_execution=False,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Is our documentation up to date?",
        example_precise_prompt="Compare the README and API docs against the current implementation and flag discrepancies",
    ),
    
    "changelog_generation": TaskType(
        name="changelog_generation",
        category=TaskCategory.REASONING,
        sdlc_phase=SDLCPhase.DOCUMENTATION,
        description="Generate accurate changelog from commits",
        requires_execution=False,
        requires_human_judgment=True,
        can_be_automated=False,
        enterprise_relevant=True,
        ml_trainable=True,
        example_vague_prompt="Write the changelog",
        example_precise_prompt="Generate a changelog for the v2.3.0 release from the 47 commits since v2.2.0",
    ),
}


# =============================================================================
# Enterprise Realism Bar
# =============================================================================

@dataclass
class EnterpriseRealismCriteria:
    """Minimum criteria for enterprise-realistic benchmarks."""
    
    # Repository characteristics
    min_loc: int = 100_000  # Lines of code
    min_files: int = 500
    min_contributors: int = 20
    min_commit_history_months: int = 24
    
    # Complexity indicators
    has_ci_pipeline: bool = True
    has_test_suite: bool = True
    has_multiple_languages: bool = False
    has_external_dependencies: int = 10  # Minimum deps
    
    # Activity indicators
    min_issues_closed_per_month: int = 10
    min_prs_merged_per_month: int = 20
    
    # Industry markers (at least one)
    industry_markers: list[str] = field(default_factory=lambda: [
        "regulated",  # Finance, healthcare, gov
        "distributed",  # Multiple services
        "legacy",  # Long-lived codebase
        "high_availability",  # SLA requirements
    ])


ENTERPRISE_REALISM_LEVELS = {
    "starter": EnterpriseRealismCriteria(
        min_loc=10_000,
        min_files=100,
        min_contributors=5,
        min_commit_history_months=6,
    ),
    "standard": EnterpriseRealismCriteria(
        min_loc=100_000,
        min_files=500,
        min_contributors=20,
        min_commit_history_months=24,
    ),
    "enterprise": EnterpriseRealismCriteria(
        min_loc=500_000,
        min_files=2000,
        min_contributors=100,
        min_commit_history_months=48,
        has_multiple_languages=True,
        has_external_dependencies=50,
    ),
    "hyperscale": EnterpriseRealismCriteria(
        min_loc=5_000_000,
        min_files=10000,
        min_contributors=500,
        min_commit_history_months=72,
        has_multiple_languages=True,
        has_external_dependencies=200,
    ),
}


# =============================================================================
# Human-Aligned Scoring Framework
# =============================================================================

@dataclass 
class ScoringDimension:
    """A dimension along which to score agent output."""
    
    name: str
    description: str
    weight: float  # 0-1, all dimensions should sum to 1
    
    # Scoring approach
    automated: bool  # Can be scored automatically
    scoring_method: str  # "binary", "partial", "rubric", "human"
    
    # Partial credit criteria (for partial scoring)
    partial_credit_levels: Optional[dict[str, float]] = None


@dataclass
class HumanAlignedScorer:
    """Scoring framework aligned with human judgment of 'good enough to ship'."""
    
    task_type: str
    dimensions: list[ScoringDimension]
    
    # Human-in-the-loop considerations
    perfect_not_required: bool  # Can be useful without being perfect
    partial_credit_meaningful: bool  # Partial solutions have value
    
    # Contamination resistance
    requires_fresh_context: bool  # Needs repo state, not just cached
    answer_in_prompt_risk: str  # "low", "medium", "high"


# Example scorers for key task types
HUMAN_ALIGNED_SCORERS = {
    "stack_trace_to_fix": HumanAlignedScorer(
        task_type="stack_trace_to_fix",
        dimensions=[
            ScoringDimension(
                name="correct_root_cause",
                description="Identified the actual root cause, not a symptom",
                weight=0.4,
                automated=False,
                scoring_method="rubric",
                partial_credit_levels={
                    "exact": 1.0,
                    "correct_file_wrong_function": 0.6,
                    "correct_module": 0.3,
                    "wrong": 0.0,
                },
            ),
            ScoringDimension(
                name="fix_correctness",
                description="Proposed fix actually resolves the issue",
                weight=0.4,
                automated=True,  # Can run tests
                scoring_method="binary",
            ),
            ScoringDimension(
                name="explanation_quality",
                description="Explanation helps developer understand the issue",
                weight=0.2,
                automated=False,
                scoring_method="rubric",
                partial_credit_levels={
                    "educational": 1.0,
                    "correct_but_terse": 0.6,
                    "confusing": 0.2,
                },
            ),
        ],
        perfect_not_required=True,
        partial_credit_meaningful=True,
        requires_fresh_context=True,
        answer_in_prompt_risk="low",
    ),
    
    "api_migration": HumanAlignedScorer(
        task_type="api_migration",
        dimensions=[
            ScoringDimension(
                name="functionality_preserved",
                description="Migrated code has same behavior as original",
                weight=0.5,
                automated=True,
                scoring_method="binary",
            ),
            ScoringDimension(
                name="migration_completeness",
                description="All usages of old API are migrated",
                weight=0.3,
                automated=True,
                scoring_method="partial",
                partial_credit_levels={
                    "complete": 1.0,
                    "most": 0.7,
                    "some": 0.4,
                    "few": 0.1,
                },
            ),
            ScoringDimension(
                name="idiomatic_usage",
                description="New API is used idiomatically",
                weight=0.2,
                automated=False,
                scoring_method="rubric",
            ),
        ],
        perfect_not_required=True,
        partial_credit_meaningful=True,
        requires_fresh_context=False,
        answer_in_prompt_risk="medium",
    ),
    
    "test_gap_identification": HumanAlignedScorer(
        task_type="test_gap_identification",
        dimensions=[
            ScoringDimension(
                name="gaps_found",
                description="Identified real testing gaps",
                weight=0.5,
                automated=False,
                scoring_method="rubric",
                partial_credit_levels={
                    "critical_gaps_found": 1.0,
                    "some_gaps_found": 0.6,
                    "trivial_gaps_only": 0.2,
                },
            ),
            ScoringDimension(
                name="no_false_positives",
                description="Didn't flag well-tested code as untested",
                weight=0.3,
                automated=True,
                scoring_method="partial",
            ),
            ScoringDimension(
                name="actionable_recommendations",
                description="Recommendations can be acted on",
                weight=0.2,
                automated=False,
                scoring_method="rubric",
            ),
        ],
        perfect_not_required=True,
        partial_credit_meaningful=True,
        requires_fresh_context=True,
        answer_in_prompt_risk="low",
    ),
}


# =============================================================================
# Portfolio-Based Benchmark Structure
# =============================================================================

@dataclass
class BenchmarkPortfolio:
    """
    A portfolio-based benchmark that balances multiple concerns.
    
    Unlike a single leaderboard, a portfolio acknowledges that:
    - Different tasks require different evaluation approaches
    - Enterprise needs differ from academic ML needs
    - Partial credit matters for real-world usefulness
    """
    
    name: str
    version: str
    
    # Task composition
    task_types: list[str]
    total_tasks: int
    
    # Balance constraints
    sdlc_coverage: dict[str, int]  # min tasks per SDLC phase
    difficulty_distribution: dict[str, float]  # target percentages
    enterprise_realism_level: str
    
    # Scoring approach
    scoring_dimensions: list[str]
    requires_human_eval_pct: float  # % of tasks needing human scoring
    
    # Refresh policy
    refresh_cadence_days: int
    contamination_resistance: str  # "low", "medium", "high"
    
    # Dual-purpose compatibility
    ml_training_compatible: bool
    product_eval_compatible: bool


# Proposed portfolio structure
PROPOSED_PORTFOLIO = BenchmarkPortfolio(
    name="IR-SDLC-Bench",
    version="1.0.0",
    
    task_types=[
        # Well-covered (existing)
        "bug_triage", "code_review", "feature_location",
        "security_audit", "dependency_analysis",
        
        # CACM gaps (new)
        "stack_trace_to_fix", "log_analysis", "flaky_test_diagnosis",
        "api_migration", "framework_upgrade", "extract_module",
        "merge_conflict_resolution",
        "test_gap_identification", "test_maintenance",
        "build_failure_diagnosis", "dependency_conflict_resolution",
        "api_versioning_strategy", "service_dependency_mapping",
        "doc_staleness_detection",
        
        # Advanced reasoning
        "requirement_disambiguation", "architecture_delta",
        "invariant_discovery", "downstream_impact",
    ],
    
    total_tasks=200,  # Target
    
    sdlc_coverage={
        "requirements": 5,
        "design": 20,
        "implementation": 40,
        "testing": 30,
        "code_review": 30,
        "security": 15,
        "ci_cd": 20,
        "maintenance": 25,
        "debugging": 20,
        "documentation": 10,
    },
    
    difficulty_distribution={
        "easy": 0.1,
        "medium": 0.25,
        "hard": 0.35,
        "expert": 0.30,
    },
    
    enterprise_realism_level="standard",
    
    scoring_dimensions=[
        "correctness",
        "completeness", 
        "explanation_quality",
        "convention_adherence",
        "safety_awareness",
    ],
    
    requires_human_eval_pct=0.30,  # 30% need human judgment
    
    refresh_cadence_days=90,
    contamination_resistance="high",
    
    ml_training_compatible=True,
    product_eval_compatible=True,
)


def export_taxonomy():
    """Export the taxonomy for documentation."""
    
    print("=" * 80)
    print("IR-SDLC-Bench: Task Type Taxonomy")
    print("=" * 80)
    
    # Group by category
    by_category = {}
    for name, task_type in TASK_TYPE_TAXONOMY.items():
        cat = task_type.category.value
        by_category.setdefault(cat, []).append((name, task_type))
    
    for category, tasks in sorted(by_category.items()):
        print(f"\n## {category.upper()}")
        print("-" * 40)
        for name, tt in tasks:
            auto = "✓" if tt.can_be_automated else "✗"
            enterprise = "✓" if tt.enterprise_relevant else "✗"
            print(f"  {name}")
            print(f"    Phase: {tt.sdlc_phase.value}")
            print(f"    Automated: {auto} | Enterprise: {enterprise}")
            print(f"    Example: \"{tt.example_vague_prompt}\"")
    
    print("\n" + "=" * 80)
    print("SDLC Coverage Target")
    print("=" * 80)
    
    for phase, count in PROPOSED_PORTFOLIO.sdlc_coverage.items():
        print(f"  {phase}: {count} tasks")
    
    print(f"\nTotal target: {PROPOSED_PORTFOLIO.total_tasks} tasks")
    print(f"Human eval required: {PROPOSED_PORTFOLIO.requires_human_eval_pct:.0%}")


if __name__ == "__main__":
    export_taxonomy()
