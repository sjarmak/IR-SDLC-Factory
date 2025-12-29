"""
IR-SDLC-Bench: Information Retrieval for Software Development Lifecycle Tasks

This module provides the core functionality for generating and evaluating
information retrieval tasks across enterprise-scale software repositories,
with full compatibility with the Harbor evaluation framework.
"""

from app.ir_sdlc.task_types import (
    SDLCTaskType,
    BugTriageTask,
    CodeReviewTask,
    DependencyAnalysisTask,
    ArchitectureUnderstandingTask,
    SecurityAuditTask,
    RefactoringAnalysisTask,
    TestCoverageTask,
    DocumentationLinkingTask,
)

from app.ir_sdlc.data_structures import (
    IRTask,
    RetrievalResult,
    GroundTruth,
    IREvaluationResult,
)

from app.ir_sdlc.harbor_adapter import (
    HarborTaskGenerator,
    generate_harbor_task,
    generate_harbor_dataset,
)

from app.ir_sdlc.metrics import (
    IRMetrics,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_mrr,
    compute_ndcg,
    compute_map,
)

from app.ir_sdlc.dashboard_schema import (
    IRSDLCTaskResult,
    IRSDLCBenchmarkRun,
    IRComparison,
    IRRetrievalMetrics,
    AgentExecutionMetrics,
    LLMJudgeScore,
    IRToolType,
    SDLCTaskType as DashboardSDLCTaskType,
)

from app.ir_sdlc.dashboard_exporter import (
    CodeContextBenchExporter,
    generate_run_id,
)

__all__ = [
    # Task Types
    "SDLCTaskType",
    "BugTriageTask",
    "CodeReviewTask",
    "DependencyAnalysisTask",
    "ArchitectureUnderstandingTask",
    "SecurityAuditTask",
    "RefactoringAnalysisTask",
    "TestCoverageTask",
    "DocumentationLinkingTask",
    # Data Structures
    "IRTask",
    "RetrievalResult",
    "GroundTruth",
    "IREvaluationResult",
    # Harbor Adapter
    "HarborTaskGenerator",
    "generate_harbor_task",
    "generate_harbor_dataset",
    # Metrics
    "IRMetrics",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_mrr",
    "compute_ndcg",
    "compute_map",
    # Dashboard Integration
    "IRSDLCTaskResult",
    "IRSDLCBenchmarkRun",
    "IRComparison",
    "IRRetrievalMetrics",
    "AgentExecutionMetrics",
    "LLMJudgeScore",
    "IRToolType",
    "CodeContextBenchExporter",
    "generate_run_id",
]
