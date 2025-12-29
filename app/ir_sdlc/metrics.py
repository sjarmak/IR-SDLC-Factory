"""
IR Evaluation Metrics for IR-SDLC-Bench.

Provides standard information retrieval metrics as well as
SDLC-specific metrics for comprehensive evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from app.ir_sdlc.data_structures import (
    RetrievalResult,
    GroundTruth,
    CodeLocation,
    RetrievalGranularity,
    IREvaluationResult,
)


def compute_precision_at_k(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """
    Compute Precision@K.

    Precision@K = (# of relevant items in top K) / K
    """
    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    relevant_count = 0
    for result in top_k:
        if _is_relevant(result.location, ground_truth):
            relevant_count += 1

    return relevant_count / k


def compute_recall_at_k(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """
    Compute Recall@K.

    Recall@K = (# of relevant items in top K) / (total # of relevant items)
    """
    if not ground_truth.locations:
        return 1.0  # No relevant items means perfect recall

    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    # Track which ground truth items are found
    found_gt_indices = set()
    for result in top_k:
        for i, gt_loc in enumerate(ground_truth.locations):
            if result.location.matches(gt_loc, ground_truth.granularity):
                found_gt_indices.add(i)

    return len(found_gt_indices) / len(ground_truth.locations)


def compute_mrr(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    MRR = 1 / (rank of first relevant result)
    """
    for i, result in enumerate(retrieved):
        if _is_relevant(result.location, ground_truth):
            return 1.0 / (i + 1)

    return 0.0


def compute_ndcg(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: Optional[int] = None,
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG = DCG / IDCG where:
    - DCG = sum of (relevance_i / log2(i + 1)) for i in 1..K
    - IDCG = ideal DCG (sorted by relevance)
    """
    if k is None:
        k = len(retrieved)

    if k <= 0 or not ground_truth.locations:
        return 0.0

    top_k = retrieved[:k]

    # Compute DCG
    dcg = 0.0
    for i, result in enumerate(top_k):
        relevance = _get_relevance_score(result.location, ground_truth)
        if relevance > 0:
            dcg += relevance / math.log2(i + 2)  # +2 because i is 0-indexed

    # Compute IDCG (ideal ranking)
    # Get all relevance scores and sort descending
    relevances = ground_truth.relevance_scores or [1.0] * len(ground_truth.locations)
    sorted_relevances = sorted(relevances, reverse=True)[:k]

    idcg = 0.0
    for i, rel in enumerate(sorted_relevances):
        idcg += rel / math.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_map(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
) -> float:
    """
    Compute Mean Average Precision (MAP).

    For a single query, this is Average Precision.
    MAP = (1/R) * sum of (P@k * rel(k)) for all k
    where R is total number of relevant items.
    """
    if not ground_truth.locations:
        return 1.0

    num_relevant = len(ground_truth.locations)
    sum_precision = 0.0
    relevant_found = 0

    for i, result in enumerate(retrieved):
        if _is_relevant(result.location, ground_truth):
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            sum_precision += precision_at_i

    if relevant_found == 0:
        return 0.0

    return sum_precision / num_relevant


def compute_hit_rate(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """
    Compute Hit Rate (Success@K).

    Returns 1.0 if at least one relevant item is in top K, else 0.0.
    """
    top_k = retrieved[:k]

    for result in top_k:
        if _is_relevant(result.location, ground_truth):
            return 1.0

    return 0.0


def compute_f1_at_k(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """
    Compute F1 Score at K.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    precision = compute_precision_at_k(retrieved, ground_truth, k)
    recall = compute_recall_at_k(retrieved, ground_truth, k)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


# SDLC-Specific Metrics

def compute_file_level_recall(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: Optional[int] = None,
) -> float:
    """
    Compute file-level recall regardless of original granularity.
    """
    if not ground_truth.locations:
        return 1.0

    results = retrieved[:k] if k else retrieved

    # Get unique ground truth files
    gt_files = set(loc.file_path for loc in ground_truth.locations)

    # Get unique retrieved files
    retrieved_files = set(r.location.file_path for r in results)

    # Compute intersection
    found_files = gt_files & retrieved_files

    return len(found_files) / len(gt_files)


def compute_function_level_precision(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: int,
) -> float:
    """
    Compute function-level precision.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0

    # Filter to results with function information
    results_with_func = [r for r in top_k if r.location.function_name]
    if not results_with_func:
        return 0.0

    gt_functions = set(
        (loc.file_path, loc.function_name)
        for loc in ground_truth.locations
        if loc.function_name
    )

    relevant_count = 0
    for result in results_with_func:
        key = (result.location.file_path, result.location.function_name)
        if key in gt_functions:
            relevant_count += 1

    return relevant_count / len(results_with_func)


def compute_cross_module_coverage(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    k: Optional[int] = None,
) -> float:
    """
    Compute ability to find files across different modules/directories.
    """
    if not ground_truth.locations:
        return 1.0

    results = retrieved[:k] if k else retrieved

    # Extract top-level directories (modules) from ground truth
    gt_modules = set()
    for loc in ground_truth.locations:
        parts = loc.file_path.split("/")
        if len(parts) > 1:
            gt_modules.add(parts[0])

    if not gt_modules:
        return 1.0

    # Extract modules from retrieved results
    retrieved_modules = set()
    for r in results:
        parts = r.location.file_path.split("/")
        if len(parts) > 1:
            retrieved_modules.add(parts[0])

    # Compute module coverage
    covered_modules = gt_modules & retrieved_modules

    return len(covered_modules) / len(gt_modules)


def compute_context_efficiency(
    retrieved: list[RetrievalResult],
    ground_truth: GroundTruth,
    max_context_tokens: int = 8000,
    avg_tokens_per_line: int = 10,
) -> float:
    """
    Compute context window efficiency.

    Measures the ratio of relevant tokens to total retrieved tokens,
    important for LLM-based systems with limited context windows.
    """
    total_lines = 0
    relevant_lines = 0

    for result in retrieved:
        if result.snippet:
            snippet_lines = result.snippet.count("\n") + 1
        elif result.location.start_line and result.location.end_line:
            snippet_lines = result.location.end_line - result.location.start_line + 1
        else:
            snippet_lines = 50  # Estimate for full file context

        total_lines += snippet_lines

        if _is_relevant(result.location, ground_truth):
            relevant_lines += snippet_lines

        # Stop if we've exceeded the context window
        if total_lines * avg_tokens_per_line >= max_context_tokens:
            break

    if total_lines == 0:
        return 0.0

    return relevant_lines / total_lines


def _is_relevant(location: CodeLocation, ground_truth: GroundTruth) -> bool:
    """Check if a location is relevant according to ground truth."""
    for gt_loc in ground_truth.locations:
        if location.matches(gt_loc, ground_truth.granularity):
            return True
    return False


def _get_relevance_score(location: CodeLocation, ground_truth: GroundTruth) -> float:
    """Get the relevance score for a location."""
    for i, gt_loc in enumerate(ground_truth.locations):
        if location.matches(gt_loc, ground_truth.granularity):
            if ground_truth.relevance_scores and i < len(ground_truth.relevance_scores):
                return ground_truth.relevance_scores[i]
            return 1.0
    return 0.0


@dataclass
class IRMetrics:
    """
    Container for computing and storing IR metrics.

    Provides a convenient interface for computing all metrics at once
    and converting to various output formats including Harbor rewards.
    """

    # Standard IR metrics
    precision_at_1: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    precision_at_20: float = 0.0

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0

    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    ndcg_at_20: float = 0.0
    map_score: float = 0.0

    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0

    f1_at_10: float = 0.0

    # SDLC-specific metrics
    file_level_recall: float = 0.0
    function_level_precision: float = 0.0
    cross_module_coverage: float = 0.0
    context_efficiency: float = 0.0

    # Timing metrics
    retrieval_time_ms: float = 0.0

    @classmethod
    def compute_all(
        cls,
        retrieved: list[RetrievalResult],
        ground_truth: GroundTruth,
        retrieval_time_ms: float = 0.0,
    ) -> "IRMetrics":
        """Compute all metrics from retrieval results and ground truth."""
        return cls(
            precision_at_1=compute_precision_at_k(retrieved, ground_truth, 1),
            precision_at_5=compute_precision_at_k(retrieved, ground_truth, 5),
            precision_at_10=compute_precision_at_k(retrieved, ground_truth, 10),
            precision_at_20=compute_precision_at_k(retrieved, ground_truth, 20),

            recall_at_1=compute_recall_at_k(retrieved, ground_truth, 1),
            recall_at_5=compute_recall_at_k(retrieved, ground_truth, 5),
            recall_at_10=compute_recall_at_k(retrieved, ground_truth, 10),
            recall_at_20=compute_recall_at_k(retrieved, ground_truth, 20),

            mrr=compute_mrr(retrieved, ground_truth),
            ndcg_at_10=compute_ndcg(retrieved, ground_truth, 10),
            ndcg_at_20=compute_ndcg(retrieved, ground_truth, 20),
            map_score=compute_map(retrieved, ground_truth),

            hit_at_1=compute_hit_rate(retrieved, ground_truth, 1),
            hit_at_5=compute_hit_rate(retrieved, ground_truth, 5),
            hit_at_10=compute_hit_rate(retrieved, ground_truth, 10),

            f1_at_10=compute_f1_at_k(retrieved, ground_truth, 10),

            file_level_recall=compute_file_level_recall(retrieved, ground_truth),
            function_level_precision=compute_function_level_precision(retrieved, ground_truth, 10),
            cross_module_coverage=compute_cross_module_coverage(retrieved, ground_truth),
            context_efficiency=compute_context_efficiency(retrieved, ground_truth),

            retrieval_time_ms=retrieval_time_ms,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "precision@1": self.precision_at_1,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "precision@20": self.precision_at_20,
            "recall@1": self.recall_at_1,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "mrr": self.mrr,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
            "map": self.map_score,
            "hit@1": self.hit_at_1,
            "hit@5": self.hit_at_5,
            "hit@10": self.hit_at_10,
            "f1@10": self.f1_at_10,
            "file_level_recall": self.file_level_recall,
            "function_level_precision": self.function_level_precision,
            "cross_module_coverage": self.cross_module_coverage,
            "context_efficiency": self.context_efficiency,
            "retrieval_time_ms": self.retrieval_time_ms,
        }

    def to_harbor_reward(self, primary_metric: str = "mrr") -> dict:
        """
        Convert to Harbor reward format.

        Returns a dictionary suitable for /logs/verifier/reward.json
        """
        all_metrics = self.to_dict()

        # Harbor expects numeric values
        reward = {}
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                # Convert metric names to valid identifiers
                safe_key = key.replace("@", "_at_")
                reward[safe_key] = float(value)

        return reward

    def get_primary_score(self, metric: str = "mrr") -> float:
        """Get a single primary score for ranking."""
        metrics_map = self.to_dict()
        return metrics_map.get(metric, metrics_map.get("mrr", 0.0))


def aggregate_metrics(metrics_list: list[IRMetrics]) -> dict:
    """
    Aggregate metrics across multiple tasks.

    Returns mean, std, min, max for each metric.
    """
    if not metrics_list:
        return {}

    import statistics

    # Collect all metric values
    all_values = {}
    for metrics in metrics_list:
        for key, value in metrics.to_dict().items():
            if key not in all_values:
                all_values[key] = []
            all_values[key].append(value)

    # Compute aggregates
    aggregates = {}
    for key, values in all_values.items():
        aggregates[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    return aggregates
