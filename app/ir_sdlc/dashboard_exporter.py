"""
CodeContextBench Exporter.

This module exports IR-SDLC benchmark results to CodeContextBench-compatible
formats for dashboard integration and analysis.

Outputs:
1. .dashboard_runs/{run_id}.json - Run status for dashboard
2. jobs/{run_id}/result.json - Harbor-compatible results
3. artifacts/ir_sdlc_comparison.json - A/B comparison metrics
4. artifacts/ir_sdlc_llm_judge.json - LLM judge results
5. benchmarks/{task_id}/ - Harbor-compatible task directories:
   - task.toml - Task metadata and configuration
   - Dockerfile - Container setup with repo clone
   - test.sh - IR impact metrics capture script
"""

from __future__ import annotations

import json
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import hashlib

from app.ir_sdlc.dashboard_schema import (
    IRSDLCBenchmarkRun,
    IRSDLCTaskResult,
    IRComparison,
    IRToolType,
    SDLCTaskType,
)


class CodeContextBenchExporter:
    """Exports IR-SDLC results to CodeContextBench format."""
    
    def __init__(self, output_dir: str):
        """Initialize exporter.
        
        Args:
            output_dir: Base directory for output (typically CodeContextBench root)
        """
        self.output_dir = Path(output_dir)
        self.dashboard_runs_dir = self.output_dir / ".dashboard_runs"
        self.jobs_dir = self.output_dir / "jobs"
        self.artifacts_dir = self.output_dir / "artifacts"
        
        # Create directories
        self.dashboard_runs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    def export_run(self, run: IRSDLCBenchmarkRun) -> Dict[str, Path]:
        """Export a complete benchmark run.
        
        Args:
            run: The benchmark run to export
            
        Returns:
            Dict mapping output type to file path
        """
        outputs = {}
        
        # Update aggregates
        run.update_aggregates()
        
        # 1. Export dashboard run status
        dashboard_path = self._export_dashboard_run(run)
        outputs["dashboard_run"] = dashboard_path
        
        # 2. Export Harbor-compatible job results
        job_path = self._export_job_results(run)
        outputs["job_results"] = job_path
        
        # 3. Export detailed task results
        tasks_path = self._export_task_details(run)
        outputs["task_details"] = tasks_path
        
        return outputs
    
    def _export_dashboard_run(self, run: IRSDLCBenchmarkRun) -> Path:
        """Export to .dashboard_runs/ format."""
        output_path = self.dashboard_runs_dir / f"{run.run_id}.json"
        
        data = run.to_dashboard_format()
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def _export_job_results(self, run: IRSDLCBenchmarkRun) -> Path:
        """Export to jobs/ Harbor format."""
        job_dir = self.jobs_dir / run.run_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Main result.json
        result_path = job_dir / "result.json"
        data = run.to_harbor_result_format()
        
        with open(result_path, "w") as f:
            json.dump(data, f, indent=4)
        
        # Config.json
        config_path = job_dir / "config.json"
        config = {
            "job_name": run.run_id,
            "jobs_dir": str(job_dir),
            "benchmark_name": run.benchmark_name,
            "ir_tool_type": run.ir_tool_type.value,
            "model_name": run.model_name,
            "agents": [{
                "name": run.agent_name,
                "import_path": run.task_results[0].agent_import_path if run.task_results else None,
                "model_name": run.model_name,
            }],
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        
        return result_path
    
    def _export_task_details(self, run: IRSDLCBenchmarkRun) -> Path:
        """Export detailed task-level results."""
        job_dir = self.jobs_dir / run.run_id
        tasks_path = job_dir / "ir_sdlc_tasks.json"
        
        data = {
            "run_id": run.run_id,
            "benchmark_name": run.benchmark_name,
            "ir_tool_type": run.ir_tool_type.value,
            "tasks": [r.to_dict() for r in run.task_results],
        }
        
        with open(tasks_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return tasks_path
    
    def export_comparison(
        self,
        comparisons: List[IRComparison],
        filename: str = "ir_sdlc_comparison.json",
    ) -> Path:
        """Export A/B comparison results.
        
        Compatible with CodeContextBench's llm_judge_results.json format.
        
        Args:
            comparisons: List of comparison results
            filename: Output filename
            
        Returns:
            Path to output file
        """
        output_path = self.artifacts_dir / filename
        
        data = [c.to_dict() for c in comparisons]
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return output_path
    
    def export_metrics_summary(
        self,
        runs: List[IRSDLCBenchmarkRun],
        filename: str = "ir_sdlc_metrics.json",
    ) -> Path:
        """Export aggregate metrics summary.
        
        Compatible with CodeContextBench's enterprise_metrics_comparison.json format.
        
        Args:
            runs: List of benchmark runs to summarize
            filename: Output filename
            
        Returns:
            Path to output file
        """
        output_path = self.artifacts_dir / filename
        
        # Group by IR tool type
        by_tool: Dict[str, Dict] = {}
        
        for run in runs:
            tool_key = run.ir_tool_type.value
            if tool_key not in by_tool:
                by_tool[tool_key] = {
                    "summary": {
                        "total_tasks": 0,
                        "avg_steps": 0.0,
                        "avg_file_reads": 0.0,
                        "avg_grep_searches": 0.0,
                        "avg_glob_searches": 0.0,
                        "avg_bash_commands": 0.0,
                        "avg_mcp_deep_search": 0.0,
                        "avg_mcp_keyword_search": 0.0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "avg_files_mentioned": 0.0,
                        "success_rate": 0.0,
                        "mean_reward": 0.0,
                    },
                    "tasks": [],
                }
            
            # Aggregate metrics
            for result in run.task_results:
                by_tool[tool_key]["tasks"].append({
                    "task_id": result.task_id,
                    "sdlc_type": result.sdlc_type.value,
                    "success": result.execution_metrics.success,
                    "reward": result.execution_metrics.reward,
                    "total_steps": result.execution_metrics.total_steps,
                    "file_reads": result.execution_metrics.file_reads,
                    "grep_searches": result.execution_metrics.grep_searches,
                    "glob_searches": result.execution_metrics.glob_searches,
                    "bash_commands": result.execution_metrics.bash_commands,
                    "mcp_deep_search": result.execution_metrics.mcp_deep_search,
                    "mcp_keyword_search": result.execution_metrics.mcp_keyword_search,
                    "total_prompt_tokens": result.execution_metrics.total_prompt_tokens,
                    "total_completion_tokens": result.execution_metrics.total_completion_tokens,
                    "files_mentioned": result.execution_metrics.files_mentioned,
                    "ir_metrics": result.ir_metrics.to_dict(),
                })
                
                by_tool[tool_key]["summary"]["total_tasks"] += 1
                by_tool[tool_key]["summary"]["total_prompt_tokens"] += result.execution_metrics.total_prompt_tokens
                by_tool[tool_key]["summary"]["total_completion_tokens"] += result.execution_metrics.total_completion_tokens
        
        # Calculate averages
        for tool_key, data in by_tool.items():
            n = data["summary"]["total_tasks"]
            if n > 0:
                tasks = data["tasks"]
                data["summary"]["avg_steps"] = sum(t["total_steps"] for t in tasks) / n
                data["summary"]["avg_file_reads"] = sum(t["file_reads"] for t in tasks) / n
                data["summary"]["avg_grep_searches"] = sum(t["grep_searches"] for t in tasks) / n
                data["summary"]["avg_glob_searches"] = sum(t["glob_searches"] for t in tasks) / n
                data["summary"]["avg_bash_commands"] = sum(t["bash_commands"] for t in tasks) / n
                data["summary"]["avg_mcp_deep_search"] = sum(t["mcp_deep_search"] for t in tasks) / n
                data["summary"]["avg_mcp_keyword_search"] = sum(t["mcp_keyword_search"] for t in tasks) / n
                data["summary"]["avg_files_mentioned"] = sum(t["files_mentioned"] for t in tasks) / n
                data["summary"]["success_rate"] = sum(1 for t in tasks if t["success"]) / n
                data["summary"]["mean_reward"] = sum(t["reward"] for t in tasks) / n
        
        with open(output_path, "w") as f:
            json.dump(by_tool, f, indent=2)
        
        return output_path
    
    def export_sdlc_analysis(
        self,
        runs: List[IRSDLCBenchmarkRun],
        filename: str = "ir_sdlc_by_task_type.json",
    ) -> Path:
        """Export analysis grouped by SDLC task type.
        
        This is IR-SDLC-Factory specific - shows how IR tools perform
        on different SDLC activities.
        
        Args:
            runs: List of benchmark runs
            filename: Output filename
            
        Returns:
            Path to output file
        """
        output_path = self.artifacts_dir / filename
        
        # Group by SDLC type and IR tool
        by_sdlc: Dict[str, Dict[str, Dict]] = {}
        
        for run in runs:
            for result in run.task_results:
                sdlc_key = result.sdlc_type.value
                tool_key = run.ir_tool_type.value
                
                if sdlc_key not in by_sdlc:
                    by_sdlc[sdlc_key] = {}
                
                if tool_key not in by_sdlc[sdlc_key]:
                    by_sdlc[sdlc_key][tool_key] = {
                        "count": 0,
                        "success_count": 0,
                        "total_reward": 0.0,
                        "total_tokens": 0,
                        "total_ir_queries": 0,
                        "tasks": [],
                    }
                
                stats = by_sdlc[sdlc_key][tool_key]
                stats["count"] += 1
                stats["success_count"] += 1 if result.execution_metrics.success else 0
                stats["total_reward"] += result.execution_metrics.reward
                stats["total_tokens"] += result.execution_metrics.total_tokens
                stats["total_ir_queries"] += result.ir_metrics.total_queries
                stats["tasks"].append(result.task_id)
        
        # Calculate summaries
        summary = {}
        for sdlc_key, tools in by_sdlc.items():
            summary[sdlc_key] = {}
            for tool_key, stats in tools.items():
                n = stats["count"]
                summary[sdlc_key][tool_key] = {
                    "count": n,
                    "success_rate": stats["success_count"] / n if n > 0 else 0,
                    "avg_reward": stats["total_reward"] / n if n > 0 else 0,
                    "avg_tokens": stats["total_tokens"] / n if n > 0 else 0,
                    "avg_ir_queries": stats["total_ir_queries"] / n if n > 0 else 0,
                }
        
        output = {
            "by_sdlc_type": by_sdlc,
            "summary": summary,
        }
        
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        return output_path

    # =========================================================================
    # Harbor Task Directory Generation
    # =========================================================================
    
    def export_task_directory(
        self,
        task: Union[IRSDLCTaskResult, Dict[str, Any]],
        benchmarks_dir: Optional[Path] = None,
    ) -> Path:
        """Export a single task to Harbor-compatible task directory.
        
        Creates:
        - {task_id}/task.toml - Task metadata
        - {task_id}/Dockerfile - Container with repo clone
        - {task_id}/test.sh - IR metrics capture script
        
        Args:
            task: Task result or task dict from benchmark JSONL
            benchmarks_dir: Output directory (default: output_dir/benchmarks)
            
        Returns:
            Path to task directory
        """
        benchmarks_dir = benchmarks_dir or self.output_dir / "benchmarks"
        benchmarks_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize to dict
        if hasattr(task, 'to_dict'):
            task_dict = task.to_dict()
        else:
            task_dict = task
        
        task_id = task_dict.get("task_id", "unknown")
        task_dir = benchmarks_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate task.toml
        task_toml_path = self._generate_task_toml(task_dict, task_dir)
        
        # Generate Dockerfile
        dockerfile_path = self._generate_dockerfile(task_dict, task_dir)
        
        # Generate test.sh
        test_sh_path = self._generate_test_script(task_dict, task_dir)
        
        # Write task metadata as JSON for easy parsing
        metadata_path = task_dir / "task_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(task_dict, f, indent=2)
        
        return task_dir
    
    def export_benchmark_tasks(
        self,
        benchmark_jsonl_path: Path,
        benchmarks_dir: Optional[Path] = None,
    ) -> List[Path]:
        """Export all tasks from a benchmark JSONL to Harbor format.
        
        Args:
            benchmark_jsonl_path: Path to benchmark .jsonl file
            benchmarks_dir: Output directory (default: output_dir/benchmarks)
            
        Returns:
            List of paths to created task directories
        """
        benchmarks_dir = benchmarks_dir or self.output_dir / "benchmarks"
        task_dirs = []
        
        with open(benchmark_jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    task_dir = self.export_task_directory(task, benchmarks_dir)
                    task_dirs.append(task_dir)
        
        return task_dirs
    
    def _generate_task_toml(self, task: Dict[str, Any], task_dir: Path) -> Path:
        """Generate task.toml for Harbor compatibility.
        
        The task.toml follows CodeContextBench/Harbor format with IR-SDLC extensions.
        """
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", task.get("sdlc_type", "unknown"))
        repo_name = task.get("repo_name", "")
        repo_url = task.get("repo_url", "")
        commit_hash = task.get("commit_hash", "HEAD")
        difficulty = task.get("difficulty", "medium")
        
        # Extract prompt/scenario
        scenario = task.get("scenario", task.get("task_title", ""))
        vague_prompt = task.get("vague_prompt", "")
        
        # SDLC-specific metadata
        category = task.get("category", "")
        tags = task.get("tags", [])
        
        # Evaluation criteria
        eval_criteria = task.get("evaluation_criteria", {})
        scoring_rubric = eval_criteria.get("scoring_rubric", {})
        
        # Ground truth
        ground_truth = task.get("ground_truth", {})
        ground_truth_files = task.get("ground_truth_files", [])
        if not ground_truth_files and isinstance(ground_truth, dict):
            # Extract files from various ground truth formats
            ground_truth_files = (
                ground_truth.get("relevant_code", []) +
                ground_truth.get("existing_abstractions", []) +
                ground_truth.get("target_files", [])
            )
        
        # Build TOML content
        toml_lines = [
            f'# Harbor-compatible task definition for IR-SDLC-Bench',
            f'# Generated by IR-SDLC-Factory CodeContextBench Exporter',
            f'',
            f'[task]',
            f'id = "{task_id}"',
            f'type = "{task_type}"',
            f'difficulty = "{difficulty}"',
            f'',
            f'[repository]',
            f'name = "{repo_name}"',
            f'url = "{repo_url}"',
            f'commit = "{commit_hash}"',
            f'',
            f'[prompt]',
            f'scenario = """{scenario}"""',
            f'vague = """{vague_prompt}"""',
            f'',
            f'[sdlc]',
            f'category = "{category}"',
            f'tags = {json.dumps(tags)}',
            f'',
            f'[evaluation]',
        ]
        
        # Add scoring rubric
        for key, weight in scoring_rubric.items():
            toml_lines.append(f'{key} = {weight}')
        
        # Add ground truth section
        toml_lines.extend([
            f'',
            f'[ground_truth]',
            f'files = {json.dumps(ground_truth_files[:10])}',  # Limit for readability
        ])
        
        # Add success/failure signals if available
        success_signals = eval_criteria.get("success_signals", [])
        failure_signals = eval_criteria.get("failure_signals", [])
        if success_signals:
            toml_lines.append(f'success_signals = {json.dumps(success_signals[:5])}')
        if failure_signals:
            toml_lines.append(f'failure_signals = {json.dumps(failure_signals[:5])}')
        
        # Write TOML
        toml_path = task_dir / "task.toml"
        with open(toml_path, "w") as f:
            f.write("\n".join(toml_lines))
        
        return toml_path
    
    def _generate_dockerfile(self, task: Dict[str, Any], task_dir: Path) -> Path:
        """Generate Dockerfile that clones enterprise repo at specific commit.
        
        The Dockerfile:
        1. Uses Python base image
        2. Clones the repository at the specified commit
        3. Installs dependencies if requirements.txt exists
        4. Sets up working directory
        """
        repo_url = task.get("repo_url", "")
        repo_name = task.get("repo_name", "").replace("/", "-")
        commit_hash = task.get("commit_hash", "HEAD")
        
        # Determine language/base image from repo or task
        tags = task.get("tags", [])
        base_image = "python:3.11-slim"
        
        # Adjust base image based on repo characteristics
        if "kubernetes" in repo_name.lower() or "go" in tags:
            base_image = "golang:1.21-bullseye"
        elif "vscode" in repo_name.lower() or "typescript" in tags or "javascript" in tags:
            base_image = "node:20-slim"
        elif "elasticsearch" in repo_name.lower() or "java" in tags:
            base_image = "eclipse-temurin:17-jdk-jammy"
        
        dockerfile_content = textwrap.dedent(f'''\
            # Harbor-compatible Dockerfile for IR-SDLC-Bench task
            # Auto-generated by IR-SDLC-Factory
            
            FROM {base_image}
            
            # Install git and common tools
            RUN apt-get update && apt-get install -y \\
                git \\
                curl \\
                jq \\
                && rm -rf /var/lib/apt/lists/*
            
            # Set working directory
            WORKDIR /workspace
            
            # Clone repository at specific commit
            RUN git clone {repo_url} repo && \\
                cd repo && \\
                git checkout {commit_hash}
            
            # Install Python dependencies if present (for metrics scripts)
            RUN if [ -f repo/requirements.txt ]; then \\
                pip install --no-cache-dir -r repo/requirements.txt 2>/dev/null || true; \\
            fi
            
            # Copy test script
            COPY test.sh /workspace/test.sh
            RUN chmod +x /workspace/test.sh
            
            # Set repo as working directory
            WORKDIR /workspace/repo
            
            # Default command runs test script
            CMD ["/workspace/test.sh"]
        ''')
        
        dockerfile_path = task_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_content)
        
        return dockerfile_path
    
    def _generate_test_script(self, task: Dict[str, Any], task_dir: Path) -> Path:
        """Generate test.sh script that captures IR impact metrics.
        
        The test script:
        1. Records baseline metrics (file count, LOC, etc.)
        2. Runs the agent with IR tool
        3. Captures retrieval quality metrics
        4. Outputs JSON metrics for dashboard integration
        """
        task_id = task.get("task_id", "unknown")
        task_type = task.get("task_type", task.get("sdlc_type", "unknown"))
        
        # Get evaluation criteria for scoring
        eval_criteria = task.get("evaluation_criteria", {})
        scoring_rubric = eval_criteria.get("scoring_rubric", {})
        rubric_json = json.dumps(scoring_rubric)
        
        # Ground truth for validation
        ground_truth = task.get("ground_truth", {})
        ground_truth_files = []
        if isinstance(ground_truth, dict):
            for key in ["relevant_code", "existing_abstractions", "target_files"]:
                ground_truth_files.extend(ground_truth.get(key, []))
        ground_truth_json = json.dumps(ground_truth_files[:20])
        
        test_script = textwrap.dedent(f'''\
            #!/bin/bash
            # IR Impact Metrics Capture Script for IR-SDLC-Bench
            # Task: {task_id}
            # Type: {task_type}
            
            set -e
            
            # Output file for metrics
            METRICS_FILE="${{METRICS_FILE:-/workspace/metrics.json}}"
            AGENT_OUTPUT_DIR="${{AGENT_OUTPUT_DIR:-/workspace/agent_output}}"
            
            # Start timer
            START_TIME=$(date +%s.%N)
            
            # ==================================================================
            # Baseline Metrics
            # ==================================================================
            
            echo "Collecting baseline metrics..."
            
            # Count source files
            TOTAL_FILES=$(find . -type f \\( -name "*.py" -o -name "*.go" -o -name "*.ts" -o -name "*.java" \\) | wc -l)
            
            # Lines of code (approximate)
            TOTAL_LOC=$(find . -type f \\( -name "*.py" -o -name "*.go" -o -name "*.ts" -o -name "*.java" \\) -exec cat {{}} \\; 2>/dev/null | wc -l || echo 0)
            
            # ==================================================================
            # Ground Truth Files
            # ==================================================================
            
            GROUND_TRUTH_FILES='{ground_truth_json}'
            
            # Count ground truth files that exist
            GT_EXISTS=0
            for file in $(echo "$GROUND_TRUTH_FILES" | jq -r '.[]' 2>/dev/null); do
                if [ -f "$file" ]; then
                    GT_EXISTS=$((GT_EXISTS + 1))
                fi
            done
            
            # ==================================================================
            # IR Retrieval Metrics Capture
            # ==================================================================
            
            # These will be populated by the agent harness
            IR_QUERIES_MADE=0
            IR_RESULTS_RETURNED=0
            IR_FILES_RETRIEVED=""
            FIRST_HIT_RANK=-1
            
            # Check if agent output exists
            if [ -d "$AGENT_OUTPUT_DIR" ]; then
                # Parse agent trajectory for IR tool usage
                if [ -f "$AGENT_OUTPUT_DIR/trajectory.json" ]; then
                    IR_QUERIES_MADE=$(jq '.tool_calls | map(select(.name | contains("search"))) | length' "$AGENT_OUTPUT_DIR/trajectory.json" 2>/dev/null || echo 0)
                    IR_FILES_RETRIEVED=$(jq -r '.tool_calls | map(select(.name | contains("search"))) | .[].result.files // []' "$AGENT_OUTPUT_DIR/trajectory.json" 2>/dev/null | jq -s 'add | unique' || echo "[]")
                fi
            fi
            
            # ==================================================================
            # Scoring Rubric
            # ==================================================================
            
            SCORING_RUBRIC='{rubric_json}'
            
            # ==================================================================
            # Calculate IR Metrics
            # ==================================================================
            
            # Precision: ground truth files found / files retrieved
            RETRIEVED_COUNT=$(echo "$IR_FILES_RETRIEVED" | jq 'length' 2>/dev/null || echo 0)
            
            # Calculate hits (retrieved files that are in ground truth)
            HITS=0
            for gt_file in $(echo "$GROUND_TRUTH_FILES" | jq -r '.[]' 2>/dev/null); do
                if echo "$IR_FILES_RETRIEVED" | jq -e ".[] | select(. == \\"$gt_file\\")" > /dev/null 2>&1; then
                    HITS=$((HITS + 1))
                fi
            done
            
            # Precision and recall
            if [ "$RETRIEVED_COUNT" -gt 0 ]; then
                PRECISION=$(echo "scale=4; $HITS / $RETRIEVED_COUNT" | bc)
            else
                PRECISION=0
            fi
            
            GT_COUNT=$(echo "$GROUND_TRUTH_FILES" | jq 'length' 2>/dev/null || echo 0)
            if [ "$GT_COUNT" -gt 0 ]; then
                RECALL=$(echo "scale=4; $HITS / $GT_COUNT" | bc)
            else
                RECALL=0
            fi
            
            # End timer
            END_TIME=$(date +%s.%N)
            DURATION=$(echo "$END_TIME - $START_TIME" | bc)
            
            # ==================================================================
            # Output Metrics JSON
            # ==================================================================
            
            cat > "$METRICS_FILE" << EOF
            {{
                "task_id": "{task_id}",
                "task_type": "{task_type}",
                "baseline": {{
                    "total_files": $TOTAL_FILES,
                    "total_loc": $TOTAL_LOC,
                    "ground_truth_files": $GROUND_TRUTH_FILES,
                    "ground_truth_exists": $GT_EXISTS
                }},
                "ir_metrics": {{
                    "queries_made": $IR_QUERIES_MADE,
                    "files_retrieved": $IR_FILES_RETRIEVED,
                    "retrieved_count": $RETRIEVED_COUNT,
                    "hits": $HITS,
                    "precision": $PRECISION,
                    "recall": $RECALL,
                    "first_hit_rank": $FIRST_HIT_RANK
                }},
                "execution": {{
                    "duration_sec": $DURATION,
                    "scoring_rubric": $SCORING_RUBRIC
                }},
                "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            }}
            EOF
            
            echo "Metrics written to $METRICS_FILE"
            cat "$METRICS_FILE"
            
            # Exit with success if metrics were captured
            exit 0
        ''')
        
        test_sh_path = task_dir / "test.sh"
        with open(test_sh_path, "w") as f:
            f.write(test_script)
        
        # Make executable
        os.chmod(test_sh_path, 0o755)
        
        return test_sh_path


def generate_run_id(benchmark_name: str = "IR-SDLC") -> str:
    """Generate a unique run ID compatible with CodeContextBench format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{benchmark_name}_{timestamp}"
