"""
CodeContextBench Exporter.

This module exports IR-SDLC benchmark results to CodeContextBench-compatible
formats for dashboard integration and analysis.

Outputs:
1. .dashboard_runs/{run_id}.json - Run status for dashboard
2. jobs/{run_id}/result.json - Harbor-compatible results
3. artifacts/ir_sdlc_comparison.json - A/B comparison metrics
4. artifacts/ir_sdlc_llm_judge.json - LLM judge results
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

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


def generate_run_id(benchmark_name: str = "IR-SDLC") -> str:
    """Generate a unique run ID compatible with CodeContextBench format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{benchmark_name}_{timestamp}"
