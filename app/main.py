#!/usr/bin/env python3
"""
IR-SDLC-Factory: Main entry point for IR evaluation on SDLC tasks.

This tool provides commands to:
- Generate IR tasks from repository data
- Evaluate IR tools on benchmark tasks
- Generate Harbor-compatible task directories
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command-line arguments."""
    parser = ArgumentParser(
        prog="ir-sdlc-factory",
        description="IR-SDLC-Factory: Evaluate IR tools on enterprise SDLC tasks",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ir-evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        aliases=["eval"],
        help="Evaluate an IR tool on IR-SDLC tasks",
    )
    set_ir_eval_parser_args(eval_parser)

    # ir-generate-tasks command
    generate_parser = subparsers.add_parser(
        "generate-tasks",
        aliases=["gen"],
        help="Generate IR tasks from repository data",
    )
    set_ir_generate_parser_args(generate_parser)

    # ir-generate-harbor command
    harbor_parser = subparsers.add_parser(
        "generate-harbor",
        aliases=["harbor"],
        help="Generate Harbor-compatible task directories",
    )
    set_ir_harbor_parser_args(harbor_parser)

    # collect-repos command
    collect_parser = subparsers.add_parser(
        "collect-repos",
        aliases=["collect"],
        help="Collect large enterprise-scale repositories from GitHub",
    )
    set_collect_repos_parser_args(collect_parser)

    return parser.parse_args()


def set_ir_eval_parser_args(parser: ArgumentParser) -> None:
    """Set up argument parser for IR evaluation command."""
    parser.add_argument(
        "--tasks-file",
        type=str,
        required=True,
        help="Path to JSONL file containing IR tasks",
    )
    parser.add_argument(
        "--ir-tool",
        type=str,
        default="grep-baseline",
        help="Name of the IR tool to evaluate (default: grep-baseline)",
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        required=True,
        help="Directory containing cloned repositories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ir_eval",
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="mrr",
        help="Primary metric for evaluation (default: mrr)",
    )
    parser.add_argument(
        "--generate-harbor",
        action="store_true",
        default=True,
        help="Generate Harbor-compatible output files",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )


def set_ir_generate_parser_args(parser: ArgumentParser) -> None:
    """Set up argument parser for IR task generation command."""
    parser.add_argument(
        "--source-file",
        type=str,
        required=True,
        help="Path to source data file (issues, PRs, etc.)",
    )
    parser.add_argument(
        "--task-types",
        type=str,
        default="bug_triage,code_review",
        help="Comma-separated list of task types to generate",
    )
    parser.add_argument(
        "--repos-dir",
        type=str,
        required=True,
        help="Directory containing cloned repositories",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSONL file for generated tasks",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ir-sdlc-bench",
        help="Name for the generated dataset",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default="1.0",
        help="Version for the generated dataset",
    )


def set_ir_harbor_parser_args(parser: ArgumentParser) -> None:
    """Set up argument parser for Harbor task generation command."""
    parser.add_argument(
        "--tasks-file",
        type=str,
        required=True,
        help="Path to JSONL file containing IR tasks",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for Harbor task directories",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ir-sdlc-bench",
        help="Name for the dataset",
    )
    parser.add_argument(
        "--author-name",
        type=str,
        default="IR-SDLC-Factory",
        help="Author name for task metadata",
    )
    parser.add_argument(
        "--author-email",
        type=str,
        default="ir-sdlc@example.com",
        help="Author email for task metadata",
    )
    parser.add_argument(
        "--generate-registry",
        action="store_true",
        default=False,
        help="Generate Harbor registry.json entry",
    )
    parser.add_argument(
        "--git-url",
        type=str,
        default="https://github.com/sjarmak/IR-SDLC-Factory.git",
        help="Git URL for registry entry",
    )


def set_collect_repos_parser_args(parser: ArgumentParser) -> None:
    """Set up argument parser for repository collection command."""
    parser.add_argument(
        "--languages",
        type=str,
        default="python,javascript,typescript,java,go,rust,cpp",
        help="Comma-separated list of programming languages",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=1000,
        help="Minimum number of stars",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=1000,
        help="Minimum number of files for enterprise-scale",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="large_repos.jsonl",
        help="Output file for collected repository metadata",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=100,
        help="Maximum number of repositories to collect per language",
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub API token (or set GITHUB_TOKEN env var)",
    )


def run_ir_evaluation_command(args) -> None:
    """Run IR evaluation command."""
    from app.ir_sdlc.evaluation_runner import run_ir_evaluation

    logger.info(f"Running IR evaluation with tool: {args.ir_tool}")

    results = run_ir_evaluation(
        tasks_file=args.tasks_file,
        ir_tool_name=args.ir_tool,
        repos_dir=args.repos_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        generate_harbor=args.generate_harbor,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("IR-SDLC-Factory Evaluation Results")
    print("=" * 60)
    print(f"Tool: {results.get('tool_name', 'Unknown')}")
    print(f"Tasks evaluated: {results.get('num_tasks', 0)}")

    if results.get("metrics"):
        print("\nAggregate Metrics:")
        for metric_name, values in results["metrics"].items():
            if isinstance(values, dict) and "mean" in values:
                print(f"  {metric_name}: {values['mean']:.4f} (±{values.get('std', 0):.4f})")

    print(f"\nResults saved to: {args.output_dir}")


def run_ir_generate_tasks_command(args) -> None:
    """Run IR task generation command."""
    from app.ir_sdlc.task_types import get_task_generator, SDLCTaskType
    from app.ir_sdlc.data_structures import IRDataset

    logger.info(f"Generating IR tasks from: {args.source_file}")

    # Parse task types
    task_types = [t.strip() for t in args.task_types.split(",")]

    # Load source data
    source_data = []
    with open(args.source_file, "r") as f:
        for line in f:
            if line.strip():
                source_data.append(json.loads(line))

    # Generate tasks
    tasks = []
    for item in source_data:
        for task_type_str in task_types:
            try:
                task_type = SDLCTaskType(task_type_str)
                generator = get_task_generator(task_type)

                repo_name = item.get("repo", "")
                repo_url = f"https://github.com/{repo_name}.git"
                commit_hash = item.get("base_commit", item.get("commit", "HEAD"))

                task = generator.create_task(
                    source_data=item,
                    repo_name=repo_name,
                    repo_url=repo_url,
                    commit_hash=commit_hash,
                )
                tasks.append(task)

            except Exception as e:
                logger.warning(f"Error generating task: {e}")
                continue

    # Create dataset
    dataset = IRDataset(
        name=args.dataset_name,
        version=args.dataset_version,
        description=f"IR-SDLC tasks generated from {args.source_file}",
        tasks=tasks,
    )

    # Save tasks
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_jsonl(output_path)

    print(f"\nGenerated {len(tasks)} IR tasks")
    print(f"Saved to: {args.output_file}")


def run_ir_generate_harbor_command(args) -> None:
    """Run Harbor task generation command."""
    from app.ir_sdlc.harbor_adapter import (
        HarborTaskGenerator,
        HarborConfig,
        generate_registry_entry,
    )
    from app.ir_sdlc.data_structures import IRDataset

    logger.info(f"Generating Harbor tasks from: {args.tasks_file}")

    # Load dataset
    dataset = IRDataset.load_jsonl(Path(args.tasks_file), name=args.dataset_name)

    # Create Harbor config
    harbor_config = HarborConfig(
        author_name=args.author_name,
        author_email=args.author_email,
    )

    # Generate Harbor tasks
    generator = HarborTaskGenerator(harbor_config)
    generated_paths = generator.generate_dataset(dataset, Path(args.output_dir))

    print(f"\nGenerated {len(generated_paths)} Harbor task directories")
    print(f"Output: {args.output_dir}")

    # Generate registry entry if requested
    if args.generate_registry:
        registry_entry = generate_registry_entry(
            dataset=dataset,
            git_url=args.git_url,
            git_commit_id="head",
        )

        registry_path = Path(args.output_dir) / "registry_entry.json"
        with open(registry_path, "w") as f:
            json.dump(registry_entry, f, indent=2)

        print(f"Registry entry saved to: {registry_path}")

    print("\nTo run with Harbor:")
    print(f"  harbor jobs start -p {args.output_dir}/{args.dataset_name} -a <agent> -m <model>")


def run_collect_repos_command(args) -> None:
    """Run repository collection command."""
    import os
    from data_collection.collect.collect_large_repos import LargeRepoCollector

    logger.info("Collecting large enterprise-scale repositories...")

    # Get GitHub token
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        logger.warning("No GitHub token provided. API rate limits will apply.")

    languages = [lang.strip() for lang in args.languages.split(",")]

    collector = LargeRepoCollector(github_token=github_token)

    repos = collector.collect_repos(
        languages=languages,
        min_stars=args.min_stars,
        min_files=args.min_files,
        max_repos_per_language=args.max_repos,
    )

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for repo in repos:
            f.write(json.dumps(repo.to_dict()) + "\n")

    print(f"\nCollected {len(repos)} repositories")
    print(f"Saved to: {args.output_file}")


def main():
    """Main entry point."""
    args = get_args()

    if args.command is None:
        print("IR-SDLC-Factory: Evaluate IR tools on enterprise SDLC tasks")
        print("\nUsage: python -m app.main <command> [options]")
        print("\nAvailable commands:")
        print("  evaluate (eval)       - Evaluate an IR tool on benchmark tasks")
        print("  generate-tasks (gen)  - Generate IR tasks from repository data")
        print("  generate-harbor       - Generate Harbor-compatible task directories")
        print("  collect-repos         - Collect large repositories from GitHub")
        print("\nUse --help with any command for more details.")
        return

    command = args.command

    if command in ("evaluate", "eval"):
        run_ir_evaluation_command(args)
    elif command in ("generate-tasks", "gen"):
        run_ir_generate_tasks_command(args)
    elif command in ("generate-harbor", "harbor"):
        run_ir_generate_harbor_command(args)
    elif command in ("collect-repos", "collect"):
        run_collect_repos_command(args)
    else:
        print(f"Unknown command: {command}")
        print("Use --help for available commands.")


if __name__ == "__main__":
    main()
