#!/usr/bin/env python3
"""
Collect large, complex repositories for IR-SDLC-Bench.

This script identifies and collects metadata about enterprise-scale
open-source repositories suitable for information retrieval benchmarking.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RepoStats:
    """Statistics for a repository."""
    full_name: str
    url: str
    clone_url: str
    default_branch: str
    stars: int
    forks: int
    watchers: int
    open_issues: int
    language: str
    languages: dict  # language -> bytes
    size_kb: int
    created_at: str
    updated_at: str
    description: str
    topics: list
    license: Optional[str]

    # Computed complexity metrics
    file_count: Optional[int] = None
    directory_count: Optional[int] = None
    contributor_count: Optional[int] = None
    commit_count: Optional[int] = None
    lines_of_code: Optional[int] = None

    # Complexity score (0-100)
    complexity_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class LargeRepoCollector:
    """
    Collects and analyzes large GitHub repositories.
    """

    def __init__(self, github_token: str):
        self.token = github_token
        self.headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def search_repos(
        self,
        languages: list[str],
        min_stars: int = 5000,
        min_forks: int = 500,
        max_results: int = 100,
    ) -> list[dict]:
        """
        Search for repositories matching criteria.
        """
        all_repos = []

        for language in languages:
            logger.info(f"Searching for {language} repositories...")

            query = f"language:{language} stars:>={min_stars} forks:>={min_forks}"
            repos = self._search_github(query, max_results // len(languages))
            all_repos.extend(repos)

            # Rate limiting
            time.sleep(1)

        # Deduplicate by full_name
        seen = set()
        unique_repos = []
        for repo in all_repos:
            if repo["full_name"] not in seen:
                seen.add(repo["full_name"])
                unique_repos.append(repo)

        logger.info(f"Found {len(unique_repos)} unique repositories")
        return unique_repos

    def _search_github(self, query: str, max_results: int) -> list[dict]:
        """Execute GitHub search API query."""
        repos = []
        page = 1
        per_page = min(100, max_results)

        while len(repos) < max_results:
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                if not items:
                    break

                repos.extend(items)
                page += 1

                # Rate limiting
                if "X-RateLimit-Remaining" in response.headers:
                    remaining = int(response.headers["X-RateLimit-Remaining"])
                    if remaining < 10:
                        logger.warning("Rate limit low, waiting...")
                        time.sleep(60)

                time.sleep(0.5)

            except requests.exceptions.RequestException as e:
                logger.error(f"Search error: {e}")
                break

        return repos[:max_results]

    def get_repo_stats(self, full_name: str) -> Optional[RepoStats]:
        """
        Get detailed statistics for a repository.
        """
        try:
            # Get basic repo info
            url = f"https://api.github.com/repos/{full_name}"
            response = self.session.get(url)
            response.raise_for_status()
            repo_data = response.json()

            # Get languages
            languages_url = repo_data.get("languages_url")
            languages = {}
            if languages_url:
                lang_response = self.session.get(languages_url)
                if lang_response.status_code == 200:
                    languages = lang_response.json()

            time.sleep(0.5)

            # Get contributor count (approximate)
            contributors_url = f"https://api.github.com/repos/{full_name}/contributors"
            params = {"per_page": 1, "anon": "true"}
            contrib_response = self.session.get(contributors_url, params=params)
            contributor_count = None
            if contrib_response.status_code == 200:
                # Parse Link header for total count
                link_header = contrib_response.headers.get("Link", "")
                if "last" in link_header:
                    import re
                    match = re.search(r'page=(\d+)>; rel="last"', link_header)
                    if match:
                        contributor_count = int(match.group(1))

            time.sleep(0.5)

            stats = RepoStats(
                full_name=repo_data["full_name"],
                url=repo_data["html_url"],
                clone_url=repo_data["clone_url"],
                default_branch=repo_data.get("default_branch", "main"),
                stars=repo_data["stargazers_count"],
                forks=repo_data["forks_count"],
                watchers=repo_data["watchers_count"],
                open_issues=repo_data["open_issues_count"],
                language=repo_data.get("language", ""),
                languages=languages,
                size_kb=repo_data["size"],
                created_at=repo_data["created_at"],
                updated_at=repo_data["updated_at"],
                description=repo_data.get("description", "") or "",
                topics=repo_data.get("topics", []),
                license=repo_data.get("license", {}).get("spdx_id") if repo_data.get("license") else None,
                contributor_count=contributor_count,
            )

            return stats

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting stats for {full_name}: {e}")
            return None

    def analyze_local_repo(self, repo_path: Path) -> dict:
        """
        Analyze a locally cloned repository for complexity metrics.
        """
        stats = {
            "file_count": 0,
            "directory_count": 0,
            "lines_of_code": 0,
        }

        try:
            # Count files and directories
            for root, dirs, files in os.walk(repo_path):
                # Skip hidden and common non-code directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                          ['node_modules', 'vendor', 'venv', '__pycache__', 'dist', 'build']]

                stats["directory_count"] += len(dirs)

                for file in files:
                    if not file.startswith('.'):
                        stats["file_count"] += 1

                        # Count lines for code files
                        file_path = Path(root) / file
                        if self._is_code_file(file):
                            try:
                                with open(file_path, 'r', errors='ignore') as f:
                                    stats["lines_of_code"] += sum(1 for _ in f)
                            except:
                                pass

            # Get git stats if available
            if (repo_path / ".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "rev-list", "--count", "HEAD"],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        stats["commit_count"] = int(result.stdout.strip())
                except:
                    pass

        except Exception as e:
            logger.error(f"Error analyzing repo: {e}")

        return stats

    def _is_code_file(self, filename: str) -> bool:
        """Check if a file is a code file based on extension."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
            '.c', '.cpp', '.h', '.hpp', '.cs', '.rb', '.php', '.swift',
            '.kt', '.scala', '.clj', '.ex', '.exs', '.ml', '.hs',
        }
        return Path(filename).suffix.lower() in code_extensions

    def compute_complexity_score(self, stats: RepoStats) -> float:
        """
        Compute a complexity score (0-100) based on various metrics.

        Higher score = more complex/larger repository.
        """
        score = 0.0

        # Size component (max 25 points)
        size_mb = stats.size_kb / 1024
        if size_mb >= 500:
            score += 25
        elif size_mb >= 100:
            score += 20
        elif size_mb >= 50:
            score += 15
        elif size_mb >= 10:
            score += 10
        else:
            score += 5

        # Stars/popularity (max 20 points)
        if stats.stars >= 50000:
            score += 20
        elif stats.stars >= 20000:
            score += 15
        elif stats.stars >= 10000:
            score += 10
        elif stats.stars >= 5000:
            score += 5

        # Contributors (max 20 points)
        if stats.contributor_count:
            if stats.contributor_count >= 1000:
                score += 20
            elif stats.contributor_count >= 500:
                score += 15
            elif stats.contributor_count >= 100:
                score += 10
            elif stats.contributor_count >= 50:
                score += 5

        # File count (max 20 points)
        if stats.file_count:
            if stats.file_count >= 10000:
                score += 20
            elif stats.file_count >= 5000:
                score += 15
            elif stats.file_count >= 1000:
                score += 10
            elif stats.file_count >= 500:
                score += 5

        # Language diversity (max 15 points)
        num_languages = len(stats.languages)
        if num_languages >= 10:
            score += 15
        elif num_languages >= 5:
            score += 10
        elif num_languages >= 3:
            score += 5

        return min(100.0, score)

    def filter_enterprise_repos(
        self,
        repos: list[RepoStats],
        min_complexity: float = 50.0,
        min_files: int = 500,
        min_contributors: int = 50,
    ) -> list[RepoStats]:
        """
        Filter repositories to keep only enterprise-scale ones.
        """
        filtered = []

        for repo in repos:
            # Compute complexity if not already done
            if repo.complexity_score is None:
                repo.complexity_score = self.compute_complexity_score(repo)

            # Apply filters
            if repo.complexity_score < min_complexity:
                continue

            if repo.file_count and repo.file_count < min_files:
                continue

            if repo.contributor_count and repo.contributor_count < min_contributors:
                continue

            filtered.append(repo)

        return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Collect large repositories for IR-SDLC-Bench"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="python,javascript,typescript,java,go,rust,cpp",
        help="Comma-separated list of languages",
    )
    parser.add_argument(
        "--min-stars",
        type=int,
        default=5000,
        help="Minimum star count",
    )
    parser.add_argument(
        "--min-forks",
        type=int,
        default=500,
        help="Minimum fork count",
    )
    parser.add_argument(
        "--min-files",
        type=int,
        default=500,
        help="Minimum file count for enterprise filter",
    )
    parser.add_argument(
        "--min-contributors",
        type=int,
        default=50,
        help="Minimum contributor count",
    )
    parser.add_argument(
        "--min-complexity",
        type=float,
        default=50.0,
        help="Minimum complexity score (0-100)",
    )
    parser.add_argument(
        "--max-repos",
        type=int,
        default=200,
        help="Maximum repositories to collect",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Get detailed stats for each repo (slower)",
    )

    args = parser.parse_args()

    # Get GitHub token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.error("GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    collector = LargeRepoCollector(token)

    # Parse languages
    languages = [lang.strip() for lang in args.languages.split(",")]

    # Search for repositories
    logger.info("Searching for repositories...")
    repos = collector.search_repos(
        languages=languages,
        min_stars=args.min_stars,
        min_forks=args.min_forks,
        max_results=args.max_repos * 2,  # Get more to filter
    )

    # Get detailed stats
    repo_stats = []
    for i, repo in enumerate(repos):
        full_name = repo["full_name"]
        logger.info(f"[{i+1}/{len(repos)}] Getting stats for {full_name}...")

        stats = collector.get_repo_stats(full_name)
        if stats:
            stats.complexity_score = collector.compute_complexity_score(stats)
            repo_stats.append(stats)

        # Rate limiting
        time.sleep(0.5)

    # Filter for enterprise-scale repos
    logger.info("Filtering for enterprise-scale repositories...")
    filtered_repos = collector.filter_enterprise_repos(
        repo_stats,
        min_complexity=args.min_complexity,
        min_files=args.min_files if args.detailed else 0,
        min_contributors=args.min_contributors,
    )

    # Sort by complexity score
    filtered_repos.sort(key=lambda r: r.complexity_score or 0, reverse=True)

    # Limit to max_repos
    filtered_repos = filtered_repos[:args.max_repos]

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for repo in filtered_repos:
            f.write(json.dumps(repo.to_dict()) + "\n")

    logger.info(f"Saved {len(filtered_repos)} repositories to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("IR-SDLC-Bench Repository Collection Summary")
    print("=" * 60)
    print(f"Total repositories collected: {len(filtered_repos)}")
    print(f"Languages: {', '.join(languages)}")
    print(f"Minimum stars: {args.min_stars}")
    print(f"Minimum complexity score: {args.min_complexity}")
    print("\nTop 10 repositories by complexity:")
    for i, repo in enumerate(filtered_repos[:10]):
        print(f"  {i+1}. {repo.full_name} (score: {repo.complexity_score:.1f}, stars: {repo.stars})")


if __name__ == "__main__":
    main()
