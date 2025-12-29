#!/usr/bin/env python3
"""
Enterprise Repository Selector for IR-SDLC-Bench.

This module selects enterprise-scale repositories suitable for SDLC benchmark
task generation. Inspired by GitTaskBench but focused on enterprise software
development patterns.

Selection criteria:
1. Size: >= 500k LOC (enterprise-scale complexity)
2. Languages: Coverage of enterprise stack (Java, C#, TypeScript, Python, Go, etc.)
3. Frameworks: Enterprise frameworks (Spring, ASP.NET, React, Kubernetes, etc.)
4. Industries: Finance, SaaS/Cloud, Government, Healthcare, Telecom, E-commerce
5. Polyglot: Multiple languages, cross-service references, infrastructure code
6. Complexity: Dead code likelihood, legacy APIs, non-obvious entry points

Uses GitHub API for metadata and tokei for authoritative LOC counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


# =============================================================================
# Enterprise Taxonomy
# =============================================================================

class EnterpriseLanguage(str, Enum):
    """Enterprise programming languages by priority."""
    # Tier 1: Highest enterprise coverage
    JAVA = "Java"
    CSHARP = "C#"
    TYPESCRIPT = "TypeScript"
    JAVASCRIPT = "JavaScript"
    PYTHON = "Python"
    GO = "Go"
    
    # Tier 2: Systems and data
    CPP = "C++"
    C = "C"
    SQL = "SQL"
    KOTLIN = "Kotlin"
    SCALA = "Scala"
    RUST = "Rust"
    
    # Tier 3: Legacy/migration targets
    PERL = "Perl"
    COBOL = "COBOL"
    FORTRAN = "Fortran"
    VB = "Visual Basic"
    DELPHI = "Delphi"
    
    # Infrastructure and scripting
    SHELL = "Shell"
    POWERSHELL = "PowerShell"
    RUBY = "Ruby"
    PHP = "PHP"


# Enterprise language detection patterns (tokei names -> our taxonomy)
LANGUAGE_ALIASES = {
    # Java ecosystem
    "Java": EnterpriseLanguage.JAVA,
    "Kotlin": EnterpriseLanguage.KOTLIN,
    "Scala": EnterpriseLanguage.SCALA,
    
    # .NET ecosystem
    "C#": EnterpriseLanguage.CSHARP,
    "CSharp": EnterpriseLanguage.CSHARP,
    "Visual Basic": EnterpriseLanguage.VB,
    "VB.NET": EnterpriseLanguage.VB,
    
    # JavaScript/TypeScript ecosystem
    "TypeScript": EnterpriseLanguage.TYPESCRIPT,
    "JavaScript": EnterpriseLanguage.JAVASCRIPT,
    "JSX": EnterpriseLanguage.JAVASCRIPT,
    "TSX": EnterpriseLanguage.TYPESCRIPT,
    
    # Python ecosystem
    "Python": EnterpriseLanguage.PYTHON,
    
    # Go ecosystem
    "Go": EnterpriseLanguage.GO,
    
    # Systems languages
    "C++": EnterpriseLanguage.CPP,
    "C": EnterpriseLanguage.C,
    "Rust": EnterpriseLanguage.RUST,
    
    # Legacy languages
    "Perl": EnterpriseLanguage.PERL,
    "COBOL": EnterpriseLanguage.COBOL,
    "Fortran": EnterpriseLanguage.FORTRAN,
    
    # Infrastructure
    "Shell": EnterpriseLanguage.SHELL,
    "Bash": EnterpriseLanguage.SHELL,
    "PowerShell": EnterpriseLanguage.POWERSHELL,
    
    # Web
    "Ruby": EnterpriseLanguage.RUBY,
    "PHP": EnterpriseLanguage.PHP,
}

# Priority weights for language scoring (higher = more valuable for enterprise benchmark)
LANGUAGE_PRIORITY = {
    EnterpriseLanguage.JAVA: 1.0,
    EnterpriseLanguage.CSHARP: 1.0,
    EnterpriseLanguage.TYPESCRIPT: 0.95,
    EnterpriseLanguage.PYTHON: 0.95,
    EnterpriseLanguage.GO: 0.9,
    EnterpriseLanguage.JAVASCRIPT: 0.85,
    EnterpriseLanguage.CPP: 0.8,
    EnterpriseLanguage.KOTLIN: 0.8,
    EnterpriseLanguage.SCALA: 0.75,
    EnterpriseLanguage.RUST: 0.7,
    EnterpriseLanguage.C: 0.65,
    EnterpriseLanguage.PERL: 0.6,  # Legacy value
    EnterpriseLanguage.COBOL: 0.7,  # High legacy value
    EnterpriseLanguage.FORTRAN: 0.5,
    EnterpriseLanguage.SHELL: 0.4,
    EnterpriseLanguage.RUBY: 0.5,
    EnterpriseLanguage.PHP: 0.5,
    EnterpriseLanguage.POWERSHELL: 0.4,
}


class EnterpriseIndustry(str, Enum):
    """Enterprise industry verticals."""
    FINANCE = "finance"
    SAAS_CLOUD = "saas_cloud"
    GOVERNMENT_DEFENSE = "gov_defense"
    HEALTHCARE = "healthcare"
    TELECOM = "telecom"
    ECOMMERCE_RETAIL = "ecommerce_retail"
    MEDIA_ENTERTAINMENT = "media"
    EDUCATION = "education"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    DATA_ML = "data_ml"
    GAMING = "gaming"


# Industry detection keywords
INDUSTRY_KEYWORDS = {
    EnterpriseIndustry.FINANCE: [
        "bank", "banking", "fintech", "trading", "insurance", "payments",
        "lending", "credit", "forex", "stock", "hedge", "investment",
        "portfolio", "compliance", "kyc", "aml", "swift", "ach",
    ],
    EnterpriseIndustry.SAAS_CLOUD: [
        "cloud", "saas", "platform", "developer tools", "observability",
        "devops", "kubernetes", "container", "orchestration", "serverless",
        "iaas", "paas", "multi-tenant", "api gateway",
    ],
    EnterpriseIndustry.GOVERNMENT_DEFENSE: [
        "gov", "government", "defense", "military", "navy", "airforce",
        "army", "nasa", "noaa", "faa", "dhs", "nih", "cdc", "usgs",
        "national lab", "federal", "public sector",
    ],
    EnterpriseIndustry.HEALTHCARE: [
        "health", "medical", "ehr", "clinical", "pharma", "biotech",
        "hospital", "patient", "fhir", "hl7", "dicom", "hipaa",
        "genomics", "drug", "therapy", "diagnostic",
    ],
    EnterpriseIndustry.TELECOM: [
        "telecom", "carrier", "5g", "lte", "network operator", "mobile",
        "wireless", "spectrum", "voip", "sip", "diameter", "ss7",
    ],
    EnterpriseIndustry.ECOMMERCE_RETAIL: [
        "ecommerce", "retail", "shopping", "commerce", "storefront",
        "checkout", "cart", "inventory", "fulfillment", "pos",
        "point of sale", "omnichannel",
    ],
    EnterpriseIndustry.MEDIA_ENTERTAINMENT: [
        "media", "streaming", "video", "music", "publisher", "news",
        "broadcast", "content", "cdn", "encoder", "transcode",
    ],
    EnterpriseIndustry.EDUCATION: [
        "education", "edtech", "learning", "school", "university",
        "lms", "mooc", "course", "student", "academic",
    ],
    EnterpriseIndustry.SECURITY: [
        "security", "auth", "iam", "encryption", "zero trust",
        "vulnerability", "siem", "soc", "threat", "malware",
        "pentest", "cryptography",
    ],
    EnterpriseIndustry.INFRASTRUCTURE: [
        "infrastructure", "terraform", "ansible", "puppet", "chef",
        "ci/cd", "pipeline", "build system", "monitoring", "logging",
        "metrics", "tracing", "service mesh",
    ],
    EnterpriseIndustry.DATA_ML: [
        "data", "analytics", "machine learning", "ml", "ai",
        "deep learning", "etl", "data warehouse", "lakehouse",
        "spark", "flink", "kafka", "airflow", "dbt",
    ],
    EnterpriseIndustry.GAMING: [
        "game", "gaming", "esports", "unity", "unreal", "graphics",
        "rendering", "multiplayer",
    ],
}


# Framework detection patterns
FRAMEWORK_PATTERNS = {
    # Backend frameworks
    "spring": {
        "files": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "content": ["org.springframework", "spring-boot", "@SpringBootApplication"],
        "category": "backend",
        "language": EnterpriseLanguage.JAVA,
    },
    "aspnet": {
        "files": ["*.csproj", "*.sln"],
        "content": ["Microsoft.AspNetCore", "Microsoft.NET", "<TargetFramework>"],
        "category": "backend",
        "language": EnterpriseLanguage.CSHARP,
    },
    "django": {
        "files": ["manage.py", "settings.py", "requirements.txt"],
        "content": ["django", "Django", "DJANGO_SETTINGS_MODULE"],
        "category": "backend",
        "language": EnterpriseLanguage.PYTHON,
    },
    "fastapi": {
        "files": ["requirements.txt", "pyproject.toml"],
        "content": ["fastapi", "FastAPI", "uvicorn"],
        "category": "backend",
        "language": EnterpriseLanguage.PYTHON,
    },
    "express": {
        "files": ["package.json"],
        "content": ["express", "\"express\""],
        "category": "backend",
        "language": EnterpriseLanguage.JAVASCRIPT,
    },
    "nestjs": {
        "files": ["package.json", "nest-cli.json"],
        "content": ["@nestjs", "NestFactory"],
        "category": "backend",
        "language": EnterpriseLanguage.TYPESCRIPT,
    },
    "gin": {
        "files": ["go.mod", "go.sum"],
        "content": ["github.com/gin-gonic/gin"],
        "category": "backend",
        "language": EnterpriseLanguage.GO,
    },
    
    # Frontend frameworks
    "react": {
        "files": ["package.json"],
        "content": ["react", "\"react\"", "react-dom"],
        "category": "frontend",
        "language": EnterpriseLanguage.TYPESCRIPT,
    },
    "angular": {
        "files": ["angular.json", "package.json"],
        "content": ["@angular/core", "@angular/cli"],
        "category": "frontend",
        "language": EnterpriseLanguage.TYPESCRIPT,
    },
    "vue": {
        "files": ["package.json", "vue.config.js"],
        "content": ["vue", "\"vue\"", "@vue/"],
        "category": "frontend",
        "language": EnterpriseLanguage.JAVASCRIPT,
    },
    "svelte": {
        "files": ["svelte.config.js", "package.json"],
        "content": ["svelte", "@sveltejs"],
        "category": "frontend",
        "language": EnterpriseLanguage.JAVASCRIPT,
    },
    
    # Data/messaging frameworks
    "kafka": {
        "files": ["pom.xml", "build.gradle", "package.json", "requirements.txt"],
        "content": ["kafka", "confluent", "apache-kafka"],
        "category": "data",
        "language": None,
    },
    "spark": {
        "files": ["pom.xml", "build.sbt", "requirements.txt"],
        "content": ["org.apache.spark", "pyspark", "spark-sql"],
        "category": "data",
        "language": None,
    },
    "flink": {
        "files": ["pom.xml", "build.gradle"],
        "content": ["org.apache.flink", "flink-streaming"],
        "category": "data",
        "language": None,
    },
    "airflow": {
        "files": ["requirements.txt", "airflow.cfg", "dags/"],
        "content": ["apache-airflow", "airflow.models", "DAG"],
        "category": "data",
        "language": EnterpriseLanguage.PYTHON,
    },
    
    # Infrastructure
    "kubernetes": {
        "files": ["*.yaml", "*.yml", "Chart.yaml", "kustomization.yaml"],
        "content": ["apiVersion:", "kind: Deployment", "kind: Service", "kind: Pod"],
        "category": "infrastructure",
        "language": None,
    },
    "terraform": {
        "files": ["*.tf", "terraform.tfvars"],
        "content": ["resource \"", "provider \"", "terraform {"],
        "category": "infrastructure",
        "language": None,
    },
    "helm": {
        "files": ["Chart.yaml", "values.yaml"],
        "content": ["apiVersion: v2", "{{ .Values", "{{ .Release"],
        "category": "infrastructure",
        "language": None,
    },
    "docker": {
        "files": ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"],
        "content": ["FROM ", "docker-compose", "services:"],
        "category": "infrastructure",
        "language": None,
    },
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class RepoLanguageProfile:
    """Language breakdown for a repository."""
    primary_language: Optional[str] = None
    language_count: int = 0
    enterprise_languages: List[str] = field(default_factory=list)
    legacy_languages: List[str] = field(default_factory=list)
    loc_by_language: Dict[str, int] = field(default_factory=dict)
    language_diversity_score: float = 0.0  # 0-1, higher = more diverse
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RepoComplexityProfile:
    """Complexity indicators for a repository."""
    total_loc: int = 0
    total_files: int = 0
    directory_depth_max: int = 0
    
    # Polyglot indicators
    has_backend: bool = False
    has_frontend: bool = False
    has_data_layer: bool = False
    has_infrastructure: bool = False
    
    # Complexity signals
    has_tests: bool = False
    has_ci_cd: bool = False
    has_docs: bool = False
    
    # Detected frameworks
    frameworks_detected: List[str] = field(default_factory=list)
    
    # Score
    complexity_score: float = 0.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EnterpriseRepoCandidate:
    """A repository candidate for enterprise benchmark."""
    # Identity
    full_name: str
    html_url: str
    description: str = ""
    topics: List[str] = field(default_factory=list)
    
    # Basic metadata
    default_branch: str = "main"
    archived: bool = False
    fork: bool = False
    stars: int = 0
    forks: int = 0
    watchers: int = 0
    
    # Activity
    pushed_at: Optional[str] = None
    created_at: Optional[str] = None
    
    # Issues
    open_issues_count: int = 0
    recent_issues_count: int = 0
    
    # Profiles
    language_profile: RepoLanguageProfile = field(default_factory=RepoLanguageProfile)
    complexity_profile: RepoComplexityProfile = field(default_factory=RepoComplexityProfile)
    
    # Classification
    industry_guess: Optional[str] = None
    industry_confidence: float = 0.0
    
    # Final scores
    enterprise_score: float = 0.0
    selection_score: float = 0.0
    
    def to_dict(self) -> dict:
        d = asdict(self)
        return d
    
    def meets_minimum_criteria(
        self,
        min_loc: int = 500_000,
        min_languages: int = 2,
        require_enterprise_lang: bool = True,
    ) -> bool:
        """Check if repo meets minimum selection criteria."""
        if self.archived or self.fork:
            return False
        
        if self.complexity_profile.total_loc < min_loc:
            return False
        
        if self.language_profile.language_count < min_languages:
            return False
        
        if require_enterprise_lang and not self.language_profile.enterprise_languages:
            return False
        
        return True


# =============================================================================
# GitHub API Helpers
# =============================================================================

class GitHubClient:
    """GitHub API client with pagination and rate limiting."""
    
    def __init__(self, token: str, rate_limit_delay: float = 0.5):
        self.token = token
        self.rate_limit_delay = rate_limit_delay
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "IR-SDLC-Bench-RepoSelector/1.0",
        })
    
    def paginate_get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 10,
    ) -> Iterable[Any]:
        """Paginate through GitHub API results."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        
        page_count = 0
        while page_count < max_pages:
            try:
                response = self.session.get(url, params=params, timeout=60)
                self._handle_rate_limit(response)
                response.raise_for_status()
                
                data = response.json()
                
                # Handle search endpoints
                if isinstance(data, dict) and "items" in data:
                    yield from data["items"]
                else:
                    yield from data
                
                # Check for next page
                link = response.headers.get("Link", "")
                match = re.search(r'<([^>]+)>;\s*rel="next"', link)
                if not match:
                    break
                
                url = match.group(1)
                params = None  # Already encoded in next link
                page_count += 1
                
                time.sleep(self.rate_limit_delay)
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"API error: {e}")
                break
    
    def get(self, url: str, **kwargs) -> Optional[dict]:
        """Make a single GET request."""
        try:
            response = self.session.get(url, timeout=60, **kwargs)
            self._handle_rate_limit(response)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"GET {url} failed: {e}")
            return None
    
    def get_stream(self, url: str) -> Optional[requests.Response]:
        """Get streaming response for large downloads."""
        try:
            response = self.session.get(url, stream=True, timeout=120)
            self._handle_rate_limit(response)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Stream GET {url} failed: {e}")
            return None
    
    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Handle GitHub rate limiting."""
        if "X-RateLimit-Remaining" in response.headers:
            remaining = int(response.headers["X-RateLimit-Remaining"])
            if remaining < 10:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(0, reset_time - time.time()) + 1
                logger.warning(f"Rate limit low ({remaining}), waiting {wait_time:.0f}s")
                time.sleep(min(wait_time, 60))
    
    def list_org_repos(self, org: str, max_repos: int = 250) -> List[dict]:
        """List repositories for an organization."""
        url = f"{GITHUB_API}/orgs/{org}/repos"
        repos = list(self.paginate_get(url, params={"type": "all"}))
        return repos[:max_repos]
    
    def search_repos(
        self,
        query: str,
        sort: str = "stars",
        max_results: int = 100,
    ) -> List[dict]:
        """Search for repositories."""
        url = f"{GITHUB_API}/search/repositories"
        params = {"q": query, "sort": sort, "order": "desc"}
        
        results = []
        for item in self.paginate_get(url, params=params, max_pages=max_results // 100 + 1):
            results.append(item)
            if len(results) >= max_results:
                break
        
        return results
    
    def get_repo_languages(self, full_name: str) -> Dict[str, int]:
        """Get language breakdown for a repository (bytes)."""
        url = f"{GITHUB_API}/repos/{full_name}/languages"
        result = self.get(url)
        return result if result else {}
    
    def get_repo_topics(self, full_name: str) -> List[str]:
        """Get repository topics."""
        url = f"{GITHUB_API}/repos/{full_name}/topics"
        result = self.get(url)
        return result.get("names", []) if result else []
    
    def count_recent_issues(
        self,
        full_name: str,
        since: datetime,
    ) -> int:
        """Count issues created since a date (excludes PRs)."""
        url = f"{GITHUB_API}/repos/{full_name}/issues"
        params = {
            "state": "all",
            "since": since.isoformat() + "Z",
            "per_page": 100,
        }
        
        count = 0
        for item in self.paginate_get(url, params=params, max_pages=5):
            # Exclude pull requests
            if isinstance(item, dict) and "pull_request" not in item:
                count += 1
        
        return count
    
    def get_repo_contributors_count(self, full_name: str) -> int:
        """Get approximate contributor count."""
        url = f"{GITHUB_API}/repos/{full_name}/contributors"
        
        try:
            response = self.session.get(
                url,
                params={"per_page": 1, "anon": "1"},
                timeout=30,
            )
            self._handle_rate_limit(response)
            response.raise_for_status()
            
            # Parse count from Link header
            link = response.headers.get("Link", "")
            match = re.search(r'page=(\d+)>; rel="last"', link)
            if match:
                return int(match.group(1))
            return len(response.json())
            
        except requests.exceptions.RequestException:
            return 0


# =============================================================================
# LOC Computation
# =============================================================================

class LOCComputer:
    """Computes lines of code using tokei."""
    
    def __init__(self, github_client: GitHubClient):
        self.client = github_client
        self._check_tokei()
    
    def _check_tokei(self) -> bool:
        """Check if tokei is installed."""
        try:
            subprocess.run(
                ["tokei", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("tokei not found. Install with: brew install tokei")
            return False
    
    def compute_loc(
        self,
        full_name: str,
        branch: str = "main",
    ) -> Tuple[int, Dict[str, int]]:
        """
        Compute LOC by downloading tarball and running tokei.
        
        Returns:
            Tuple of (total_loc, loc_by_language)
        """
        tar_url = f"{GITHUB_API}/repos/{full_name}/tarball/{branch}"
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tar_path = Path(tmp_dir) / "repo.tar.gz"
            src_dir = Path(tmp_dir) / "src"
            
            # Download tarball
            response = self.client.get_stream(tar_url)
            if response is None:
                return 0, {}
            
            with open(tar_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            
            # Extract
            src_dir.mkdir(exist_ok=True)
            try:
                subprocess.run(
                    ["tar", "-xzf", str(tar_path), "-C", str(src_dir), "--strip-components=1"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract tarball: {e}")
                return 0, {}
            
            # Run tokei
            try:
                result = subprocess.run(
                    ["tokei", str(src_dir), "-o", "json"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                data = json.loads(result.stdout)
                
                loc_by_lang: Dict[str, int] = {}
                total = 0
                
                for lang, stats in data.items():
                    if lang == "Total":
                        total = int(stats.get("code", 0))
                    else:
                        loc_by_lang[lang] = int(stats.get("code", 0))
                
                return total, loc_by_lang
                
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                logger.warning(f"tokei failed: {e}")
                return 0, {}
    
    def estimate_loc_from_bytes(self, language_bytes: Dict[str, int]) -> int:
        """
        Estimate LOC from GitHub language bytes.
        
        Very rough approximation: ~40 bytes per line average.
        """
        total_bytes = sum(language_bytes.values())
        return total_bytes // 40


# =============================================================================
# Repository Analyzer
# =============================================================================

class RepoAnalyzer:
    """Analyzes repositories for enterprise suitability."""
    
    def __init__(self, github_client: GitHubClient, compute_loc: bool = False):
        self.client = github_client
        self.loc_computer = LOCComputer(github_client) if compute_loc else None
    
    def analyze_repo(self, repo_data: dict) -> EnterpriseRepoCandidate:
        """
        Analyze a repository and return candidate with scores.
        
        Args:
            repo_data: Raw repository data from GitHub API
            
        Returns:
            EnterpriseRepoCandidate with analysis results
        """
        full_name = repo_data["full_name"]
        
        candidate = EnterpriseRepoCandidate(
            full_name=full_name,
            html_url=repo_data.get("html_url", ""),
            description=repo_data.get("description") or "",
            topics=repo_data.get("topics") or [],
            default_branch=repo_data.get("default_branch", "main"),
            archived=bool(repo_data.get("archived")),
            fork=bool(repo_data.get("fork")),
            stars=int(repo_data.get("stargazers_count") or 0),
            forks=int(repo_data.get("forks_count") or 0),
            watchers=int(repo_data.get("watchers_count") or 0),
            pushed_at=repo_data.get("pushed_at"),
            created_at=repo_data.get("created_at"),
            open_issues_count=int(repo_data.get("open_issues_count") or 0),
        )
        
        # Get topics if not included
        if not candidate.topics:
            candidate.topics = self.client.get_repo_topics(full_name)
        
        # Analyze languages
        candidate.language_profile = self._analyze_languages(full_name, repo_data)
        
        # Analyze complexity
        candidate.complexity_profile = self._analyze_complexity(full_name, candidate)
        
        # Classify industry
        candidate.industry_guess, candidate.industry_confidence = self._classify_industry(
            candidate
        )
        
        # Get recent issues
        since = datetime.now(timezone.utc) - timedelta(days=180)
        try:
            candidate.recent_issues_count = self.client.count_recent_issues(full_name, since)
        except Exception:
            candidate.recent_issues_count = 0
        
        # Compute final scores
        candidate.enterprise_score = self._compute_enterprise_score(candidate)
        candidate.selection_score = self._compute_selection_score(candidate)
        
        return candidate
    
    def _analyze_languages(
        self,
        full_name: str,
        repo_data: dict,
    ) -> RepoLanguageProfile:
        """Analyze language breakdown."""
        profile = RepoLanguageProfile()
        
        # Get language bytes from API
        lang_bytes = self.client.get_repo_languages(full_name)
        
        if not lang_bytes:
            profile.primary_language = repo_data.get("language")
            return profile
        
        # Map to enterprise languages
        enterprise_langs = set()
        legacy_langs = set()
        loc_by_lang: Dict[str, int] = {}
        
        for lang, bytes_count in lang_bytes.items():
            # Estimate LOC
            estimated_loc = bytes_count // 40
            loc_by_lang[lang] = estimated_loc
            
            # Classify
            if lang in LANGUAGE_ALIASES:
                ent_lang = LANGUAGE_ALIASES[lang]
                enterprise_langs.add(ent_lang.value)
                
                if ent_lang in [EnterpriseLanguage.COBOL, EnterpriseLanguage.FORTRAN,
                               EnterpriseLanguage.PERL, EnterpriseLanguage.VB]:
                    legacy_langs.add(ent_lang.value)
        
        # Compute diversity score
        total_bytes = sum(lang_bytes.values())
        if total_bytes > 0:
            # Normalized entropy-like score
            proportions = [b / total_bytes for b in lang_bytes.values() if b > 0]
            if len(proportions) > 1:
                import math
                entropy = -sum(p * math.log(p) for p in proportions if p > 0)
                max_entropy = math.log(len(proportions))
                profile.language_diversity_score = entropy / max_entropy if max_entropy > 0 else 0
        
        profile.primary_language = max(lang_bytes.keys(), key=lambda k: lang_bytes[k]) if lang_bytes else None
        profile.language_count = len(lang_bytes)
        profile.enterprise_languages = list(enterprise_langs)
        profile.legacy_languages = list(legacy_langs)
        profile.loc_by_language = loc_by_lang
        
        return profile
    
    def _analyze_complexity(
        self,
        full_name: str,
        candidate: EnterpriseRepoCandidate,
    ) -> RepoComplexityProfile:
        """Analyze repository complexity."""
        profile = RepoComplexityProfile()
        
        # Estimate LOC
        if self.loc_computer:
            try:
                profile.total_loc, loc_breakdown = self.loc_computer.compute_loc(
                    full_name, candidate.default_branch
                )
                if loc_breakdown:
                    candidate.language_profile.loc_by_language = loc_breakdown
            except Exception as e:
                logger.warning(f"LOC computation failed for {full_name}: {e}")
        
        if profile.total_loc == 0:
            # Fallback to estimate from bytes
            profile.total_loc = self.loc_computer.estimate_loc_from_bytes(
                candidate.language_profile.loc_by_language
            ) if self.loc_computer else sum(candidate.language_profile.loc_by_language.values())
        
        # Detect frameworks from topics and description
        detected = self._detect_frameworks(candidate)
        profile.frameworks_detected = detected
        
        # Determine layer coverage
        for fw in detected:
            if fw in FRAMEWORK_PATTERNS:
                category = FRAMEWORK_PATTERNS[fw]["category"]
                if category == "backend":
                    profile.has_backend = True
                elif category == "frontend":
                    profile.has_frontend = True
                elif category == "data":
                    profile.has_data_layer = True
                elif category == "infrastructure":
                    profile.has_infrastructure = True
        
        # Check for common patterns
        topics_lower = " ".join(candidate.topics).lower()
        desc_lower = (candidate.description or "").lower()
        
        profile.has_tests = any(
            kw in topics_lower or kw in desc_lower
            for kw in ["test", "testing", "pytest", "jest", "junit"]
        )
        profile.has_ci_cd = any(
            kw in topics_lower
            for kw in ["ci", "cd", "github-actions", "jenkins", "travis"]
        )
        profile.has_docs = any(
            kw in topics_lower or kw in desc_lower
            for kw in ["docs", "documentation", "wiki"]
        )
        
        # Compute complexity score
        profile.complexity_score = self._compute_complexity_score(profile, candidate)
        
        return profile
    
    def _detect_frameworks(self, candidate: EnterpriseRepoCandidate) -> List[str]:
        """Detect frameworks from metadata."""
        detected = []
        
        text = " ".join([
            candidate.description or "",
            " ".join(candidate.topics),
        ]).lower()
        
        for fw_name, fw_info in FRAMEWORK_PATTERNS.items():
            # Check content patterns
            for pattern in fw_info.get("content", []):
                if pattern.lower() in text:
                    detected.append(fw_name)
                    break
        
        return list(set(detected))
    
    def _classify_industry(
        self,
        candidate: EnterpriseRepoCandidate,
    ) -> Tuple[Optional[str], float]:
        """Classify repository by industry."""
        text = " ".join([
            candidate.full_name,
            candidate.description or "",
            " ".join(candidate.topics),
        ]).lower()
        
        best_industry = None
        best_score = 0.0
        
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw.lower() in text)
            score = hits / len(keywords) if keywords else 0
            
            if score > best_score:
                best_score = score
                best_industry = industry.value
        
        return best_industry, min(best_score * 3, 1.0)  # Scale up for visibility
    
    def _compute_complexity_score(
        self,
        profile: RepoComplexityProfile,
        candidate: EnterpriseRepoCandidate,
    ) -> float:
        """Compute overall complexity score (0-100)."""
        score = 0.0
        
        # LOC contribution (0-30)
        if profile.total_loc >= 1_000_000:
            score += 30
        elif profile.total_loc >= 500_000:
            score += 20
        elif profile.total_loc >= 100_000:
            score += 10
        
        # Language diversity (0-20)
        score += candidate.language_profile.language_diversity_score * 20
        
        # Enterprise language coverage (0-15)
        ent_langs = len(candidate.language_profile.enterprise_languages)
        score += min(15, ent_langs * 3)
        
        # Layer coverage (0-20)
        layers = sum([
            profile.has_backend,
            profile.has_frontend,
            profile.has_data_layer,
            profile.has_infrastructure,
        ])
        score += layers * 5
        
        # Framework coverage (0-10)
        score += min(10, len(profile.frameworks_detected) * 2)
        
        # Legacy indicators (0-5)
        if candidate.language_profile.legacy_languages:
            score += 5
        
        return min(100, score)
    
    def _compute_enterprise_score(self, candidate: EnterpriseRepoCandidate) -> float:
        """Compute enterprise suitability score (0-1)."""
        score = 0.0
        
        # Language priority score
        lang_score = 0.0
        for lang in candidate.language_profile.enterprise_languages:
            for ent_lang, priority in LANGUAGE_PRIORITY.items():
                if ent_lang.value == lang:
                    lang_score = max(lang_score, priority)
        score += lang_score * 0.3
        
        # Complexity score
        score += (candidate.complexity_profile.complexity_score / 100) * 0.3
        
        # Activity score
        if candidate.pushed_at:
            try:
                pushed = datetime.fromisoformat(candidate.pushed_at.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - pushed).days
                activity = max(0, 1 - (days_ago / 365))
                score += activity * 0.2
            except ValueError:
                pass
        
        # Industry confidence
        score += candidate.industry_confidence * 0.1
        
        # Polyglot bonus
        if candidate.language_profile.language_count >= 5:
            score += 0.1
        
        return min(1.0, score)
    
    def _compute_selection_score(self, candidate: EnterpriseRepoCandidate) -> float:
        """Compute final selection score for ranking."""
        if candidate.archived or candidate.fork:
            return 0.0
        
        score = candidate.enterprise_score
        
        # Issue activity bonus
        if candidate.recent_issues_count > 0:
            kloc = max(1, candidate.complexity_profile.total_loc / 1000)
            issue_density = candidate.recent_issues_count / kloc
            score += min(0.1, issue_density * 0.01)
        
        # Stars as quality indicator (small bonus)
        if candidate.stars >= 1000:
            score += 0.05
        
        return min(1.0, score)


# =============================================================================
# Repository Selector
# =============================================================================

@dataclass
class SelectionCriteria:
    """Criteria for repository selection."""
    min_loc: int = 500_000
    min_languages: int = 2
    require_enterprise_language: bool = True
    min_selection_score: float = 0.3
    
    # Language requirements
    required_languages: Set[str] = field(default_factory=set)
    preferred_languages: Set[str] = field(default_factory=lambda: {
        "Java", "C#", "TypeScript", "JavaScript", "Python", "Go"
    })
    
    # Industry coverage targets
    target_industries: Set[str] = field(default_factory=lambda: {
        "finance", "saas_cloud", "gov_defense", "healthcare",
        "telecom", "ecommerce_retail", "infrastructure", "data_ml"
    })
    
    # Diversity constraints
    max_repos_per_org: int = 5
    max_repos_per_industry: int = 10
    min_repos_per_industry: int = 2


class EnterpriseRepoSelector:
    """
    Selects enterprise repositories for SDLC benchmark generation.
    
    Implements GitTaskBench-style selection with enterprise SDLC focus.
    """
    
    # Known enterprise-scale organizations
    ENTERPRISE_ORGS = [
        # Cloud/SaaS
        "kubernetes", "hashicorp", "elastic", "grafana", "prometheus",
        "apache", "cncf", "envoyproxy", "istio", "linkerd",
        
        # Finance (open source)
        "finos", "goldmansachs", "bloomberg", "stripe", "square",
        
        # E-commerce/Retail
        "shopify", "etsy", "walmart", "alibaba",
        
        # Tech giants (open source arms)
        "microsoft", "google", "facebook", "amazon", "netflix",
        "uber", "lyft", "airbnb", "twitter", "linkedin",
        
        # Government/Public sector
        "nasa", "usgs", "gsa", "18f",
        
        # Telecom
        "opnfv", "onap",
        
        # Data/ML
        "apache", "databricks", "dbt-labs",
    ]
    
    # High-value individual repositories
    ENTERPRISE_REPOS = [
        # Kubernetes ecosystem
        "kubernetes/kubernetes",
        "helm/helm",
        "argoproj/argo-cd",
        "istio/istio",
        
        # Databases
        "cockroachdb/cockroach",
        "pingcap/tidb",
        "yugabyte/yugabyte-db",
        "apache/cassandra",
        
        # Messaging
        "apache/kafka",
        "apache/pulsar",
        "nats-io/nats-server",
        
        # Data processing
        "apache/spark",
        "apache/flink",
        "apache/airflow",
        "trinodb/trino",
        
        # Backend frameworks
        "spring-projects/spring-framework",
        "spring-projects/spring-boot",
        "dotnet/aspnetcore",
        "django/django",
        "pallets/flask",
        "tiangolo/fastapi",
        "nestjs/nest",
        "gin-gonic/gin",
        
        # Frontend
        "facebook/react",
        "angular/angular",
        "vuejs/vue",
        "sveltejs/svelte",
        
        # Infrastructure
        "hashicorp/terraform",
        "ansible/ansible",
        "pulumi/pulumi",
        
        # Languages/Runtimes
        "golang/go",
        "python/cpython",
        "microsoft/TypeScript",
        "rust-lang/rust",
        
        # Observability
        "grafana/grafana",
        "prometheus/prometheus",
        "elastic/elasticsearch",
        
        # Security
        "hashicorp/vault",
        "keycloak/keycloak",
        
        # Finance-related
        "finos/legend-studio",
        "quantlib/QuantLib",
        
        # E-commerce
        "prestashop/PrestaShop",
        "spree/spree",
        "magento/magento2",
        
        # Healthcare
        "openmrs/openmrs-core",
        "hapifhir/hapi-fhir",
        
        # Government
        "nasa/openmct",
        "usnistgov/NIST-Tech-Pubs",
    ]
    
    def __init__(
        self,
        github_token: str,
        compute_loc: bool = False,
        rate_limit_delay: float = 0.5,
    ):
        self.client = GitHubClient(github_token, rate_limit_delay)
        self.analyzer = RepoAnalyzer(self.client, compute_loc)
        self.compute_loc = compute_loc
    
    def discover_candidates(
        self,
        orgs: Optional[List[str]] = None,
        repos: Optional[List[str]] = None,
        search_queries: Optional[List[str]] = None,
        max_per_source: int = 50,
    ) -> List[EnterpriseRepoCandidate]:
        """
        Discover candidate repositories from multiple sources.
        
        Args:
            orgs: Organizations to scan
            repos: Specific repos to include
            search_queries: GitHub search queries
            max_per_source: Max repos per source
            
        Returns:
            List of analyzed candidates
        """
        candidates: Dict[str, EnterpriseRepoCandidate] = {}
        
        # Use defaults if no sources specified
        if not orgs and not repos and not search_queries:
            orgs = self.ENTERPRISE_ORGS[:10]
            repos = self.ENTERPRISE_REPOS[:20]
        
        # Process organizations
        if orgs:
            for org in orgs:
                logger.info(f"Scanning organization: {org}")
                try:
                    org_repos = self.client.list_org_repos(org, max_repos=max_per_source)
                    
                    # Pre-filter
                    org_repos = [
                        r for r in org_repos
                        if not r.get("archived") and not r.get("fork")
                    ]
                    
                    # Sort by activity
                    org_repos.sort(key=lambda r: r.get("pushed_at") or "", reverse=True)
                    org_repos = org_repos[:max_per_source]
                    
                    for repo_data in org_repos:
                        full_name = repo_data["full_name"]
                        if full_name not in candidates:
                            candidate = self.analyzer.analyze_repo(repo_data)
                            candidates[full_name] = candidate
                            
                except Exception as e:
                    logger.warning(f"Failed to scan org {org}: {e}")
        
        # Process specific repos
        if repos:
            for repo_name in repos:
                if repo_name in candidates:
                    continue
                
                logger.info(f"Analyzing repo: {repo_name}")
                try:
                    repo_data = self.client.get(f"{GITHUB_API}/repos/{repo_name}")
                    if repo_data:
                        candidate = self.analyzer.analyze_repo(repo_data)
                        candidates[repo_name] = candidate
                except Exception as e:
                    logger.warning(f"Failed to analyze {repo_name}: {e}")
        
        # Process search queries
        if search_queries:
            for query in search_queries:
                logger.info(f"Searching: {query}")
                try:
                    results = self.client.search_repos(query, max_results=max_per_source)
                    for repo_data in results:
                        full_name = repo_data["full_name"]
                        if full_name not in candidates:
                            candidate = self.analyzer.analyze_repo(repo_data)
                            candidates[full_name] = candidate
                except Exception as e:
                    logger.warning(f"Search failed for '{query}': {e}")
        
        return list(candidates.values())
    
    def select_repos(
        self,
        candidates: List[EnterpriseRepoCandidate],
        criteria: Optional[SelectionCriteria] = None,
        max_total: int = 50,
    ) -> List[EnterpriseRepoCandidate]:
        """
        Select repositories based on criteria with diversity.
        
        Args:
            candidates: Candidate repositories to select from
            criteria: Selection criteria
            max_total: Maximum total repos to select
            
        Returns:
            Selected repositories
        """
        criteria = criteria or SelectionCriteria()
        
        # Filter by minimum criteria
        eligible = [
            c for c in candidates
            if c.meets_minimum_criteria(
                min_loc=criteria.min_loc,
                min_languages=criteria.min_languages,
                require_enterprise_lang=criteria.require_enterprise_language,
            )
            and c.selection_score >= criteria.min_selection_score
        ]
        
        if not eligible:
            logger.warning("No candidates meet minimum criteria")
            return []
        
        # Sort by score
        eligible.sort(key=lambda c: c.selection_score, reverse=True)
        
        # Apply diversity constraints
        selected: List[EnterpriseRepoCandidate] = []
        org_counts: Dict[str, int] = {}
        industry_counts: Dict[str, int] = {}
        
        for candidate in eligible:
            if len(selected) >= max_total:
                break
            
            org = candidate.full_name.split("/")[0]
            industry = candidate.industry_guess or "unknown"
            
            # Check org limit
            if org_counts.get(org, 0) >= criteria.max_repos_per_org:
                continue
            
            # Check industry limit
            if industry_counts.get(industry, 0) >= criteria.max_repos_per_industry:
                continue
            
            selected.append(candidate)
            org_counts[org] = org_counts.get(org, 0) + 1
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        # Log selection summary
        logger.info(f"Selected {len(selected)} repos from {len(candidates)} candidates")
        logger.info(f"Industry distribution: {industry_counts}")
        
        return selected
    
    def generate_selection_report(
        self,
        selected: List[EnterpriseRepoCandidate],
        output_path: Optional[Path] = None,
    ) -> dict:
        """
        Generate a selection report with statistics.
        
        Args:
            selected: Selected repositories
            output_path: Optional path to save report
            
        Returns:
            Report as dict
        """
        # Compute statistics
        by_industry: Dict[str, List[str]] = {}
        by_language: Dict[str, List[str]] = {}
        total_loc = 0
        
        for repo in selected:
            industry = repo.industry_guess or "unknown"
            if industry not in by_industry:
                by_industry[industry] = []
            by_industry[industry].append(repo.full_name)
            
            for lang in repo.language_profile.enterprise_languages:
                if lang not in by_language:
                    by_language[lang] = []
                by_language[lang].append(repo.full_name)
            
            total_loc += repo.complexity_profile.total_loc
        
        report = {
            "summary": {
                "total_repos": len(selected),
                "total_loc_estimated": total_loc,
                "industries_covered": len(by_industry),
                "languages_covered": len(by_language),
            },
            "by_industry": {k: len(v) for k, v in by_industry.items()},
            "by_language": {k: len(v) for k, v in by_language.items()},
            "repos": [
                {
                    "full_name": r.full_name,
                    "score": round(r.selection_score, 3),
                    "loc": r.complexity_profile.total_loc,
                    "industry": r.industry_guess,
                    "languages": r.language_profile.enterprise_languages,
                    "frameworks": r.complexity_profile.frameworks_detected,
                }
                for r in selected
            ],
        }
        
        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def save_candidates_csv(
        self,
        candidates: List[EnterpriseRepoCandidate],
        output_path: Path,
    ) -> None:
        """Save candidates to CSV for manual review."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "full_name", "selection_score", "enterprise_score",
                "loc_estimated", "language_count", "primary_language",
                "enterprise_languages", "frameworks",
                "industry_guess", "industry_confidence",
                "pushed_at", "stars", "html_url"
            ])
            
            for c in sorted(candidates, key=lambda x: x.selection_score, reverse=True):
                writer.writerow([
                    c.full_name,
                    f"{c.selection_score:.4f}",
                    f"{c.enterprise_score:.4f}",
                    c.complexity_profile.total_loc,
                    c.language_profile.language_count,
                    c.language_profile.primary_language or "",
                    ";".join(c.language_profile.enterprise_languages),
                    ";".join(c.complexity_profile.frameworks_detected),
                    c.industry_guess or "",
                    f"{c.industry_confidence:.3f}",
                    c.pushed_at or "",
                    c.stars,
                    c.html_url,
                ])
        
        logger.info(f"Saved {len(candidates)} candidates to {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for repository selection."""
    parser = argparse.ArgumentParser(
        description="Select enterprise repositories for IR-SDLC benchmark generation"
    )
    parser.add_argument(
        "--org", action="append", dest="orgs",
        help="GitHub organization to scan (repeatable)"
    )
    parser.add_argument(
        "--repo", action="append", dest="repos",
        help="Specific repository to include (repeatable)"
    )
    parser.add_argument(
        "--search", action="append", dest="searches",
        help="GitHub search query (repeatable)"
    )
    parser.add_argument(
        "--use-defaults", action="store_true",
        help="Use default enterprise orgs and repos"
    )
    parser.add_argument(
        "--min-loc", type=int, default=500_000,
        help="Minimum lines of code"
    )
    parser.add_argument(
        "--compute-loc", action="store_true",
        help="Compute actual LOC with tokei (slower but accurate)"
    )
    parser.add_argument(
        "--max-repos", type=int, default=50,
        help="Maximum repos to select"
    )
    parser.add_argument(
        "--out-csv", default="repo_candidates.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--out-json", default="repo_selection.json",
        help="Output JSON report path"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable required")
        return 1
    
    # Initialize selector
    selector = EnterpriseRepoSelector(
        github_token=token,
        compute_loc=args.compute_loc,
    )
    
    # Determine sources
    orgs = args.orgs
    repos = args.repos
    searches = args.searches
    
    if args.use_defaults:
        orgs = (orgs or []) + selector.ENTERPRISE_ORGS[:15]
        repos = (repos or []) + selector.ENTERPRISE_REPOS[:30]
    
    # Discover candidates
    candidates = selector.discover_candidates(
        orgs=orgs,
        repos=repos,
        search_queries=searches,
    )
    
    # Select repos
    criteria = SelectionCriteria(min_loc=args.min_loc)
    selected = selector.select_repos(
        candidates,
        criteria=criteria,
        max_total=args.max_repos,
    )
    
    # Save outputs
    selector.save_candidates_csv(candidates, Path(args.out_csv))
    selector.generate_selection_report(selected, Path(args.out_json))
    
    print(f"\nSelected {len(selected)} repos from {len(candidates)} candidates")
    print(f"CSV: {args.out_csv}")
    print(f"Report: {args.out_json}")
    
    return 0


if __name__ == "__main__":
    exit(main())
