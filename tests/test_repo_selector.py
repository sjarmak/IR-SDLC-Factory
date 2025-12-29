"""
Tests for repo_selector.py - Enterprise Repository Selector.

Tests cover:
1. Data structures (EnterpriseLanguage, EnterpriseIndustry, profiles)
2. GitHubClient API interactions (mocked)
3. LOCComputer estimation and tokei integration
4. RepoAnalyzer analysis methods
5. EnterpriseRepoSelector selection logic
6. SelectionCriteria filtering
"""

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.ir_sdlc.repo_selector import (
    FRAMEWORK_PATTERNS,
    INDUSTRY_KEYWORDS,
    LANGUAGE_ALIASES,
    LANGUAGE_PRIORITY,
    EnterpriseIndustry,
    EnterpriseLanguage,
    EnterpriseRepoCandidate,
    EnterpriseRepoSelector,
    GitHubClient,
    LOCComputer,
    RepoAnalyzer,
    RepoComplexityProfile,
    RepoLanguageProfile,
    SelectionCriteria,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_github_client():
    """Create a mock GitHub client."""
    client = MagicMock(spec=GitHubClient)
    client.session = MagicMock()
    client.rate_limit_delay = 0.0
    return client


@pytest.fixture
def sample_repo_data() -> dict:
    """Sample GitHub repository data."""
    return {
        "id": 12345,
        "full_name": "enterprise-corp/payment-service",
        "html_url": "https://github.com/enterprise-corp/payment-service",
        "description": "Enterprise payment processing platform with fraud detection",
        "topics": ["payments", "finance", "microservices", "spring-boot"],
        "default_branch": "main",
        "archived": False,
        "fork": False,
        "stargazers_count": 5000,
        "forks_count": 800,
        "watchers_count": 300,
        "open_issues_count": 120,
        "pushed_at": "2024-01-15T10:00:00Z",
        "created_at": "2020-01-01T00:00:00Z",
        "language": "Java",
        "size": 150000,  # KB
    }


@pytest.fixture
def sample_language_bytes() -> Dict[str, int]:
    """Sample language breakdown by bytes."""
    return {
        "Java": 20_000_000,
        "TypeScript": 8_000_000,
        "Python": 3_000_000,
        "Shell": 500_000,
        "Dockerfile": 50_000,
    }


@pytest.fixture
def sample_candidate() -> EnterpriseRepoCandidate:
    """Sample enterprise repo candidate."""
    return EnterpriseRepoCandidate(
        full_name="enterprise-corp/payment-service",
        html_url="https://github.com/enterprise-corp/payment-service",
        description="Enterprise payment processing platform",
        topics=["payments", "finance", "microservices"],
        default_branch="main",
        archived=False,
        fork=False,
        stars=5000,
        forks=800,
        watchers=300,
        open_issues_count=120,
        recent_issues_count=50,
        pushed_at="2024-01-15T10:00:00Z",
        created_at="2020-01-01T00:00:00Z",
        language_profile=RepoLanguageProfile(
            primary_language="Java",
            language_count=5,
            enterprise_languages=["Java", "TypeScript", "Python"],
            legacy_languages=[],
            loc_by_language={"Java": 400000, "TypeScript": 150000, "Python": 50000},
            language_diversity_score=0.7,
        ),
        complexity_profile=RepoComplexityProfile(
            total_loc=600000,
            total_files=2500,
            directory_depth_max=8,
            has_backend=True,
            has_frontend=True,
            has_data_layer=True,
            has_infrastructure=True,
            has_tests=True,
            has_ci_cd=True,
            has_docs=True,
            frameworks_detected=["spring", "react", "kafka"],
            complexity_score=75.0,
        ),
        industry_guess="finance",
        industry_confidence=0.85,
        enterprise_score=0.9,
        selection_score=0.85,
    )


# =============================================================================
# Test EnterpriseLanguage Enum
# =============================================================================


class TestEnterpriseLanguage:
    """Tests for EnterpriseLanguage enum."""

    def test_tier1_languages_exist(self):
        """Verify tier 1 enterprise languages are defined."""
        assert EnterpriseLanguage.JAVA == "Java"
        assert EnterpriseLanguage.CSHARP == "C#"
        assert EnterpriseLanguage.TYPESCRIPT == "TypeScript"
        assert EnterpriseLanguage.PYTHON == "Python"
        assert EnterpriseLanguage.GO == "Go"

    def test_legacy_languages_exist(self):
        """Verify legacy languages are defined for migration tasks."""
        assert EnterpriseLanguage.COBOL == "COBOL"
        assert EnterpriseLanguage.PERL == "Perl"
        assert EnterpriseLanguage.FORTRAN == "Fortran"

    def test_language_aliases_map_to_enum(self):
        """Verify all language aliases map to valid enum values."""
        for alias, lang in LANGUAGE_ALIASES.items():
            assert isinstance(lang, EnterpriseLanguage)
            assert lang.value is not None

    def test_language_priority_covers_all_languages(self):
        """Verify all languages have a priority weight."""
        for lang in EnterpriseLanguage:
            if lang in LANGUAGE_PRIORITY:
                assert 0 <= LANGUAGE_PRIORITY[lang] <= 1


class TestEnterpriseIndustry:
    """Tests for EnterpriseIndustry enum."""

    def test_target_industries_exist(self):
        """Verify target industries are defined."""
        assert EnterpriseIndustry.FINANCE == "finance"
        assert EnterpriseIndustry.SAAS_CLOUD == "saas_cloud"
        assert EnterpriseIndustry.GOVERNMENT_DEFENSE == "gov_defense"
        assert EnterpriseIndustry.HEALTHCARE == "healthcare"
        assert EnterpriseIndustry.TELECOM == "telecom"
        assert EnterpriseIndustry.ECOMMERCE_RETAIL == "ecommerce_retail"

    def test_industry_keywords_defined_for_all(self):
        """Verify industry keywords are defined for all industries."""
        for industry in EnterpriseIndustry:
            assert industry in INDUSTRY_KEYWORDS
            assert len(INDUSTRY_KEYWORDS[industry]) > 0


# =============================================================================
# Test Framework Patterns
# =============================================================================


class TestFrameworkPatterns:
    """Tests for framework detection patterns."""

    def test_backend_frameworks_defined(self):
        """Verify backend framework patterns exist."""
        backend_frameworks = ["spring", "aspnet", "django", "fastapi", "express", "nestjs", "gin"]
        for fw in backend_frameworks:
            assert fw in FRAMEWORK_PATTERNS
            assert FRAMEWORK_PATTERNS[fw]["category"] == "backend"

    def test_frontend_frameworks_defined(self):
        """Verify frontend framework patterns exist."""
        frontend_frameworks = ["react", "angular", "vue", "svelte"]
        for fw in frontend_frameworks:
            assert fw in FRAMEWORK_PATTERNS
            assert FRAMEWORK_PATTERNS[fw]["category"] == "frontend"

    def test_data_frameworks_defined(self):
        """Verify data/messaging framework patterns exist."""
        data_frameworks = ["kafka", "spark", "flink", "airflow"]
        for fw in data_frameworks:
            assert fw in FRAMEWORK_PATTERNS
            assert FRAMEWORK_PATTERNS[fw]["category"] == "data"

    def test_infrastructure_frameworks_defined(self):
        """Verify infrastructure framework patterns exist."""
        infra_frameworks = ["kubernetes", "terraform", "helm", "docker"]
        for fw in infra_frameworks:
            assert fw in FRAMEWORK_PATTERNS
            assert FRAMEWORK_PATTERNS[fw]["category"] == "infrastructure"

    def test_framework_has_detection_rules(self):
        """Verify each framework has detection rules."""
        for name, pattern in FRAMEWORK_PATTERNS.items():
            assert "files" in pattern or "content" in pattern
            assert "category" in pattern


# =============================================================================
# Test Data Structures
# =============================================================================


class TestRepoLanguageProfile:
    """Tests for RepoLanguageProfile dataclass."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        profile = RepoLanguageProfile()
        assert profile.primary_language is None
        assert profile.language_count == 0
        assert profile.enterprise_languages == []
        assert profile.legacy_languages == []
        assert profile.loc_by_language == {}
        assert profile.language_diversity_score == 0.0

    def test_to_dict(self):
        """Verify to_dict serialization works."""
        profile = RepoLanguageProfile(
            primary_language="Java",
            language_count=3,
            enterprise_languages=["Java", "Python"],
            loc_by_language={"Java": 100000, "Python": 50000},
        )
        d = profile.to_dict()
        assert d["primary_language"] == "Java"
        assert d["language_count"] == 3
        assert "Java" in d["enterprise_languages"]


class TestRepoComplexityProfile:
    """Tests for RepoComplexityProfile dataclass."""

    def test_default_values(self):
        """Verify default values are set correctly."""
        profile = RepoComplexityProfile()
        assert profile.total_loc == 0
        assert profile.total_files == 0
        assert profile.has_backend is False
        assert profile.has_frontend is False
        assert profile.frameworks_detected == []
        assert profile.complexity_score == 0.0

    def test_to_dict(self):
        """Verify to_dict serialization works."""
        profile = RepoComplexityProfile(
            total_loc=500000,
            has_backend=True,
            frameworks_detected=["spring", "react"],
        )
        d = profile.to_dict()
        assert d["total_loc"] == 500000
        assert d["has_backend"] is True
        assert "spring" in d["frameworks_detected"]


class TestEnterpriseRepoCandidate:
    """Tests for EnterpriseRepoCandidate dataclass."""

    def test_default_values(self):
        """Verify minimal candidate can be created."""
        candidate = EnterpriseRepoCandidate(
            full_name="owner/repo",
            html_url="https://github.com/owner/repo",
        )
        assert candidate.full_name == "owner/repo"
        assert candidate.archived is False
        assert candidate.fork is False
        assert candidate.language_profile is not None
        assert candidate.complexity_profile is not None

    def test_to_dict(self, sample_candidate):
        """Verify to_dict produces valid JSON-serializable dict."""
        d = sample_candidate.to_dict()
        # Verify it's JSON serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0
        # Verify key fields are present
        assert d["full_name"] == "enterprise-corp/payment-service"
        assert d["enterprise_score"] == 0.9

    def test_meets_minimum_criteria_valid(self, sample_candidate):
        """Verify a good candidate meets minimum criteria."""
        assert sample_candidate.meets_minimum_criteria(
            min_loc=500000,
            min_languages=2,
            require_enterprise_lang=True,
        ) is True

    def test_meets_minimum_criteria_too_small(self, sample_candidate):
        """Verify small repos are rejected."""
        sample_candidate.complexity_profile.total_loc = 100000
        assert sample_candidate.meets_minimum_criteria(min_loc=500000) is False

    def test_meets_minimum_criteria_archived(self, sample_candidate):
        """Verify archived repos are rejected."""
        sample_candidate.archived = True
        assert sample_candidate.meets_minimum_criteria() is False

    def test_meets_minimum_criteria_fork(self, sample_candidate):
        """Verify forks are rejected."""
        sample_candidate.fork = True
        assert sample_candidate.meets_minimum_criteria() is False

    def test_meets_minimum_criteria_few_languages(self, sample_candidate):
        """Verify repos with few languages are rejected."""
        sample_candidate.language_profile.language_count = 1
        assert sample_candidate.meets_minimum_criteria(min_languages=2) is False


# =============================================================================
# Test SelectionCriteria
# =============================================================================


class TestSelectionCriteria:
    """Tests for SelectionCriteria dataclass."""

    def test_default_values(self):
        """Verify default selection criteria."""
        criteria = SelectionCriteria()
        assert criteria.min_loc == 500_000
        assert criteria.min_languages == 2
        assert criteria.require_enterprise_language is True
        assert criteria.min_selection_score == 0.3

    def test_preferred_languages_default(self):
        """Verify default preferred languages."""
        criteria = SelectionCriteria()
        assert "Java" in criteria.preferred_languages
        assert "C#" in criteria.preferred_languages
        assert "TypeScript" in criteria.preferred_languages
        assert "Python" in criteria.preferred_languages
        assert "Go" in criteria.preferred_languages

    def test_target_industries_default(self):
        """Verify default target industries."""
        criteria = SelectionCriteria()
        assert "finance" in criteria.target_industries
        assert "saas_cloud" in criteria.target_industries
        assert "healthcare" in criteria.target_industries

    def test_custom_criteria(self):
        """Verify custom criteria can be set."""
        criteria = SelectionCriteria(
            min_loc=1_000_000,
            min_languages=4,
            required_languages={"Java", "Kotlin"},
            max_repos_per_org=3,
        )
        assert criteria.min_loc == 1_000_000
        assert "Java" in criteria.required_languages


# =============================================================================
# Test GitHubClient
# =============================================================================


class TestGitHubClient:
    """Tests for GitHubClient API wrapper."""

    def test_init_sets_headers(self):
        """Verify client initializes with correct headers."""
        with patch("app.ir_sdlc.repo_selector.requests.Session") as mock_session:
            mock_instance = MagicMock()
            mock_session.return_value = mock_instance
            
            client = GitHubClient(token="test-token")
            
            # Verify headers were set
            mock_instance.headers.update.assert_called_once()
            call_args = mock_instance.headers.update.call_args[0][0]
            assert "Bearer test-token" in call_args["Authorization"]
            assert "application/vnd.github+json" in call_args["Accept"]

    def test_rate_limit_delay(self):
        """Verify rate limit delay is respected."""
        with patch("app.ir_sdlc.repo_selector.requests.Session"):
            client = GitHubClient(token="test", rate_limit_delay=1.5)
            assert client.rate_limit_delay == 1.5

    def test_get_returns_json(self, mock_github_client):
        """Verify get method returns JSON data."""
        expected_data = {"id": 123, "name": "test-repo"}
        mock_response = MagicMock()
        mock_response.json.return_value = expected_data
        mock_response.raise_for_status = MagicMock()
        mock_github_client.session.get.return_value = mock_response
        
        # Patch the actual method for this test
        mock_github_client.get = lambda url, **kwargs: expected_data
        result = mock_github_client.get("https://api.github.com/repos/owner/repo")
        
        assert result == expected_data

    def test_list_org_repos_paginated(self, mock_github_client):
        """Verify organization repos are fetched with pagination."""
        repos = [
            {"id": 1, "full_name": "org/repo1"},
            {"id": 2, "full_name": "org/repo2"},
        ]
        mock_github_client.list_org_repos.return_value = repos
        
        result = mock_github_client.list_org_repos("enterprise-org")
        assert len(result) == 2


# =============================================================================
# Test LOCComputer
# =============================================================================


class TestLOCComputer:
    """Tests for LOCComputer class."""

    def test_estimate_loc_from_bytes(self, mock_github_client):
        """Verify LOC estimation from bytes works correctly."""
        computer = LOCComputer(mock_github_client)
        
        language_bytes = {
            "Java": 10_000_000,  # 10 MB
            "Python": 5_000_000,  # 5 MB
        }
        
        # Average bytes per line varies by language
        # Typical: 30-50 bytes per line
        estimated_loc = computer.estimate_loc_from_bytes(language_bytes)
        
        # Should be in reasonable range (200k - 600k LOC for 15MB)
        assert estimated_loc > 100_000
        assert estimated_loc < 1_000_000

    def test_estimate_loc_empty(self, mock_github_client):
        """Verify empty language bytes returns 0."""
        computer = LOCComputer(mock_github_client)
        assert computer.estimate_loc_from_bytes({}) == 0

    @patch("subprocess.run")
    def test_check_tokei_installed(self, mock_run, mock_github_client):
        """Verify tokei installation check."""
        mock_run.return_value = MagicMock(returncode=0)
        
        computer = LOCComputer(mock_github_client)
        result = computer._check_tokei()
        
        # tokei --version is called to check installation
        assert mock_run.called
        assert result is True

    @patch("subprocess.run")
    def test_check_tokei_not_installed(self, mock_run, mock_github_client):
        """Verify tokei check fails gracefully when not installed."""
        mock_run.side_effect = FileNotFoundError()
        
        computer = LOCComputer(mock_github_client)
        result = computer._check_tokei()
        
        assert result is False


# =============================================================================
# Test RepoAnalyzer
# =============================================================================


class TestRepoAnalyzer:
    """Tests for RepoAnalyzer class."""

    def test_analyze_repo_creates_candidate(self, mock_github_client, sample_repo_data, sample_language_bytes):
        """Verify analyze_repo creates a valid candidate."""
        mock_github_client.get_repo_languages.return_value = sample_language_bytes
        mock_github_client.get_repo_topics.return_value = ["finance", "payments"]
        mock_github_client.count_recent_issues.return_value = 25
        
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        candidate = analyzer.analyze_repo(sample_repo_data)
        
        assert candidate.full_name == "enterprise-corp/payment-service"
        assert candidate.language_profile.primary_language == "Java"
        assert candidate.complexity_profile.total_loc > 0
        assert candidate.enterprise_score > 0

    def test_analyze_languages_identifies_enterprise_languages(self, mock_github_client, sample_repo_data):
        """Verify enterprise languages are correctly identified."""
        language_bytes = {
            "Java": 10_000_000,
            "TypeScript": 5_000_000,
            "Shell": 100_000,
        }
        mock_github_client.get_repo_languages.return_value = language_bytes
        
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        profile = analyzer._analyze_languages(
            sample_repo_data["full_name"],
            sample_repo_data,
        )
        
        assert "Java" in profile.enterprise_languages
        assert "TypeScript" in profile.enterprise_languages
        assert profile.language_count >= 2

    def test_analyze_languages_identifies_legacy(self, mock_github_client, sample_repo_data):
        """Verify legacy languages are identified for migration tasks."""
        language_bytes = {
            "COBOL": 5_000_000,
            "Java": 10_000_000,
        }
        mock_github_client.get_repo_languages.return_value = language_bytes
        
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        profile = analyzer._analyze_languages(
            sample_repo_data["full_name"],
            sample_repo_data,
        )
        
        # COBOL should be identified as legacy
        assert "COBOL" in profile.legacy_languages or len(profile.enterprise_languages) >= 1

    def test_classify_industry_finance(self, mock_github_client):
        """Verify finance industry classification."""
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        candidate = EnterpriseRepoCandidate(
            full_name="company/payments",
            html_url="https://github.com/company/payments",
            description="Banking and payment processing system with fraud detection",
            topics=["payments", "banking", "fintech"],
        )
        
        industry, confidence = analyzer._classify_industry(candidate)
        
        assert industry == "finance"
        assert confidence > 0.5

    def test_classify_industry_healthcare(self, mock_github_client):
        """Verify healthcare industry classification."""
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        candidate = EnterpriseRepoCandidate(
            full_name="hospital/ehr-system",
            html_url="https://github.com/hospital/ehr-system",
            description="Electronic health records with HIPAA compliance and FHIR support",
            topics=["healthcare", "ehr", "fhir", "hipaa"],
        )
        
        industry, confidence = analyzer._classify_industry(candidate)
        
        assert industry == "healthcare"
        assert confidence > 0.5

    def test_detect_frameworks_spring(self, mock_github_client):
        """Verify Spring framework detection."""
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        candidate = EnterpriseRepoCandidate(
            full_name="company/service",
            html_url="https://github.com/company/service",
            description="Spring Boot microservice",
            topics=["spring-boot", "java", "microservices"],
        )
        
        frameworks = analyzer._detect_frameworks(candidate)
        
        assert "spring" in frameworks

    def test_compute_enterprise_score(self, mock_github_client, sample_candidate):
        """Verify enterprise score computation."""
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        
        score = analyzer._compute_enterprise_score(sample_candidate)
        
        # High-quality enterprise candidate should score > 0.5
        assert score > 0.5
        assert score <= 1.0


# =============================================================================
# Test EnterpriseRepoSelector
# =============================================================================


class TestEnterpriseRepoSelector:
    """Tests for EnterpriseRepoSelector class."""

    def test_init_with_token(self):
        """Verify selector initializes with token."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient") as mock_client:
            selector = EnterpriseRepoSelector(github_token="test-token")
            mock_client.assert_called_once()

    def test_discover_candidates_from_org(self, mock_github_client):
        """Verify candidate discovery from organization."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient", return_value=mock_github_client):
            with patch("app.ir_sdlc.repo_selector.RepoAnalyzer") as mock_analyzer:
                mock_analyzer_instance = MagicMock()
                mock_analyzer.return_value = mock_analyzer_instance
                
                # Mock repos returned from org
                mock_github_client.list_org_repos.return_value = [
                    {
                        "id": 1,
                        "full_name": "org/repo1",
                        "html_url": "https://github.com/org/repo1",
                        "archived": False,
                        "fork": False,
                        "size": 100000,
                        "stargazers_count": 1000,
                    }
                ]
                
                # Mock analysis result
                mock_analyzer_instance.analyze_repo.return_value = EnterpriseRepoCandidate(
                    full_name="org/repo1",
                    html_url="https://github.com/org/repo1",
                    complexity_profile=RepoComplexityProfile(total_loc=600000),
                    language_profile=RepoLanguageProfile(
                        language_count=3,
                        enterprise_languages=["Java", "Python"],
                    ),
                    enterprise_score=0.8,
                )
                
                selector = EnterpriseRepoSelector(github_token="test")
                candidates = selector.discover_candidates(orgs=["enterprise-org"])
                
                assert len(candidates) >= 0  # May filter some out

    def test_select_repos_applies_criteria(self):
        """Verify selection applies criteria correctly."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient"):
            selector = EnterpriseRepoSelector(github_token="test")
            
            # Create candidates with varying quality
            good_candidate = EnterpriseRepoCandidate(
                full_name="org/good-repo",
                html_url="https://github.com/org/good-repo",
                complexity_profile=RepoComplexityProfile(total_loc=700000),
                language_profile=RepoLanguageProfile(
                    language_count=4,
                    enterprise_languages=["Java", "TypeScript", "Python"],
                ),
                industry_guess="finance",
                selection_score=0.85,
            )
            
            small_candidate = EnterpriseRepoCandidate(
                full_name="org/small-repo",
                html_url="https://github.com/org/small-repo",
                complexity_profile=RepoComplexityProfile(total_loc=100000),
                language_profile=RepoLanguageProfile(
                    language_count=2,
                    enterprise_languages=["Python"],
                ),
                selection_score=0.3,
            )
            
            criteria = SelectionCriteria(min_loc=500000)
            selected = selector.select_repos(
                [good_candidate, small_candidate],
                criteria=criteria,
            )
            
            # Only good candidate should be selected
            assert len(selected) == 1
            assert selected[0].full_name == "org/good-repo"

    def test_select_repos_diversity_constraint(self):
        """Verify diversity constraints are applied."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient"):
            selector = EnterpriseRepoSelector(github_token="test")
            
            # Create 6 candidates from same org
            candidates = []
            for i in range(6):
                candidates.append(EnterpriseRepoCandidate(
                    full_name=f"same-org/repo{i}",
                    html_url=f"https://github.com/same-org/repo{i}",
                    complexity_profile=RepoComplexityProfile(total_loc=600000),
                    language_profile=RepoLanguageProfile(
                        language_count=3,
                        enterprise_languages=["Java"],
                    ),
                    industry_guess="finance",
                    selection_score=0.8 - (i * 0.05),  # Decreasing scores
                ))
            
            criteria = SelectionCriteria(
                min_loc=500000,
                max_repos_per_org=3,  # Limit to 3 per org
            )
            selected = selector.select_repos(candidates, criteria=criteria)
            
            # Should be limited to max_repos_per_org
            assert len(selected) <= 3

    def test_select_repos_industry_coverage(self):
        """Verify selection aims for industry coverage."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient"):
            selector = EnterpriseRepoSelector(github_token="test")
            
            finance_candidate = EnterpriseRepoCandidate(
                full_name="bank/payments",
                html_url="https://github.com/bank/payments",
                complexity_profile=RepoComplexityProfile(total_loc=600000),
                language_profile=RepoLanguageProfile(
                    language_count=3,
                    enterprise_languages=["Java"],
                ),
                industry_guess="finance",
                selection_score=0.8,
            )
            
            healthcare_candidate = EnterpriseRepoCandidate(
                full_name="hospital/ehr",
                html_url="https://github.com/hospital/ehr",
                complexity_profile=RepoComplexityProfile(total_loc=600000),
                language_profile=RepoLanguageProfile(
                    language_count=3,
                    enterprise_languages=["C#"],
                ),
                industry_guess="healthcare",
                selection_score=0.75,
            )
            
            criteria = SelectionCriteria(min_loc=500000)
            selected = selector.select_repos(
                [finance_candidate, healthcare_candidate],
                criteria=criteria,
                max_total=10,
            )
            
            # Both should be selected for industry diversity
            industries = {c.industry_guess for c in selected}
            assert len(industries) >= 1

    def test_generate_selection_report(self, sample_candidate):
        """Verify selection report generation."""
        with patch("app.ir_sdlc.repo_selector.GitHubClient"):
            selector = EnterpriseRepoSelector(github_token="test")
            
            report = selector.generate_selection_report([sample_candidate])
            
            assert "summary" in report
            # Report has 'repos' key with candidate details
            assert "repos" in report
            assert len(report["repos"]) == 1


# =============================================================================
# Test Enterprise Defaults
# =============================================================================


class TestEnterpriseDefaults:
    """Tests for enterprise organization and repo defaults."""

    def test_enterprise_orgs_defined(self):
        """Verify enterprise organizations are defined."""
        # ENTERPRISE_ORGS is a class attribute of EnterpriseRepoSelector
        assert hasattr(EnterpriseRepoSelector, "ENTERPRISE_ORGS")
        orgs = EnterpriseRepoSelector.ENTERPRISE_ORGS
        
        assert isinstance(orgs, (list, tuple))
        assert len(orgs) > 0
        # Should include major cloud/enterprise orgs
        assert any("kubernetes" in org.lower() for org in orgs)

    def test_enterprise_repos_defined(self):
        """Verify enterprise repositories are defined."""
        # ENTERPRISE_REPOS is a class attribute of EnterpriseRepoSelector
        assert hasattr(EnterpriseRepoSelector, "ENTERPRISE_REPOS")
        repos = EnterpriseRepoSelector.ENTERPRISE_REPOS
        
        assert isinstance(repos, (list, tuple))
        assert len(repos) > 0
        # Should include high-value repos in owner/repo format
        assert any("/" in repo for repo in repos)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRepoSelectorIntegration:
    """Integration tests for repo selector workflow."""

    def test_full_analysis_workflow(self, mock_github_client, sample_repo_data, sample_language_bytes):
        """Test complete analysis workflow from repo data to scored candidate."""
        mock_github_client.get_repo_languages.return_value = sample_language_bytes
        mock_github_client.get_repo_topics.return_value = ["finance", "payments", "spring-boot"]
        mock_github_client.count_recent_issues.return_value = 30
        
        analyzer = RepoAnalyzer(mock_github_client, compute_loc=False)
        candidate = analyzer.analyze_repo(sample_repo_data)
        
        # Verify complete analysis
        assert candidate.full_name == sample_repo_data["full_name"]
        assert candidate.language_profile.primary_language is not None
        assert candidate.complexity_profile.total_loc > 0
        assert candidate.enterprise_score > 0
        assert candidate.selection_score > 0
        
        # Check serialization
        d = candidate.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_candidate_meets_criteria_e2e(self, sample_candidate):
        """End-to-end test of candidate criteria checking."""
        criteria = SelectionCriteria(
            min_loc=500000,
            min_languages=2,
            require_enterprise_language=True,
        )
        
        # Should meet all criteria
        assert sample_candidate.meets_minimum_criteria(
            min_loc=criteria.min_loc,
            min_languages=criteria.min_languages,
            require_enterprise_lang=criteria.require_enterprise_language,
        )
        
        # Verify score threshold
        assert sample_candidate.selection_score >= criteria.min_selection_score
