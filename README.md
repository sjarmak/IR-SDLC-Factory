# 🔍 IR-SDLC-Factory

**Information Retrieval Evaluation for Enterprise SDLC Tasks**

A comprehensive benchmark framework for evaluating **information retrieval (IR) tools** on **software development lifecycle (SDLC) tasks** across enterprise-scale codebases. Fully compatible with the [Harbor](https://harborframework.com) evaluation framework.

---

## 📚 Documentation

- [ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture and component overview
- [DEVELOPMENT.md](docs/DEVELOPMENT.md): Development, testing, and extension guide
- [MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md): Model Context Protocol (MCP) integration instructions

See also the [docs/](docs/) directory for additional documentation.

## ✨ Key Features

- **Enterprise-Scale Repositories**: Target repos with 100K+ files across Python, JavaScript, TypeScript, Java, Go, Rust, and C++
- **10 SDLC Task Types**: Bug triage, code review, dependency analysis, architecture understanding, security audit, refactoring, test coverage, documentation linking, and more
- **Standard IR Metrics**: Precision@K, Recall@K, MRR, NDCG, MAP, F1@K, plus SDLC-specific metrics
- **Harbor Framework Compatible**: Generate tasks compatible with [Harbor](https://harborframework.com) evaluation harness
- **Extensible IR Tool Interface**: Easy integration of new IR tools for benchmarking

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/sjarmak/IR-SDLC-Factory.git
cd IR-SDLC-Factory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Collect Enterprise-Scale Repositories

```bash
python -m app.main collect-repos \
    --languages "python,javascript,java" \
    --min-stars 1000 \
    --min-files 1000 \
    --output-file "repos/large_repos.jsonl"
```

### 2. Generate IR Tasks from Source Data

```bash
python -m app.main generate-tasks \
    --source-file "data/issues.jsonl" \
    --task-types "bug_triage,code_review,security_audit" \
    --repos-dir "./repos" \
    --output-file "ir_tasks.jsonl" \
    --dataset-name "my-ir-benchmark"
```

### 3. Run IR Evaluation

```bash
python -m app.main evaluate \
    --tasks-file "ir_tasks.jsonl" \
    --ir-tool "grep-baseline" \
    --repos-dir "./repos" \
    --output-dir "results/ir_eval" \
    --generate-harbor
```

### 4. Generate Harbor-Compatible Tasks

```bash
python -m app.main generate-harbor \
    --tasks-file "ir_tasks.jsonl" \
    --output-dir "./harbor_tasks" \
    --dataset-name "ir-sdlc-bench" \
    --generate-registry
```

### 5. Run with Harbor

```bash
harbor jobs start -p ./harbor_tasks/ir-sdlc-bench -a your-agent -m your-model
```

---

## 📊 Supported SDLC Task Types

| Task Type | Description | Ground Truth |
|-----------|-------------|--------------|
| `bug_triage` | Locate relevant files for bug reports | Files mentioned in fix |
| `code_review` | Find files requiring review | Changed files in PR |
| `dependency_analysis` | Identify dependency impact | Import graph analysis |
| `architecture_understanding` | Navigate codebase structure | Module/package hierarchy |
| `security_audit` | Find security-relevant code | Files with security patterns |
| `refactoring_analysis` | Locate refactoring targets | Call graph & dependencies |
| `test_coverage` | Find untested code | Coverage reports |
| `documentation_linking` | Match docs to code | Doc-code references |
| `api_discovery` | Locate API definitions | API endpoint files |
| `performance_analysis` | Find performance bottlenecks | Profiling-relevant code |

---

## 📈 IR Metrics

### Standard IR Metrics
- **Precision@K**: Fraction of retrieved items that are relevant
- **Recall@K**: Fraction of relevant items that are retrieved
- **F1@K**: Harmonic mean of Precision and Recall
- **Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Hit Rate@K**: Fraction of queries with at least one relevant result

### SDLC-Specific Metrics
- **File-level Recall**: Recall at the file granularity
- **Function-level Precision**: Precision at the function/method level
- **Cross-module Coverage**: Coverage of relevant modules
- **Context Efficiency**: Ratio of relevant tokens to total tokens

---

## 🔌 Integrating Your IR Tool

Create a custom IR tool by implementing the `IRToolInterface`:

```python
from app.ir_sdlc import IRToolInterface, IRTask, RetrievalResult

class MyIRTool(IRToolInterface):
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
        # Your retrieval logic here
        return [
            RetrievalResult(
                file_path="src/module.py",
                score=0.95,
                start_line=10,
                end_line=50,
            )
        ]

    def get_name(self) -> str:
        return "my-ir-tool"

# Register for CLI usage
from app.ir_sdlc import register_ir_tool
register_ir_tool("my-ir-tool", MyIRTool)
```

---

## 🐳 Harbor Integration

IR-SDLC-Factory generates fully Harbor-compatible task directories:

```
task_directory/
├── task.toml              # Task metadata
├── instruction.md         # Task description
├── ground_truth.json      # Expected retrieval results
├── environment/
│   └── Dockerfile         # Evaluation environment
├── solution/
│   └── solve.sh          # Reference solution
└── tests/
    ├── test.sh           # Evaluation script
    └── evaluate_retrieval.py  # Metric computation
```

### Harbor Task Format

The generated `task.toml` follows the Harbor specification:

```toml
version = "1.0"

[task]
name = "bug_triage_repo_issue_123"
type = "ir_retrieval"
description = "Locate relevant files for issue #123"

[task.metadata]
sdlc_task_type = "bug_triage"
repo_name = "owner/repo"
difficulty = "medium"
```

---

## 📁 Project Structure

```
IR-SDLC-Factory/
├── app/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── ir_sdlc/
│   │   ├── __init__.py
│   │   ├── data_structures.py     # Core data models
│   │   ├── task_types.py          # SDLC task type definitions
│   │   ├── metrics.py             # IR evaluation metrics
│   │   ├── harbor_adapter.py      # Harbor format generator
│   │   ├── ir_tool_interface.py   # Tool abstraction layer
│   │   └── evaluation_runner.py   # Evaluation orchestration
│   └── model/                     # LLM model interfaces
├── data_collection/
│   └── collect/
│       └── collect_large_repos.py # Repository collector
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🛠️ CLI Reference

### `evaluate` (alias: `eval`)
Evaluate an IR tool on benchmark tasks.

```bash
python -m app.main evaluate \
    --tasks-file <path>        # Required: JSONL file with IR tasks
    --ir-tool <name>           # IR tool name (default: grep-baseline)
    --repos-dir <path>         # Required: Directory with cloned repos
    --output-dir <path>        # Results directory (default: results/ir_eval)
    --primary-metric <metric>  # Primary metric (default: mrr)
    --generate-harbor          # Generate Harbor output (default: true)
    --num-workers <n>          # Parallel workers (default: 1)
```

### `generate-tasks` (alias: `gen`)
Generate IR tasks from repository data.

```bash
python -m app.main generate-tasks \
    --source-file <path>       # Required: Source data file
    --task-types <types>       # Comma-separated task types
    --repos-dir <path>         # Required: Directory with repos
    --output-file <path>       # Required: Output JSONL file
    --dataset-name <name>      # Dataset name
    --dataset-version <ver>    # Dataset version
```

### `generate-harbor` (alias: `harbor`)
Generate Harbor-compatible task directories.

```bash
python -m app.main generate-harbor \
    --tasks-file <path>        # Required: JSONL file with tasks
    --output-dir <path>        # Required: Output directory
    --dataset-name <name>      # Dataset name
    --author-name <name>       # Author name for metadata
    --author-email <email>     # Author email
    --generate-registry        # Generate registry entry
    --git-url <url>            # Git URL for registry
```

### `collect-repos` (alias: `collect`)
Collect large enterprise-scale repositories from GitHub.

```bash
python -m app.main collect-repos \
    --languages <langs>        # Comma-separated languages
    --min-stars <n>            # Minimum stars (default: 1000)
    --min-files <n>            # Minimum files (default: 1000)
    --output-file <path>       # Output file
    --max-repos <n>            # Max repos per language
    --github-token <token>     # GitHub API token
```

---

## 🙏 Acknowledgements

This project is adapted from and builds upon [SWE-Factory](https://github.com/SWE-Factory/SWE-Factory), an automated factory for GitHub Issue Resolution Training Data and Evaluation Benchmarks.

If you use this work, please consider citing the original SWE-Factory paper:

```bibtex
@article{guo2025swefactory,
  title={SWE-Factory: Your Automated Factory for Issue Resolution Training Data and Evaluation Benchmarks},
  author={Lianghong Guo and Yanlin Wang and Caihua Li and Pengyu Yang and Jiachi Chen and Wei Tao and Yingtian Zou and Duyu Tang and Zibin Zheng},
  journal={arXiv preprint arXiv:2506.10954},
  year={2025},
  url={https://arxiv.org/abs/2506.10954},
}
```

We also acknowledge the following foundational works:
- [Harbor Framework](https://harborframework.com) - Evaluation harness for AI coding agents
- [SWE-bench](https://arxiv.org/abs/2310.06770) - Software engineering benchmark
- [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym/) - Repository to environment framework

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
