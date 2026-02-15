# IR Benchmark Development Guide

This guide provides instructions for developing, testing, and extending the IR SDLC Benchmark system.

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests:**
   ```bash
   pytest tests/
   ```
3. **Run a benchmark:**
   ```bash
   python app/main.py --benchmark benchmarks/ir-sdlc-advanced-reasoning.jsonl
   ```

## Code Structure

- **Core logic:** `app/ir_sdlc/`
- **Adapters:** Integrations for external tools and datasets
- **Models:** LLM and agent backends in `app/model/`
- **Scripts:** Utilities for data generation and ablation studies in `scripts/`
- **Tests:** All new features must include tests in `tests/`

## Adding a New Agent or Model

1. Implement the agent/model in `app/model/` or as an adapter in `app/ir_sdlc/`.
2. Register the new agent/model in `app/model/register.py`.
3. Add tests in `tests/`.

## Adding a New Benchmark

1. Create a new JSONL file in `benchmarks/` following the existing schema.
2. Update scripts or pipeline as needed to support new task types.

## Testing & Quality

- All code changes must include unit tests.
- Use real implementations for tests (no mocks unless required).
- Run `pytest tests/` before submitting changes.

## Documentation

- Update `README.md` and `docs/` for any new features or changes.
- Follow the design principles in `AGENTS.md`.

---
