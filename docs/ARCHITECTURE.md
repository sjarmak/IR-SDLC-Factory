# IR Benchmark Architecture

This document describes the architecture of the IR (Information Retrieval) SDLC Benchmark system.

## Overview

The IR SDLC Benchmark is designed to evaluate and compare the performance of various information retrieval agents and tools across a standardized set of software engineering tasks. The system is modular, extensible, and supports integration with multiple LLMs, agent frameworks, and external tools.

## Key Components

- **Benchmark Pipeline**: Orchestrates the execution of benchmarks, manages task distribution, and collects results.
- **Task Types**: Defines the structure and requirements for different benchmark tasks (e.g., advanced reasoning, gap filling).
- **Adapters**: Integrate external systems (e.g., Harbor, Swebench) for data collection and agent execution.
- **Metrics & Evaluation**: Computes quantitative and qualitative metrics for agent performance.
- **Dashboard & Exporters**: Provides visualization and export of benchmark results.
- **Model Registry**: Manages available LLMs and agent backends.

## Data Flow

1. **Benchmark Definition**: Benchmarks are defined in JSONL files specifying tasks, metadata, and ground truth.
2. **Agent Execution**: Agents are invoked on each task, with results and telemetry collected.
3. **Evaluation**: Results are scored using automated and (optionally) human-in-the-loop metrics.
4. **Reporting**: Results are aggregated and exported for analysis.

## Extensibility

- New agent types, task types, and adapters can be added by implementing the appropriate interfaces in `app/ir_sdlc/`.
- Metrics and evaluation logic are modular and can be extended for new research needs.

## Directory Structure

- `app/ir_sdlc/`: Core logic, adapters, metrics, and evaluation code
- `benchmarks/`: Benchmark definitions
- `outputs/`: Results and reports
- `tests/`: Automated test suite
- `docs/`: Documentation

---
