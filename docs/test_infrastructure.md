"""
docs/test_infrastructure.md

# Test Infrastructure for Benchmark Evaluation

This document describes the fixtures and utilities provided for testing benchmark pipeline components in this repository.

## Location
- All fixtures/utilities are defined in `tests/conftest.py`

## Provided Fixtures
- `dummy_harbor_registry_config`: Returns a dummy config dict for harbor_registry tests.
- `dummy_observability_context`: Returns a dummy context dict for observability tests.
- `dummy_swebench_input`: Returns a dummy input dict for swebench_adapter tests.
- `dummy_registry_client`: Returns a dummy registry client object (uses dummy_harbor_registry_config).
- `dummy_observability_logger`: Returns a dummy logger object (uses dummy_observability_context).
- `dummy_swebench_adapter`: Returns a dummy swebench adapter object (uses dummy_swebench_input).

## Usage Example
In your test file:

```python
import pytest

def test_registry(dummy_registry_client):
    assert dummy_registry_client.is_connected()

def test_logger(dummy_observability_logger):
    dummy_observability_logger.log("msg")
    assert dummy_observability_logger.logs
```

## Purpose
These fixtures/utilities enable isolated, reliable tests for:
- `harbor_registry` (mock registry config/client)
- `observability` (mock context/logger)
- `swebench_adapter` (mock input/adapter)

They are reusable in any test module by simply requesting the fixture as a test argument.
