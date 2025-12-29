"""
test_benchmark_fixtures.py
Tests that the benchmark evaluation fixtures/utilities are usable and correct.
"""
import pytest

def test_dummy_harbor_registry_config(dummy_harbor_registry_config):
    assert isinstance(dummy_harbor_registry_config, dict)
    assert "url" in dummy_harbor_registry_config
    assert "username" in dummy_harbor_registry_config
    assert "password" in dummy_harbor_registry_config

def test_dummy_observability_context(dummy_observability_context):
    assert isinstance(dummy_observability_context, dict)
    assert "trace_id" in dummy_observability_context
    assert "span_id" in dummy_observability_context

def test_dummy_swebench_input(dummy_swebench_input):
    assert isinstance(dummy_swebench_input, dict)
    assert "repo" in dummy_swebench_input
    assert "commit" in dummy_swebench_input
    assert "task_id" in dummy_swebench_input

def test_dummy_registry_client(dummy_registry_client):
    assert hasattr(dummy_registry_client, "is_connected")
    assert dummy_registry_client.is_connected() is True

def test_dummy_observability_logger(dummy_observability_logger):
    dummy_observability_logger.log("test message")
    assert dummy_observability_logger.logs
    context, message = dummy_observability_logger.logs[0]
    assert "trace_id" in context
    assert message == "test message"

def test_dummy_swebench_adapter(dummy_swebench_adapter):
    result = dummy_swebench_adapter.run()
    assert result["status"] == "success"
    assert "input" in result
