"""
conftest.py for benchmark evaluation test infrastructure.
Provides fixtures and utilities for testing benchmark pipeline components:
- harbor_registry
- observability
- swebench_adapter
"""
import pytest

# Example: Dummy config or data for harbor_registry
@pytest.fixture
def dummy_harbor_registry_config():
    return {
        "url": "http://localhost:5000",
        "username": "testuser",
        "password": "testpass"
    }

# Example: Dummy observability context
@pytest.fixture
def dummy_observability_context():
    return {
        "trace_id": "test-trace-id",
        "span_id": "test-span-id"
    }

# Example: Dummy swebench_adapter input
@pytest.fixture
def dummy_swebench_input():
    return {
        "repo": "example/repo",
        "commit": "abcdef1234567890",
        "task_id": "task-001"
    }

# Utility: Mock a simple registry client
class DummyRegistryClient:
    def __init__(self, config):
        self.config = config
        self.connected = False
    def connect(self):
        self.connected = True
    def is_connected(self):
        return self.connected

@pytest.fixture
def dummy_registry_client(dummy_harbor_registry_config):
    client = DummyRegistryClient(dummy_harbor_registry_config)
    client.connect()
    return client

# Utility: Observability logger stub
class DummyObservabilityLogger:
    def __init__(self, context):
        self.context = context
        self.logs = []
    def log(self, message):
        self.logs.append((self.context, message))

@pytest.fixture
def dummy_observability_logger(dummy_observability_context):
    return DummyObservabilityLogger(dummy_observability_context)

# Utility: Swebench adapter stub
class DummySwebenchAdapter:
    def __init__(self, input_data):
        self.input_data = input_data
    def run(self):
        return {"status": "success", "input": self.input_data}

@pytest.fixture
def dummy_swebench_adapter(dummy_swebench_input):
    return DummySwebenchAdapter(dummy_swebench_input)
