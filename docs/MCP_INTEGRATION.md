# MCP Integration Guide

This document describes how to integrate the IR SDLC Benchmark system with Model Context Protocol (MCP) servers.

## Overview

MCP integration enables the IR benchmark to:
- Invoke LLMs and agents via MCP-compliant servers
- Collect telemetry and results in a standardized format
- Support distributed and scalable evaluation

## Integration Steps

1. **Configure MCP server connection:**
   - Set MCP server URL and credentials in the environment or config file.
2. **Implement MCP Adapter:**
   - Extend or use `ir_tool_interface.py` to communicate with MCP endpoints.
   - Ensure all required MCP methods are implemented (e.g., `run_task`, `get_status`).
3. **Register MCP Agents:**
   - Add MCP-based agents to the model registry in `app/model/register.py`.
4. **Run Benchmarks:**
   - Use the main pipeline to execute tasks via MCP agents.

## Example

```python
from app.ir_sdlc.ir_tool_interface import MCPAdapter
adapter = MCPAdapter(server_url="http://mcp-server:8000")
result = adapter.run_task(task)
```

## Notes

- Ensure MCP server is running and accessible before starting benchmarks.
- For advanced telemetry, extend `ir_telemetry.py` to capture additional signals.

---
