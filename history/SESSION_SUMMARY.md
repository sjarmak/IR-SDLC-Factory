# Session Summary: IR-SDLC-Factory (Dec 29, 2025)

## What Was Done
- Created and updated documentation: `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/MCP_INTEGRATION.md`, and updated `README.md` to reference these docs.
- Investigated and confirmed that benchmark JSONL files (e.g., `benchmarks/ir-sdlc-advanced-reasoning.jsonl`) target specific public repositories, but do not include a local `repos/` directory by default.
- Cloned all referenced repositories (`kubernetes`, `grafana`, `elasticsearch`, `vscode`) into `repos/` for evaluation.
- Attempted to run both the IR evaluation and Harbor export pipeline, but both failed due to missing `query` or `instruction` fields in the benchmark JSONL tasks.
- Patched the code to allow fallback to `instruction`, but the benchmark files still lack both fields.

## Next Steps
1. **Add Instruction Field to Benchmarks**
   - Update all tasks in your benchmark JSONL files to include an `instruction` field (recommended: use the `scenario` or `vague_prompt` as the value).
   - This is required for both IR evaluation and Harbor export.
   - You can automate this with a script that copies `scenario` or `vague_prompt` to `instruction` for each task.

2. **Re-run Harbor Export and Evaluation**
   - After updating the JSONL files, re-run:
     ```bash
     python -m app.main generate-harbor --tasks-file benchmarks/ir-sdlc-advanced-reasoning.jsonl --output-dir outputs/harbor_advanced_reasoning --dataset-name ir-sdlc-advanced-reasoning --generate-registry
     ```
   - And/or run the IR evaluation pipeline as needed.

3. **Verify Output**
   - Check that the output directories contain Harbor-compatible task files (task.toml, instruction.md, etc.).
   - Confirm that the evaluation pipeline runs without errors.

4. **(Optional) Automate Instruction Field Addition**
   - Write a Python script to patch all benchmark JSONL files in `benchmarks/` to add the `instruction` field if missing.

## Ready-to-Use Prompt for Next Session
Continue work on IR-SDLC-Factory: Patch all benchmark JSONL files to add an `instruction` field (using `scenario` or `vague_prompt`), then re-run Harbor export and evaluation. Verify that all tasks are Harbor-compatible and the evaluation pipeline works end-to-end.

---
