# PR-SC MCP Architecture Validation Runner

This folder contains a runner for the PR-SC MCP **Architecture Validation** stage.

## Methodological boundary

This runner is **not** a live LLM evaluation and does **not** call live MCP servers. It implements a controlled / semi-real benchmark harness that estimates whether each architecture variant selects and retains sufficient evidence to support the expected answer.

It is intended to accompany the study as a reproducibility asset for the architecture-validation stage.

## Benchmark scope

- 24 controlled / semi-real benchmark cases
- 8 architecture variants
- 192 deterministic case × variant runs
- Tool domains: `metrics`, `logs`, `traces`, `github`, `topology`, `docs`, `tickets`, `security`
- Failure modes: `timeout`, `partial_response`, `malformed_json`, `stale_data`, `conflicting_evidence`, `overlong_output`, `duplicated_records`, `missing_required_tool`, `rate_limited`, `schema_drift`, `irrelevant_high_confidence_signal`, `mixed_granularity`

## Architecture variants

1. Naive MCP
2. Direct Long-Context
3. Compression-only MCP
4. Traditional RAG
5. Routing MCP
6. Routing + Compression, no packaging
7. Routing + Packaging, no compression
8. Full PR-SC MCP

## How to run

```bash
python pr_sc_mcp_architecture_validation_runner.py --out-dir architecture_validation_outputs
```

Optional quiet mode:

```bash
python pr_sc_mcp_architecture_validation_runner.py --out-dir architecture_validation_outputs --quiet
```

No external Python packages are required.

## Outputs

The runner writes:

- `architecture_validation_run_records.json`
- `architecture_validation_run_summary.csv`
- `architecture_validation_summary_by_variant.csv`
- `architecture_validation_case_summary.csv`
- `architecture_validation_manifest.json`

