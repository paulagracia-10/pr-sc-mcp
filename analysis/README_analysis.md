# Analysis Files

This folder contains the analysis artifacts used to compute the main reported tables, confidence intervals, and Figure 2 in the paper.

## Purpose

The analysis layer connects raw validation outputs to the paper's reported results.

It supports:

- Table 5: Live Validation mean total tokens and latency;
- Table 6: Full PR-SC MCP efficiency deltas versus the Exhaustive MCP Baseline;
- Table 7: run-level confidence intervals;
- Figure 2: token and latency deltas across three models;
- quality and robustness summaries.

## Expected inputs

The script `reproduce_tables_and_figures.py` assumes that the Live Validation folders are organised as follows:

```text
results/live_validation/gpt_oss_120b/run_summaries/
results/live_validation/gemma_4_31b/run_summaries/
results/live_validation/zai_glm_4_7/run_summaries/
```

Each `run_summaries/` folder should contain the five final CSV batches used in the paper.

## Key output files

Recommended analysis outputs:

```text
live_validation_table5_model_variant_summary.csv
live_validation_table6_full_vs_exhaustive_delta.csv
statistical_ci_run_level_differences.csv
figure2_full_vs_exhaustive_mcp.png
```

## Naming note

If older filenames include `naive`, interpret this as the baseline later renamed in the paper:

```text
Naive MCP -> Exhaustive MCP Baseline
```

## Statistical approach

The primary uncertainty analysis uses paired run-level aggregation. Individual generations are not treated as fully independent replicates because they are nested within fixed case definitions, repeated run batches, and architecture variants.


## Reproducing the analysis

From the repository root:

```bash
python analysis/reproduce_tables_and_figures.py
```

The script writes outputs into the `analysis/` folder.
