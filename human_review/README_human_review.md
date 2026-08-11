# Human Semantic Review Materials

This folder contains the semantic review materials used to calibrate deterministic quality checks against human-reviewed answer quality.

## Purpose

The semantic review complements deterministic post-checks by evaluating whether generated answers are faithful, complete, causally disciplined, evidence-aware, and actionable.

## Review design

The review sample contains:

```text
3 models × 4 architecture variants × 5 case families = 60 responses
```

## Selected case families

The reviewed case families are:

- latency degradation;
- conflicting evidence;
- prompt injection in tool output;
- missing required tool;
- stale data.

These case families were selected because they are diagnostically important for evaluating MCP context governance.

##  files


```text
MMCP_human_review_sample_60 6.xlsx
README_human_review.md
```

```text
The following sheets are included in  MMCP_human_review_sample_60 6.xlsx

```
## Readme

Item and description

## Scoring rubric

Each response is scored from 1 to 5 on:

1. faithfulness;
2. completeness;
3. causal discipline;
4. evidence prioritization;
5. actionability.

## Review_Sample_60

60 cases to review by Human


## Blind_Review_Template

Blinf template to be reviewed by a Human expert


## Aggregate_By_Variant

Final results of the Human review

## Interpretation

The semantic review is a calibration layer, not a full manual evaluation of all live generations. Its purpose is to check whether deterministic evidence-use metrics are directionally aligned with human-reviewed semantic answer quality.
