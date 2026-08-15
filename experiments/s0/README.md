# S0 runner area

This directory is reserved for executable S0 experiment drivers and frozen configuration files.

The normative experiment definition lives in [`research/s0/EXPERIMENT.md`](../../research/s0/EXPERIMENT.md).

Planned sequence:

1. deterministic smoke run with a fake provider;
2. hosted/local provider adapter;
3. calibration task manifest;
4. empirical matched-random decision pool;
5. pilot runner with JSONL receipts;
6. frozen confirmatory runner and analysis.

Do not place credentials, checkpoints, raw private data, or generated run artifacts in git.
