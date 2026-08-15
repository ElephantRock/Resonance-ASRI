# Research Program

## Core question

Can heterogeneous, adaptive computation produce more useful work per unit resource than uniform computation?

The primary quantity of interest is not raw benchmark score. It is the quality-cost frontier under explicit resource accounting.

## Experimental stages

### S0 — System-level adaptive compute

No weight modification. Establish the causal value of conditional computation using provider-backed reasoning paths, optional memory, specialists, iterative passes, verification, and telemetry.

Required controls:

- shallow fixed path;
- deep fixed path;
- random matched-budget policy;
- adaptive policy;
- oracle policy where feasible.

Primary hypothesis:

> At matched average resource use, an adaptive heterogeneous policy improves task quality or solved-task rate over fixed and random allocation.

### S1 — Trainable recurrence

Introduce a small open-weight model and a reusable recurrent reasoning block. First establish whether additional recurrent depth can improve difficult cases without degrading easy cases.

### S2 — Learned compute control

Learn halting/depth/resource allocation. Compare against the S0/S1 controls at matched compute.

### S3 — Sparse specialists

Introduce adapter or expert specialization. Measure specialization, routing entropy, expert collapse, load balance, and incremental capability per active parameter.

### S4 — Failure-directed distillation

Use a stronger teacher primarily where the student fails or is uncertain. Compare capability gain per teacher token and per training FLOP against uniform distillation.

### S5 — Integrated ASRI

Integrate only mechanisms with positive isolated evidence. Re-run ablations because mechanisms may interact destructively.

## Measurement contract

Every experiment should record, where available:

- task outcome / quality score;
- input and output tokens;
- model/provider calls;
- reasoning iterations;
- specialist activations;
- retrieval operations;
- verifier operations;
- tool calls;
- wall-clock latency;
- GPU/CPU memory for local runs;
- FLOPs or a documented proxy;
- monetary cost where applicable;
- artifact/config/code identifiers.

## Evidence discipline

Exploratory calibration and confirmatory runs are separate. A mechanism advances only when its claim is bounded, falsifiable, reproducible, and supported by controls that isolate the mechanism rather than merely demonstrating an integrated system.
