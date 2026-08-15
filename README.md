# Resonance ASRI

**Adaptive Sparse Recurrent Intelligence (ASRI)** is a research program for building compute-efficient AI systems that allocate expensive computation only when a problem demonstrates that it needs it.

This repository is the neural and runtime research laboratory for ASRI. It is intentionally separate from [`ElephantRock/Resonance-World`](https://github.com/ElephantRock/Resonance-World): ASRI owns model/runtime mechanisms and training artifacts; Resonance World owns higher-level capability governance, experimental integration, provenance, composition, and world-scale evaluation.

## Research thesis

ASRI tests whether a system composed of a compact core, adaptive recurrent reasoning, sparse specialization, memory, tools, verification, and learned compute allocation can improve **useful work per unit compute** relative to uniform-compute baselines.

The governing rule is:

> Never perform expensive computation unless the current problem demonstrates that it needs it.

The primary research objective is the quality-cost frontier, not parameter count.

## Experimental progression

- **S0 — System-level adaptive compute:** external controller, iterative reasoning, specialist paths, memory, conditional verification, and complete compute telemetry. No model-weight surgery.
- **S1 — Trainable recurrence:** small open-weight model with a reusable recurrent reasoning block and fixed-depth controls.
- **S2 — Learned compute control:** learned controller/halting policy against fixed, random matched-cost, oracle, always-shallow, and always-deep controls.
- **S3 — Sparse specialists:** trainable adapter/expert specialization with explicit routing and load/specialization measurements.
- **S4 — Failure-directed distillation:** teacher supervision concentrated on student failures and uncertainty.
- **S5 — Integrated ASRI:** combine only mechanisms that survived isolated validation.

## Local hardware target

The initial research node is designed to be practical on a single **RTX 3080 Ti 12 GB** class GPU. S0 intentionally requires no local weight training. S1+ should prefer small open models, frozen-base methods, parameter-efficient training, quantization where scientifically appropriate, and cloud GPU bursts only for confirmatory scaling experiments.

## Repository boundary

```text
Resonance-ASRI
  owns: architecture, runtime, routing, recurrence, experts, memory interface,
        verification, training, distillation, model artifacts, compute telemetry

Resonance-World
  owns: capability contracts, provenance, experiment governance, composition,
        resource allocation, world integration, cross-system evaluation
```

The integration boundary is artifact- and receipt-based. Resonance World should not import ASRI tensor/training internals.

## Development rules

1. Every adaptive mechanism must have a matched-cost non-adaptive control.
2. Routing is a hypothesis, not an assumed benefit.
3. Report quality and resource use together.
4. Keep private reasoning state private; export metrics, artifacts, manifests, and reproducible evidence.
5. Separate exploratory calibration from confirmatory experiments.
6. Prefer small falsifiable experiments over integrated demonstrations that cannot identify causal mechanisms.

## Status

Repository bootstrap is complete on `main`. The first implementation line is `research/asri-s0`.
