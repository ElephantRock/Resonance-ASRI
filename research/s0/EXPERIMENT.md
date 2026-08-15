# ASRI-S0 Experimental Protocol

## Question

Can request-conditioned allocation of heterogeneous inference resources improve task quality at matched average resource use compared with non-adaptive allocation?

S0 is a system-level experiment. It does **not** modify model weights and therefore must not be interpreted as evidence for trainable neural recurrence or learned sparse MoE behavior.

## Primary hypothesis

At matched average provider-call/token cost, the adaptive policy achieves a higher solved-task rate or quality score than a request-independent random allocation policy drawn from the same empirical compute-allocation distribution.

## Conditions

1. `fixed-shallow-v0` — one answer pass.
2. `fixed-deep-v0` — four answer/refinement passes, one specialist critique+synthesis, and verification.
3. `matched-random-v0` — request-independent sampling from the adaptive policy's empirical decision pool.
4. `adaptive-heuristic-v0` — transparent request-conditioned allocation.
5. Oracle allocation — offline analysis only where task difficulty labels or exhaustive counterfactual runs permit it.

The matched-random condition is the central causal control: it preserves the marginal allocation distribution while breaking the relationship between the request and the allocated computation.

## Stages

### S0-A — Harness validation

Use fake/local deterministic providers to establish execution semantics, receipt accounting, policy determinism, and artifact persistence. No capability claim is permitted.

### S0-B — Provider calibration

Run a small heterogeneous task set through all fixed paths. Estimate cost distributions, identify evaluator failures, and construct the empirical adaptive decision pool. Calibration data cannot enter the confirmatory test set.

### S0-C — Matched-cost pilot

Compare adaptive vs matched-random allocation on held-out tasks. Tune only infrastructure/evaluator defects; do not tune policy thresholds on confirmatory outcomes.

### S0-D — Confirmatory run

Freeze task set, policy versions, provider/model version, prompts, evaluator, random seeds, and analysis. Run the predeclared comparison once.

## Required measurements

Per request:

- task and split identifier;
- model/provider identifier;
- policy identifier;
- reasoning iterations;
- specialist activations;
- retrieval count;
- verifier count;
- provider calls;
- input/output tokens;
- latency;
- estimated monetary cost when available;
- quality/solved score;
- code/config revision.

Aggregate:

- solved-task rate / mean quality;
- provider calls per task;
- tokens per task;
- cost per solved task;
- latency distribution;
- quality-cost frontier;
- allocation distribution by task class/difficulty;
- bootstrap or randomization-based uncertainty for paired comparisons.

## Guardrails

- Adaptive routing is a hypothesis, not a default production policy.
- Hidden chain-of-thought is not logged or required for evaluation.
- Provider prompts request final answers, critiques, or revisions rather than private reasoning traces.
- A positive S0 result establishes the value of system-level conditional allocation only.
- S1 is required before making claims about recurrent model architecture.
