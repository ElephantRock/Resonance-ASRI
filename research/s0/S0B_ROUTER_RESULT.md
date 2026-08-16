# ASRI S0-B Router Characterization Result — frozen `adaptive-heuristic-v0`

**Verdict: the frozen controller is perfectly safe but perfectly unhelpful on this
distribution.** It avoided 100% of harm, dodged the cost spiral entirely, and spent only
21% more tokens than shallow — but realized **zero of the four available rescues**, so
its quality is byte-for-byte shallow's (mean Q 0.694). This is characterization data for
designing S0-C, not evidence for or against the ASRI claim.

Run: 2026-08-16 · branch `research/asri-s0` @ `968052b` · artifacts:
[`artifacts/s0b_router/`](artifacts/s0b_router/) — `analysis.json`, `matched_random_pool.json`,
`receipts.jsonl` (36), `environment.json`. Same frozen substrate and envelope as S0-B.

## Provenance (why this freeze is defensible)

`adaptive-heuristic-v0` was created in scaffold commit `e7eb10e` and last touched by the
Ruff import fix `896df71` — both before any S0-B calibration measurement existed. It was
**not modified using S0-B outcomes** and is frozen prospectively for the adaptive
evaluation. Its apparent alignment (or misalignment) with the observed class gradient is
an observation, not a tuning decision. The characterization runner recomputes the
controller's features from the policy's own constants and asserts consistency with its
actual decisions on every task (no drift; all 36 tasks passed the check).

## Routing confusion matrix (controller routing vs empirical value label)

| | extra compute useful (HELP) | not useful |
|---|---|---|
| **routed DEEP** | 2 (routing TP) | 4 (routing FP) |
| **routed SHALLOW** | 2 (routing FN) | 28 (routing TN) |

Three-state crosstab: deep-routed → {HELP 2, NEUTRAL 4, **HARM 0**}; shallow-routed →
{HELP 2, NEUTRAL 26, HARM 2}.

**Realized-rescue correction (the matrix above overstates success):** routing at a task
is not solving it. Both deep-routed HELP tasks (`code-02`, `code-03`) received only
depth-2 refinement and **still scored 0** — the B1 rescues required the full 7-call path
(refine ×3 + specialist critique/synthesis + verify). Realized rescues: **0 of 4**.
Realized harm: **0 of 2 possible**. Net per-task outcome: `adaptive_quality ==
shallow_quality` on all 36 tasks — nothing broken, nothing gained.

## The three routing questions

1. **Helps routed deep enough?** 2 of 4 aimed at (`code-02/03`, via the multiline
   feature — all six code tasks were the only deep-routed tasks); `arith-04`/`word-06`
   routed shallow. And "deep" here means depth 2, which is insufficient: the rescues
   live at specialist/verify depth this heuristic never allocates on this set (max
   observed controller score = 1).
2. **Harm cases routed shallow?** Yes, 2 of 2 (`instr-03`, `logic-06` stayed Q = 1.0 at
   50/97 tokens). Deep-routed set contained zero HARM-label tasks.
3. **`logic-04` spiral?** Fully avoided: routed shallow, depth 1, **80 tokens / 1.4 s**
   versus B1's 1,405 tokens / 44.7 s failure spiral.

## Empirical decision distribution (the matched-random pool)

```text
iter=1; specialists=none; verify=no   30/36
iter=2; specialists=none; verify=no    6/36
```

The only feature that fired on this task set is `multiline` (the six code-snippet
prompts). No allocation ever reached iteration 3+, specialists, memory, or verification.

Pool persisted in `artifacts/s0b_router/matched_random_pool.json`:
**36 decisions, sampling seed 20260816, pool sha256 `d7f270cc8c146fa6…`** (full hash in
the artifact). S0-C rule: exact allocation-count matching — shuffle this fixed 36-item
vector without replacement so adaptive and random spend identical logical allocation
counts by construction; actual token accounting remains the primary cost measure.

## Economics (calibration set)

| Condition | Mean Q | Mean tokens | Mean latency |
|---|---|---|---|
| B0 shallow | 0.694 | 66 | 0.6 s |
| **adaptive-v0** | **0.694** | **80** | **0.8 s** |
| B1 deep | 0.750 | 631 | 9.8 s |

## Reading and S0-C design implications

1. **Safety properties are real**: no harm realized, no spiral, +21% token overhead over
   shallow. As a *cost containment* policy v0 already works on this distribution.
2. **Recovery capability is absent**: the value in this task set sits at
   specialist/verify depth; the frozen feature ladder (short prompts, plain classes)
   never climbs there. On held-out tasks with similar surface statistics, adaptive-v0
   and matched-random will be quality-equivalent by construction — **S0-C's held-out
   manifest must include tasks whose prompts actually trigger the deeper rungs**
   (long/multiline/marker-rich instances), or the pilot cannot distinguish anything.
3. The matched-random pool contains no specialist/verify allocations, so the S0-C
   comparison family is "shallow vs shallow+retry" on both arms. That is a faithful test
   of *this* frozen controller — and an honest baseline expectation of ≈0 quality delta
   on shallow-surfaced task sets.
4. A learned/next controller faces a concrete, quantified bar: recover part of the
   4/36 rescue rate (11.1%) while preserving the 0/36 harm rate and near-shallow cost.

## Limitations

- Characterization on 36 calibration tasks only; not a confirmatory comparison.
- Depth-2 insufficiency is confounded with task difficulty: code-02/03 might also fail
  at depth 2 under matched-random; that is exactly what S0-C's paired design will test.
- The heuristic's `task_type` feature never fired because calibration class names
  (`arithmetic`, `code_reasoning`, …) are not in its difficult-set vocabulary
  (`math`, `code`, `planning`, `research`); on S0-C manifests the caller controls
  task_type strings, so this mapping must be frozen explicitly in the manifest design.
