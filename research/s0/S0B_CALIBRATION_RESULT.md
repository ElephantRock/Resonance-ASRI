# ASRI S0-B Calibration Result — B0 fixed-shallow vs B1 fixed-deep

**Verdict: a value-of-compute gradient exists and is selective.** Extra computation helps
three classes, damages two tasks that shallow already solved, and cannot rescue a hard
core of tasks at any tested depth — while costing 9.5× tokens and 15.7× wall-clock latency
on average. This satisfies the gate for freezing the adaptive heuristic and constructing
the matched-random pool from its empirical decision distribution. No claim is made here
about adaptive-vs-random; that comparison is S0-C.

Run: 2026-08-16 14:20–14:5x UTC · branch `research/asri-s0` @ `774758f` (working tree
contained the runner keyword-arg fix, uncommitted at run time) · frozen substrate per
[PHASE0_RESULT.md](PHASE0_RESULT.md), envelope per [PHASE0B2_RESULT.md](PHASE0B2_RESULT.md).
Artifacts: [`artifacts/s0b/`](artifacts/s0b/) — `manifest.json` (36 tasks,
sha256 `7bfd0112d60bdbf8…`), `environment.json`, `receipts.jsonl` (72 receipts via
`JsonlReceiptLedger`), `analysis.json`.

## Setup

36 tasks × 2 conditions, one process, model loaded once. `b0-fixed-shallow` =
`fixed-shallow-v0` (1 call); `b1-fixed-deep` = `fixed-deep-v0` (4 reasoning iterations +
1 specialist critique/synthesis + 1 verification = 7 calls). Standard `ASRIRuntime`
through `QwenLocalProvider` (greedy, non-thinking, ≤ 256 output tokens per call).
Deterministic evaluators only (exact / numeric / option / keywords). Cost accounting:
C_tokens = input+output; C_calls = provider calls; C_latency = summed seconds. Residency
guard checked after every task; never triggered (all runs resident).

## Four populations (per task, ΔQ = Q_deep − Q_shallow)

| Population | Count | Members |
|---|---|---|
| shallow already solves (Q_shallow = 1) | 25 | includes 2 tasks deep **damaged** (below) |
| compute helps (ΔQ > 0) | 4 | arith-04, code-02, code-03, word-06 |
| compute does nothing (ΔQ = 0, unsolved) | 7 | arith-06, code-01/04/05/06, logic-02, logic-04 |
| compute hurts (Q_shallow < 1, ΔQ < 0) | 0 | — |

**Deep damaged previously-correct answers on 2 of 25 shallow-solved tasks:**
`instr-03` (is 51 prime → YES/NO drifted) and `logic-06` (middle-seat answer flipped
during refinement). These sit inside "shallow already solves" but are the concrete
overthinking cost.

## Per-class aggregates

| Class | Q shallow | Q deep | ΔQ | Δtokens | Δlatency |
|---|---|---|---|---|---|
| arithmetic | 0.67 | 0.83 | +0.17 | 479 | 6.2 s |
| code_reasoning | 0.00 | 0.33 | +0.33 | 643 | 12.3 s |
| word_problem | 0.83 | 1.00 | +0.17 | 767 | 16.8 s |
| lookup | 1.00 | 1.00 | 0.00 | 412 | 3.4 s |
| instruction_following | 1.00 | 0.83 | −0.17 | 412 | 3.8 s |
| reasoning | 0.67 | 0.50 | −0.17 | 677 | 12.6 s |

## Aggregate economics

- Mean quality: 0.694 (shallow) → 0.750 (deep), +0.056 over all 36 tasks.
- Mean C_tokens: 66 → 631 (**9.5×**) · mean C_calls: 1 → 7 · mean C_latency: 0.6 s →
  9.8 s (**15.7×**).
- Worst case: `logic-04` (two-dice probability) consumed 1,405 tokens and 44.7 s of deep
  compute and still failed — refinement can ramble without correcting.
- Selective-value signal: class-level (three classes improve, two degrade) and task-level
  (4 helps, 2 damages). Both levels carry routing information a controller could use.

## Reading (calibration, not confirmation)

1. Deep compute genuinely rescues specific arithmetic/code/word-problem failures — the
   value is real but concentrated: 4 tasks of 36.
2. Instruction-following and lookup are already at ceiling shallow; deep adds pure cost
   and occasionally damages format-sensitive correctness. A controller should route these
   shallow unconditionally.
3. A hard core (7 tasks) fails at every depth — mostly multi-step code tracing and two
   logic items. Extra iterations do not fix them on this substrate; they are cost sinks,
   and an ideal policy stops early on them.
4. Because deep's mean gain (+0.056) is small against its 9.5×/15.7× cost, uniform-deep
   is not a rational default — precisely the regime where selective allocation can win.

## Limitations

- n = 36, one deterministic pass per (task, condition). Greedy decoding makes reruns
  reproducible in principle, but there is no repetition-based variance estimate yet.
- Evaluators are strict (exact/numeric match); verbose-but-correct answers score 0
  equally in both conditions — fair for comparison, harsh in absolute terms.
- `logic-04`'s 44.7 s is one task; per-class latency means smooth over such outliers.
- Calibration data cannot be reused as confirmatory test material (per
  [EXPERIMENT.md](EXPERIMENT.md), S0-B data is excluded from the S0-C/S0-D sets).

## Gate decision

The step-5 inspection is satisfied: extra computation has empirically demonstrated
selective value on this task set. Proceed to: freeze `adaptive-heuristic-v0` routing
(reads: lookup/instruction-following shallow; arithmetic/word-problem/code deeper; early
stop tendency on repeated-failure tasks), run it on the calibration set to record its
empirical decision distribution, then construct the matched-random pool from that
distribution. The adaptive-vs-matched-random pilot (S0-C) starts only after all four
conditions are operationally stable.
