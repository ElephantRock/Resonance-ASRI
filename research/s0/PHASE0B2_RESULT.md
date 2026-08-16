# ASRI Phase-0B2 Forced Generation-Length Residency Result

**Verdict: generation-side memory growth is negligible on this substrate; the envelope is
prompt-bound. No WDDM spill at any probed condition.** No ASRI performance claims.

Run: 2026-08-16 13:48–13:56 UTC (21 measured runs) · branch `research/asri-s0` @
`0e656d7` · artifacts: [`artifacts/phase0b2/`](artifacts/phase0b2/) (`environment.json`, `runs.jsonl`,
`summary.csv`, `summary.json`). Same frozen substrate and environment as
[Phase-0B](PHASE0B_RESULT.md): Qwen3-4B @ `1cfa9a7…f60c`, BF16, `cuda:0`, non-thinking,
greedy; RTX 3080 Ti 12.0 GiB, WDDM, desktop live (~2.2 GiB occupied).

## Purpose and control

Phase-0B established the prompt-side envelope but outputs stopped at ~32 tokens (EOS), so
`max_new_tokens` was an allowance, not a demonstrated length. B2 forces actual output
length with `min_new_tokens == max_new_tokens == target` so generation-side KV-cache and
decoder costs are exercised for real. **`min_new_tokens` is a benchmark-only residency
control, recorded in every record; no S0 policy uses it.** Prompts stay in the
known-resident regime (512 and 1024 tokens, same deterministic word-corpus documents as
Phase-0B). 1 discarded warm-up + 3 measured repetitions per condition, batch 1, same
synchronized latency/CUDA-memory protocol. Stop rule: first run whose peak allocated
exceeds the resident budget (12.0 GiB − 2.0 GiB desktop reserve = 10.0 GiB) halts the
sweep. It never triggered.

## Results (mean of 3 measured repetitions; gen = actual generated tokens)

| Prompt | Gen target | Actual gen | Latency (s) | Tok/s | Peak alloc (GiB) |
|---|---|---|---|---|---|
| 512 | 64 | 64 | 6.321 | 10.13 | 7.74 |
| 512 | 128 | 128 | 12.508 | 10.31 | 7.74 |
| 512 | 256 | 256 | 21.577 | 11.87 | 7.74 |
| 512 | 512 | 512 | 42.039 | 12.18 | 7.75 |
| 1024 | 64 | 64 | 5.367 | 11.93 | 8.10 |
| 1024 | 128 | 128 | 11.627 | 11.04 | 8.10 |
| 1024 | 256 | 256 | 22.402 | 11.43 | 8.10 |

## Findings

1. **Generation length does not move peak memory at these scales.** At prompt 512, going
   from 32 (Phase-0B) to 512 actual generated tokens changed peak allocated by +8.1 MB
   (8,312,902,144 → 8,321,371,136 B); at prompt 1024, 256 forced tokens changed it by
   +4.2 MB. This is at or below allocator segment granularity — KV-cache growth
   (~0.15 MB/token theoretical for Qwen3-4B) is real but invisible next to the 7.56 GiB
   weights and prompt-side allocation steps. Verified, no longer extrapolated.
2. **Latency is the binding cost of generation, not VRAM.** Marginal decode cost is
   ~80–90 ms/token (~11–12.5 tok/s) with ~1–1.5 s fixed prefill/launch overhead at these
   prompt sizes. A 256-token output pass costs ~22 s wall clock at prompt 1024.
3. **Condition-level throughput varies ~±15%** (10.1–12.2 tok/s means across conditions;
   e.g. the (512, 64) row ran slower than (1024, 64)) — WDDM/desktop sharing noise, worth
   remembering when single-pass latencies feed S0 comparisons.

## Frozen ceilings (supersede the Phase-0B caveat)

- **Routine S0 generation ceiling: 256 actual output tokens** — now *measured* at the
  routine prompt ceiling (1024, peak 8.10 GiB, 3/3 reps resident) rather than assumed.
  Operational margin unchanged: 1.90 GiB above the 2.0 GiB desktop reserve.
- **Absolute resident generation maximum tested: 512 actual tokens** at prompt 512
  (peak 7.75 GiB). Terminology preserved: this is 512 *generated* tokens — distinct from
  Phase-0B's "prompt 2048 with a 512-token *allowance*, actual output ≈32".
- **Prompt-side ceilings unchanged** from Phase-0B: routine 1024 / absolute resident
  2048 / spill regime at 4096.

## Limitations

- Generation beyond 512 tokens and generation at prompt 2048+ are untested; the flat-KV
  finding should not be extrapolated to multi-thousand-token outputs.
- Forced continuation after the natural EOS makes text semantically degenerate — irrelevant
  for residency, but throughput under forced continuation may differ slightly from natural
  prose generation.
- Single node, single session, 3 reps; WDDM noise as above.

## Scope statement

Substrate characterization only. With prompt and generation ceilings now both measured,
the frozen substrate operating envelope is: **prompt ≤ 1024, output ≤ 256, ~3–22 s per
call, peak ≈ 8.10 GiB, resident with 1.90 GiB margin.** S0-B calibration design can
proceed inside this envelope with per-pass cost C = Σ(input + output tokens) plus latency
and peak-residency telemetry.
