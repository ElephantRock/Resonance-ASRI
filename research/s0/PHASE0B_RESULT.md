# ASRI Phase-0B Qwen3-4B Inference Envelope Result

**Phase-0 prerequisite: PASS** (12/12 invariants, see [PHASE0_RESULT.md](PHASE0_RESULT.md)).
**Phase-0B: valid characterization of the resident regime; grid terminated early at the
WDDM shared-memory spill boundary.** No ASRI performance claims are made here — this
phase characterizes the substrate only.

Run: 2026-08-16 ~10:20–10:40 UTC · branch `research/asri-s0` @ `33bec2d`
(working tree contained the runner entry-point fix, uncommitted at measurement time) ·
artifacts: [`artifacts/phase0b/`](artifacts/phase0b/) (`environment.json`, `runs.jsonl`,
`summary.csv`, `summary.json`).

## Exact environment and model

Windows 11 (10.0.26200), AMD64 6C/12T, 32 GiB RAM, Python 3.12.10, torch 2.13.0+cu130,
transformers 5.15.0, huggingface-hub 1.27.0, psutil 7.2.2. GPU: RTX 3080 Ti, 12.0 GiB
(12,884,246,528 B), capability 8.6, driver 610.47, CUDA 13.0, WDDM mode.
Model: `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, `cuda:0`,
`trust_remote_code=False`, chat template `enable_thinking=False`. Weights alone occupy
7.56 GiB allocated / 7.57 GiB reserved (Phase-0 smoke).

## Benchmark methodology

Batch 1, greedy (`do_sample=False`, `use_cache=True`, eos 151645, pad 151643 — the only
varied setting is `max_new_tokens`). Deterministic word-corpus documents (seeded per
prompt target) rendered through the non-thinking chat template; prompt lengths hit the
nominal token targets exactly (verified per run; recorded as `actual_prompt_tokens`).
Per condition: 1 warm-up (discarded) + 3 measured repetitions. Latency brackets the full
`generate` call with `torch.cuda.synchronize()` on both sides — **not TTFT** (`ttft: null`;
no streaming instrumentation). CUDA peak stats are reset per repetition after the
steady-state snapshot; host RSS/system RAM and device-wide VRAM (nvidia-smi) are sampled
around each call. Model loaded once per process.

## Actual grid executed

Nominal grid: prompt {512, 1024, 2048, 4096, 8192} × max_new {32, 64, 128, 256, 512}.

- **15 conditions fully measured and resident-valid** (45 runs): all five generation
  targets at prompt 512, 1024, and 2048.
- **1 condition measured but invalidated** (3 runs): (4096, 32) — see OOM boundary.
- **9 conditions not attempted**: (4096, 64…512) and the whole 8192 row — skipped when the
  sweep was stopped at the spill boundary.

Generated text stopped at EOS after 30–36 tokens under the one-sentence summarization
instruction, so `actual_generated_tokens` ≪ `max_new_tokens` except at the 32 cap.

## Latency, throughput, memory curves (resident regime, mean per condition)

| Prompt tokens | Latency (s) | Tokens/s | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---|---|---|---|
| 512  | 2.401–2.448 | 12.26–12.50 | 7.74 | 7.78 |
| 1024 | 2.740–3.109 | 11.58–12.03 | 8.10 | 8.19 |
| 2048 | 3.022–3.051 | 10.49–10.59 | 9.16 | 9.31 |
| 4096 | 47.275 *(invalidated)* | 0.66 *(invalidated)* | 13.05 *(> physical)* | 13.31 *(> physical)* |

Generation count barely affects latency/memory here because outputs stop at ~32 tokens:
latency is prefill-plus-overhead dominated, and memory grows with prompt length.
Device-wide free VRAM bottomed at 311 MiB during the 2048 row (desktop baseline ~1.7–2.2 GiB).
Repetition scatter within conditions is tiny (e.g. (2048, 512): 3.026–3.044 s).

## OOM boundary (and why no OOM was recorded)

No CUDA OOM ever fired. On Windows WDDM, the driver satisfied the 4096-token condition's
allocations beyond physical VRAM by paging into shared system memory: PyTorch reported
`peak_allocated` = 13.05 GiB on a 12.0 GiB device while latency regressed 15.6×
(3.0 s → 47.3 s) and throughput collapsed to 0.66 tok/s. Those runs are mechanically
successful but measure PCIe paging, not GPU inference; they are invalidated in
`summary.json` (`wddm_shared_memory_spill_regime`, 3 runs) and excluded from aggregation.
The sweep was then stopped manually — hardware safety (this GPU drives the desktop, ~2 GiB
already occupied outside PyTorch) outranks completing grid points whose data would be
invalid. The physical-residency boundary therefore lies between 2048 and 4096 prompt
tokens; on a Linux/TCC-class setup the same allocation would likely have raised a hard
CUDA OOM instead.

## Envelopes

**A. Absolute tested maximum (resident):** prompt **2048** × max_new **512** — all 3
repetitions successful, mean latency 3.033 s, peak allocated 9.16 GiB, peak reserved
9.31 GiB. (The largest mechanically-successful condition, (4096, 32), is excluded as
spill-regime.)

**B. Recommended routine S0 envelope:** prompt ceiling **1024 tokens**, generation
ceiling **256 tokens**. Decision rule (recorded in `summary.json` → `envelopes`):
among conditions where every repetition succeeded and max peak allocated fits within
total VRAM minus the 2.0 GiB desktop reserve minus a 1.0 GiB orchestration margin
(≤ 9.0 GiB), take the largest prompt with generation capped at 256. The 2048 row (9.16 GiB
peak, 311 MiB device-wide free) is deliberately not recommended: it leaves no room for
runtime variation, additional ASRI calls, or orchestration overhead.

**Operational margin at the routine envelope:** 1.90 GiB (12.0 − 2.0 desktop − 8.10 peak).
At routine conditions a single provider call costs ~3.0 s wall clock at ~12 tok/s decode.

## Desktop-VRAM caveat

All measurements were taken with the Windows desktop live on the same GPU (~1.7–2.2 GiB
occupied by the OS/compositor outside PyTorch). The 2048-row ran with as little as
311 MiB device-wide free. Latencies are representative of this shared-GPU reality, not of
a dedicated-GPU deployment.

## Unresolved limitations

- The spill boundary is bracketed (2048 < boundary < 4096 prompt tokens) but not bisected;
  no run pinned the exact last-resident prompt length.
- Generation-side memory is untested at scale: outputs stopped at 30–36 tokens, so
  KV-cache growth from long generations did not exercise the envelope. Prompts that elicit
  200–500-token answers will sit higher on the memory curve at the same prompt length.
- Single machine, single session, batch 1; no cross-run variance data beyond the 3
  repetitions per condition.
- TTFT was not measured (would require streaming instrumentation) and stays null.
- The 8192 row was never attempted; it is presumably deep in the spill regime.

## Scope statement

This document characterizes the frozen BF16 inference substrate only. It makes no claim
about adaptive compute, routing benefit, or any ASRI mechanism. S0-B calibration design
must fit within the routine envelope above.
