# ASRI Phase-0 Reference Validation Result

**Verdict: PASS** (12/12 identity invariants passed)

Run: 2026-08-16 09:36 UTC · branch `research/asri-s0` @ `4384148` (working tree contained
this implementation, uncommitted at measurement time) · artifacts:
[`artifacts/phase0/`](artifacts/phase0/) (`environment.json`, `smoke.json`).

## Frozen substrate identity

| Field | Value |
|---|---|
| model_id | `Qwen/Qwen3-4B` |
| requested revision | `1cfa9a7208912126459214e8b04321603b3df60c` |
| resolved revision | `1cfa9a7208912126459214e8b04321603b3df60c` (snapshot dir name, exact match) |
| snapshot path | `C:\huggingface_cache\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7…f60c` |
| tokenizer | `Qwen2Tokenizer` (`transformers.models.qwen2.tokenization_qwen2`, native) |
| model | `Qwen3ForCausalLM` (`transformers.models.qwen3.modeling_qwen3`, native) |
| dtype | `torch.bfloat16` (observed on parameters) |
| device | `cuda:0`, all parameters resident |
| trust_remote_code | `False` (tokenizer and model) |
| enable_thinking | `False` |

## Environment

Windows 11 (10.0.26200) · AMD64, 6C/12T, 3.9 GHz · 32 GiB RAM · Python 3.12.10 (CPython)
· torch 2.13.0+cu130 · transformers 5.15.0 · huggingface-hub 1.27.0 · psutil 7.2.2 ·
NVIDIA GeForce RTX 3080 Ti 12 GiB (capability 8.6, driver 610.47, CUDA 13.0).
Full record: `artifacts/phase0/environment.json`.

## Invariant results (all PASS)

| # | Invariant | Result |
|---|---|---|
| 1 | model_id is `Qwen/Qwen3-4B` | PASS |
| 2 | requested revision is an exact commit SHA | PASS |
| 3 | resolved snapshot revision is exact | PASS |
| 4 | tokenizer native, `trust_remote_code=False` | PASS |
| 5 | model native, `trust_remote_code=False` | PASS |
| 6 | `enable_thinking=False` | PASS |
| 7 | rendered chat template is Qwen non-thinking mode (`<think>\n\n</think>` closed block) | PASS |
| 8 | generated response contains no `<think>` tags | PASS |
| 9 | generation non-empty | PASS |
| 10 | CUDA available | PASS |
| 11 | model resides on `cuda:0` | PASS |
| 12 | dtype is BF16 | PASS |

## Smoke generation

Prompt `"Return exactly the word READY."`, chat template with `add_generation_prompt=True`,
`enable_thinking=False`, greedy decoding (`do_sample=False`), `max_new_tokens=16`,
`use_cache=True`.

| Metric | Observed | Historical reference | Match |
|---|---|---|---|
| response | `READY` | `READY` | exact |
| prompt tokens | 18 | 18 | exact |
| generated tokens | 2 | 2 | exact |
| peak allocated | 8,115,154,432 B | 8,115,154,432 B | exact |
| peak reserved | 8,128,561,152 B | 8,128,561,152 B | exact |
| latency | 1.4023 s | 1.2402 s | no (informational) |

The byte-identical peak memory values are expected: identical weights, dtype, and
allocation sequence produce identical allocator peaks. Latency is not an identity
invariant and was not required to match.

## Scope statement

This result validates the frozen reference substrate only. It is not evidence about
adaptive compute, routing, or any ASRI mechanism. Phase-0B characterizes the inference
envelope of this substrate; S0 experiments follow only after that.
