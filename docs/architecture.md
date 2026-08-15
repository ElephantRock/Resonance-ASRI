# Architecture

## Purpose

Resonance ASRI is the model/runtime research layer for Adaptive Sparse Recurrent Intelligence. Its job is to test mechanisms that trade additional computation for additional capability only when the input justifies the expense.

## Architectural layers

```text
request
  │
  ▼
compute policy ───────────────┐
  │                           │
  ├─ base path                │
  ├─ memory path              │
  ├─ specialist path(s)       │
  ├─ additional reasoning     │
  └─ verification             │
                              ▼
                         telemetry ledger
                              │
                              ▼
                         execution receipt
```

### Control plane

The control plane decides which expensive mechanisms may run. In S0 this policy is explicit and externally inspectable. Later stages may learn the policy, but learned policies must remain measurable against matched-cost controls.

### Execution plane

The execution plane invokes model providers, memory, specialists, iterative reasoning, tools, and verification. Provider implementations are replaceable and must expose usage information sufficient to build resource receipts.

### Evidence plane

Every run produces an execution receipt describing what was activated and what it cost. Receipts are designed to be exported into Resonance World without exposing private model reasoning state.

## Boundary with Resonance World

Resonance ASRI owns model/runtime mechanisms, training, checkpoints, and low-level resource telemetry. Resonance World owns capability governance, provenance, composition, allocation across societies, and cross-system evaluation.

The interface is therefore data-oriented rather than import-oriented. A World integration should consume versioned ASRI artifacts and receipts, not ASRI's tensor objects or training implementation.

## Architectural invariants

1. Adaptive execution must be observable.
2. Cost must be measured at the same granularity as quality.
3. Every adaptive policy must have matched-cost controls.
4. Model/provider-specific code stays behind provider interfaces.
5. Experimental policy and production policy are distinct artifacts.
6. Private chain-of-thought or hidden strategy state is never required in exported receipts.
