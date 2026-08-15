# Resonance World Interface

## Principle

Resonance ASRI is a capability-producing substrate. Resonance World consumes its declared capabilities, artifacts, manifests, and execution evidence without taking a dependency on ASRI training internals.

## Exported artifact concepts

A stable integration should eventually expose records equivalent to:

```text
ASRIModelArtifact
  model_id
  architecture_version
  checkpoint_digest
  training_manifest_digest
  total_parameters
  active_parameter_policy

ASRIExecutionReceipt
  run_id
  model_id
  policy_id
  input_digest
  reasoning_iterations
  specialists_activated
  retrieval_count
  verifier_count
  tool_calls
  input_tokens
  output_tokens
  latency_ms
  resource_metrics
  outcome_metrics
```

## Privacy boundary

Execution receipts report decisions and resource use, not hidden chain-of-thought. Private learned strategy state, internal activations, or scratch reasoning are not required for World-level provenance.

## Versioning

The World-facing contract should be versioned independently from ASRI implementation versions. Breaking changes to exported schemas require a new contract version.

## Integration sequence

1. Prove S0 internally in Resonance ASRI.
2. Freeze the minimum receipt/artifact schema required by the evidence.
3. Add an ASRI provider/capability adapter to Resonance World on a short-lived integration branch.
4. Validate replay/provenance/resource accounting across the repository boundary.
5. Only then expose ASRI as a World capability source.
