from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from resonance_asri.contracts import ExecutionReceipt, ExecutionRequest, ResourceUsage
from resonance_asri.controller import ComputePolicy
from resonance_asri.memory import MemoryProvider
from resonance_asri.providers import CompletionProvider, ProviderResponse


class ASRIRuntime:
    """Provider-agnostic S0 execution engine with explicit resource accounting."""

    def __init__(
        self,
        *,
        provider: CompletionProvider,
        policy: ComputePolicy,
        memory: MemoryProvider | None = None,
    ) -> None:
        self._provider = provider
        self._policy = policy
        self._memory = memory

    def run(self, request: ExecutionRequest) -> ExecutionReceipt:
        decision = self._policy.decide(request)
        started = perf_counter()

        responses: list[ProviderResponse] = []
        retrieval_count = 0
        memory_context = ""

        if decision.use_memory:
            if self._memory is None:
                raise RuntimeError(
                    "policy requested memory but no MemoryProvider was configured"
                )
            snippets = tuple(self._memory.retrieve(request, limit=4))
            retrieval_count = 1
            if snippets:
                memory_context = "\n\nRetrieved context:\n" + "\n".join(
                    f"- {item}" for item in snippets
                )

        answer = ""
        for iteration in range(decision.reasoning_iterations):
            if iteration == 0:
                prompt = (
                    "Answer the task directly and accurately. Return only the answer that "
                    "should be shown to the user."
                    f"{memory_context}\n\nTask:\n{request.prompt}"
                )
                purpose = "answer"
            else:
                prompt = (
                    "Improve the candidate answer for correctness, completeness, and precision. "
                    "Return only the revised answer; do not describe the revision process."
                    f"{memory_context}\n\nTask:\n{request.prompt}\n\n"
                    f"Candidate answer:\n{answer}"
                )
                purpose = "refine"

            response = self._provider.complete(prompt=prompt, purpose=purpose)
            responses.append(response)
            answer = response.text

        for specialist in decision.specialists:
            critique = self._provider.complete(
                prompt=(
                    f"Act as the {specialist} specialist. Critique the candidate answer for "
                    "specific errors or omissions. Be concise."
                    f"\n\nTask:\n{request.prompt}\n\nCandidate answer:\n{answer}"
                ),
                purpose=f"specialist:{specialist}",
            )
            responses.append(critique)

            synthesis = self._provider.complete(
                prompt=(
                    "Revise the candidate answer using the specialist critique. Return only the "
                    "final revised answer."
                    f"\n\nTask:\n{request.prompt}\n\nCandidate answer:\n{answer}"
                    f"\n\nSpecialist critique:\n{critique.text}"
                ),
                purpose="specialist-synthesis",
            )
            responses.append(synthesis)
            answer = synthesis.text

        if decision.verify:
            verification = self._provider.complete(
                prompt=(
                    "Verify the candidate answer against the task. Correct any error you find and "
                    "return only the final answer."
                    f"\n\nTask:\n{request.prompt}\n\nCandidate answer:\n{answer}"
                ),
                purpose="verify",
            )
            responses.append(verification)
            answer = verification.text

        latency_ms = (perf_counter() - started) * 1000.0
        input_tokens = sum(response.input_tokens for response in responses)
        output_tokens = sum(response.output_tokens for response in responses)

        cost_known = all(response.estimated_cost_usd is not None for response in responses)
        estimated_cost_usd = (
            sum(response.estimated_cost_usd or 0.0 for response in responses)
            if cost_known
            else None
        )

        usage = ResourceUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            provider_calls=len(responses),
            reasoning_iterations=decision.reasoning_iterations,
            retrieval_count=retrieval_count,
            verifier_count=int(decision.verify),
            latency_ms=latency_ms,
            estimated_cost_usd=estimated_cost_usd,
        )

        return ExecutionReceipt(
            run_id=str(uuid4()),
            request_id=request.request_id,
            policy_id=self._policy.policy_id,
            model_id=self._provider.model_id,
            answer=answer,
            usage=usage,
            specialists_activated=decision.specialists,
            metadata={"nominal_compute_units": str(decision.nominal_units)},
        )
