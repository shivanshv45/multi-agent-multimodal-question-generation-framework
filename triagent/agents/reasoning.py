# Phase 2: Reasoning Agent — BLIND, never sees the image, only works with the IR

from __future__ import annotations

import json

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR, ReasoningOutput, QuestionType, DistractorStrategy


REASONING_SYSTEM_PROMPT = """You are a Logical Reasoning Module for constructing educational assessment items from structured data. You are NOT a conversational assistant.

Given a structured IR of a visual scene, construct the LOGICAL SKELETON of a challenging, culturally-grounded MCQ. Output ONLY the reasoning structure as JSON.

RULES:
1. You are BLIND — you work ONLY with structured IR, never the image.
2. Every question MUST test inferential logic, not recognition.
3. Use cultural markers for deep reasoning, not identification.
4. Distractors must be PLAUSIBLE logical misinterpretations.

QUESTION TYPES: analogical, causal, counterfactual, compositional, cultural_inference, spatial_reasoning
DISTRACTOR STRATEGIES: cultural_misattribution, spatial_inversion, functional_swap, temporal_confusion, analogical_mismatch

OUTPUT JSON:
{
    "question_type": "string",
    "reasoning_chain": ["step1", "step2", "..."],
    "correct_answer_logic": "string",
    "distractor_strategies": ["strategy1", "strategy2", "strategy3"],
    "distractor_rationales": ["rationale1", "rationale2", "rationale3"],
    "difficulty_level": 1-5,
    "required_knowledge": ["domain1", "domain2"],
    "analogical_mapping": {"source": "target"} or null
}"""


REASONING_TASK_PROMPT = """Analyze this IR and construct a logical skeleton for a challenging MCQ.

## INTERMEDIATE REPRESENTATION
```json
{ir_json}
```

1. Study focal entities, cultural markers, and spatial relations
2. Identify the most interesting reasoning pathway
3. Construct a multi-step reasoning chain
4. Design 3 distractor strategies exploiting reasoning fallacies
5. Estimate difficulty (1=basic, 5=expert)

Output JSON only."""


class ReasoningAgent(BaseAgent):

    def __init__(self, backend: ModelBackend, verbose: bool = True):
        super().__init__(backend=backend, verbose=verbose)

    @property
    def agent_name(self) -> str:
        return "Reasoning Agent (Logician)"

    @property
    def agent_role(self) -> str:
        return "Structured IR → Logical question skeleton"

    @property
    def phase(self) -> int:
        return 2

    @property
    def system_prompt(self) -> str:
        return REASONING_SYSTEM_PROMPT

    async def process(self, visual_ir: VisualIR, temperature: float = 0.5) -> ReasoningOutput:
        self.log_phase_start(
            f"IR with {len(visual_ir.focal_entities)} entities, "
            f"{len(visual_ir.cultural_markers)} cultural markers"
        )

        ir_json = visual_ir.model_dump_json(indent=2)
        prompt = REASONING_TASK_PROMPT.format(ir_json=ir_json)

        # NO image_path — this agent is intentionally blind
        response = await self._call_backend(
            prompt=prompt, image_path=None, temperature=temperature,
            max_tokens=4096, json_mode=True,
        )

        self.log("Parsing reasoning structure...")
        raw_data = self._parse_json_response(response.text)

        try:
            q_type = raw_data.get("question_type", "cultural_inference")
            if q_type not in [qt.value for qt in QuestionType]:
                q_type = "cultural_inference"

            strategies = []
            for s in raw_data.get("distractor_strategies", []):
                if s in [ds.value for ds in DistractorStrategy]:
                    strategies.append(s)
                else:
                    strategies.append("functional_swap")
            if len(strategies) < 2:
                strategies = ["cultural_misattribution", "functional_swap", "temporal_confusion"]

            rationales = raw_data.get("distractor_rationales", [])
            while len(rationales) < len(strategies):
                rationales.append("Plausible misinterpretation of cultural context")

            reasoning_output = ReasoningOutput(
                question_type=q_type,
                reasoning_chain=raw_data.get("reasoning_chain", ["Identify cultural elements", "Map functional relationships"]),
                correct_answer_logic=raw_data.get("correct_answer_logic", "Based on cultural marker analysis"),
                distractor_strategies=strategies,
                distractor_rationales=rationales,
                difficulty_level=min(5, max(1, raw_data.get("difficulty_level", 3))),
                required_knowledge=raw_data.get("required_knowledge", []),
                analogical_mapping=raw_data.get("analogical_mapping"),
            )
            self.log(f"✓ type={reasoning_output.question_type.value}, difficulty={reasoning_output.difficulty_level}", style="green")

        except Exception as e:
            self.log(f"⚠ Validation issue: {e}", style="yellow")
            reasoning_output = ReasoningOutput(
                question_type=QuestionType.CULTURAL_INFERENCE,
                reasoning_chain=["Analyze cultural markers", "Determine functional significance", "Map to reasoning task"],
                correct_answer_logic="Derived from cultural marker analysis",
                distractor_strategies=[DistractorStrategy.CULTURAL_MISATTRIBUTION, DistractorStrategy.FUNCTIONAL_SWAP, DistractorStrategy.TEMPORAL_CONFUSION],
                distractor_rationales=["Misattributes origin", "Swaps roles", "Confuses temporal significance"],
                difficulty_level=3,
                required_knowledge=["Indic cultural traditions"],
            )

        self.log_phase_complete(f"Type: {reasoning_output.question_type.value}, Difficulty: {reasoning_output.difficulty_level}/5")
        return reasoning_output
