# Phase 3: Synthesis Agent — takes IR + logic and writes the actual Tamil-English MCQ

from __future__ import annotations

import uuid
from typing import Optional

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR, ReasoningOutput, BenchmarkItem


SYNTHESIS_SYSTEM_PROMPT = """You are a Linguistic Synthesis Module specialized in code-mixed Tamil-English educational content generation.

Given a Visual IR and Reasoning skeleton, produce a complete MCQ benchmark item as JSON.

RULES:
1. Question stem MUST be code-mixed Tamil-English (Tanglish). Use Tamil for cultural terms and English for technical/logical connectors.
2. Provide BOTH a code-mixed version and English-only version.
3. Generate exactly 4 choices (A/B/C/D): 1 correct + 3 distractors.
4. Each distractor must follow the distractor strategy from the reasoning skeleton.
5. Include a detailed explanation in both English and Tamil.
6. The question must test the reasoning type specified, NOT simple recognition.

CODE-MIXING GUIDELINES:
- Cultural nouns stay in Tamil transliteration: "kolam", "kuthu vilakku", "pongal"
- Logical connectors in English: "because", "therefore", "if...then"
- Technical terms in English: "spatial relationship", "geometric pattern"
- Example: "Indha kolam-la irukura geometric pattern-oda significance enna?"

OUTPUT JSON:
{
    "question_id": "unique-id",
    "question_stem": "code-mixed Tamil-English question",
    "question_stem_english": "English-only version",
    "choices": {"A": "choice1", "B": "choice2", "C": "choice3", "D": "choice4"},
    "correct_answer": "A/B/C/D",
    "explanation": "detailed English explanation",
    "explanation_tamil": "code-mixed explanation",
    "question_type": "from reasoning skeleton",
    "difficulty": 1-5,
    "cultural_tags": ["tag1", "tag2"]
}"""


SYNTHESIS_TASK_PROMPT = """Generate a complete benchmark MCQ from the following inputs.

## VISUAL IR (Scene Structure)
```json
{ir_json}
```

## REASONING SKELETON (Logical Constraints)
```json
{reasoning_json}
```

Create a code-mixed Tamil-English MCQ that:
1. Tests {question_type} reasoning at difficulty {difficulty}/5
2. Uses the reasoning chain to construct the question stem
3. Generates distractors using strategies: {strategies}
4. Includes cultural tags from the scene

Output the complete benchmark item as JSON."""


class SynthesisAgent(BaseAgent):

    def __init__(self, backend: ModelBackend, verbose: bool = True):
        super().__init__(backend=backend, verbose=verbose)

    @property
    def agent_name(self) -> str:
        return "Synthesis Agent (Linguist)"

    @property
    def agent_role(self) -> str:
        return "IR + Logic → Code-mixed Tamil-English MCQ"

    @property
    def phase(self) -> int:
        return 3

    @property
    def system_prompt(self) -> str:
        return SYNTHESIS_SYSTEM_PROMPT

    async def process(self, visual_ir: VisualIR, reasoning_output: ReasoningOutput, image_path: Optional[str] = None, temperature: float = 0.7) -> BenchmarkItem:
        self.log_phase_start(f"Type: {reasoning_output.question_type.value}, Difficulty: {reasoning_output.difficulty_level}")

        ir_json = visual_ir.model_dump_json(indent=2)
        reasoning_json = reasoning_output.model_dump_json(indent=2)

        strategies = ", ".join(
            s if isinstance(s, str) else s.value
            for s in reasoning_output.distractor_strategies
        )

        prompt = SYNTHESIS_TASK_PROMPT.format(
            ir_json=ir_json, reasoning_json=reasoning_json,
            question_type=reasoning_output.question_type.value,
            difficulty=reasoning_output.difficulty_level, strategies=strategies,
        )

        response = await self._call_backend(
            prompt=prompt, image_path=None, temperature=temperature,
            max_tokens=4096, json_mode=True,
        )

        self.log("Parsing benchmark item...")
        raw_data = self._parse_json_response(response.text)

        question_id = raw_data.get("question_id", f"taq-{uuid.uuid4().hex[:8]}")

        # Make sure we always have 4 choices
        choices = raw_data.get("choices", {})
        if not isinstance(choices, dict) or len(choices) < 4:
            choices = {
                "A": choices.get("A", "Option A"), "B": choices.get("B", "Option B"),
                "C": choices.get("C", "Option C"), "D": choices.get("D", "Option D"),
            }

        correct = raw_data.get("correct_answer", "A")
        if correct not in ("A", "B", "C", "D"):
            correct = "A"

        benchmark_item = BenchmarkItem(
            question_id=question_id,
            question_stem=raw_data.get("question_stem", ""),
            question_stem_english=raw_data.get("question_stem_english", ""),
            choices=choices,
            correct_answer=correct,
            explanation=raw_data.get("explanation", ""),
            explanation_tamil=raw_data.get("explanation_tamil"),
            question_type=reasoning_output.question_type,
            difficulty=reasoning_output.difficulty_level,
            cultural_tags=raw_data.get("cultural_tags", []),
            source_image_path=str(image_path) if image_path else None,
            visual_ir=visual_ir,
            reasoning_output=reasoning_output,
        )

        self.log(f"✓ Generated: {benchmark_item.question_stem_english[:80]}...", style="green")
        self.log_phase_complete(f"Question ID: {question_id}, Answer: {correct}")

        return benchmark_item
