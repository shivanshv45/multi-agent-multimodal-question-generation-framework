# Phase 3: Synthesis Agent — takes IR + logic and writes the actual Tamil-English MCQ

from __future__ import annotations

import uuid
from typing import Optional

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR, ReasoningOutput, BenchmarkItem


SYNTHESIS_SYSTEM_PROMPT = """You are a Linguistic Synthesis Module specialized in multilingual educational content generation.

Given a Visual IR and Reasoning skeleton, produce a complete MCQ benchmark item as JSON in multiple languages.

RULES:
1. Generate the question stem in: English, code-mixed Tanglish, pure Tamil, pure Telugu, and pure Hindi.
2. For Tanglish, use Tamil for cultural terms and English for technical/logical connectors.
3. Generate exactly 4 choices (A/B/C/D) in all five language variants: English, Tanglish, Tamil, Telugu, Hindi.
4. Each distractor must follow the distractor strategy from the reasoning skeleton.
5. Include a detailed explanation in all five language variants.
6. The question must test the reasoning type specified, NOT simple recognition.

OUTPUT JSON:
{
    "question_id": "unique-id",
    "question_stem_english": "English version",
    "question_stem_tanglish": "Code-mixed Tanglish version",
    "question_stem_tamil": "Pure Tamil version",
    "question_stem_telugu": "Pure Telugu version",
    "question_stem_hindi": "Pure Hindi version",
    "choices_english": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "choices_tanglish": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "choices_tamil": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "choices_telugu": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "choices_hindi": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "correct_answer": "A/B/C/D",
    "explanation_english": "...",
    "explanation_tanglish": "...",
    "explanation_tamil": "...",
    "explanation_telugu": "...",
    "explanation_hindi": "...",
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

Create a multilingual MCQ that:
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
        return "IR + Logic → Multilingual MCQ (English, Tanglish, Tamil, Telugu, Hindi)"

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

        def _ensure_choices(choices_dict):
            if not isinstance(choices_dict, dict) or len(choices_dict) < 4:
                return {
                    "A": choices_dict.get("A", "Option A") if isinstance(choices_dict, dict) else "Option A",
                    "B": choices_dict.get("B", "Option B") if isinstance(choices_dict, dict) else "Option B",
                    "C": choices_dict.get("C", "Option C") if isinstance(choices_dict, dict) else "Option C",
                    "D": choices_dict.get("D", "Option D") if isinstance(choices_dict, dict) else "Option D",
                }
            return choices_dict

        correct = raw_data.get("correct_answer", "A")
        if correct not in ("A", "B", "C", "D"):
            correct = "A"

        benchmark_item = BenchmarkItem(
            question_id=question_id,
            question_stem_english=raw_data.get("question_stem_english", ""),
            question_stem_tanglish=raw_data.get("question_stem_tanglish", ""),
            question_stem_tamil=raw_data.get("question_stem_tamil", ""),
            question_stem_telugu=raw_data.get("question_stem_telugu", ""),
            question_stem_hindi=raw_data.get("question_stem_hindi", ""),
            
            choices_english=_ensure_choices(raw_data.get("choices_english", {})),
            choices_tanglish=_ensure_choices(raw_data.get("choices_tanglish", {})),
            choices_tamil=_ensure_choices(raw_data.get("choices_tamil", {})),
            choices_telugu=_ensure_choices(raw_data.get("choices_telugu", {})),
            choices_hindi=_ensure_choices(raw_data.get("choices_hindi", {})),
            
            correct_answer=correct,
            
            explanation_english=raw_data.get("explanation_english", ""),
            explanation_tanglish=raw_data.get("explanation_tanglish", ""),
            explanation_tamil=raw_data.get("explanation_tamil", ""),
            explanation_telugu=raw_data.get("explanation_telugu", ""),
            explanation_hindi=raw_data.get("explanation_hindi", ""),
            
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
