# Phase 3: Synthesis Agent — takes IR + logic and writes the actual code-mixed MCQ

from __future__ import annotations

import uuid
from typing import Optional

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR, VideoIR, AudioIR, ReasoningOutput, BenchmarkItem


# The {target_language} placeholder is filled at runtime so the same agent
# works for Tanglish, Hinglish, Teluguish, pure Hindi, pure Tamil, etc.
SYNTHESIS_SYSTEM_PROMPT = """You are a Linguistic Synthesis Module specialized in code-mixed educational content generation.

Your current target language is: **{target_language}**

Given a Visual IR and Reasoning skeleton, produce a complete MCQ benchmark item as JSON in multiple languages.

RULES:
1. Question stem MUST be code-mixed in the requested target language. Use the regional language for cultural terms and English for technical/logical connectors.
2. Provide BOTH a code-mixed version and English-only version.
3. Generate exactly 4 choices (A/B/C/D): 1 correct + 3 distractors.
4. Each distractor must follow the distractor strategy from the reasoning skeleton.
5. Include a detailed explanation in both English and the target code-mixed language.
6. The question must test the reasoning type specified, NOT simple recognition.

CODE-MIXING GUIDELINES (adapt to the target language):
- Cultural nouns stay in regional transliteration (e.g., for Tanglish: "kolam", "kuthu vilakku", "pongal"; for Hinglish: "rangoli", "diya", "aarti"; for Teluguish: "muggulu", "deepam")
- Logical connectors in English: "because", "therefore", "if...then"
- Technical terms in English: "spatial relationship", "geometric pattern"

OUTPUT JSON:
{{
    "question_id": "unique-id",
    "question_stem": "code-mixed question in target language",
    "question_stem_english": "English-only version",
    "choices": {{"A": "choice1", "B": "choice2", "C": "choice3", "D": "choice4"}},
    "correct_answer": "A/B/C/D",
    "explanation": "detailed English explanation",
    "explanation_code_mixed": "code-mixed explanation in target language",
    "target_language": "{target_language}",
    "question_type": "from reasoning skeleton",
    "difficulty": 1-5,
    "cultural_tags": ["tag1", "tag2"]
}}"""


SYNTHESIS_TASK_PROMPT = """Generate a complete benchmark MCQ from the following inputs.

## VISUAL IR (Scene Structure)
```json
{ir_json}
```

## REASONING SKELETON (Logical Constraints)
```json
{reasoning_json}
```

Target language for code-mixing: **{target_language}**

Create a code-mixed MCQ in {target_language} that:
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
        return "IR + Logic → Code-mixed MCQ"

    @property
    def phase(self) -> int:
        return 3

    @property
    def system_prompt(self) -> str:
        # Returns the template; callers must .format(target_language=...) before use
        return SYNTHESIS_SYSTEM_PROMPT

    async def process(
        self,
        reasoning_output: ReasoningOutput,
        visual_ir: VisualIR | None = None,
        video_ir: VideoIR | None = None,
        audio_ir: AudioIR | None = None,
        media_path: Optional[str] = None,
        temperature: float = 0.7,
        target_language: str = "Tanglish (Tamil-English)",
    ) -> BenchmarkItem:
        # Determine which IR to serialize
        ir = video_ir or audio_ir or visual_ir
        media_type = "video" if video_ir else ("audio" if audio_ir else "image")

        self.log_phase_start(
            f"Type: {reasoning_output.question_type.value}, "
            f"Difficulty: {reasoning_output.difficulty_level}, "
            f"Language: {target_language}, Source: {media_type}"
        )

        ir_json = ir.model_dump_json(indent=2) if ir else "{}"
        reasoning_json = reasoning_output.model_dump_json(indent=2)

        strategies = ", ".join(
            s if isinstance(s, str) else s.value
            for s in reasoning_output.distractor_strategies
        )

        # Inject the target language into the system prompt
        resolved_system_prompt = self.system_prompt.format(target_language=target_language)

        prompt = SYNTHESIS_TASK_PROMPT.format(
            ir_json=ir_json, reasoning_json=reasoning_json,
            question_type=reasoning_output.question_type.value,
            difficulty=reasoning_output.difficulty_level,
            strategies=strategies,
            target_language=target_language,
        )

        response = await self._call_backend(
            prompt=prompt, image_path=None, temperature=temperature,
            max_tokens=4096, json_mode=True,
            system_prompt_override=resolved_system_prompt,
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
            explanation=raw_data.get("explanation", ""),
            explanation_code_mixed=raw_data.get("explanation_code_mixed") or raw_data.get("explanation_tamil"),
            target_language=raw_data.get("target_language", target_language),
            question_type=reasoning_output.question_type,
            difficulty=reasoning_output.difficulty_level,
            cultural_tags=raw_data.get("cultural_tags", []),
            source_media_path=str(media_path) if media_path else None,
            source_media_type=media_type,
            visual_ir=visual_ir,
            video_ir=video_ir,
            audio_ir=audio_ir,
            reasoning_output=reasoning_output,
        )

        self.log(f"✓ Generated: {benchmark_item.question_stem_english[:80]}...", style="green")
        self.log_phase_complete(f"Question ID: {question_id}, Answer: {correct}")

        return benchmark_item
