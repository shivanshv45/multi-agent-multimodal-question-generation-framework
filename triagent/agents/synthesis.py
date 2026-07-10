# Phase 3: Synthesis Agent — takes IR + logic and writes the actual code-mixed MCQ

from __future__ import annotations

import uuid
from typing import Optional

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR, VideoIR, AudioIR, ReasoningOutput, BenchmarkItem


# The {base_language} placeholder is filled at runtime (e.g., "Tamil", "Hindi", "Telugu")
SYNTHESIS_SYSTEM_PROMPT = """You are a Linguistic Synthesis Module specialized in multilingual educational content generation.

Your current base language is: **{base_language}**

Given a Visual IR and Reasoning skeleton, produce a complete MCQ benchmark item as JSON in three formats simultaneously: English, Pure {base_language}, and Code-Mixed (e.g., Tanglish/Hinglish).

RULES:
1. Generate exactly 4 choices (A/B/C/D): 1 correct + 3 distractors matching the logic.
2. Produce the question stem, choices, and explanation in ALL THREE formats.
3. Code-Mixed version: Blend {base_language} cultural terms/verbs with English technical connectors.
4. Pure version: Use authentic, formal {base_language} script (e.g., தமிழ், हिंदी, తెలుగు).

OUTPUT JSON:
{{
    "question_id": "unique-id",
    "question_stem_english": "English version",
    "choices_english": {{"A": "choice1", "B": "choice2", "C": "choice3", "D": "choice4"}},
    "explanation_english": "English explanation",
    
    "question_stem_pure": "Pure {base_language} script version",
    "choices_pure": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "explanation_pure": "Pure {base_language} explanation",
    
    "question_stem_mixed": "Code-mixed version (e.g. Tanglish/Hinglish)",
    "choices_mixed": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "explanation_mixed": "Code-mixed explanation",
    
    "correct_answer": "A/B/C/D",
    "base_language": "{base_language}",
    "question_type": "from reasoning skeleton",
    "difficulty": 1-5,
    "cultural_tags": ["tag1", "tag2"]
}}

## EXAMPLE OUTPUT (for base_language: Tamil)
{{
    "question_id": "taq-001",
    "question_stem_english": "A freshly drawn Kolam on the threshold is flanked by a lit Kuthu Vilakku. What implicit action does this indicate?",
    "choices_english": {{
        "A": "The household has just completed the dawn welcoming ritual.",
        "B": "A night-time Diwali celebration is taking place.",
        "C": "The lamp is being used as a reading light.",
        "D": "They are preparing for an evening-only ritual."
    }},
    "explanation_english": "The combination signifies the completion of the morning auspicious welcoming ritual.",
    
    "question_stem_pure": "வாசலில் வரையப்பட்ட கோலத்திற்கு அருகில் ஏற்றப்பட்ட குத்துவிளக்கு உள்ளது. இது எந்தச் செயலைக் குறிக்கிறது?",
    "choices_pure": {{
        "A": "குடும்பத்தினர் காலை நேர மங்களகரமான வரவேற்புச் சடங்கை முடித்துள்ளனர்.",
        "B": "இரவு நேர தீபாவளி கொண்டாட்டம் நடைபெறுகிறது.",
        "C": "விளக்கு படிப்பதற்காகப் பயன்படுத்தப்படுகிறது.",
        "D": "அவர்கள் மாலை நேரச் சடங்கிற்கு மட்டுமே தயாராகிறார்கள்."
    }},
    "explanation_pure": "கோலம் மற்றும் குத்துவிளக்கு ஆகியவை காலை நேர வரவேற்புச் சடங்கு நிறைவடைந்ததைக் குறிக்கின்றன.",
    
    "question_stem_mixed": "Oru veetu vaasal-la freshly drawn Kolam-um, pakkathula lit aana Kuthu Vilakku-m irukku. Idhu endha implicit action-a indicate pannudhu?",
    "choices_mixed": {{
        "A": "Household dawn auspicious welcoming ritual-a complete pannitaanga.",
        "B": "Night-time Diwali celebration nadakkudhu.",
        "C": "Lamp-a just reading light-ah use pandraanga.",
        "D": "Avanga evening-only ritual-kku prepare aagaraanga."
    }},
    "explanation_mixed": "Correct answer A dhaan. Kaalaila vaasal-la Kolam podradhum, Kuthu Vilakku yethuradhum morning ritual-oda conclusion-a mark pannudhu.",
    
    "correct_answer": "A",
    "base_language": "Tamil",
    "question_type": "cultural_inference",
    "difficulty": 4,
    "cultural_tags": ["Kolam", "Kuthu Vilakku"]
}}"""""


SYNTHESIS_TASK_PROMPT = """Generate a complete benchmark MCQ from the following inputs.

## VISUAL IR (Scene Structure)
```json
{ir_json}
```

## REASONING SKELETON (Logical Constraints)
```json
{reasoning_json}
```

Target Base Language: **{base_language}**

Create a multi-format MCQ (English, Pure {base_language}, and Code-Mixed) that:
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
        base_language: str = "Tamil",
    ) -> BenchmarkItem:
        ir = video_ir or audio_ir or visual_ir
        media_type = "video" if video_ir else ("audio" if audio_ir else "image")

        self.log_phase_start(
            f"Type: {reasoning_output.question_type.value}, "
            f"Language: {base_language} (All 3 formats)"
        )

        ir_json = ir.model_dump_json(indent=2) if ir else "{}"
        reasoning_json = reasoning_output.model_dump_json(indent=2)
        strategies = ", ".join(s if isinstance(s, str) else s.value for s in reasoning_output.distractor_strategies)

        resolved_system_prompt = self.system_prompt.format(base_language=base_language)
        prompt = SYNTHESIS_TASK_PROMPT.format(
            ir_json=ir_json, reasoning_json=reasoning_json,
            question_type=reasoning_output.question_type.value,
            difficulty=reasoning_output.difficulty_level,
            strategies=strategies,
            base_language=base_language,
        )

        response = await self._call_backend(
            prompt=prompt, image_path=None, temperature=temperature,
            max_tokens=4096, json_mode=True,
            system_prompt_override=resolved_system_prompt,
        )

        self.log("Parsing multi-language benchmark item...")
        raw = self._parse_json_response(response.text)

        def _ensure_4(d):
            if not isinstance(d, dict): d = {}
            return {
                "A": d.get("A", "Option A"), "B": d.get("B", "Option B"),
                "C": d.get("C", "Option C"), "D": d.get("D", "Option D"),
            }

        # Force a unique UUID so we never overwrite files, even if LLM hallucinated the same ID
        unique_id = f"taq-{uuid.uuid4().hex[:8]}"
        
        benchmark_item = BenchmarkItem(
            question_id=unique_id,
            
            question_stem_english=raw.get("question_stem_english", ""),
            choices_english=_ensure_4(raw.get("choices_english", {})),
            explanation_english=raw.get("explanation_english", raw.get("explanation", "")),
            
            question_stem_pure=raw.get("question_stem_pure", ""),
            choices_pure=_ensure_4(raw.get("choices_pure", {})),
            explanation_pure=raw.get("explanation_pure", ""),
            
            question_stem_mixed=raw.get("question_stem_mixed", ""),
            choices_mixed=_ensure_4(raw.get("choices_mixed", {})),
            explanation_mixed=raw.get("explanation_mixed", ""),
            
            correct_answer=raw.get("correct_answer", "A"),
            base_language=raw.get("base_language", base_language),
            question_type=reasoning_output.question_type,
            difficulty=reasoning_output.difficulty_level,
            cultural_tags=raw.get("cultural_tags", []),
            source_media_path=str(media_path) if media_path else None,
            source_media_type=media_type,
            visual_ir=visual_ir,
            video_ir=video_ir,
            audio_ir=audio_ir,
            reasoning_output=reasoning_output,
        )

        self.log(f"✓ Generated 3 formats for: {benchmark_item.question_stem_english[:60]}...", style="green")
        return benchmark_item
