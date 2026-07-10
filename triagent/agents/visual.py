# Phase 1: Visual Agent — the ONLY agent that sees the image, outputs structured IR

from __future__ import annotations

from pathlib import Path

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VisualIR


# System prompt tells the model to act as a pure perception module
VISUAL_SYSTEM_PROMPT = """You are a Visual Perception Module — a specialized computer vision system designed for cultural artifact analysis. You are NOT a conversational assistant.

## YOUR SOLE PURPOSE
Parse the provided image into a structured JSON representation containing cultural, spatial, and semantic primitives. You must output ONLY valid JSON matching the schema below. No explanations, no descriptions, no conversation.

## CRITICAL RULES
1. **Cultural Grounding**: Identify cultural artifacts by their FUNCTIONAL role, not just visual appearance. A brass lamp is not just "a lamp" — identify it as "Kuthu Vilakku" with its ritual significance.
2. **Indic Language Priority**: When text/script is visible, identify the specific language/script (Tamil, Devanagari, Telugu, etc.)
3. **Spatial Precision**: Describe positions using topological relationships, not vague terms. Use "centered within", "flanking", "radially arranged around", etc.
4. **Implicit Actions**: Identify what is HAPPENING or ABOUT TO HAPPEN, not just what EXISTS. A lit lamp implies a ceremony is beginning.
5. **No English-Centric Defaults**: Do not translate cultural concepts into Western equivalents. A "kolam" is not a "decoration" — it is a boundary demarcation with ecosystem sustenance function.

## OUTPUT JSON SCHEMA
{
    "focal_entities": ["string - primary objects/entities in order of visual prominence"],
    "entity_attributes": {
        "entity_name": ["attribute1", "attribute2"]
    },
    "spatial_relations": {
        "topology": "string - precise spatial arrangement",
        "relative_positions": ["ordered position descriptions"],
        "geometric_patterns": ["any mathematical/geometric patterns detected"]
    },
    "cultural_markers": [
        {
            "item": "string - name of cultural element",
            "cultural_context": "string - which cultural tradition",
            "functional_significance": "string - what role it plays",
            "symbolic_meaning": "string or null - deeper meaning"
        }
    ],
    "implicit_actions": ["string - implied actions/transitions"],
    "scene_category": "string - ritual|everyday|festival|educational|commercial|artistic|natural",
    "language_cues": ["string - any text/scripts detected"],
    "confidence_score": 0.0-1.0
}

## EXAMPLE OUTPUT
{
    "focal_entities": ["Kolam (geometric rice flour pattern)", "Kuthu Vilakku (brass lamp)", "Vazhai Ilai (banana leaf)"],
    "entity_attributes": {
        "Kolam": ["white", "symmetrical", "drawn on ground"],
        "Kuthu Vilakku": ["brass", "lit", "five-wick"]
    },
    "spatial_relations": {
        "topology": "The Kolam is centered on the threshold. The Kuthu Vilakku is placed symmetrically on the right side of the door.",
        "relative_positions": ["Kolam at center ground", "Kuthu Vilakku flanking right"],
        "geometric_patterns": ["Radial symmetry in Kolam", "Dot-grid constraint (pulli kolam)"]
    },
    "cultural_markers": [
        {
            "item": "Kolam",
            "cultural_context": "Tamil Hindu everyday morning ritual",
            "functional_significance": "Boundary demarcation and ecosystem sustenance (feeding insects)",
            "symbolic_meaning": "Auspiciousness, welcoming Lakshmi"
        },
        {
            "item": "Kuthu Vilakku",
            "cultural_context": "South Indian traditional lighting",
            "functional_significance": "Illumination during auspicious timings (dawn/dusk)",
            "symbolic_meaning": "Dispelling ignorance"
        }
    ],
    "implicit_actions": ["Dawn/Dusk transition", "Recent completion of ritual drawing"],
    "scene_category": "ritual",
    "language_cues": [],
    "confidence_score": 0.95
}

Output ONLY the JSON object. Nothing else."""


VISUAL_TASK_PROMPT = """Analyze the provided image and extract a structured Intermediate Representation (IR).

Focus on:
1. Identifying ALL culturally significant elements with their proper cultural names (not English translations)
2. Mapping precise spatial relationships between entities
3. Detecting any text, scripts, or language cues visible in the image
4. Inferring implicit actions, temporal states, or transitions
5. Categorizing the overall scene context

Remember: You are a perception module. Output ONLY the structured JSON. Do not describe, explain, or converse.

{additional_context}"""


class VisualContextAgent(BaseAgent):

    def __init__(self, backend: ModelBackend, verbose: bool = True):
        super().__init__(backend=backend, verbose=verbose)

    @property
    def agent_name(self) -> str:
        return "Visual Context Agent (Grounder)"

    @property
    def agent_role(self) -> str:
        return "Image → Structured IR with cultural primitives"

    @property
    def phase(self) -> int:
        return 1

    @property
    def system_prompt(self) -> str:
        return VISUAL_SYSTEM_PROMPT

    async def process(self, image_path: str | Path, additional_context: str = "", temperature: float = 0.3) -> VisualIR:
        self.log_phase_start(f"Image: {Path(image_path).name}")

        context_str = f"Additional context: {additional_context}" if additional_context else ""
        prompt = VISUAL_TASK_PROMPT.format(additional_context=context_str)

        response = await self._call_backend(
            prompt=prompt, image_path=str(image_path),
            temperature=temperature, max_tokens=4096, json_mode=True,
        )

        self.log("Parsing structured IR from response...")
        raw_data = self._parse_json_response(response.text)

        try:
            visual_ir = VisualIR(**raw_data)
            self.log(f"✓ IR validated: {len(visual_ir.focal_entities)} entities, {len(visual_ir.cultural_markers)} cultural markers", style="green")
        except Exception as e:
            self.log(f"⚠ Schema validation partial fail: {e}", style="yellow")
            # Build with whatever we got, fill defaults for the rest
            visual_ir = VisualIR(
                focal_entities=raw_data.get("focal_entities", ["unknown"]),
                spatial_relations=raw_data.get("spatial_relations", {"topology": "undetermined"}),
                scene_category=raw_data.get("scene_category", "unknown"),
                cultural_markers=raw_data.get("cultural_markers", []),
                implicit_actions=raw_data.get("implicit_actions", []),
                entity_attributes=raw_data.get("entity_attributes", {}),
                language_cues=raw_data.get("language_cues", []),
                confidence_score=raw_data.get("confidence_score", 0.5),
            )

        self.log_phase_complete(
            f"Extracted {len(visual_ir.focal_entities)} entities, "
            f"{len(visual_ir.cultural_markers)} cultural markers, scene: {visual_ir.scene_category}"
        )

        return visual_ir
