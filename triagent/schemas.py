# Data models for what gets passed between agents (the IR contract)

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Phase 1 output: what the Visual Agent extracts from an image ---

class CulturalMarker(BaseModel):
    item: str = Field(..., description="Name of the cultural artifact or element")
    cultural_context: str = Field(..., description="Which cultural tradition this belongs to")
    functional_significance: str = Field(..., description="What role this element plays")
    symbolic_meaning: Optional[str] = Field(None, description="Deeper symbolic meaning")


class SpatialRelation(BaseModel):
    topology: str = Field(..., description="Spatial arrangement description")
    relative_positions: Optional[list[str]] = Field(None, description="Ordered entity positions")
    geometric_patterns: Optional[list[str]] = Field(None, description="Geometric patterns detected")


class VisualIR(BaseModel):
    # This structured JSON is what Phase 2 receives instead of the raw image
    focal_entities: list[str] = Field(..., description="Primary objects in the image", min_length=1)
    entity_attributes: dict[str, list[str]] = Field(default_factory=dict, description="Attributes per entity")
    spatial_relations: SpatialRelation = Field(..., description="How entities are arranged")
    cultural_markers: list[CulturalMarker] = Field(default_factory=list, description="Cultural elements found")
    implicit_actions: list[str] = Field(default_factory=list, description="Implied actions or transitions")
    scene_category: str = Field(..., description="Scene type: ritual, everyday, festival, etc.")
    language_cues: list[str] = Field(default_factory=list, description="Any text/scripts in the image")
    confidence_score: float = Field(default=0.0, description="Confidence 0.0 to 1.0", ge=0.0, le=1.0)


# --- Phase 2 output: the logical skeleton for question generation ---

class QuestionType(str, Enum):
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    COMPOSITIONAL = "compositional"
    CULTURAL_INFERENCE = "cultural_inference"
    SPATIAL_REASONING = "spatial_reasoning"


class DistractorStrategy(str, Enum):
    CULTURAL_MISATTRIBUTION = "cultural_misattribution"
    SPATIAL_INVERSION = "spatial_inversion"
    FUNCTIONAL_SWAP = "functional_swap"
    TEMPORAL_CONFUSION = "temporal_confusion"
    ANALOGICAL_MISMATCH = "analogical_mismatch"


class ReasoningOutput(BaseModel):
    question_type: QuestionType = Field(..., description="What reasoning this tests")
    reasoning_chain: list[str] = Field(..., description="Step-by-step logic", min_length=2)
    correct_answer_logic: str = Field(..., description="Why the answer is correct")
    distractor_strategies: list[DistractorStrategy] = Field(..., description="How wrong answers are made", min_length=2)
    distractor_rationales: list[str] = Field(..., description="Why each distractor is plausible", min_length=2)
    difficulty_level: int = Field(..., description="1=easy, 5=expert", ge=1, le=5)
    required_knowledge: list[str] = Field(default_factory=list, description="Knowledge needed to answer")
    analogical_mapping: Optional[dict[str, str]] = Field(None, description="Source→target mapping")


# --- Phase 3 output: the final MCQ question ---

class BenchmarkItem(BaseModel):
    question_id: str = Field(..., description="Unique ID")
    question_stem: str = Field(..., description="Code-mixed Tamil-English question")
    question_stem_english: str = Field(..., description="English-only version")
    choices: dict[str, str] = Field(..., description="A/B/C/D answer choices")
    correct_answer: str = Field(..., description="Correct key A/B/C/D")
    explanation: str = Field(..., description="Why the answer is correct")
    explanation_tamil: Optional[str] = Field(None, description="Tamil explanation")
    question_type: QuestionType = Field(..., description="Reasoning type tested")
    difficulty: int = Field(..., description="Difficulty 1-5", ge=1, le=5)
    cultural_tags: list[str] = Field(default_factory=list, description="Cultural context tags")
    source_image_path: Optional[str] = Field(None, description="Source image path")

    # Full trace so we can debug the pipeline
    visual_ir: Optional[VisualIR] = Field(None, description="Phase 1 IR")
    reasoning_output: Optional[ReasoningOutput] = Field(None, description="Phase 2 reasoning")
