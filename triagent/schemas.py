# Data models for what gets passed between agents (the IR contract)

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 output: what the Visual Agent extracts from an image
# ═══════════════════════════════════════════════════════════════════════════════

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
    """Structured intermediate representation from a single image."""
    focal_entities: list[str] = Field(..., description="Primary objects in the image", min_length=1)
    entity_attributes: dict[str, list[str]] = Field(default_factory=dict, description="Attributes per entity")
    spatial_relations: SpatialRelation = Field(..., description="How entities are arranged")
    cultural_markers: list[CulturalMarker] = Field(default_factory=list, description="Cultural elements found")
    implicit_actions: list[str] = Field(default_factory=list, description="Implied actions or transitions")
    scene_category: str = Field(..., description="Scene type: ritual, everyday, festival, etc.")
    language_cues: list[str] = Field(default_factory=list, description="Any text/scripts in the image")
    confidence_score: float = Field(default=0.0, description="Confidence 0.0 to 1.0", ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Video IR: structured representation of a video clip
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalEvent(BaseModel):
    """A single event or scene change detected in the video."""
    timestamp: str = Field(..., description="Timestamp in MM:SS format")
    description: str = Field(..., description="What happens at this moment")
    entities_involved: list[str] = Field(default_factory=list, description="Entities active in this event")
    cultural_significance: Optional[str] = Field(None, description="Cultural meaning of the event")


class VideoIR(BaseModel):
    """Structured intermediate representation from a video.

    Extends the spatial primitives of VisualIR into the temporal dimension.
    The Reasoning Agent receives this instead of the raw video, maintaining
    the architectural principle that Phase 2 never sees the raw media.
    """
    # Scene-level (inherited from image IR concepts)
    focal_entities: list[str] = Field(..., description="Primary entities across the video", min_length=1)
    entity_attributes: dict[str, list[str]] = Field(default_factory=dict, description="Attributes per entity")
    scene_category: str = Field(..., description="Dominant scene type")
    cultural_markers: list[CulturalMarker] = Field(default_factory=list, description="Cultural elements found")
    language_cues: list[str] = Field(default_factory=list, description="Text/scripts/spoken language detected")

    # Temporal (video-specific)
    duration_seconds: float = Field(default=0.0, description="Video duration in seconds")
    temporal_events: list[TemporalEvent] = Field(default_factory=list, description="Key timestamped events")
    narrative_arc: str = Field(default="", description="Overall story/progression: setup → climax → resolution")
    transitions: list[str] = Field(default_factory=list, description="Scene transitions observed")

    # Audio stream analysis
    audio_summary: str = Field(default="", description="Summary of audio content (speech, music, effects)")
    spoken_language: Optional[str] = Field(None, description="Detected spoken language(s)")
    transcript_excerpt: Optional[str] = Field(None, description="Key speech transcript excerpt")

    confidence_score: float = Field(default=0.0, description="Confidence 0.0 to 1.0", ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Audio IR: structured representation of an audio file
# ═══════════════════════════════════════════════════════════════════════════════

class SpeechSegment(BaseModel):
    """A segment of detected speech in the audio."""
    timestamp_start: str = Field(..., description="Start timestamp MM:SS")
    timestamp_end: str = Field(..., description="End timestamp MM:SS")
    speaker: str = Field(default="unknown", description="Speaker identifier if detectable")
    text: str = Field(..., description="Transcribed text")
    language: str = Field(default="unknown", description="Detected language of this segment")


class AudioIR(BaseModel):
    """Structured intermediate representation from an audio file.

    Captures speech, music, sound effects, and their cultural significance.
    The Reasoning Agent receives this instead of the raw audio.
    """
    duration_seconds: float = Field(default=0.0, description="Audio duration in seconds")
    audio_type: str = Field(default="speech", description="Primary type: speech, music, mixed, environmental")

    # Speech content
    full_transcript: str = Field(default="", description="Full transcription of speech content")
    speech_segments: list[SpeechSegment] = Field(default_factory=list, description="Timestamped speech segments")
    detected_languages: list[str] = Field(default_factory=list, description="Languages detected in audio")

    # Non-speech audio
    music_description: Optional[str] = Field(None, description="Description of music (genre, instruments, mood)")
    sound_effects: list[str] = Field(default_factory=list, description="Notable sound effects detected")

    # Cultural analysis
    cultural_markers: list[CulturalMarker] = Field(default_factory=list, description="Cultural elements in audio")
    contextual_cues: list[str] = Field(default_factory=list, description="Contextual inferences from audio")

    # Topic analysis
    key_topics: list[str] = Field(default_factory=list, description="Main topics discussed")
    sentiment: Optional[str] = Field(None, description="Overall sentiment: positive, negative, neutral, mixed")

    confidence_score: float = Field(default=0.0, description="Confidence 0.0 to 1.0", ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 output: the logical skeleton for question generation
# ═══════════════════════════════════════════════════════════════════════════════

class QuestionType(str, Enum):
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    COMPOSITIONAL = "compositional"
    CULTURAL_INFERENCE = "cultural_inference"
    SPATIAL_REASONING = "spatial_reasoning"
    TEMPORAL_REASONING = "temporal_reasoning"          # video-specific
    AUDIO_COMPREHENSION = "audio_comprehension"        # audio-specific
    CROSS_MODAL_INFERENCE = "cross_modal_inference"    # video audio+visual


class DistractorStrategy(str, Enum):
    CULTURAL_MISATTRIBUTION = "cultural_misattribution"
    SPATIAL_INVERSION = "spatial_inversion"
    FUNCTIONAL_SWAP = "functional_swap"
    TEMPORAL_CONFUSION = "temporal_confusion"
    ANALOGICAL_MISMATCH = "analogical_mismatch"
    SEQUENCE_REORDERING = "sequence_reordering"        # video-specific
    SPEAKER_MISATTRIBUTION = "speaker_misattribution"  # audio-specific


class ReasoningOutput(BaseModel):
    question_type: QuestionType = Field(..., description="What reasoning this tests")
    reasoning_chain: list[str] = Field(..., description="Step-by-step logic", min_length=2)
    correct_answer_logic: str = Field(..., description="Why the answer is correct")
    distractor_strategies: list[DistractorStrategy] = Field(..., description="How wrong answers are made", min_length=2)
    distractor_rationales: list[str] = Field(..., description="Why each distractor is plausible", min_length=2)
    difficulty_level: int = Field(..., description="1=easy, 5=expert", ge=1, le=5)
    required_knowledge: list[str] = Field(default_factory=list, description="Knowledge needed to answer")
    analogical_mapping: Optional[dict[str, str]] = Field(None, description="Source→target mapping")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 output: the final MCQ question
# ═══════════════════════════════════════════════════════════════════════════════

class BenchmarkItem(BaseModel):
    question_id: str = Field(..., description="Unique ID")
    
    # English baseline
    question_stem_english: str = Field(..., description="English-only version")
    choices_english: dict[str, str] = Field(..., description="English A/B/C/D choices")
    explanation_english: str = Field(..., description="Why the answer is correct in English")
    
    # Pure Regional Language
    question_stem_pure: str = Field(..., description="Pure regional language version")
    choices_pure: dict[str, str] = Field(..., description="Pure regional A/B/C/D choices")
    explanation_pure: str = Field(..., description="Explanation in pure regional language")
    
    # Code-Mixed Language (e.g., Tanglish, Hinglish)
    question_stem_mixed: str = Field(..., description="Code-mixed version")
    choices_mixed: dict[str, str] = Field(..., description="Code-mixed A/B/C/D choices")
    explanation_mixed: str = Field(..., description="Explanation in code-mixed language")
    
    correct_answer: str = Field(..., description="Correct key A/B/C/D")
    base_language: str = Field(default="Tamil", description="The base language (e.g., Tamil, Hindi, Telugu)")

    question_type: QuestionType = Field(..., description="Reasoning type tested")
    difficulty: int = Field(..., description="Difficulty 1-5", ge=1, le=5)
    cultural_tags: list[str] = Field(default_factory=list, description="Cultural context tags")
    source_media_path: Optional[str] = Field(None, description="Source media file path")
    source_media_type: str = Field(default="image", description="Type of source: image, video, audio")

    # Full trace so we can debug the pipeline
    visual_ir: Optional[VisualIR] = Field(None, description="Phase 1 IR (image)")
    video_ir: Optional[VideoIR] = Field(None, description="Phase 1 IR (video)")
    audio_ir: Optional[AudioIR] = Field(None, description="Phase 1 IR (audio)")
    reasoning_output: Optional[ReasoningOutput] = Field(None, description="Phase 2 reasoning")
