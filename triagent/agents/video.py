# Phase 1 (Video): Video Context Agent — processes video into a temporal+spatial IR
#
# Strategy:
#   - If the backend supports native video (Gemini): upload the whole video
#     via the File API and let the model process both visual frames (1 FPS)
#     and the audio track simultaneously.
#   - If the backend does NOT support native video (Grok, Ollama): fall back
#     to extracting keyframes with ffmpeg, then analyzing them as images.
#     Audio is lost in this fallback — the user should run the audio pipeline
#     separately if needed.

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from triagent.agents.base import BaseAgent
from triagent.backends.base import ModelBackend
from triagent.schemas import VideoIR


VIDEO_SYSTEM_PROMPT = """You are a Video Perception Module — a specialized multimodal system for temporal and cultural analysis of video content. You are NOT a conversational assistant.

## YOUR SOLE PURPOSE
Parse the provided video into a structured JSON representation containing temporal events, cultural primitives, spatial arrangements, and audio analysis. Output ONLY valid JSON matching the schema below.

## CRITICAL RULES
1. **Temporal Precision**: Identify key events with timestamps in MM:SS format. Note transitions, scene changes, and pivotal moments.
2. **Cultural Grounding**: Identify cultural artifacts by their FUNCTIONAL role. A traditional dance is not just "dancing" — identify the specific form (Bharatanatyam, Kathak, etc.) with its ritual significance.
3. **Audio-Visual Integration**: Analyze BOTH the visual content and the audio track. Note speech, music, sound effects, and their relationship to visual events.
4. **Narrative Arc**: Describe the overall progression of the video — setup, development, climax, resolution.
5. **Indic Language Priority**: When speech or text is detected, identify the specific language/script.
6. **No English-Centric Defaults**: Do not translate cultural concepts into Western equivalents.

## OUTPUT JSON SCHEMA
{
    "focal_entities": ["primary entities appearing across the video"],
    "entity_attributes": {"entity_name": ["attribute1", "attribute2"]},
    "scene_category": "ritual|everyday|festival|educational|commercial|artistic|natural",
    "cultural_markers": [
        {
            "item": "name",
            "cultural_context": "which tradition",
            "functional_significance": "what role it plays",
            "symbolic_meaning": "deeper meaning or null"
        }
    ],
    "language_cues": ["text/scripts/spoken languages detected"],
    "duration_seconds": 0.0,
    "temporal_events": [
        {
            "timestamp": "MM:SS",
            "description": "what happens",
            "entities_involved": ["entity1"],
            "cultural_significance": "significance or null"
        }
    ],
    "narrative_arc": "setup → development → climax → resolution description",
    "transitions": ["scene transition descriptions"],
    "audio_summary": "summary of audio content",
    "spoken_language": "detected spoken language(s) or null",
    "transcript_excerpt": "key speech excerpt or null",
    "confidence_score": 0.0-1.0
}

Output ONLY the JSON object. Nothing else."""


VIDEO_TASK_PROMPT = """Analyze the provided video and extract a structured Intermediate Representation (IR).

Focus on:
1. Identifying ALL temporally significant events with precise timestamps
2. Mapping cultural artifacts to their functional roles (not English translations)
3. Analyzing the audio track: speech content, music, sound effects
4. Describing the narrative progression of the video
5. Detecting scene transitions and their significance
6. Identifying any text, scripts, or spoken language cues

Remember: You are a perception module. Output ONLY the structured JSON.

{additional_context}"""


# Fallback prompt used when we extract keyframes and feed them as images
KEYFRAME_ANALYSIS_PROMPT = """You are analyzing {num_frames} keyframes extracted from a video at the timestamps shown.

For each frame, identify:
1. Cultural artifacts and their functional significance
2. Spatial arrangements and entities
3. Any visible text or language cues

Then synthesize across ALL frames to produce a unified VideoIR JSON that captures:
- The temporal progression across frames
- Consistent entity tracking
- Cultural markers that appear repeatedly
- The overall narrative arc

Frames are provided in chronological order.

{additional_context}

Output ONLY the JSON matching the VideoIR schema."""


class VideoContextAgent(BaseAgent):

    def __init__(self, backend: ModelBackend, verbose: bool = True):
        super().__init__(backend=backend, verbose=verbose)

    @property
    def agent_name(self) -> str:
        return "Video Context Agent (Grounder)"

    @property
    def agent_role(self) -> str:
        return "Video → Structured temporal+spatial IR"

    @property
    def phase(self) -> int:
        return 1

    @property
    def system_prompt(self) -> str:
        return VIDEO_SYSTEM_PROMPT

    async def process(
        self,
        video_path: str | Path,
        additional_context: str = "",
        temperature: float = 0.3,
        max_keyframes: int = 8,
    ) -> VideoIR:
        video_path = Path(video_path)
        self.log_phase_start(f"Video: {video_path.name}")

        # Try native video processing first (Gemini supports this)
        try:
            return await self._process_native(video_path, additional_context, temperature)
        except NotImplementedError:
            self.log("Backend does not support native video — falling back to keyframe extraction", style="yellow")
            return await self._process_keyframes(video_path, additional_context, temperature, max_keyframes)

    async def _process_native(self, video_path: Path, additional_context: str, temperature: float) -> VideoIR:
        """Use the backend's native video understanding (Gemini File API)."""
        context_str = f"Additional context: {additional_context}" if additional_context else ""
        prompt = VIDEO_TASK_PROMPT.format(additional_context=context_str)

        self.log(f"Uploading video to {self.backend.name} (native processing)...")

        response = await self.backend.generate_with_video(
            prompt=prompt,
            video_path=str(video_path),
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=8192,
            json_mode=True,
        )

        self.log("Parsing structured VideoIR from response...")
        raw_data = self._parse_json_response(response.text)
        video_ir = self._build_video_ir(raw_data)

        self.log_phase_complete(
            f"Extracted {len(video_ir.temporal_events)} events, "
            f"{len(video_ir.cultural_markers)} cultural markers, "
            f"scene: {video_ir.scene_category}"
        )
        return video_ir

    async def _process_keyframes(self, video_path: Path, additional_context: str, temperature: float, max_keyframes: int) -> VideoIR:
        """Fallback: extract keyframes via ffmpeg, then analyze as images."""
        self.log(f"Extracting up to {max_keyframes} keyframes with ffmpeg...")
        keyframe_paths = self._extract_keyframes(video_path, max_keyframes)

        if not keyframe_paths:
            raise RuntimeError(
                f"Failed to extract keyframes from {video_path.name}. "
                "Ensure ffmpeg is installed and the video file is valid."
            )

        self.log(f"Extracted {len(keyframe_paths)} keyframes, analyzing...")

        context_str = f"Additional context: {additional_context}" if additional_context else ""
        prompt = KEYFRAME_ANALYSIS_PROMPT.format(
            num_frames=len(keyframe_paths),
            additional_context=context_str,
        )

        # Analyze the first keyframe with the prompt (backend only supports one image)
        # Then append analysis of remaining frames
        all_frame_analyses = []
        for i, frame_path in enumerate(keyframe_paths):
            frame_prompt = f"Frame {i+1}/{len(keyframe_paths)} (approx timestamp {i * 5}s):\nAnalyze this frame and extract cultural/spatial primitives as JSON."

            response = await self.backend.generate_with_image(
                prompt=frame_prompt,
                image_path=str(frame_path),
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=4096,
                json_mode=True,
            )
            all_frame_analyses.append(response.text)

        # Synthesize all frame analyses into a unified VideoIR
        synthesis_prompt = f"""Synthesize these {len(all_frame_analyses)} keyframe analyses into a single unified VideoIR JSON.

Frame analyses:
{json.dumps(all_frame_analyses, indent=2)}

{context_str}

Output a single VideoIR JSON that captures the temporal progression, consistent entities, cultural markers, and narrative arc across all frames."""

        response = await self.backend.generate(
            prompt=synthesis_prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=8192,
            json_mode=True,
        )

        raw_data = self._parse_json_response(response.text)
        video_ir = self._build_video_ir(raw_data)

        # Clean up temp frames
        for fp in keyframe_paths:
            try:
                fp.unlink()
            except OSError:
                pass

        self.log_phase_complete(
            f"Extracted {len(video_ir.temporal_events)} events from {len(keyframe_paths)} keyframes"
        )
        return video_ir

    def _extract_keyframes(self, video_path: Path, max_frames: int) -> list[Path]:
        """Extract keyframes from a video using ffmpeg.

        Uses the -skip_frame nokey flag which skips decoding of non-keyframes
        entirely, making it much faster than the select filter approach.
        Falls back to uniform sampling if keyframe extraction yields too few frames.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="triagent_keyframes_"))

        # First try: extract I-frames (keyframes)
        output_pattern = str(tmp_dir / "keyframe_%03d.jpg")
        cmd = [
            "ffmpeg", "-skip_frame", "nokey",
            "-i", str(video_path),
            "-vsync", "vfr",
            "-q:v", "2",
            "-frames:v", str(max_frames),
            output_pattern,
            "-y", "-loglevel", "error",
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            self.log(f"Keyframe extraction failed: {e}. Trying uniform sampling...", style="yellow")
            # Fallback: uniform sampling at 0.2 fps (1 frame every 5 seconds)
            output_pattern_uniform = str(tmp_dir / "frame_%03d.jpg")
            cmd_uniform = [
                "ffmpeg", "-i", str(video_path),
                "-vf", f"fps=0.2",
                "-q:v", "2",
                "-frames:v", str(max_frames),
                output_pattern_uniform,
                "-y", "-loglevel", "error",
            ]
            try:
                subprocess.run(cmd_uniform, check=True, capture_output=True, timeout=60)
            except Exception:
                return []

        frames = sorted(tmp_dir.glob("*.jpg"))
        return frames[:max_frames]

    def _build_video_ir(self, raw_data: dict) -> VideoIR:
        """Build a validated VideoIR from raw parsed JSON, with fallback defaults."""
        try:
            return VideoIR(**raw_data)
        except Exception as e:
            self.log(f"⚠ VideoIR validation partial fail: {e}", style="yellow")
            return VideoIR(
                focal_entities=raw_data.get("focal_entities", ["unknown"]),
                scene_category=raw_data.get("scene_category", "unknown"),
                cultural_markers=raw_data.get("cultural_markers", []),
                temporal_events=raw_data.get("temporal_events", []),
                narrative_arc=raw_data.get("narrative_arc", ""),
                audio_summary=raw_data.get("audio_summary", ""),
                spoken_language=raw_data.get("spoken_language"),
                transcript_excerpt=raw_data.get("transcript_excerpt"),
                language_cues=raw_data.get("language_cues", []),
                transitions=raw_data.get("transitions", []),
                entity_attributes=raw_data.get("entity_attributes", {}),
                duration_seconds=raw_data.get("duration_seconds", 0.0),
                confidence_score=raw_data.get("confidence_score", 0.5),
            )
