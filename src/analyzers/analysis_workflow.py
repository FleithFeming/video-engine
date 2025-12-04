"""Video analysis workflow - from scene detection to tag generation."""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .scene_detection import SceneDetector, SceneDetectionResult, DetectedScene
from .face_detection import FaceDetector, FaceAnalysisResult
from .tag_templates import TagTaxonomy, get_taxonomy, TagCategory

logger = logging.getLogger(__name__)


class AnalysisStage(Enum):
    """Stages in the analysis workflow."""
    SCENE_DETECTION = "scene_detection"
    FACE_DETECTION = "face_detection"
    COLOR_ANALYSIS = "color_analysis"
    MOTION_ANALYSIS = "motion_analysis"
    TAG_GENERATION = "tag_generation"
    TAG_NORMALIZATION = "tag_normalization"


@dataclass
class WorkflowConfig:
    """Configuration for the analysis workflow."""
    # Scene detection settings
    scene_threshold: float = 30.0
    min_scene_duration: float = 1.0
    max_scenes: int = 100

    # Face detection settings
    detect_faces: bool = True
    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    detect_eyes: bool = True
    detect_smile: bool = False

    # Analysis settings
    analyze_colors: bool = True
    analyze_motion: bool = True
    classify_scenes: bool = True

    # Tag generation settings
    normalize_tags: bool = True
    expand_tag_hierarchy: bool = True
    min_tag_confidence: float = 0.3

    # Performance settings
    max_frames_for_faces: int = 50
    face_sample_rate: int = 3


@dataclass
class WorkflowProgress:
    """Progress tracking for the workflow."""
    current_stage: AnalysisStage
    stage_progress: float  # 0.0 to 1.0
    stages_completed: List[AnalysisStage] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis result from the workflow."""
    # Scene data
    scenes: SceneDetectionResult
    scene_count: int
    avg_scene_duration: float

    # Face data
    face_analysis: Dict[str, Any]
    has_faces: bool
    face_frequency: float

    # Tags (organized by category)
    tags_by_category: Dict[str, List[str]]
    all_tags: List[str]
    normalized_tags: List[str]

    # Metadata
    duration: float
    frames_analyzed: int

    # Raw data for further processing
    raw_scene_tags: List[str]
    raw_face_tags: List[str]


class AnalysisWorkflow:
    """
    Complete video analysis workflow.

    Orchestrates scene detection, face detection, and tag generation
    into a unified analysis pipeline.
    """

    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        progress_callback: Optional[Callable[[WorkflowProgress], None]] = None
    ):
        """
        Initialize the analysis workflow.

        Args:
            config: Workflow configuration
            progress_callback: Optional callback for progress updates
        """
        self.config = config or WorkflowConfig()
        self.progress_callback = progress_callback
        self.taxonomy = get_taxonomy()

        # Initialize detectors
        self.scene_detector = SceneDetector(
            threshold=self.config.scene_threshold,
            min_scene_duration=self.config.min_scene_duration,
            classify_scenes=self.config.classify_scenes
        )

        if self.config.detect_faces:
            self.face_detector = FaceDetector(
                scale_factor=self.config.face_scale_factor,
                min_neighbors=self.config.face_min_neighbors,
                detect_eyes=self.config.detect_eyes,
                detect_smile=self.config.detect_smile
            )
        else:
            self.face_detector = None

    def analyze(self, video_path: str) -> AnalysisResult:
        """
        Run complete analysis workflow on a video.

        Args:
            video_path: Path to the video file

        Returns:
            AnalysisResult with all analysis data
        """
        logger.info("Starting analysis workflow for: %s", video_path)

        # Stage 1: Scene Detection
        self._update_progress(AnalysisStage.SCENE_DETECTION, 0.0, "Detecting scenes...")
        scene_result = self.scene_detector.detect_scenes(
            video_path,
            max_scenes=self.config.max_scenes
        )
        self._update_progress(AnalysisStage.SCENE_DETECTION, 1.0, f"Found {scene_result.scene_count} scenes")

        # Stage 2: Face Detection (on scene keyframes)
        face_analysis = {"has_faces": False, "face_frequency": 0, "face_tags": []}
        if self.config.detect_faces and self.face_detector:
            self._update_progress(AnalysisStage.FACE_DETECTION, 0.0, "Detecting faces...")
            face_analysis = self._analyze_faces_in_scenes(scene_result.scenes)
            self._update_progress(AnalysisStage.FACE_DETECTION, 1.0,
                                  f"Faces found in {face_analysis.get('frames_with_faces', 0)} frames")

        # Stage 3: Tag Generation
        self._update_progress(AnalysisStage.TAG_GENERATION, 0.0, "Generating tags...")
        tags_by_category, all_tags = self._generate_tags(scene_result, face_analysis)
        self._update_progress(AnalysisStage.TAG_GENERATION, 1.0, f"Generated {len(all_tags)} tags")

        # Stage 4: Tag Normalization
        if self.config.normalize_tags:
            self._update_progress(AnalysisStage.TAG_NORMALIZATION, 0.0, "Normalizing tags...")
            normalized_tags = self._normalize_tags(all_tags)
            self._update_progress(AnalysisStage.TAG_NORMALIZATION, 1.0,
                                  f"Normalized to {len(normalized_tags)} tags")
        else:
            normalized_tags = all_tags

        # Compile result
        result = AnalysisResult(
            scenes=scene_result,
            scene_count=scene_result.scene_count,
            avg_scene_duration=scene_result.avg_scene_duration,
            face_analysis=face_analysis,
            has_faces=face_analysis.get("has_faces", False),
            face_frequency=face_analysis.get("face_frequency", 0),
            tags_by_category=tags_by_category,
            all_tags=all_tags,
            normalized_tags=normalized_tags,
            duration=scene_result.total_duration,
            frames_analyzed=sum(1 for s in scene_result.scenes if s.keyframe is not None),
            raw_scene_tags=scene_result.all_tags,
            raw_face_tags=face_analysis.get("face_tags", [])
        )

        logger.info("Analysis workflow complete: %d scenes, %d tags",
                    result.scene_count, len(result.normalized_tags))

        return result

    def _analyze_faces_in_scenes(self, scenes: List[DetectedScene]) -> Dict[str, Any]:
        """Analyze faces in scene keyframes."""
        if not self.face_detector:
            return {"has_faces": False, "face_frequency": 0, "face_tags": []}

        # Collect keyframes
        frames_with_timestamps = []
        for scene in scenes[:self.config.max_frames_for_faces]:
            if scene.keyframe is not None:
                frames_with_timestamps.append((scene.keyframe_time, scene.keyframe))

        if not frames_with_timestamps:
            return {"has_faces": False, "face_frequency": 0, "face_tags": []}

        # Analyze faces
        return self.face_detector.analyze_video_faces(
            frames_with_timestamps,
            sample_rate=1  # Analyze all keyframes
        )

    def _generate_tags(
        self,
        scene_result: SceneDetectionResult,
        face_analysis: Dict[str, Any]
    ) -> tuple[Dict[str, List[str]], List[str]]:
        """Generate organized tags from analysis results."""
        tags_by_category: Dict[str, List[str]] = {
            "scene": [],
            "object": [],
            "person": [],
            "activity": [],
            "technical": [],
            "color": [],
            "lighting": [],
            "composition": [],
            "style": [],
        }

        # Process scene tags
        for tag in scene_result.all_tags:
            category = self._categorize_tag(tag)
            if category in tags_by_category:
                if tag not in tags_by_category[category]:
                    tags_by_category[category].append(tag)

        # Process individual scene classifications
        for scene in scene_result.scenes:
            if scene.classification:
                # Environment tags
                for tag in scene.classification.environment_tags:
                    self._add_tag_to_category(tag, tags_by_category)

                # Color tags
                for tag in scene.classification.color_tags:
                    self._add_tag_to_category(tag, tags_by_category)

                # Composition tags
                for tag in scene.classification.composition_tags:
                    self._add_tag_to_category(tag, tags_by_category)

            # Color analysis tags
            if scene.color_analysis:
                color_tags = self._generate_color_tags(scene.color_analysis)
                for tag in color_tags:
                    self._add_tag_to_category(tag, tags_by_category)

        # Process face tags
        for tag in face_analysis.get("face_tags", []):
            self._add_tag_to_category(tag, tags_by_category)

        # Add technical tags based on analysis
        technical_tags = self._generate_technical_tags(scene_result)
        for tag in technical_tags:
            if tag not in tags_by_category["technical"]:
                tags_by_category["technical"].append(tag)

        # Add style/genre inference
        style_tags = self._infer_style_tags(tags_by_category, scene_result, face_analysis)
        for tag in style_tags:
            if tag not in tags_by_category["style"]:
                tags_by_category["style"].append(tag)

        # Compile all tags
        all_tags = []
        for category_tags in tags_by_category.values():
            all_tags.extend(category_tags)

        return tags_by_category, list(set(all_tags))

    def _categorize_tag(self, tag: str) -> str:
        """Determine the category for a tag."""
        tag_lower = tag.lower()

        if tag_lower.startswith(("indoor", "outdoor", "underwater", "aerial")):
            return "scene"
        elif tag_lower.startswith(("person", "face")):
            return "person"
        elif tag_lower.startswith("activity"):
            return "activity"
        elif tag_lower.startswith(("quality", "duration", "framerate", "stability")):
            return "technical"
        elif tag_lower.startswith(("color", "color_style")):
            return "color"
        elif tag_lower.startswith("lighting"):
            return "lighting"
        elif tag_lower.startswith(("shot", "camera", "framing")):
            return "composition"
        elif tag_lower.startswith(("style", "genre")):
            return "style"
        else:
            return "scene"  # Default

    def _add_tag_to_category(self, tag: str, tags_by_category: Dict[str, List[str]]):
        """Add a tag to the appropriate category."""
        category = self._categorize_tag(tag)
        if category in tags_by_category and tag not in tags_by_category[category]:
            tags_by_category[category].append(tag)

    def _generate_color_tags(self, color_analysis) -> List[str]:
        """Generate tags from color analysis."""
        tags = []

        # Dominant colors
        for color_name in color_analysis.color_names[:3]:
            if color_name != "unknown":
                tags.append(f"color/{color_name}")

        # Temperature
        tags.append(f"color_style/{color_analysis.temperature.value}")

        # Saturation
        if color_analysis.saturation_level == "high":
            tags.append("color_style/vibrant")
        elif color_analysis.saturation_level == "low":
            tags.append("color_style/muted")

        # Monochrome
        if color_analysis.is_monochrome:
            tags.append("color_style/monochrome")

        # Contrast
        if color_analysis.contrast_level == "high":
            tags.append("color_style/high_contrast")

        return tags

    def _generate_technical_tags(self, scene_result: SceneDetectionResult) -> List[str]:
        """Generate technical quality tags."""
        tags = []

        # Duration tags
        duration = scene_result.total_duration
        if duration < 30:
            tags.append("duration/short")
        elif duration < 300:
            tags.append("duration/medium")
        else:
            tags.append("duration/long")

        # Pacing tags (based on average scene duration)
        if scene_result.avg_scene_duration < 2:
            tags.append("style/fast_paced")
        elif scene_result.avg_scene_duration > 10:
            tags.append("style/slow_paced")

        # Scene complexity
        if scene_result.scene_count > 20:
            tags.append("style/complex_editing")
        elif scene_result.scene_count <= 3:
            tags.append("style/simple_editing")

        return tags

    def _infer_style_tags(
        self,
        tags_by_category: Dict[str, List[str]],
        scene_result: SceneDetectionResult,
        face_analysis: Dict[str, Any]
    ) -> List[str]:
        """Infer style and genre tags from other tags."""
        style_tags = []

        # Check for vlog style (faces + talking head indicators)
        if face_analysis.get("face_frequency", 0) > 0.5:
            has_close_up = any("close_up" in t for t in tags_by_category.get("composition", []))
            has_centered = any("centered" in t for t in tags_by_category.get("composition", []))
            if has_close_up or has_centered:
                style_tags.append("style/vlog")

        # Check for documentary style
        has_outdoor = any("outdoor" in t for t in tags_by_category.get("scene", []))
        has_handheld = any("handheld" in t for t in tags_by_category.get("composition", []))
        if has_outdoor and scene_result.scene_count > 5:
            style_tags.append("style/documentary")

        # Check for cinematic style
        if scene_result.avg_scene_duration > 5 and scene_result.scene_count > 3:
            has_wide = any("wide" in t for t in tags_by_category.get("composition", []))
            if has_wide:
                style_tags.append("style/cinematic")

        # Check for tutorial style
        if face_analysis.get("face_frequency", 0) > 0.3:
            has_static = scene_result.scene_summary.get("static_scenes", 0) > scene_result.scene_count * 0.5
            if has_static:
                style_tags.append("style/tutorial")

        # Genre inference based on scene content
        scene_tags = tags_by_category.get("scene", [])

        if any("nature" in t for t in scene_tags):
            style_tags.append("genre/travel")

        if any("gym" in t or "exercise" in t for t in scene_tags + tags_by_category.get("activity", [])):
            style_tags.append("genre/fitness")

        if any("kitchen" in t or "food" in t for t in scene_tags + tags_by_category.get("object", [])):
            style_tags.append("genre/food")

        return style_tags

    def _normalize_tags(self, tags: List[str]) -> List[str]:
        """Normalize tags using the taxonomy."""
        normalized = []

        for tag in tags:
            # Try to find canonical form
            canonical = self.taxonomy.get_canonical_name(tag)
            if canonical:
                if canonical not in normalized:
                    normalized.append(canonical)
            else:
                # Keep tag as-is if not in taxonomy
                if tag not in normalized:
                    normalized.append(tag)

        # Optionally expand with ancestors
        if self.config.expand_tag_hierarchy:
            expanded = []
            for tag in normalized:
                if tag not in expanded:
                    expanded.append(tag)
                # Add parent tags
                ancestors = self.taxonomy.get_ancestors(tag)
                for ancestor in ancestors:
                    if ancestor not in expanded:
                        expanded.append(ancestor)
            normalized = expanded

        return normalized

    def _update_progress(self, stage: AnalysisStage, progress: float, message: str):
        """Update progress and notify callback."""
        if self.progress_callback:
            workflow_progress = WorkflowProgress(
                current_stage=stage,
                stage_progress=progress,
                messages=[message]
            )
            self.progress_callback(workflow_progress)

        logger.debug("[%s] %.0f%% - %s", stage.value, progress * 100, message)


# Convenience functions
def analyze_video(video_path: str, config: Optional[WorkflowConfig] = None) -> AnalysisResult:
    """Run complete video analysis workflow."""
    workflow = AnalysisWorkflow(config=config)
    return workflow.analyze(video_path)


def quick_analyze(video_path: str) -> Dict[str, Any]:
    """Quick analysis returning a simple dictionary of results."""
    result = analyze_video(video_path)

    return {
        "duration": result.duration,
        "scene_count": result.scene_count,
        "avg_scene_duration": result.avg_scene_duration,
        "has_faces": result.has_faces,
        "face_frequency": result.face_frequency,
        "tags": result.normalized_tags,
        "tags_by_category": result.tags_by_category,
    }
