"""Enhanced automatic scene detection with classification for video analysis."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class SceneTransitionType(Enum):
    """Types of scene transitions."""
    CUT = "cut"                    # Hard cut between scenes
    FADE = "fade"                  # Fade in/out transition
    DISSOLVE = "dissolve"          # Cross-dissolve
    WIPE = "wipe"                  # Wipe transition
    GRADUAL = "gradual"            # Slow gradual change


class ColorTemperature(Enum):
    """Color temperature classification."""
    WARM = "warm"
    NEUTRAL = "neutral"
    COOL = "cool"


class BrightnessLevel(Enum):
    """Brightness level classification."""
    DARK = "dark"
    LOW = "low"
    MEDIUM = "medium"
    BRIGHT = "bright"
    OVEREXPOSED = "overexposed"


@dataclass
class ColorAnalysis:
    """Color analysis results for a frame."""
    dominant_colors: List[Tuple[int, int, int]]  # RGB values
    color_names: List[str]
    temperature: ColorTemperature
    saturation_level: str  # "low", "medium", "high"
    brightness: BrightnessLevel
    contrast_level: str  # "low", "medium", "high"
    is_monochrome: bool


@dataclass
class SceneClassification:
    """Classification results for a scene."""
    indoor_outdoor: str  # "indoor", "outdoor", "unknown"
    lighting_type: str   # "natural", "artificial", "mixed", "low_light"
    environment_tags: List[str]
    color_tags: List[str]
    composition_tags: List[str]
    confidence: float


@dataclass
class DetectedScene:
    """Information about a detected scene."""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    duration: float
    transition_type: SceneTransitionType
    keyframe: Optional[np.ndarray]
    keyframe_time: float
    classification: Optional[SceneClassification]
    color_analysis: Optional[ColorAnalysis]
    motion_level: str  # "static", "low", "medium", "high"
    tags: List[str] = field(default_factory=list)


@dataclass
class SceneDetectionResult:
    """Complete scene detection result for a video."""
    scenes: List[DetectedScene]
    scene_count: int
    avg_scene_duration: float
    total_duration: float
    dominant_transition: SceneTransitionType
    all_tags: List[str]
    scene_summary: Dict[str, Any]


class SceneDetector:
    """
    Enhanced scene detection with visual classification.

    Detects scene boundaries and classifies scenes based on visual features.
    """

    # Color name mapping (approximate RGB ranges)
    COLOR_NAMES = {
        "red": [(150, 0, 0), (255, 100, 100)],
        "orange": [(200, 100, 0), (255, 180, 100)],
        "yellow": [(200, 200, 0), (255, 255, 100)],
        "green": [(0, 150, 0), (100, 255, 100)],
        "cyan": [(0, 200, 200), (100, 255, 255)],
        "blue": [(0, 0, 150), (100, 100, 255)],
        "purple": [(100, 0, 150), (200, 100, 255)],
        "pink": [(200, 100, 150), (255, 200, 220)],
        "brown": [(100, 50, 0), (180, 120, 80)],
        "white": [(220, 220, 220), (255, 255, 255)],
        "gray": [(100, 100, 100), (180, 180, 180)],
        "black": [(0, 0, 0), (50, 50, 50)],
    }

    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_duration: float = 1.0,
        fade_threshold: float = 10.0,
        motion_threshold: float = 5.0,
        classify_scenes: bool = True,
    ):
        """
        Initialize scene detector.

        Args:
            threshold: Scene change detection threshold (higher = fewer scenes)
            min_scene_duration: Minimum duration between scenes in seconds
            fade_threshold: Threshold for detecting fade transitions
            motion_threshold: Threshold for motion detection
            classify_scenes: Whether to classify detected scenes
        """
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.fade_threshold = fade_threshold
        self.motion_threshold = motion_threshold
        self.classify_scenes = classify_scenes

    def detect_scenes(
        self,
        video_path: str,
        max_scenes: int = 100
    ) -> SceneDetectionResult:
        """
        Detect scenes in a video file.

        Args:
            video_path: Path to video file
            max_scenes: Maximum number of scenes to detect

        Returns:
            SceneDetectionResult with detected scenes and metadata
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            min_frames_between = int(self.min_scene_duration * fps)

            scenes: List[DetectedScene] = []
            scene_boundaries: List[Tuple[int, float, SceneTransitionType]] = []

            prev_frame = None
            prev_hist = None
            frame_idx = 0
            last_scene_idx = 0

            # Collect frame differences for adaptive thresholding
            frame_diffs: List[float] = []

            logger.info("Detecting scenes in video...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert to grayscale and HSV
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Calculate histogram
                hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()

                if prev_frame is not None and prev_hist is not None:
                    # Frame difference (content change)
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_score = np.mean(diff)
                    frame_diffs.append(diff_score)

                    # Histogram comparison (color distribution change)
                    hist_diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

                    # Combined score
                    combined_score = diff_score + (hist_diff * 50)

                    # Detect scene change
                    if (combined_score > self.threshold and
                            frame_idx - last_scene_idx >= min_frames_between):

                        # Determine transition type
                        transition = self._detect_transition_type(
                            diff_score, hist_diff, frame_diffs[-min(10, len(frame_diffs)):]
                        )

                        scene_boundaries.append((frame_idx, frame_idx / fps, transition))
                        last_scene_idx = frame_idx

                        if len(scene_boundaries) >= max_scenes:
                            break

                prev_frame = gray
                prev_hist = hist
                frame_idx += 1

            # Create scene objects from boundaries
            scenes = self._create_scenes_from_boundaries(
                cap, fps, total_frames, duration, scene_boundaries
            )

            # Generate result
            return self._create_result(scenes, duration)

        finally:
            cap.release()

    def _detect_transition_type(
        self,
        diff_score: float,
        hist_diff: float,
        recent_diffs: List[float]
    ) -> SceneTransitionType:
        """Determine the type of scene transition."""
        if len(recent_diffs) < 3:
            return SceneTransitionType.CUT

        avg_recent = np.mean(recent_diffs[:-1]) if len(recent_diffs) > 1 else 0

        # Hard cut: sudden large change
        if diff_score > self.threshold * 1.5 and diff_score > avg_recent * 3:
            return SceneTransitionType.CUT

        # Fade: gradual brightness change
        if all(d < self.fade_threshold for d in recent_diffs[:-1]) and diff_score > self.threshold:
            return SceneTransitionType.FADE

        # Dissolve: moderate change over multiple frames
        if np.std(recent_diffs) < self.threshold * 0.3:
            return SceneTransitionType.DISSOLVE

        # Gradual: slow consistent change
        if len(recent_diffs) > 5:
            trend = np.polyfit(range(len(recent_diffs)), recent_diffs, 1)[0]
            if trend > 0.5:
                return SceneTransitionType.GRADUAL

        return SceneTransitionType.CUT

    def _create_scenes_from_boundaries(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        total_frames: int,
        duration: float,
        boundaries: List[Tuple[int, float, SceneTransitionType]]
    ) -> List[DetectedScene]:
        """Create scene objects from detected boundaries."""
        scenes: List[DetectedScene] = []

        # Add implicit start boundary
        all_boundaries = [(0, 0.0, SceneTransitionType.CUT)] + list(boundaries)

        # Add implicit end boundary
        all_boundaries.append((total_frames, duration, SceneTransitionType.CUT))

        for i in range(len(all_boundaries) - 1):
            start_frame, start_time, _ = all_boundaries[i]
            end_frame, end_time, transition = all_boundaries[i + 1]

            scene_duration = end_time - start_time
            keyframe_time = start_time + scene_duration / 2
            keyframe_idx = int(keyframe_time * fps)

            # Extract keyframe
            cap.set(cv2.CAP_PROP_POS_FRAMES, keyframe_idx)
            ret, keyframe = cap.read()

            if ret:
                keyframe_rgb = cv2.cvtColor(keyframe, cv2.COLOR_BGR2RGB)

                # Analyze the keyframe
                color_analysis = self.analyze_colors(keyframe_rgb) if self.classify_scenes else None
                classification = self.classify_scene(keyframe_rgb) if self.classify_scenes else None
                motion_level = self._estimate_motion_level(cap, start_frame, end_frame, fps)

                # Generate tags
                tags = self._generate_scene_tags(color_analysis, classification, motion_level, scene_duration)
            else:
                keyframe_rgb = None
                color_analysis = None
                classification = None
                motion_level = "unknown"
                tags = []

            scene = DetectedScene(
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                duration=scene_duration,
                transition_type=transition,
                keyframe=keyframe_rgb,
                keyframe_time=keyframe_time,
                classification=classification,
                color_analysis=color_analysis,
                motion_level=motion_level,
                tags=tags
            )
            scenes.append(scene)

        return scenes

    def analyze_colors(self, frame: np.ndarray) -> ColorAnalysis:
        """
        Analyze colors in a frame.

        Args:
            frame: RGB frame

        Returns:
            ColorAnalysis with color information
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        # Calculate average brightness
        brightness_value = np.mean(hsv[:, :, 2])
        brightness = self._classify_brightness(brightness_value)

        # Calculate saturation
        saturation_value = np.mean(hsv[:, :, 1])
        saturation_level = self._classify_saturation(saturation_value)

        # Calculate contrast
        contrast = np.std(lab[:, :, 0])
        contrast_level = self._classify_contrast(contrast)

        # Check if monochrome
        is_monochrome = saturation_value < 20

        # Find dominant colors using k-means
        dominant_colors = self._find_dominant_colors(frame, k=5)
        color_names = [self._get_color_name(c) for c in dominant_colors]

        # Determine color temperature
        temperature = self._determine_temperature(dominant_colors, hsv)

        return ColorAnalysis(
            dominant_colors=dominant_colors,
            color_names=color_names,
            temperature=temperature,
            saturation_level=saturation_level,
            brightness=brightness,
            contrast_level=contrast_level,
            is_monochrome=is_monochrome
        )

    def classify_scene(self, frame: np.ndarray) -> SceneClassification:
        """
        Classify a scene based on visual features.

        Args:
            frame: RGB frame

        Returns:
            SceneClassification with environment and style tags
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Analyze various features
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        hue_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()

        # Edge density (indicator of complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255

        # Sky detection (blue regions in upper portion)
        upper_third = hsv[:frame.shape[0]//3, :, :]
        blue_mask = cv2.inRange(upper_third, np.array([100, 50, 50]), np.array([130, 255, 255]))
        sky_ratio = np.mean(blue_mask) / 255

        # Green detection (nature indicator)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        green_ratio = np.mean(green_mask) / 255

        # Indoor/outdoor classification
        indoor_outdoor = self._classify_indoor_outdoor(sky_ratio, green_ratio, brightness, edge_density)

        # Lighting type
        lighting_type = self._classify_lighting(brightness, saturation, hsv)

        # Generate environment tags
        environment_tags = self._generate_environment_tags(
            indoor_outdoor, sky_ratio, green_ratio, edge_density, brightness
        )

        # Generate color tags
        color_tags = self._generate_color_tags(hsv, saturation, brightness)

        # Generate composition tags
        composition_tags = self._generate_composition_tags(frame, edge_density)

        # Calculate confidence based on feature distinctiveness
        confidence = min(1.0, (abs(sky_ratio - 0.5) + abs(green_ratio - 0.5) + edge_density) / 1.5)

        return SceneClassification(
            indoor_outdoor=indoor_outdoor,
            lighting_type=lighting_type,
            environment_tags=environment_tags,
            color_tags=color_tags,
            composition_tags=composition_tags,
            confidence=confidence
        )

    def _classify_brightness(self, value: float) -> BrightnessLevel:
        """Classify brightness level."""
        if value < 40:
            return BrightnessLevel.DARK
        elif value < 80:
            return BrightnessLevel.LOW
        elif value < 170:
            return BrightnessLevel.MEDIUM
        elif value < 220:
            return BrightnessLevel.BRIGHT
        else:
            return BrightnessLevel.OVEREXPOSED

    def _classify_saturation(self, value: float) -> str:
        """Classify saturation level."""
        if value < 50:
            return "low"
        elif value < 150:
            return "medium"
        else:
            return "high"

    def _classify_contrast(self, value: float) -> str:
        """Classify contrast level."""
        if value < 30:
            return "low"
        elif value < 60:
            return "medium"
        else:
            return "high"

    def _find_dominant_colors(self, frame: np.ndarray, k: int = 5) -> List[Tuple[int, int, int]]:
        """Find dominant colors using k-means clustering."""
        # Reshape and convert to float32
        pixels = frame.reshape(-1, 3).astype(np.float32)

        # Subsample for performance
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

        # Sort by frequency
        label_counts = np.bincount(labels.flatten(), minlength=k)
        sorted_indices = np.argsort(-label_counts)

        dominant_colors = [
            tuple(int(c) for c in centers[i])
            for i in sorted_indices
        ]

        return dominant_colors

    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Get the closest color name for an RGB value."""
        r, g, b = rgb

        # Check for grayscale
        if abs(r - g) < 20 and abs(g - b) < 20 and abs(r - b) < 20:
            if (r + g + b) / 3 > 200:
                return "white"
            elif (r + g + b) / 3 < 50:
                return "black"
            else:
                return "gray"

        # Find closest named color
        min_dist = float('inf')
        closest_name = "unknown"

        for name, (low, high) in self.COLOR_NAMES.items():
            center = tuple((l + h) // 2 for l, h in zip(low, high))
            dist = sum((a - b) ** 2 for a, b in zip(rgb, center)) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name

    def _determine_temperature(
        self,
        colors: List[Tuple[int, int, int]],
        hsv: np.ndarray
    ) -> ColorTemperature:
        """Determine the color temperature of the frame."""
        # Average hue (0-30 and 150-180 are warm, 90-130 are cool)
        avg_hue = np.mean(hsv[:, :, 0])

        warm_count = sum(1 for r, g, b in colors if r > b)
        cool_count = sum(1 for r, g, b in colors if b > r)

        if warm_count > cool_count * 1.5 or avg_hue < 30 or avg_hue > 150:
            return ColorTemperature.WARM
        elif cool_count > warm_count * 1.5 or 90 < avg_hue < 130:
            return ColorTemperature.COOL
        else:
            return ColorTemperature.NEUTRAL

    def _classify_indoor_outdoor(
        self,
        sky_ratio: float,
        green_ratio: float,
        brightness: float,
        edge_density: float
    ) -> str:
        """Classify scene as indoor or outdoor."""
        outdoor_score = sky_ratio * 2 + green_ratio * 1.5 + (brightness / 255) * 0.5
        indoor_score = edge_density * 1.5 + (1 - sky_ratio) + (1 - green_ratio) * 0.5

        if outdoor_score > indoor_score * 1.3:
            return "outdoor"
        elif indoor_score > outdoor_score * 1.3:
            return "indoor"
        else:
            return "unknown"

    def _classify_lighting(self, brightness: float, saturation: float, hsv: np.ndarray) -> str:
        """Classify lighting type."""
        if brightness < 60:
            return "low_light"

        # Check for natural light characteristics
        upper_brightness = np.mean(hsv[:hsv.shape[0]//2, :, 2])
        lower_brightness = np.mean(hsv[hsv.shape[0]//2:, :, 2])

        if upper_brightness > lower_brightness * 1.2:
            return "natural"
        elif saturation < 80 and brightness > 150:
            return "artificial"
        else:
            return "mixed"

    def _estimate_motion_level(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        fps: float
    ) -> str:
        """Estimate motion level in a scene."""
        sample_frames = min(10, end_frame - start_frame)
        if sample_frames < 2:
            return "static"

        frame_indices = np.linspace(start_frame, end_frame - 1, sample_frames, dtype=int)
        motion_scores = []

        prev_gray = None
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_scores.append(np.mean(magnitude))

            prev_gray = gray

        if not motion_scores:
            return "unknown"

        avg_motion = np.mean(motion_scores)

        if avg_motion < 0.5:
            return "static"
        elif avg_motion < 2:
            return "low"
        elif avg_motion < 5:
            return "medium"
        else:
            return "high"

    def _generate_environment_tags(
        self,
        indoor_outdoor: str,
        sky_ratio: float,
        green_ratio: float,
        edge_density: float,
        brightness: float
    ) -> List[str]:
        """Generate environment-related tags."""
        tags = []

        if indoor_outdoor == "outdoor":
            tags.append("outdoor")
            if sky_ratio > 0.3:
                tags.append("outdoor/nature")
            if green_ratio > 0.2:
                tags.append("outdoor/nature/field")
        elif indoor_outdoor == "indoor":
            tags.append("indoor")
            if edge_density > 0.15:
                tags.append("indoor/commercial")

        if brightness < 60:
            tags.append("lighting/low_light")
        elif brightness > 200:
            tags.append("lighting/natural")

        return tags

    def _generate_color_tags(
        self,
        hsv: np.ndarray,
        saturation: float,
        brightness: float
    ) -> List[str]:
        """Generate color-related tags."""
        tags = []

        if saturation < 30:
            tags.append("color_style/muted")
            if brightness < 50 or brightness > 200:
                tags.append("color_style/monochrome")
        elif saturation > 150:
            tags.append("color_style/vibrant")

        # Temperature
        avg_hue = np.mean(hsv[:, :, 0])
        if avg_hue < 30 or avg_hue > 150:
            tags.append("color_style/warm")
        elif 90 < avg_hue < 130:
            tags.append("color_style/cool")

        return tags

    def _generate_composition_tags(self, frame: np.ndarray, edge_density: float) -> List[str]:
        """Generate composition-related tags."""
        tags = []

        height, width = frame.shape[:2]

        # Check for centered composition (brightness in center vs edges)
        center_region = frame[height//4:3*height//4, width//4:3*width//4]
        edge_regions = np.concatenate([
            frame[:height//4, :].flatten(),
            frame[3*height//4:, :].flatten(),
            frame[:, :width//4].flatten(),
            frame[:, 3*width//4:].flatten()
        ])

        center_brightness = np.mean(center_region)
        edge_brightness = np.mean(edge_regions)

        if center_brightness > edge_brightness * 1.2:
            tags.append("framing/centered")

        # Complexity based on edge density
        if edge_density < 0.05:
            tags.append("shot/wide")
        elif edge_density > 0.15:
            tags.append("shot/close_up")

        return tags

    def _generate_scene_tags(
        self,
        color_analysis: Optional[ColorAnalysis],
        classification: Optional[SceneClassification],
        motion_level: str,
        duration: float
    ) -> List[str]:
        """Generate tags for a scene."""
        tags = []

        # Motion tags
        if motion_level == "static":
            tags.append("camera/static")
        elif motion_level == "high":
            tags.append("activity/action")

        # Duration tags
        if duration < 2:
            tags.append("duration/short")
        elif duration > 10:
            tags.append("duration/long")

        # Color analysis tags
        if color_analysis:
            if color_analysis.is_monochrome:
                tags.append("color_style/monochrome")
            tags.append(f"color_style/{color_analysis.temperature.value}")

            if color_analysis.brightness == BrightnessLevel.DARK:
                tags.append("lighting/low_light")
            elif color_analysis.brightness == BrightnessLevel.BRIGHT:
                tags.append("lighting/natural")

        # Classification tags
        if classification:
            tags.extend(classification.environment_tags)
            tags.extend(classification.color_tags)
            tags.extend(classification.composition_tags)

        return list(set(tags))  # Remove duplicates

    def _create_result(
        self,
        scenes: List[DetectedScene],
        total_duration: float
    ) -> SceneDetectionResult:
        """Create the final scene detection result."""
        if not scenes:
            return SceneDetectionResult(
                scenes=[],
                scene_count=0,
                avg_scene_duration=0,
                total_duration=total_duration,
                dominant_transition=SceneTransitionType.CUT,
                all_tags=[],
                scene_summary={}
            )

        # Calculate statistics
        durations = [s.duration for s in scenes]
        avg_duration = np.mean(durations)

        # Find dominant transition type
        transition_counts: Dict[SceneTransitionType, int] = {}
        for scene in scenes:
            transition_counts[scene.transition_type] = transition_counts.get(scene.transition_type, 0) + 1
        dominant_transition = max(transition_counts, key=transition_counts.get)

        # Collect all tags
        all_tags: Dict[str, int] = {}
        for scene in scenes:
            for tag in scene.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        # Keep frequent tags (appear in >20% of scenes)
        threshold = len(scenes) * 0.2
        frequent_tags = [tag for tag, count in all_tags.items() if count >= threshold]

        # Scene summary
        scene_summary = {
            "indoor_scenes": sum(1 for s in scenes if s.classification and s.classification.indoor_outdoor == "indoor"),
            "outdoor_scenes": sum(1 for s in scenes if s.classification and s.classification.indoor_outdoor == "outdoor"),
            "static_scenes": sum(1 for s in scenes if s.motion_level == "static"),
            "high_motion_scenes": sum(1 for s in scenes if s.motion_level == "high"),
            "short_scenes": sum(1 for s in scenes if s.duration < 2),
            "long_scenes": sum(1 for s in scenes if s.duration > 10),
            "transition_types": {t.value: c for t, c in transition_counts.items()},
        }

        return SceneDetectionResult(
            scenes=scenes,
            scene_count=len(scenes),
            avg_scene_duration=round(avg_duration, 2),
            total_duration=total_duration,
            dominant_transition=dominant_transition,
            all_tags=frequent_tags,
            scene_summary=scene_summary
        )


# Convenience function
def detect_scenes(video_path: str, **kwargs) -> SceneDetectionResult:
    """Quick scene detection on a video file."""
    detector = SceneDetector(**kwargs)
    return detector.detect_scenes(video_path)
