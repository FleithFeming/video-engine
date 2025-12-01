"""Video analysis using frame-based ML models and temporal analysis."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol
import numpy as np
from collections import Counter
from tqdm import tqdm
import tempfile
import cv2

from ..utils.video_io import (
    get_video_info,
    extract_frames,
    extract_keyframes,
    save_frame,
)


# Optional ML analyzer protocols for type checking
class MLAnalyzerProtocol(Protocol):
    """Protocol for ML analyzers (optional dependency)."""
    def analyze(self, image_path: str) -> Dict[str, Any]: ...


class ContentAnalyzerProtocol(Protocol):
    """Protocol for content analyzers (optional dependency)."""
    def analyze(self, image_path: str) -> Dict[str, Any]: ...


# Type aliases for optional analyzers
MLVisionAnalyzer = Optional[MLAnalyzerProtocol]
ContentAnalyzer = Optional[ContentAnalyzerProtocol]


class VideoAnalyzer:
    """
    Analyze videos by extracting frames and applying ML models.

    Combines frame-level analysis with temporal consistency checks.
    """

    def __init__(
        self,
        ml_analyzer: Optional[MLAnalyzerProtocol] = None,
        content_analyzer: Optional[ContentAnalyzerProtocol] = None,
        keyframe_threshold: float = 30.0,
        min_scene_duration: float = 1.0,
        frames_per_scene: int = 3,
    ):
        """
        Initialize video analyzer.

        Args:
            ml_analyzer: ML vision analyzer for frame analysis
            content_analyzer: Content analyzer for quality metrics
            keyframe_threshold: Scene change detection threshold
            min_scene_duration: Minimum duration between scenes (seconds)
            frames_per_scene: Number of frames to analyze per scene
        """
        self.ml_analyzer = ml_analyzer
        self.content_analyzer = content_analyzer
        self.keyframe_threshold = keyframe_threshold
        self.min_scene_duration = min_scene_duration
        self.frames_per_scene = frames_per_scene

    def analyze(
        self,
        video_path: str,
        extract_keyframes_only: bool = True,
        max_frames: int = 50,
    ) -> Dict[str, Any]:
        """
        Analyze a video file.

        Args:
            video_path: Path to video file
            extract_keyframes_only: Use scene detection vs uniform sampling
            max_frames: Maximum frames to analyze (for long videos)

        Returns:
            Comprehensive video analysis results
        """
        # Input validation
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")

        print(f"\nAnalyzing video: {os.path.basename(video_path)}")

        # Step 1: Extract video metadata
        video_info = get_video_info(video_path)
        print(f"Duration: {video_info['duration_formatted']}, "
              f"Resolution: {video_info['resolution']}, "
              f"FPS: {video_info['fps']:.1f}")

        # Step 2: Extract frames
        if extract_keyframes_only:
            print("Extracting keyframes (scene detection)...")
            frames = extract_keyframes(
                video_path,
                threshold=self.keyframe_threshold,
                min_scene_duration=self.min_scene_duration,
            )
        else:
            print(f"Extracting {max_frames} evenly-spaced frames...")
            frames = extract_frames(video_path, num_frames=max_frames)

        print(f"Extracted {len(frames)} frames for analysis")

        # Limit total frames analyzed
        if len(frames) > max_frames:
            print(f"Limiting to {max_frames} frames for performance")
            # Keep evenly distributed frames
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Step 3: Analyze frames
        frame_analyses = []
        scene_tags = []
        object_detections = []
        quality_scores = []

        print("Analyzing frames with ML models...")
        for timestamp, frame in tqdm(frames, desc="Frame analysis", unit="frame"):
            frame_result = {
                "timestamp": timestamp,
                "timestamp_formatted": self._format_timestamp(timestamp),
            }

            # Content analysis (quality, blur, colors)
            if self.content_analyzer:
                # Save frame temporarily for analysis
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp_path = tmp.name
                    save_frame(frame, tmp_path)
                    content_result = self.content_analyzer.analyze(tmp_path)
                    frame_result.update(content_result)
                    if "quality_score" in content_result:
                        quality_scores.append(content_result["quality_score"])
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass  # Ignore cleanup errors

            # ML analysis (scenes, objects)
            if self.ml_analyzer:
                # Save frame temporarily for ML analysis
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                        tmp_path = tmp.name
                    save_frame(frame, tmp_path)
                    ml_result = self.ml_analyzer.analyze(tmp_path)
                except Exception as e:
                    # If ML fails, continue without ML tags
                    ml_result = {"ml_error": str(e)}
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass  # Ignore cleanup errors

                frame_result.update(ml_result)

                # Collect scene tags
                if "primary_scene" in ml_result:
                    scene_tags.append(ml_result["primary_scene"])

                # Collect object detections
                if "detected_objects" in ml_result:
                    object_detections.extend(ml_result["detected_objects"])

            frame_analyses.append(frame_result)

        # Step 4: Aggregate temporal analysis
        temporal_analysis = self._aggregate_temporal_data(
            frame_analyses,
            scene_tags,
            object_detections,
            quality_scores,
        )

        # Step 5: Generate content tags
        content_tags = self._generate_content_tags(
            video_info,
            temporal_analysis,
        )

        # Combine all results
        result = {
            **video_info,
            "num_frames_analyzed": len(frame_analyses),
            "temporal_analysis": temporal_analysis,
            "content_tags": content_tags,
            "frame_analyses": frame_analyses,  # Detailed frame-by-frame data
        }

        return result

    def analyze_batch(
        self,
        video_paths: List[str],
        show_progress: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple videos.

        Args:
            video_paths: List of video file paths
            show_progress: Show progress bar
            **kwargs: Arguments passed to analyze()

        Returns:
            List of analysis results
        """
        results = []

        iterator = tqdm(video_paths, desc="Videos") if show_progress else video_paths

        for video_path in iterator:
            try:
                result = self.analyze(video_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"\nError analyzing {video_path}: {e}")
                results.append({
                    "file_path": video_path,
                    "error": str(e),
                })

        return results

    def _aggregate_temporal_data(
        self,
        frame_analyses: List[Dict[str, Any]],
        scene_tags: List[str],
        object_detections: List[str],
        quality_scores: List[float],
    ) -> Dict[str, Any]:
        """
        Aggregate frame-level data into temporal patterns.

        Args:
            frame_analyses: List of frame analysis results
            scene_tags: List of detected scenes
            object_detections: List of detected objects
            quality_scores: List of quality scores

        Returns:
            Aggregated temporal analysis
        """
        # Count scene occurrences
        scene_counter = Counter(scene_tags)
        dominant_scenes = scene_counter.most_common(5)

        # Count object occurrences
        object_counter = Counter(object_detections)
        frequent_objects = object_counter.most_common(10)

        # Quality statistics
        avg_quality = np.mean(quality_scores) if quality_scores else None
        min_quality = np.min(quality_scores) if quality_scores else None
        max_quality = np.max(quality_scores) if quality_scores else None

        # Scene transitions
        num_scenes = len(frame_analyses)
        scene_changes = sum(
            1 for i in range(1, len(scene_tags))
            if scene_tags[i] != scene_tags[i - 1]
        ) if len(scene_tags) > 1 else 0

        return {
            "dominant_scenes": [
                {"scene": scene, "count": count, "percentage": count / len(scene_tags) * 100}
                for scene, count in dominant_scenes
            ] if scene_tags else [],
            "frequent_objects": [
                {"object": obj, "count": count}
                for obj, count in frequent_objects
            ],
            "quality_stats": {
                "average": avg_quality,
                "min": min_quality,
                "max": max_quality,
            },
            "scene_count": num_scenes,
            "scene_changes": scene_changes,
            "avg_scene_duration_sec": (
                frame_analyses[-1]["timestamp"] / scene_changes
                if scene_changes > 0 else frame_analyses[-1]["timestamp"]
            ) if frame_analyses else 0,
        }

    def _generate_content_tags(
        self,
        video_info: Dict[str, Any],
        temporal_analysis: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """
        Generate searchable content tags from analysis.

        Args:
            video_info: Video metadata
            temporal_analysis: Temporal analysis results

        Returns:
            Categorized content tags
        """
        tags = {
            "scenes": [],
            "objects": [],
            "activities": [],
            "quality": [],
            "technical": [],
        }

        # Scene tags
        for scene_data in temporal_analysis["dominant_scenes"]:
            if scene_data["percentage"] > 10:  # Present in >10% of video
                tags["scenes"].append(scene_data["scene"])

        # Object tags
        for obj_data in temporal_analysis["frequent_objects"][:5]:
            tags["objects"].append(obj_data["object"])

        # Quality tags
        avg_quality = temporal_analysis["quality_stats"].get("average")
        if avg_quality:
            if avg_quality > 80:
                tags["quality"].append("high-quality")
            elif avg_quality > 60:
                tags["quality"].append("good-quality")
            else:
                tags["quality"].append("low-quality")

        # Technical tags
        resolution = video_info.get("height", 0)
        if resolution >= 2160:
            tags["technical"].append("4k")
        elif resolution >= 1080:
            tags["technical"].append("1080p")
        elif resolution >= 720:
            tags["technical"].append("720p")

        duration = video_info.get("duration_sec", 0)
        if duration < 30:
            tags["technical"].append("short-clip")
        elif duration < 300:
            tags["technical"].append("medium-length")
        else:
            tags["technical"].append("long-video")

        # Infer activities from scenes and objects
        tags["activities"] = self._infer_activities(tags["scenes"], tags["objects"])

        return tags

    def _infer_activities(
        self,
        scenes: List[str],
        objects: List[str],
    ) -> List[str]:
        """
        Infer activities from detected scenes and objects.

        Args:
            scenes: List of scene tags
            objects: List of object tags

        Returns:
            Inferred activity tags
        """
        activities = []

        # Fitness/exercise indicators
        if any(term in " ".join(scenes + objects).lower()
               for term in ["gym", "yoga", "exercise", "workout", "mat"]):
            activities.append("fitness")

        # Food/cooking indicators
        if any(term in " ".join(scenes + objects).lower()
               for term in ["kitchen", "food", "cooking", "dining"]):
            activities.append("cooking")

        # Outdoor/nature indicators
        if any(term in " ".join(scenes).lower()
               for term in ["outdoor", "nature", "landscape", "park", "beach"]):
            activities.append("outdoor")

        # Family/social indicators
        if "person" in objects or "people" in objects:
            activities.append("social")

        # Product/commercial indicators
        if any(term in " ".join(objects).lower()
               for term in ["product", "bottle", "package", "box"]):
            activities.append("product")

        return activities

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp as HH:MM:SS or MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
