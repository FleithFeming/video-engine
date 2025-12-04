"""Facial recognition and detection patterns for video analysis."""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FacePosition(Enum):
    """Position of face in frame."""
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class FaceSize(Enum):
    """Relative size of face in frame."""
    CLOSE_UP = "close_up"      # >30% of frame
    MEDIUM = "medium"          # 10-30% of frame
    SMALL = "small"            # 3-10% of frame
    DISTANT = "distant"        # <3% of frame


@dataclass
class DetectedFace:
    """Information about a detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    position: FacePosition
    size: FaceSize
    # Optional additional detections
    has_eyes: bool = False
    has_smile: bool = False
    profile: bool = False


@dataclass
class FaceAnalysisResult:
    """Complete face analysis result for a frame."""
    faces: List[DetectedFace]
    face_count: int
    has_faces: bool
    primary_face: Optional[DetectedFace]
    face_positions: List[str]
    face_sizes: List[str]
    tags: List[str]


class FaceDetector:
    """
    Face detection using OpenCV cascade classifiers.

    Provides face detection, counting, position analysis, and pattern recognition.
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30),
        detect_eyes: bool = True,
        detect_smile: bool = False,
        detect_profile: bool = True,
    ):
        """
        Initialize face detector.

        Args:
            scale_factor: Scale factor for detection (1.1 = 10% size reduction per scale)
            min_neighbors: Minimum neighbors for detection confidence
            min_face_size: Minimum face size to detect (width, height)
            detect_eyes: Whether to detect eyes within faces
            detect_smile: Whether to detect smiles within faces
            detect_profile: Whether to detect profile (side) faces
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.detect_eyes = detect_eyes
        self.detect_smile = detect_smile
        self.detect_profile = detect_profile

        # Load cascade classifiers
        self._load_cascades()

    def _load_cascades(self):
        """Load OpenCV Haar cascade classifiers."""
        # Get OpenCV data path
        cv2_data = cv2.data.haarcascades

        # Frontal face detector (primary)
        self.face_cascade = cv2.CascadeClassifier(
            cv2_data + 'haarcascade_frontalface_default.xml'
        )

        # Alternative frontal face detector
        self.face_cascade_alt = cv2.CascadeClassifier(
            cv2_data + 'haarcascade_frontalface_alt2.xml'
        )

        # Profile face detector
        self.profile_cascade = cv2.CascadeClassifier(
            cv2_data + 'haarcascade_profileface.xml'
        )

        # Eye detector
        self.eye_cascade = cv2.CascadeClassifier(
            cv2_data + 'haarcascade_eye.xml'
        )

        # Smile detector
        self.smile_cascade = cv2.CascadeClassifier(
            cv2_data + 'haarcascade_smile.xml'
        )

        # Verify cascades loaded
        if self.face_cascade.empty():
            logger.warning("Failed to load frontal face cascade")
        if self.profile_cascade.empty():
            logger.warning("Failed to load profile face cascade")

    def detect(self, frame: np.ndarray) -> FaceAnalysisResult:
        """
        Detect faces in a frame.

        Args:
            frame: Input frame as numpy array (RGB or BGR)

        Returns:
            FaceAnalysisResult with detected faces and metadata
        """
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Enhance contrast for better detection
        gray = cv2.equalizeHist(gray)

        frame_height, frame_width = gray.shape[:2]
        frame_area = frame_height * frame_width

        detected_faces: List[DetectedFace] = []

        # Detect frontal faces
        frontal_faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Process frontal faces
        for (x, y, w, h) in frontal_faces:
            face = self._create_detected_face(
                x, y, w, h,
                frame_width, frame_height, frame_area,
                gray,
                profile=False
            )
            detected_faces.append(face)

        # Detect profile faces if enabled
        if self.detect_profile:
            profile_faces = self.profile_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
            )

            # Add profile faces that don't overlap with frontal faces
            for (x, y, w, h) in profile_faces:
                if not self._overlaps_existing(x, y, w, h, detected_faces):
                    face = self._create_detected_face(
                        x, y, w, h,
                        frame_width, frame_height, frame_area,
                        gray,
                        profile=True
                    )
                    detected_faces.append(face)

            # Also check flipped image for right-facing profiles
            gray_flipped = cv2.flip(gray, 1)
            profile_faces_right = self.profile_cascade.detectMultiScale(
                gray_flipped,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=self.min_face_size,
            )

            for (x, y, w, h) in profile_faces_right:
                # Mirror x coordinate back
                x_mirrored = frame_width - x - w
                if not self._overlaps_existing(x_mirrored, y, w, h, detected_faces):
                    face = self._create_detected_face(
                        x_mirrored, y, w, h,
                        frame_width, frame_height, frame_area,
                        gray,
                        profile=True
                    )
                    detected_faces.append(face)

        # Sort faces by size (largest first)
        detected_faces.sort(key=lambda f: f.width * f.height, reverse=True)

        # Generate result
        return self._create_result(detected_faces)

    def _create_detected_face(
        self,
        x: int, y: int, w: int, h: int,
        frame_width: int, frame_height: int, frame_area: int,
        gray: np.ndarray,
        profile: bool
    ) -> DetectedFace:
        """Create a DetectedFace object with full analysis."""
        # Calculate position
        position = self._calculate_position(x, y, w, h, frame_width, frame_height)

        # Calculate relative size
        face_area = w * h
        size_ratio = face_area / frame_area
        size = self._calculate_size(size_ratio)

        # Calculate confidence based on size and detection quality
        confidence = min(1.0, size_ratio * 10 + 0.5)

        # Detect eyes if enabled
        has_eyes = False
        if self.detect_eyes and not profile:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.1), int(h * 0.1))
            )
            has_eyes = len(eyes) >= 1

        # Detect smile if enabled
        has_smile = False
        if self.detect_smile and not profile:
            roi_gray = gray[y:y+h, x:x+w]
            # Focus on lower half of face for smile
            lower_half = roi_gray[h//2:, :]
            smiles = self.smile_cascade.detectMultiScale(
                lower_half,
                scaleFactor=1.7,
                minNeighbors=20,
                minSize=(int(w * 0.2), int(h * 0.1))
            )
            has_smile = len(smiles) >= 1

        return DetectedFace(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=confidence,
            position=position,
            size=size,
            has_eyes=has_eyes,
            has_smile=has_smile,
            profile=profile
        )

    def _calculate_position(
        self,
        x: int, y: int, w: int, h: int,
        frame_width: int, frame_height: int
    ) -> FacePosition:
        """Calculate the position of a face in the frame."""
        center_x = x + w / 2
        center_y = y + h / 2

        # Normalize to 0-1 range
        rel_x = center_x / frame_width
        rel_y = center_y / frame_height

        # Determine horizontal position
        if rel_x < 0.33:
            h_pos = "left"
        elif rel_x > 0.67:
            h_pos = "right"
        else:
            h_pos = "center"

        # Determine vertical position
        if rel_y < 0.33:
            v_pos = "top"
        elif rel_y > 0.67:
            v_pos = "bottom"
        else:
            v_pos = "center"

        # Combine positions
        if h_pos == "center" and v_pos == "center":
            return FacePosition.CENTER
        elif v_pos == "center":
            return FacePosition.LEFT if h_pos == "left" else FacePosition.RIGHT
        elif h_pos == "center":
            return FacePosition.TOP if v_pos == "top" else FacePosition.BOTTOM
        else:
            pos_map = {
                ("top", "left"): FacePosition.TOP_LEFT,
                ("top", "right"): FacePosition.TOP_RIGHT,
                ("bottom", "left"): FacePosition.BOTTOM_LEFT,
                ("bottom", "right"): FacePosition.BOTTOM_RIGHT,
            }
            return pos_map.get((v_pos, h_pos), FacePosition.CENTER)

    def _calculate_size(self, size_ratio: float) -> FaceSize:
        """Calculate the relative size category of a face."""
        if size_ratio > 0.30:
            return FaceSize.CLOSE_UP
        elif size_ratio > 0.10:
            return FaceSize.MEDIUM
        elif size_ratio > 0.03:
            return FaceSize.SMALL
        else:
            return FaceSize.DISTANT

    def _overlaps_existing(
        self,
        x: int, y: int, w: int, h: int,
        existing_faces: List[DetectedFace],
        threshold: float = 0.5
    ) -> bool:
        """Check if a face region overlaps significantly with existing detections."""
        for face in existing_faces:
            # Calculate intersection
            x1 = max(x, face.x)
            y1 = max(y, face.y)
            x2 = min(x + w, face.x + face.width)
            y2 = min(y + h, face.y + face.height)

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = w * h
                area2 = face.width * face.height
                union = area1 + area2 - intersection
                iou = intersection / union

                if iou > threshold:
                    return True

        return False

    def _create_result(self, faces: List[DetectedFace]) -> FaceAnalysisResult:
        """Create a complete face analysis result."""
        tags = []

        if not faces:
            tags.append("face/none")
            return FaceAnalysisResult(
                faces=[],
                face_count=0,
                has_faces=False,
                primary_face=None,
                face_positions=[],
                face_sizes=[],
                tags=tags
            )

        # Face count tags
        if len(faces) == 1:
            tags.append("person/single")
            tags.append("face/visible")
        elif len(faces) == 2:
            tags.append("person/couple")
            tags.append("face/multiple")
        else:
            tags.append("person/group")
            tags.append("face/multiple")

        # Primary face analysis
        primary_face = faces[0]

        # Size-based tags
        if primary_face.size == FaceSize.CLOSE_UP:
            tags.append("face/close_up")
            tags.append("shot/close_up")
        elif primary_face.size == FaceSize.MEDIUM:
            tags.append("shot/medium")

        # Position tags
        if primary_face.position == FacePosition.CENTER:
            tags.append("framing/centered")

        # Profile detection
        profile_count = sum(1 for f in faces if f.profile)
        frontal_count = len(faces) - profile_count
        if profile_count > 0:
            tags.append("face/partial")
        if frontal_count > 0 and primary_face.has_eyes:
            tags.append("face/frontal")

        # Collect all positions and sizes
        face_positions = list(set(f.position.value for f in faces))
        face_sizes = list(set(f.size.value for f in faces))

        return FaceAnalysisResult(
            faces=faces,
            face_count=len(faces),
            has_faces=True,
            primary_face=primary_face,
            face_positions=face_positions,
            face_sizes=face_sizes,
            tags=tags
        )

    def analyze_video_faces(
        self,
        frames: List[Tuple[float, np.ndarray]],
        sample_rate: int = 1
    ) -> Dict[str, Any]:
        """
        Analyze faces across multiple frames.

        Args:
            frames: List of (timestamp, frame) tuples
            sample_rate: Analyze every Nth frame

        Returns:
            Aggregated face analysis for the video
        """
        all_results: List[FaceAnalysisResult] = []
        face_counts: List[int] = []
        timestamps_with_faces: List[float] = []

        for i, (timestamp, frame) in enumerate(frames):
            if i % sample_rate != 0:
                continue

            result = self.detect(frame)
            all_results.append(result)
            face_counts.append(result.face_count)

            if result.has_faces:
                timestamps_with_faces.append(timestamp)

        # Aggregate statistics
        total_frames = len(all_results)
        frames_with_faces = sum(1 for r in all_results if r.has_faces)

        if total_frames == 0:
            return {
                "has_faces": False,
                "face_frequency": 0,
                "avg_face_count": 0,
                "max_face_count": 0,
                "face_tags": ["face/none"],
            }

        face_frequency = frames_with_faces / total_frames
        avg_face_count = np.mean(face_counts) if face_counts else 0
        max_face_count = max(face_counts) if face_counts else 0

        # Aggregate tags
        all_tags: Dict[str, int] = {}
        for result in all_results:
            for tag in result.tags:
                all_tags[tag] = all_tags.get(tag, 0) + 1

        # Keep tags that appear in >20% of frames
        threshold = total_frames * 0.2
        frequent_tags = [tag for tag, count in all_tags.items() if count >= threshold]

        # Determine person count category
        if avg_face_count < 0.5:
            person_tag = "person/none"
        elif avg_face_count < 1.5:
            person_tag = "person/single"
        elif avg_face_count < 2.5:
            person_tag = "person/couple"
        else:
            person_tag = "person/group"

        if person_tag not in frequent_tags:
            frequent_tags.append(person_tag)

        return {
            "has_faces": frames_with_faces > 0,
            "face_frequency": round(face_frequency, 2),
            "avg_face_count": round(avg_face_count, 1),
            "max_face_count": max_face_count,
            "frames_analyzed": total_frames,
            "frames_with_faces": frames_with_faces,
            "timestamps_with_faces": timestamps_with_faces[:10],  # First 10
            "face_tags": frequent_tags,
        }


# Convenience function
def detect_faces(frame: np.ndarray) -> FaceAnalysisResult:
    """Quick face detection on a single frame."""
    detector = FaceDetector()
    return detector.detect(frame)
