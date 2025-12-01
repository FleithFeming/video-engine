"""Video I/O utilities for frame extraction and metadata."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2


# Supported video formats
SUPPORTED_VIDEO_FORMATS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.ogv'
}


def is_supported_video(file_path: str) -> bool:
    """
    Check if file is a supported video format.

    Args:
        file_path: Path to file

    Returns:
        True if supported video format
    """
    if not os.path.isfile(file_path):
        return False

    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_VIDEO_FORMATS


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract basic video information using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video metadata
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))

        # Calculate duration
        duration_sec = frame_count / fps if fps > 0 else 0

        # Convert codec to string
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        info = {
            "file_path": video_path,
            "file_name": os.path.basename(video_path),
            "file_size_mb": os.path.getsize(video_path) / (1024 * 1024),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
            "duration_formatted": format_duration(duration_sec),
            "codec": codec_str,
            "aspect_ratio": width / height if height > 0 else 0,
        }

        return info

    finally:
        cap.release()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to HH:MM:SS.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def extract_frames(
    video_path: str,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    start_sec: float = 0,
    end_sec: Optional[float] = None,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames from video at specified intervals.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (evenly spaced)
        fps: Extract frames at this FPS rate
        start_sec: Start extraction at this timestamp
        end_sec: End extraction at this timestamp

    Returns:
        List of (timestamp, frame) tuples
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0

        # Determine end time
        if end_sec is None:
            end_sec = duration

        # Calculate frame indices to extract
        if num_frames is not None:
            # Evenly spaced frames
            indices = np.linspace(
                int(start_sec * video_fps),
                int(end_sec * video_fps),
                num_frames,
                dtype=int
            )
        elif fps is not None:
            # Extract at specific FPS
            interval = int(video_fps / fps)
            start_frame = int(start_sec * video_fps)
            end_frame = int(end_sec * video_fps)
            indices = range(start_frame, end_frame, interval)
        else:
            raise ValueError("Must specify either num_frames or fps")

        # Extract frames
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                timestamp = idx / video_fps
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((timestamp, frame_rgb))

        return frames

    finally:
        cap.release()


def extract_keyframes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_duration: float = 1.0,
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract keyframes based on scene changes.

    Uses frame difference to detect scene changes.

    Args:
        video_path: Path to video file
        threshold: Scene change detection threshold (higher = fewer scenes)
        min_scene_duration: Minimum duration between keyframes in seconds

    Returns:
        List of (timestamp, frame) tuples at scene boundaries
    """
    # Input validation
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    if min_scene_duration <= 0:
        raise ValueError(f"min_scene_duration must be positive, got {min_scene_duration}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_frames_between = int(min_scene_duration * fps)

        keyframes = []
        prev_frame = None
        frame_idx = 0
        last_keyframe_idx = -min_frames_between

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale for comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate frame difference
            if prev_frame is not None:
                diff = cv2.absdiff(gray, prev_frame)
                diff_score = np.mean(diff)

                # Detect scene change
                if (diff_score > threshold and
                    frame_idx - last_keyframe_idx >= min_frames_between):
                    timestamp = frame_idx / fps
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    keyframes.append((timestamp, frame_rgb))
                    last_keyframe_idx = frame_idx

            prev_frame = gray
            frame_idx += 1

        # Always include first frame if not already included
        if not keyframes or keyframes[0][0] > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                keyframes.insert(0, (0.0, frame_rgb))

        return keyframes

    finally:
        cap.release()


def save_frame(frame: np.ndarray, output_path: str, quality: int = 95):
    """
    Save frame to file.

    Args:
        frame: Frame as numpy array (RGB)
        output_path: Output file path
        quality: JPEG quality (1-100)
    """
    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save with quality settings
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        cv2.imwrite(output_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(output_path, frame_bgr)
