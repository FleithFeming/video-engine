# Video Engine

GPU-accelerated video analysis with ML frame detection.

## Features

- Frame extraction (uniform sampling or keyframe/scene detection)
- Video metadata extraction (resolution, duration, codec, FPS)
- Scene change detection
- Optional ML-based frame analysis (objects, scenes)
- SQLite database for searchable indexing

## Installation

```bash
pip install -r requirements.txt
```

For ML-based analysis, also install:
```bash
pip install torch transformers ultralytics
```

## Quick Start

```python
from src.analyzers import VideoAnalyzer
from src.utils import get_video_info, extract_keyframes

# Get video metadata
info = get_video_info("video.mp4")
print(f"Duration: {info['duration_formatted']}")
print(f"Resolution: {info['resolution']}")

# Extract keyframes
keyframes = extract_keyframes("video.mp4", threshold=30.0)
print(f"Found {len(keyframes)} scenes")

# Full analysis (without ML)
analyzer = VideoAnalyzer()
result = analyzer.analyze("video.mp4")
```

## Database Usage

```python
from src.database import VideoDatabase

# Import analysis results
with VideoDatabase("videos.db") as db:
    db.import_analysis(results)

    # Search by tags
    fitness_videos = db.search(tags=["fitness", "exercise"])

    # Search by metadata
    long_4k = db.search(min_duration=300, resolution="4k")
```

## Architecture

- `src/analyzers/video_analyzer.py` - Main analysis pipeline
- `src/utils/video_io.py` - Frame extraction and video I/O
- `src/database/video_db.py` - SQLite storage and search

## Requirements

- Python 3.10+
- OpenCV
- NumPy

Optional:
- PyTorch (for ML analysis)
- CUDA (for GPU acceleration)
