# API Reference

## Core Classes

### VideoAnalyzer

The main class for analyzing video files.

```python
from src.analyzers import VideoAnalyzer
```

#### Constructor

```python
VideoAnalyzer(
    ml_analyzer: Optional[MLAnalyzerProtocol] = None,
    content_analyzer: Optional[ContentAnalyzerProtocol] = None,
    keyframe_threshold: float = 30.0,
    min_scene_duration: float = 1.0,
    frames_per_scene: int = 3,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ml_analyzer` | `MLAnalyzerProtocol` | `None` | Optional ML vision analyzer for frame analysis |
| `content_analyzer` | `ContentAnalyzerProtocol` | `None` | Optional content analyzer for quality metrics |
| `keyframe_threshold` | `float` | `30.0` | Scene change detection threshold (higher = fewer scenes) |
| `min_scene_duration` | `float` | `1.0` | Minimum duration between scenes in seconds |
| `frames_per_scene` | `int` | `3` | Number of frames to analyze per detected scene |

#### Methods

##### analyze()

Analyze a single video file.

```python
def analyze(
    video_path: str,
    extract_keyframes_only: bool = True,
    max_frames: int = 50,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `str` | required | Path to video file |
| `extract_keyframes_only` | `bool` | `True` | Use scene detection vs uniform sampling |
| `max_frames` | `int` | `50` | Maximum frames to analyze |

**Returns:** `Dict[str, Any]` - Comprehensive analysis results

**Example:**
```python
analyzer = VideoAnalyzer(keyframe_threshold=25.0)
result = analyzer.analyze("video.mp4", max_frames=30)
print(result["content_tags"])
```

##### analyze_batch()

Analyze multiple video files.

```python
def analyze_batch(
    video_paths: List[str],
    show_progress: bool = True,
    **kwargs,
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_paths` | `List[str]` | required | List of video file paths |
| `show_progress` | `bool` | `True` | Show progress bar |
| `**kwargs` | | | Arguments passed to `analyze()` |

**Returns:** `List[Dict[str, Any]]` - List of analysis results

---

### VideoDatabase

SQLite database for storing and searching video metadata.

```python
from src.database import VideoDatabase
```

#### Constructor

```python
VideoDatabase(db_path: str = "video_library.db")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | `str` | `"video_library.db"` | Path to SQLite database file |

#### Methods

##### import_analysis()

Import video analysis results into the database.

```python
def import_analysis(
    analysis_results: List[Dict[str, Any]]
) -> Dict[str, int]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `analysis_results` | `List[Dict[str, Any]]` | List of analysis result dictionaries |

**Returns:** `Dict[str, int]` - Import statistics (`{"inserted": N, "updated": N, "errors": N}`)

**Example:**
```python
with VideoDatabase("videos.db") as db:
    stats = db.import_analysis(results)
    print(f"Imported {stats['inserted']} new videos")
```

##### search()

Search videos by tags and metadata.

```python
def search(
    tags: Optional[List[str]] = None,
    categories: Optional[Dict[str, List[str]]] = None,
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_quality: Optional[float] = None,
    resolution: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tags` | `List[str]` | `None` | Tags to search for (OR logic) |
| `categories` | `Dict[str, List[str]]` | `None` | Category-specific tag search |
| `min_duration` | `float` | `None` | Minimum duration in seconds |
| `max_duration` | `float` | `None` | Maximum duration in seconds |
| `min_quality` | `float` | `None` | Minimum average quality score |
| `resolution` | `str` | `None` | Resolution filter (e.g., "1080p", "4k") |
| `limit` | `int` | `100` | Maximum results to return |

**Returns:** `List[Dict[str, Any]]` - List of matching video records

**Example:**
```python
with VideoDatabase("videos.db") as db:
    # Find fitness videos longer than 5 minutes
    results = db.search(
        tags=["fitness", "exercise"],
        min_duration=300
    )
```

##### get_all_tags()

Get all unique tags grouped by category.

```python
def get_all_tags() -> Dict[str, List[str]]
```

**Returns:** `Dict[str, List[str]]` - Tags organized by category

##### get_stats()

Get database statistics.

```python
def get_stats() -> Dict[str, Any]
```

**Returns:** `Dict[str, Any]` - Database statistics including totals and averages

---

## Utility Functions

### Video I/O

```python
from src.utils import (
    get_video_info,
    extract_frames,
    extract_keyframes,
    save_frame,
    is_supported_video,
    format_duration,
    SUPPORTED_VIDEO_FORMATS,
)
```

#### get_video_info()

Extract video metadata.

```python
def get_video_info(video_path: str) -> Dict[str, Any]
```

**Returns:**
```python
{
    "file_path": str,
    "file_name": str,
    "file_size_mb": float,
    "width": int,
    "height": int,
    "resolution": str,  # e.g., "1920x1080"
    "fps": float,
    "frame_count": int,
    "duration_sec": float,
    "duration_formatted": str,  # e.g., "05:30"
    "codec": str,
    "aspect_ratio": float,
}
```

#### extract_frames()

Extract frames at regular intervals.

```python
def extract_frames(
    video_path: str,
    num_frames: Optional[int] = None,
    fps: Optional[float] = None,
    start_sec: float = 0,
    end_sec: Optional[float] = None,
) -> List[Tuple[float, np.ndarray]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | `str` | Path to video file |
| `num_frames` | `int` | Number of frames to extract (evenly spaced) |
| `fps` | `float` | Extract at this FPS rate |
| `start_sec` | `float` | Start extraction at this timestamp |
| `end_sec` | `float` | End extraction at this timestamp |

**Returns:** `List[Tuple[float, np.ndarray]]` - List of (timestamp, frame) tuples

#### extract_keyframes()

Extract keyframes based on scene changes.

```python
def extract_keyframes(
    video_path: str,
    threshold: float = 30.0,
    min_scene_duration: float = 1.0,
) -> List[Tuple[float, np.ndarray]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_path` | `str` | required | Path to video file |
| `threshold` | `float` | `30.0` | Scene change threshold (higher = fewer scenes) |
| `min_scene_duration` | `float` | `1.0` | Minimum seconds between keyframes |

**Returns:** `List[Tuple[float, np.ndarray]]` - List of (timestamp, frame) tuples

#### save_frame()

Save a frame to disk.

```python
def save_frame(
    frame: np.ndarray,
    output_path: str,
    quality: int = 95
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame` | `np.ndarray` | required | Frame as RGB numpy array |
| `output_path` | `str` | required | Output file path |
| `quality` | `int` | `95` | JPEG quality (1-100) |

---

## Plugin System

### AnalyzerPlugin

Base class for custom analyzers.

```python
from src.plugins import AnalyzerPlugin
```

```python
class AnalyzerPlugin:
    name: str  # Unique plugin name
    version: str  # Plugin version
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a frame and return results."""
        raise NotImplementedError
    
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
```

### ProcessorPlugin

Base class for frame processors.

```python
from src.plugins import ProcessorPlugin
```

```python
class ProcessorPlugin:
    name: str
    
    def process(
        self,
        frame: np.ndarray,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Process a frame and return modified frame."""
        raise NotImplementedError
```

### ExporterPlugin

Base class for result exporters.

```python
from src.plugins import ExporterPlugin
```

```python
class ExporterPlugin:
    name: str
    
    def export(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Export analysis results."""
        raise NotImplementedError
```

### PluginRegistry

Manages plugin discovery and loading.

```python
from src.plugins import PluginRegistry
```

```python
registry = PluginRegistry()
registry.discover()  # Auto-discover plugins
registry.register(MyPlugin())  # Manual registration
plugin = registry.get("plugin_name")
```

---

## Hook System

### HookManager

Manages event hooks for extensibility.

```python
from src.hooks import HookManager
```

#### Available Hooks

| Hook Name | Arguments | Description |
|-----------|-----------|-------------|
| `pre_analyze` | `video_path, config` | Before analysis starts |
| `post_analyze` | `video_path, result` | After analysis completes |
| `pre_frame` | `frame, timestamp` | Before frame processing |
| `post_frame` | `frame, timestamp, result` | After frame processing |
| `on_error` | `error, context` | When an error occurs |

#### Usage

```python
from src.hooks import HookManager

hooks = HookManager()

# Register a hook
@hooks.register("post_analyze")
def my_handler(video_path, result):
    print(f"Analyzed: {video_path}")

# Execute hooks
hooks.execute("post_analyze", video_path="/path/to/video", result={})
```

---

## MCP Server

### VideoEngineMCPServer

Model Context Protocol server for AI assistant integration.

```python
from src.mcp import VideoEngineMCPServer
```

#### Constructor

```python
VideoEngineMCPServer(
    db_path: str = "video_library.db",
    host: str = "localhost",
    port: int = 8080,
)
```

#### Available Tools

| Tool | Description |
|------|-------------|
| `analyze_video` | Analyze a video file |
| `search_videos` | Search video database |
| `get_video_info` | Get video metadata |
| `list_tags` | List all available tags |
| `get_stats` | Get database statistics |

#### Running the Server

```python
server = VideoEngineMCPServer(db_path="videos.db")
server.run()  # Starts the MCP server
```

---

## Protocols

### MLAnalyzerProtocol

Protocol for ML vision analyzers.

```python
class MLAnalyzerProtocol(Protocol):
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze an image and return results.
        
        Expected return format:
        {
            "primary_scene": str,
            "detected_objects": List[str],
            "confidence": float,
            ...
        }
        """
        ...
```

### ContentAnalyzerProtocol

Protocol for content quality analyzers.

```python
class ContentAnalyzerProtocol(Protocol):
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image content quality.
        
        Expected return format:
        {
            "quality_score": float,  # 0-100
            "blur_score": float,
            "brightness": float,
            ...
        }
        """
        ...
```

---

## Constants

### SUPPORTED_VIDEO_FORMATS

Set of supported video file extensions.

```python
SUPPORTED_VIDEO_FORMATS = {
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', 
    '.flv', '.webm', '.m4v', '.mpg', '.mpeg', 
    '.3gp', '.ogv'
}
```
