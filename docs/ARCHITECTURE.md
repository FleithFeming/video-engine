# Architecture Documentation

## System Overview

Video Engine is designed as a modular, extensible video analysis pipeline with a layered architecture that separates concerns and enables customization at multiple levels.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                         │
│  (CLI tools, API endpoints, Integration scripts)                │
├─────────────────────────────────────────────────────────────────┤
│                         Plugin System                            │
│  (Custom analyzers, processors, exporters)                      │
├─────────────────────────────────────────────────────────────────┤
│                          Hooks Layer                             │
│  (Pre/post processing hooks, event callbacks)                   │
├─────────────────────────────────────────────────────────────────┤
│                        Core Analyzers                            │
│  (VideoAnalyzer, ML analyzers, Content analyzers)               │
├─────────────────────────────────────────────────────────────────┤
│                         Utilities Layer                          │
│  (video_io, frame extraction, format handling)                  │
├─────────────────────────────────────────────────────────────────┤
│                        Database Layer                            │
│  (SQLite storage, search indexing, query optimization)          │
├─────────────────────────────────────────────────────────────────┤
│                         MCP Server                               │
│  (Model Context Protocol integration for AI assistants)         │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
video-engine/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── analyzers/            # Video analysis components
│   │   ├── __init__.py
│   │   └── video_analyzer.py # Main analysis pipeline
│   ├── database/             # Data persistence
│   │   ├── __init__.py
│   │   └── video_db.py       # SQLite database wrapper
│   ├── utils/                # Utility functions
│   │   ├── __init__.py
│   │   └── video_io.py       # Video I/O operations
│   ├── plugins/              # Plugin system
│   │   ├── __init__.py
│   │   ├── base.py           # Base plugin classes
│   │   └── registry.py       # Plugin discovery/registration
│   ├── hooks/                # Hook system
│   │   ├── __init__.py
│   │   └── manager.py        # Hook registration and execution
│   └── mcp/                  # MCP server
│       ├── __init__.py
│       └── server.py         # MCP protocol implementation
├── docs/                     # Documentation
├── MODEL_CARD.md             # Model documentation
├── README.md                 # Quick start guide
└── requirements.txt          # Dependencies
```

## Core Components

### 1. Video Analyzer (`src/analyzers/video_analyzer.py`)

The central component that orchestrates video analysis.

```python
class VideoAnalyzer:
    def __init__(
        self,
        ml_analyzer: Optional[MLAnalyzerProtocol] = None,
        content_analyzer: Optional[ContentAnalyzerProtocol] = None,
        keyframe_threshold: float = 30.0,
        min_scene_duration: float = 1.0,
        frames_per_scene: int = 3,
    )
```

**Responsibilities:**
- Coordinate frame extraction
- Manage ML analyzer invocations
- Aggregate temporal analysis results
- Generate content tags

**Design Patterns:**
- Strategy Pattern: ML and content analyzers are injectable
- Template Method: `analyze()` follows a fixed sequence with customizable steps

### 2. Video I/O (`src/utils/video_io.py`)

Low-level video operations using OpenCV.

**Key Functions:**
- `get_video_info()`: Extract metadata (resolution, duration, FPS, codec)
- `extract_frames()`: Uniform frame sampling
- `extract_keyframes()`: Scene-change-based extraction
- `save_frame()`: Frame persistence

**Performance Considerations:**
- Uses direct frame seeking for uniform sampling
- Frame difference calculation for scene detection
- Memory-efficient streaming for large videos

### 3. Database (`src/database/video_db.py`)

SQLite-based persistence with full-text search capabilities.

**Schema:**
```sql
videos (
    id, file_path, file_name, file_size_mb,
    duration_sec, width, height, resolution,
    fps, codec, quality_avg, analysis_date,
    full_analysis JSON
)

tags (
    id, video_id, category, tag
    -- Enables fast tag-based search
)
```

**Indexes:**
- Tags: category, tag, video_id
- Videos: resolution, duration, quality, analysis_date

### 4. Plugin System (`src/plugins/`)

Extensible plugin architecture for custom functionality.

**Plugin Types:**
- `AnalyzerPlugin`: Custom frame analysis
- `ProcessorPlugin`: Pre/post processing
- `ExporterPlugin`: Custom output formats

**Discovery:**
- Entry points: `video_engine.plugins`
- Dynamic loading from plugin directory
- Configuration-based activation

### 5. Hook System (`src/hooks/`)

Event-based extensibility for cross-cutting concerns.

**Hook Points:**
- `pre_analyze`: Before video analysis starts
- `post_analyze`: After analysis completes
- `pre_frame`: Before frame processing
- `post_frame`: After frame processing
- `on_error`: When errors occur

### 6. MCP Server (`src/mcp/`)

Model Context Protocol server for AI assistant integration.

**Capabilities:**
- Video analysis tool
- Search and query tools
- Database management tools

## Data Flow

### Analysis Pipeline

```
┌──────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────┐
│  Video   │───▶│ Frame         │───▶│ ML Analysis │───▶│ Temporal │
│  Input   │    │ Extraction    │    │ (optional)  │    │ Analysis │
└──────────┘    └───────────────┘    └─────────────┘    └──────────┘
                       │                    │                  │
                       ▼                    ▼                  ▼
                ┌─────────────┐      ┌───────────┐      ┌───────────┐
                │ Hook:       │      │ Plugin:   │      │ Content   │
                │ pre_frame   │      │ Analyzer  │      │ Tags      │
                └─────────────┘      └───────────┘      └───────────┘
                                                              │
                                                              ▼
                                                       ┌───────────┐
                                                       │ Database  │
                                                       │ Storage   │
                                                       └───────────┘
```

### Search Flow

```
┌────────────┐    ┌───────────────┐    ┌─────────────┐    ┌──────────┐
│   Query    │───▶│ Query Parser  │───▶│ SQL Builder │───▶│  SQLite  │
│            │    │               │    │             │    │  Query   │
└────────────┘    └───────────────┘    └─────────────┘    └──────────┘
                                                                │
                                                                ▼
                                                         ┌───────────┐
                                                         │  Results  │
                                                         │  + JSON   │
                                                         └───────────┘
```

## Extension Points

### Adding a Custom Analyzer

```python
from src.plugins import AnalyzerPlugin

class MyAnalyzer(AnalyzerPlugin):
    name = "my_analyzer"
    
    def analyze(self, frame: np.ndarray) -> dict:
        # Custom analysis logic
        return {"custom_metric": value}
```

### Adding a Hook

```python
from src.hooks import HookManager

hooks = HookManager()

@hooks.register("post_analyze")
def log_results(result):
    print(f"Analysis complete: {result['file_name']}")
```

### Using MCP Server

```python
from src.mcp import VideoEngineMCPServer

server = VideoEngineMCPServer(db_path="videos.db")
server.run()
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_ENGINE_DB_PATH` | Database file path | `video_library.db` |
| `VIDEO_ENGINE_PLUGIN_DIR` | Plugin directory | `~/.video_engine/plugins` |
| `VIDEO_ENGINE_LOG_LEVEL` | Logging verbosity | `INFO` |

### Configuration File

```yaml
# video_engine.yaml
database:
  path: "./videos.db"
  
analysis:
  keyframe_threshold: 30.0
  min_scene_duration: 1.0
  max_frames: 50

plugins:
  enabled:
    - quality_analyzer
    - thumbnail_generator
    
hooks:
  post_analyze:
    - log_to_file
    - send_notification
```

## Performance Considerations

### Memory Management
- Frames are processed sequentially to minimize memory footprint
- Large videos use streaming extraction
- Database uses connection pooling for concurrent access

### GPU Utilization
- OpenCV operations can leverage CUDA when available
- ML models run on GPU when PyTorch CUDA is configured
- Batch processing optimizes GPU memory usage

### Scalability
- Database indexing enables fast queries on large collections
- Plugin system allows horizontal scaling of analysis
- MCP server supports concurrent requests
