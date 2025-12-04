# Video Engine

GPU-accelerated video analysis with ML frame detection.

## Features

- Frame extraction (uniform sampling or keyframe/scene detection)
- Video metadata extraction (resolution, duration, codec, FPS)
- Scene change detection
- Optional ML-based frame analysis (objects, scenes)
- SQLite database for searchable indexing
- **Plugin system** for custom analyzers, processors, and exporters
- **Hook system** for event-based extensibility
- **MCP server** for AI assistant integration (Claude, etc.)

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

## Plugin System

Extend Video Engine with custom plugins:

```python
from src.plugins import AnalyzerPlugin, PluginRegistry

class MyAnalyzer(AnalyzerPlugin):
    name = "my_analyzer"
    
    def analyze(self, frame):
        # Custom analysis logic
        return {"custom_metric": 0.95}

# Register and use
registry = PluginRegistry()
registry.register(MyAnalyzer())
```

See [Plugin Development Guide](docs/PLUGINS.md) for more details.

## Hook System

Add custom behavior at key points in the pipeline:

```python
from src.hooks import HookManager

hooks = HookManager()

@hooks.register("post_analyze")
def log_result(video_path, result):
    print(f"Analyzed: {video_path}")

# Hooks are called automatically during analysis
```

## MCP Server (AI Integration)

Integrate with AI assistants like Claude:

```bash
# Start the MCP server
python -m src.mcp.server --db-path videos.db
```

Add to Claude Desktop configuration:
```json
{
  "mcpServers": {
    "video-engine": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "/path/to/video-engine"
    }
  }
}
```

See [MCP Server Guide](docs/MCP_SERVER.md) for more details.

## Architecture

```
src/
├── analyzers/          # Video analysis components
│   └── video_analyzer.py
├── database/           # Data persistence
│   └── video_db.py
├── utils/              # Utility functions
│   └── video_io.py
├── plugins/            # Plugin system
│   ├── base.py         # Base plugin classes
│   └── registry.py     # Plugin discovery
├── hooks/              # Hook system
│   └── manager.py      # Hook management
└── mcp/                # MCP server
    └── server.py       # AI assistant integration
```

## Documentation

| Document | Description |
|----------|-------------|
| [Model Card](MODEL_CARD.md) | Model capabilities, requirements, and limitations |
| [Architecture](docs/ARCHITECTURE.md) | System design and components |
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation |
| [Plugin Guide](docs/PLUGINS.md) | How to create custom plugins |
| [MCP Server](docs/MCP_SERVER.md) | AI assistant integration guide |

## Requirements

- Python 3.10+
- OpenCV
- NumPy

Optional:
- PyTorch (for ML analysis)
- CUDA (for GPU acceleration)
