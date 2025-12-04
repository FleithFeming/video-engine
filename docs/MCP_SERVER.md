# MCP Server Integration Guide

This guide explains how to use the Video Engine MCP (Model Context Protocol) server for AI assistant integration.

## Overview

The MCP server allows AI assistants like Claude to interact with Video Engine through a standardized protocol. The server exposes tools for:

- Analyzing videos
- Searching the video database
- Retrieving metadata
- Managing the video library

## Quick Start

### Starting the Server

```bash
# Run the MCP server
python -m src.mcp.server --db-path videos.db
```

### Claude Desktop Configuration

Add Video Engine to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "video-engine": {
      "command": "python",
      "args": ["-m", "src.mcp.server", "--db-path", "videos.db"],
      "cwd": "/path/to/video-engine"
    }
  }
}
```

## Available Tools

### analyze_video

Analyze a video file and extract metadata, scenes, and content tags.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_path` | string | Yes | Path to the video file |
| `max_frames` | integer | No | Maximum frames to analyze (default: 50) |
| `extract_keyframes_only` | boolean | No | Use scene detection (default: true) |

**Example:**
```
Use the analyze_video tool with video_path="/videos/demo.mp4"
```

**Response:**
```json
{
  "file_name": "demo.mp4",
  "duration_formatted": "02:30",
  "resolution": "1920x1080",
  "fps": 30.0,
  "num_frames_analyzed": 25,
  "content_tags": {
    "scenes": ["outdoor", "nature"],
    "objects": ["person", "tree"],
    "quality": ["high-quality"],
    "technical": ["1080p", "medium-length"]
  }
}
```

### search_videos

Search the video database by tags and metadata.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tags` | array | No | Tags to search for |
| `min_duration` | number | No | Minimum duration in seconds |
| `max_duration` | number | No | Maximum duration in seconds |
| `min_quality` | number | No | Minimum quality score |
| `resolution` | string | No | Resolution filter |
| `limit` | integer | No | Max results (default: 20) |

**Example:**
```
Search for fitness videos longer than 5 minutes using search_videos with tags=["fitness", "exercise"] and min_duration=300
```

### get_video_info

Get metadata for a video file without full analysis.

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `video_path` | string | Yes | Path to the video file |

### list_tags

List all available tags in the database, grouped by category.

**Parameters:** None

**Response:**
```json
{
  "scenes": ["indoor", "outdoor", "nature"],
  "objects": ["person", "car", "animal"],
  "activities": ["fitness", "cooking"],
  "quality": ["high-quality", "good-quality"],
  "technical": ["4k", "1080p", "720p"]
}
```

### get_stats

Get database statistics.

**Parameters:** None

**Response:**
```json
{
  "total_videos": 150,
  "avg_duration_sec": 245.5,
  "total_size_mb": 5120.0,
  "avg_quality": 78.5
}
```

## Programmatic Usage

### Python Integration

```python
from src.mcp import VideoEngineMCPServer
import json

# Create server instance
server = VideoEngineMCPServer(db_path="videos.db")

# Call tools directly
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "analyze_video",
        "arguments": {
            "video_path": "/path/to/video.mp4",
            "max_frames": 30
        }
    }
}

response = server.handle_request(request)
print(json.dumps(response, indent=2))
```

### Subprocess Communication

```python
import subprocess
import json

# Start MCP server as subprocess
process = subprocess.Popen(
    ["python", "-m", "src.mcp.server"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    text=True
)

# Send request
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}
process.stdin.write(json.dumps(request) + "\n")
process.stdin.flush()

# Read response
response = json.loads(process.stdout.readline())
print(response)
```

## MCP Protocol Details

The server implements the Model Context Protocol over stdin/stdout using JSON-RPC 2.0.

### Initialize

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {}
  }
}
```

### List Tools

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list",
  "params": {}
}
```

### Call Tool

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "analyze_video",
    "arguments": {
      "video_path": "/videos/demo.mp4"
    }
  }
}
```

## Error Handling

The server returns standard JSON-RPC errors:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Video not found: /path/to/missing.mp4"
  }
}
```

**Error Codes:**
| Code | Description |
|------|-------------|
| -32700 | Parse error (invalid JSON) |
| -32601 | Method not found |
| -32602 | Invalid params |
| -32000 | Application error |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_ENGINE_DB_PATH` | Database path | `video_library.db` |
| `VIDEO_ENGINE_LOG_LEVEL` | Log verbosity | `INFO` |

### Command Line Arguments

```bash
python -m src.mcp.server \
    --db-path /data/videos.db
```

## Security Considerations

1. **File Access**: The server can access any file the process has permissions for. Restrict file paths in production.

2. **Database**: The SQLite database is local. Consider access controls for multi-user deployments.

3. **Resource Limits**: Video analysis can be CPU/memory intensive. Monitor resource usage.

## Example Workflows

### Batch Video Analysis

```
1. Use list_tags to see existing categories
2. Use search_videos to find unanalyzed videos
3. Use analyze_video on each video
4. Use get_stats to verify import
```

### Content Discovery

```
1. Use search_videos with tags=["fitness"] to find workout videos
2. Use get_video_info to get details on interesting videos
3. Use search_videos with min_quality=80 to find high-quality content
```

### Library Management

```
1. Use get_stats to overview the library
2. Use list_tags to see content distribution
3. Use search_videos with different filters to organize content
```

## Troubleshooting

### Server Not Starting

```bash
# Check Python path
which python

# Verify dependencies
pip install -r requirements.txt

# Test import
python -c "from src.mcp import VideoEngineMCPServer"
```

### Tool Calls Failing

1. Check video path exists
2. Verify database permissions
3. Check server logs for errors

### Performance Issues

1. Reduce `max_frames` for faster analysis
2. Use `extract_keyframes_only=true`
3. Consider database indexing for large libraries
