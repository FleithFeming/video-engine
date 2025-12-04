"""
MCP (Model Context Protocol) server for Video Engine.

This module implements an MCP server that exposes Video Engine functionality
to AI assistants like Claude. The server provides tools for video analysis,
search, and database management.
"""

import json
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

# MCP server implementation
# Uses stdin/stdout for communication following MCP protocol


class VideoEngineMCPServer:
    """
    MCP server for Video Engine integration with AI assistants.
    
    The server exposes video analysis capabilities through the Model Context
    Protocol, allowing AI assistants to analyze videos, search the database,
    and retrieve video metadata.
    
    Tools provided:
        - analyze_video: Analyze a video file
        - search_videos: Search the video database
        - get_video_info: Get metadata for a video
        - list_tags: List all available tags
        - get_stats: Get database statistics
    
    Example:
        server = VideoEngineMCPServer(db_path="videos.db")
        server.run()  # Starts listening on stdin/stdout
    """
    
    def __init__(
        self,
        db_path: str = "video_library.db",
        analyzer_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the MCP server.
        
        Args:
            db_path: Path to the SQLite database
            analyzer_config: Optional configuration for video analyzer
        """
        self.db_path = db_path
        self.analyzer_config = analyzer_config or {}
        self._db = None
        self._analyzer = None
        
        # Tool definitions following MCP schema
        self._tools = {
            "analyze_video": {
                "name": "analyze_video",
                "description": "Analyze a video file and extract metadata, scenes, and content tags",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to the video file to analyze"
                        },
                        "max_frames": {
                            "type": "integer",
                            "description": "Maximum number of frames to analyze (default: 50)",
                            "default": 50
                        },
                        "extract_keyframes_only": {
                            "type": "boolean",
                            "description": "Use scene detection instead of uniform sampling (default: true)",
                            "default": True
                        }
                    },
                    "required": ["video_path"]
                }
            },
            "search_videos": {
                "name": "search_videos",
                "description": "Search the video database by tags, duration, quality, or resolution",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to search for (OR logic)"
                        },
                        "min_duration": {
                            "type": "number",
                            "description": "Minimum duration in seconds"
                        },
                        "max_duration": {
                            "type": "number",
                            "description": "Maximum duration in seconds"
                        },
                        "min_quality": {
                            "type": "number",
                            "description": "Minimum quality score (0-100)"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution filter (e.g., '1080p', '4k')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default: 20)",
                            "default": 20
                        }
                    }
                }
            },
            "get_video_info": {
                "name": "get_video_info",
                "description": "Get metadata for a video file without full analysis",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "video_path": {
                            "type": "string",
                            "description": "Path to the video file"
                        }
                    },
                    "required": ["video_path"]
                }
            },
            "list_tags": {
                "name": "list_tags",
                "description": "List all available tags in the database, grouped by category",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "get_stats": {
                "name": "get_stats",
                "description": "Get database statistics including total videos, average duration, and storage size",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def _get_db(self):
        """Get or create database connection."""
        if self._db is None:
            from ..database import VideoDatabase
            self._db = VideoDatabase(self.db_path)
        return self._db
    
    def _get_analyzer(self):
        """Get or create video analyzer."""
        if self._analyzer is None:
            from ..analyzers import VideoAnalyzer
            self._analyzer = VideoAnalyzer(**self.analyzer_config)
        return self._analyzer
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        Args:
            request: MCP request dictionary
            
        Returns:
            MCP response dictionary
        """
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                result = self._handle_initialize(params)
            elif method == "tools/list":
                result = self._handle_list_tools()
            elif method == "tools/call":
                result = self._handle_call_tool(params)
            else:
                return self._error_response(
                    request_id,
                    -32601,
                    f"Method not found: {method}"
                )
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return self._error_response(request_id, -32000, str(e))
    
    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "video-engine",
                "version": "0.1.0"
            }
        }
    
    def _handle_list_tools(self) -> Dict[str, Any]:
        """Handle tools/list request."""
        return {
            "tools": list(self._tools.values())
        }
    
    def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        # Route to appropriate handler
        handlers = {
            "analyze_video": self._tool_analyze_video,
            "search_videos": self._tool_search_videos,
            "get_video_info": self._tool_get_video_info,
            "list_tags": self._tool_list_tags,
            "get_stats": self._tool_get_stats,
        }
        
        handler = handlers.get(tool_name)
        if handler:
            result = handler(arguments)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2, default=str)
                    }
                ]
            }
        
        raise ValueError(f"No handler for tool: {tool_name}")
    
    def _tool_analyze_video(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a video file."""
        video_path = args.get("video_path")
        if not video_path:
            raise ValueError("video_path is required")
        
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        analyzer = self._get_analyzer()
        result = analyzer.analyze(
            str(path),
            max_frames=args.get("max_frames", 50),
            extract_keyframes_only=args.get("extract_keyframes_only", True)
        )
        
        # Optionally store in database
        db = self._get_db()
        db.import_analysis([result])
        
        # Return summary (full analysis is in DB)
        return {
            "file_name": result.get("file_name"),
            "duration_formatted": result.get("duration_formatted"),
            "resolution": result.get("resolution"),
            "fps": result.get("fps"),
            "num_frames_analyzed": result.get("num_frames_analyzed"),
            "content_tags": result.get("content_tags"),
            "temporal_analysis": result.get("temporal_analysis"),
        }
    
    def _tool_search_videos(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Search the video database."""
        db = self._get_db()
        
        results = db.search(
            tags=args.get("tags"),
            min_duration=args.get("min_duration"),
            max_duration=args.get("max_duration"),
            min_quality=args.get("min_quality"),
            resolution=args.get("resolution"),
            limit=args.get("limit", 20)
        )
        
        # Return simplified results
        simplified = []
        for r in results:
            simplified.append({
                "file_name": r.get("file_name"),
                "file_path": r.get("file_path"),
                "duration_formatted": r.get("duration_formatted"),
                "resolution": r.get("resolution"),
                "quality_avg": r.get("quality_avg"),
            })
        
        return {
            "count": len(simplified),
            "videos": simplified
        }
    
    def _tool_get_video_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get video metadata."""
        video_path = args.get("video_path")
        if not video_path:
            raise ValueError("video_path is required")
        
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        from ..utils import get_video_info
        return get_video_info(str(path))
    
    def _tool_list_tags(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List all tags."""
        db = self._get_db()
        return db.get_all_tags()
    
    def _tool_get_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get database statistics."""
        db = self._get_db()
        return db.get_stats()
    
    def _error_response(
        self, 
        request_id: Any, 
        code: int, 
        message: str
    ) -> Dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    def run(self) -> None:
        """
        Run the MCP server, listening on stdin/stdout.
        
        The server reads JSON-RPC requests from stdin and writes
        responses to stdout, following the MCP protocol.
        """
        while True:
            try:
                # Read line from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                # Parse JSON request
                request = json.loads(line.strip())
                
                # Handle request
                response = self.handle_request(request)
                
                # Write response to stdout
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except json.JSONDecodeError:
                # Invalid JSON, write error
                error = self._error_response(None, -32700, "Parse error")
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()
            except KeyboardInterrupt:
                break
            except Exception as e:
                error = self._error_response(None, -32000, str(e))
                sys.stdout.write(json.dumps(error) + "\n")
                sys.stdout.flush()
        
        # Cleanup
        if self._db:
            self._db.close()


def main():
    """Entry point for MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Engine MCP Server")
    parser.add_argument(
        "--db-path",
        default="video_library.db",
        help="Path to SQLite database"
    )
    
    args = parser.parse_args()
    
    server = VideoEngineMCPServer(db_path=args.db_path)
    server.run()


if __name__ == "__main__":
    main()
