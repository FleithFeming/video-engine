"""Video database for searchable indexing and filtering."""

import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def convert_to_json_serializable(obj):
    """
    Recursively convert numpy types and other non-JSON-serializable types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


class VideoDatabase:
    """SQLite database for video metadata and search."""

    def __init__(self, db_path: str = "video_library.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        """Create database schema."""
        # Main videos table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                file_size_mb REAL,
                duration_sec REAL,
                duration_formatted TEXT,
                width INTEGER,
                height INTEGER,
                resolution TEXT,
                fps REAL,
                codec TEXT,
                aspect_ratio REAL,
                quality_avg REAL,
                quality_min REAL,
                quality_max REAL,
                num_frames_analyzed INTEGER,
                analysis_date TEXT,
                full_analysis JSON
            )
        """)

        # Tags table for searchability
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(id) ON DELETE CASCADE,
                UNIQUE(video_id, category, tag)
            )
        """)

        # Create indexes for fast searching
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_category ON tags(category)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_resolution ON videos(resolution)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_duration ON videos(duration_sec)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_quality ON videos(quality_avg)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_videos_analysis_date ON videos(analysis_date)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tags_video_id ON tags(video_id)
        """)

        self.conn.commit()

    def import_analysis(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Import video analysis results into database.

        Args:
            analysis_results: List of video analysis dictionaries

        Returns:
            Stats about import (inserted, updated, errors)
        """
        stats = {"inserted": 0, "updated": 0, "errors": 0}

        for result in analysis_results:
            try:
                # Skip error results
                if "error" in result:
                    stats["errors"] += 1
                    continue

                # Extract metadata
                file_path = result.get("file_path", "")
                file_name = result.get("file_name", "")

                temporal = result.get("temporal_analysis", {})
                quality_stats = temporal.get("quality_stats", {})

                # Check if video already exists
                self.cursor.execute(
                    "SELECT id FROM videos WHERE file_path = ?", (file_path,)
                )
                existing = self.cursor.fetchone()

                video_data = {
                    "file_path": file_path,
                    "file_name": file_name,
                    "file_size_mb": result.get("file_size_mb", 0),
                    "duration_sec": result.get("duration_sec", 0),
                    "duration_formatted": result.get("duration_formatted", ""),
                    "width": result.get("width", 0),
                    "height": result.get("height", 0),
                    "resolution": result.get("resolution", ""),
                    "fps": result.get("fps", 0),
                    "codec": result.get("codec", ""),
                    "aspect_ratio": result.get("aspect_ratio", 0),
                    "quality_avg": quality_stats.get("average"),
                    "quality_min": quality_stats.get("min"),
                    "quality_max": quality_stats.get("max"),
                    "num_frames_analyzed": result.get("num_frames_analyzed", 0),
                    "analysis_date": datetime.now().isoformat(),
                    "full_analysis": json.dumps(convert_to_json_serializable(result)),
                }

                if existing:
                    # Update existing record
                    video_id = existing["id"]
                    update_sql = """
                        UPDATE videos SET
                        file_name = ?, file_size_mb = ?, duration_sec = ?,
                        duration_formatted = ?, width = ?, height = ?,
                        resolution = ?, fps = ?, codec = ?, aspect_ratio = ?,
                        quality_avg = ?, quality_min = ?, quality_max = ?,
                        num_frames_analyzed = ?, analysis_date = ?, full_analysis = ?
                        WHERE id = ?
                    """
                    self.cursor.execute(
                        update_sql,
                        (
                            video_data["file_name"],
                            video_data["file_size_mb"],
                            video_data["duration_sec"],
                            video_data["duration_formatted"],
                            video_data["width"],
                            video_data["height"],
                            video_data["resolution"],
                            video_data["fps"],
                            video_data["codec"],
                            video_data["aspect_ratio"],
                            video_data["quality_avg"],
                            video_data["quality_min"],
                            video_data["quality_max"],
                            video_data["num_frames_analyzed"],
                            video_data["analysis_date"],
                            video_data["full_analysis"],
                            video_id,
                        ),
                    )
                    stats["updated"] += 1

                    # Clear old tags
                    self.cursor.execute("DELETE FROM tags WHERE video_id = ?", (video_id,))
                else:
                    # Insert new record
                    insert_sql = """
                        INSERT INTO videos (
                            file_path, file_name, file_size_mb, duration_sec,
                            duration_formatted, width, height, resolution, fps,
                            codec, aspect_ratio, quality_avg, quality_min,
                            quality_max, num_frames_analyzed, analysis_date, full_analysis
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    self.cursor.execute(
                        insert_sql,
                        (
                            video_data["file_path"],
                            video_data["file_name"],
                            video_data["file_size_mb"],
                            video_data["duration_sec"],
                            video_data["duration_formatted"],
                            video_data["width"],
                            video_data["height"],
                            video_data["resolution"],
                            video_data["fps"],
                            video_data["codec"],
                            video_data["aspect_ratio"],
                            video_data["quality_avg"],
                            video_data["quality_min"],
                            video_data["quality_max"],
                            video_data["num_frames_analyzed"],
                            video_data["analysis_date"],
                            video_data["full_analysis"],
                        ),
                    )
                    video_id = self.cursor.lastrowid
                    stats["inserted"] += 1

                # Insert tags (batch insert for performance)
                content_tags = result.get("content_tags", {})
                tag_rows = []
                for category, tag_list in content_tags.items():
                    for tag in tag_list:
                        tag_rows.append((video_id, category, tag))

                if tag_rows:
                    self.cursor.executemany(
                        """
                        INSERT OR IGNORE INTO tags (video_id, category, tag)
                        VALUES (?, ?, ?)
                        """,
                        tag_rows
                    )

            except Exception as e:
                print(f"Error importing {result.get('file_name', 'unknown')}: {e}")
                stats["errors"] += 1

        self.conn.commit()
        return stats

    def search(
        self,
        tags: Optional[List[str]] = None,
        categories: Optional[Dict[str, List[str]]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_quality: Optional[float] = None,
        resolution: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search videos by tags and metadata.

        Args:
            tags: List of tags to search for (OR logic)
            categories: Dict of {category: [tags]} for category-specific search
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_quality: Minimum average quality score
            resolution: Resolution filter (e.g., "1080p", "4k")
            limit: Maximum results to return

        Returns:
            List of matching video records
        """
        sql_parts = ["SELECT DISTINCT v.* FROM videos v"]
        where_clauses = []
        params = []

        # Tag search
        if tags or categories:
            sql_parts.append("JOIN tags t ON v.id = t.video_id")

            tag_conditions = []
            if tags:
                placeholders = ",".join("?" * len(tags))
                tag_conditions.append(f"t.tag IN ({placeholders})")
                params.extend(tags)

            if categories:
                for category, category_tags in categories.items():
                    if category_tags:
                        placeholders = ",".join("?" * len(category_tags))
                        tag_conditions.append(
                            f"(t.category = ? AND t.tag IN ({placeholders}))"
                        )
                        params.append(category)
                        params.extend(category_tags)

            if tag_conditions:
                where_clauses.append("(" + " OR ".join(tag_conditions) + ")")

        # Duration filters
        if min_duration is not None:
            where_clauses.append("v.duration_sec >= ?")
            params.append(min_duration)

        if max_duration is not None:
            where_clauses.append("v.duration_sec <= ?")
            params.append(max_duration)

        # Quality filter
        if min_quality is not None:
            where_clauses.append("v.quality_avg >= ?")
            params.append(min_quality)

        # Resolution filter
        if resolution:
            where_clauses.append("v.resolution LIKE ?")
            params.append(f"%{resolution}%")

        # Build final query
        if where_clauses:
            sql_parts.append("WHERE " + " AND ".join(where_clauses))

        sql_parts.append("ORDER BY v.analysis_date DESC")
        sql_parts.append(f"LIMIT {limit}")

        sql = " ".join(sql_parts)

        self.cursor.execute(sql, params)
        rows = self.cursor.fetchall()

        return [dict(row) for row in rows]

    def get_all_tags(self) -> Dict[str, List[str]]:
        """Get all unique tags grouped by category."""
        self.cursor.execute("""
            SELECT DISTINCT category, tag
            FROM tags
            ORDER BY category, tag
        """)

        tags_by_category = {}
        for row in self.cursor.fetchall():
            category = row["category"]
            tag = row["tag"]
            if category not in tags_by_category:
                tags_by_category[category] = []
            tags_by_category[category].append(tag)

        return tags_by_category

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        self.cursor.execute("SELECT COUNT(*) as total FROM videos")
        total = self.cursor.fetchone()["total"]

        self.cursor.execute("""
            SELECT
                AVG(duration_sec) as avg_duration,
                SUM(file_size_mb) as total_size_mb,
                AVG(quality_avg) as avg_quality
            FROM videos
        """)
        stats_row = self.cursor.fetchone()

        return {
            "total_videos": total,
            "avg_duration_sec": stats_row["avg_duration"] or 0,
            "total_size_mb": stats_row["total_size_mb"] or 0,
            "avg_quality": stats_row["avg_quality"] or 0,
            "all_tags": self.get_all_tags(),
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
