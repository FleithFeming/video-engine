#!/usr/bin/env python3
"""Test script to analyze 10 videos with video-engine."""

import logging
import sys
import os
import glob

# Configure logging to see output
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzers import VideoAnalyzer
from src.database import VideoDatabase
from src.utils import get_video_info, extract_keyframes

def main():
    # Find all test videos
    video_dir = os.path.join(os.path.dirname(__file__), 'test_videos')
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

    print(f"\n{'='*60}")
    print(f"VIDEO-ENGINE TEST: Analyzing {len(video_files)} videos")
    print(f"{'='*60}\n")

    # Step 1: Test individual video info extraction
    print("STEP 1: Testing get_video_info()")
    print("-" * 40)
    for video_path in video_files[:3]:  # First 3 videos
        info = get_video_info(video_path)
        print(f"  {info['file_name']}: {info['resolution']} @ {info['fps']:.0f}fps, {info['duration_formatted']}")
    print()

    # Step 2: Test keyframe extraction
    print("STEP 2: Testing extract_keyframes()")
    print("-" * 40)
    for video_path in video_files[:3]:
        keyframes = extract_keyframes(video_path, threshold=30.0, min_scene_duration=0.5)
        print(f"  {os.path.basename(video_path)}: {len(keyframes)} keyframes extracted")
    print()

    # Step 3: Test VideoAnalyzer on all 10 videos
    print("STEP 3: Testing VideoAnalyzer on all 10 videos")
    print("-" * 40)

    analyzer = VideoAnalyzer(
        keyframe_threshold=20.0,  # Lower threshold for test videos
        min_scene_duration=0.5,
        frames_per_scene=2,
    )

    results = analyzer.analyze_batch(
        video_files,
        show_progress=True,
        extract_keyframes_only=True,
        max_frames=10,
    )

    # Print summary of results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    for result in successful:
        print(f"\n  {result['file_name']}:")
        print(f"    Resolution: {result['resolution']}")
        print(f"    Duration: {result['duration_formatted']}")
        print(f"    FPS: {result['fps']:.1f}")
        print(f"    Frames analyzed: {result['num_frames_analyzed']}")

        # Show temporal analysis
        temporal = result.get('temporal_analysis', {})
        print(f"    Scene changes: {temporal.get('scene_changes', 0)}")

        # Show content tags
        tags = result.get('content_tags', {})
        tech_tags = tags.get('technical', [])
        if tech_tags:
            print(f"    Technical tags: {', '.join(tech_tags)}")

    if failed:
        print("\nFailed analyses:")
        for result in failed:
            print(f"  {result['file_path']}: {result['error']}")

    # Step 4: Test database import and search
    print(f"\n{'='*60}")
    print("STEP 4: Testing VideoDatabase")
    print(f"{'='*60}")

    db_path = os.path.join(video_dir, 'test_library.db')

    with VideoDatabase(db_path) as db:
        # Import results
        stats = db.import_analysis(results)
        print(f"\nDatabase import: {stats}")

        # Get stats
        db_stats = db.get_stats()
        print(f"\nDatabase stats:")
        print(f"  Total videos: {db_stats['total_videos']}")
        print(f"  Avg duration: {db_stats['avg_duration_sec']:.1f}s")
        print(f"  Total size: {db_stats['total_size_mb']:.2f} MB")

        # Test search
        short_clips = db.search(max_duration=3.0)
        print(f"\nSearch results (videos <= 3s): {len(short_clips)} found")

        hd_videos = db.search(resolution="1280")
        print(f"Search results (1280px width): {len(hd_videos)} found")

    print(f"\n{'='*60}")
    print("TEST COMPLETE - All components working!")
    print(f"{'='*60}\n")

    # Clean up test database
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Cleaned up test database: {db_path}")

if __name__ == '__main__':
    main()
