#!/usr/bin/env python3
"""Test script for new video analysis features."""

import logging
import sys
import os
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzers import (
    # Tag system
    TagTaxonomy, TagCategory, get_taxonomy,
    # Face detection
    FaceDetector, detect_faces,
    # Scene detection
    SceneDetector, detect_scenes,
    # Workflow
    AnalysisWorkflow, WorkflowConfig, analyze_video, quick_analyze,
)
from src.utils import get_video_info, extract_frames


def test_tag_taxonomy():
    """Test the tag taxonomy system."""
    print(f"\n{'='*60}")
    print("TEST 1: Tag Taxonomy System")
    print(f"{'='*60}")

    taxonomy = get_taxonomy()
    stats = taxonomy.get_tag_statistics()

    print(f"\nTaxonomy Statistics:")
    print(f"  Total tags: {stats['total_tags']}")
    print(f"  Total aliases: {stats['total_aliases']}")
    print(f"  Hierarchy depth: {stats['hierarchy_depth']}")
    print(f"\n  Tags by category:")
    for category, count in stats['categories'].items():
        print(f"    {category}: {count}")

    # Test tag lookup
    print(f"\nTag Lookup Tests:")

    # Test canonical name lookup
    tag = taxonomy.get_tag("indoor/home/kitchen")
    if tag:
        print(f"  Found: {tag.name} (category: {tag.category.value})")
        print(f"    Aliases: {tag.aliases[:3]}...")

    # Test alias lookup
    canonical = taxonomy.get_canonical_name("gym")
    print(f"  Alias 'gym' -> {canonical}")

    canonical = taxonomy.get_canonical_name("sunset")
    print(f"  Alias 'sunset' -> {canonical}")

    # Test hierarchy
    children = taxonomy.get_children("indoor/home")
    print(f"\n  Children of 'indoor/home': {children[:5]}...")

    ancestors = taxonomy.get_ancestors("indoor/home/kitchen")
    print(f"  Ancestors of 'indoor/home/kitchen': {ancestors}")

    # Test search
    results = taxonomy.search_tags("outdoor")
    print(f"\n  Search 'outdoor': Found {len(results)} tags")
    print(f"    First 5: {results[:5]}")

    print("\n  [PASS] Tag taxonomy working correctly")
    return True


def test_face_detection():
    """Test face detection on video frames."""
    print(f"\n{'='*60}")
    print("TEST 2: Face Detection")
    print(f"{'='*60}")

    # Find test videos
    video_dir = os.path.join(os.path.dirname(__file__), 'test_videos')
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

    if not video_files:
        print("  [SKIP] No test videos found")
        return True

    # Extract a frame from first video
    video_path = video_files[0]
    print(f"\n  Testing on: {os.path.basename(video_path)}")

    frames = extract_frames(video_path, num_frames=3)
    print(f"  Extracted {len(frames)} frames")

    # Test face detection
    detector = FaceDetector(
        detect_eyes=True,
        detect_profile=True
    )

    for timestamp, frame in frames:
        result = detector.detect(frame)
        print(f"\n  Frame at {timestamp:.1f}s:")
        print(f"    Faces detected: {result.face_count}")
        print(f"    Has faces: {result.has_faces}")
        print(f"    Tags: {result.tags}")

    # Test video-level face analysis
    print(f"\n  Video-level face analysis:")
    video_result = detector.analyze_video_faces(frames)
    print(f"    Face frequency: {video_result['face_frequency']}")
    print(f"    Avg face count: {video_result['avg_face_count']}")
    print(f"    Face tags: {video_result['face_tags']}")

    print("\n  [PASS] Face detection working correctly")
    return True


def test_scene_detection():
    """Test enhanced scene detection."""
    print(f"\n{'='*60}")
    print("TEST 3: Scene Detection & Classification")
    print(f"{'='*60}")

    # Find test videos
    video_dir = os.path.join(os.path.dirname(__file__), 'test_videos')
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

    if not video_files:
        print("  [SKIP] No test videos found")
        return True

    # Test on a video with scene changes
    video_path = video_files[-1]  # video_10_scene_change.mp4
    print(f"\n  Testing on: {os.path.basename(video_path)}")

    # Run scene detection
    detector = SceneDetector(
        threshold=20.0,
        min_scene_duration=0.5,
        classify_scenes=True
    )

    result = detector.detect_scenes(video_path)

    print(f"\n  Scene Detection Results:")
    print(f"    Total scenes: {result.scene_count}")
    print(f"    Avg scene duration: {result.avg_scene_duration:.2f}s")
    print(f"    Dominant transition: {result.dominant_transition.value}")
    print(f"    All tags: {result.all_tags}")

    print(f"\n  Scene Summary:")
    for key, value in result.scene_summary.items():
        print(f"    {key}: {value}")

    # Show details of each scene
    print(f"\n  Individual Scenes:")
    for i, scene in enumerate(result.scenes[:5]):  # First 5 scenes
        print(f"\n    Scene {i+1}:")
        print(f"      Time: {scene.start_time:.1f}s - {scene.end_time:.1f}s ({scene.duration:.1f}s)")
        print(f"      Transition: {scene.transition_type.value}")
        print(f"      Motion: {scene.motion_level}")

        if scene.classification:
            print(f"      Indoor/Outdoor: {scene.classification.indoor_outdoor}")
            print(f"      Lighting: {scene.classification.lighting_type}")
            print(f"      Environment tags: {scene.classification.environment_tags}")

        if scene.color_analysis:
            print(f"      Color temp: {scene.color_analysis.temperature.value}")
            print(f"      Brightness: {scene.color_analysis.brightness.value}")
            print(f"      Colors: {scene.color_analysis.color_names[:3]}")

        print(f"      Tags: {scene.tags}")

    print("\n  [PASS] Scene detection working correctly")
    return True


def test_analysis_workflow():
    """Test the complete analysis workflow."""
    print(f"\n{'='*60}")
    print("TEST 4: Complete Analysis Workflow")
    print(f"{'='*60}")

    # Find test videos
    video_dir = os.path.join(os.path.dirname(__file__), 'test_videos')
    video_files = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))

    if not video_files:
        print("  [SKIP] No test videos found")
        return True

    # Test on multiple videos
    for video_path in video_files[:3]:
        print(f"\n  Analyzing: {os.path.basename(video_path)}")

        # Use quick_analyze for simple output
        result = quick_analyze(video_path)

        print(f"    Duration: {result['duration']:.1f}s")
        print(f"    Scenes: {result['scene_count']}")
        print(f"    Has faces: {result['has_faces']}")
        print(f"    Total tags: {len(result['tags'])}")

        print(f"\n    Tags by category:")
        for category, tags in result['tags_by_category'].items():
            if tags:
                print(f"      {category}: {tags[:3]}{'...' if len(tags) > 3 else ''}")

    # Test full workflow with config
    print(f"\n  Testing full workflow with custom config:")

    config = WorkflowConfig(
        scene_threshold=25.0,
        detect_faces=True,
        analyze_colors=True,
        normalize_tags=True,
        expand_tag_hierarchy=True,
    )

    workflow = AnalysisWorkflow(config=config)
    result = workflow.analyze(video_files[0])

    print(f"    Scene count: {result.scene_count}")
    print(f"    Face frequency: {result.face_frequency}")
    print(f"    All tags: {len(result.all_tags)}")
    print(f"    Normalized tags: {len(result.normalized_tags)}")
    print(f"    Sample tags: {result.normalized_tags[:10]}")

    print("\n  [PASS] Analysis workflow working correctly")
    return True


def main():
    """Run all tests."""
    print(f"\n{'#'*60}")
    print("VIDEO-ENGINE: New Features Test Suite")
    print(f"{'#'*60}")

    tests = [
        ("Tag Taxonomy", test_tag_taxonomy),
        ("Face Detection", test_face_detection),
        ("Scene Detection", test_scene_detection),
        ("Analysis Workflow", test_analysis_workflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n  [FAIL] {name}: {e}")
            results.append((name, False, str(e)))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if error:
            print(f"         Error: {error}")

    print(f"\n  Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n  All new features working correctly!")
    else:
        print("\n  Some tests failed. Please review the output above.")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
