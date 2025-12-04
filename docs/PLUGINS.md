# Plugin Development Guide

This guide explains how to create custom plugins for Video Engine to extend its functionality.

## Overview

Video Engine uses a plugin system that allows you to add custom:
- **Analyzers**: Process frames and extract custom information
- **Processors**: Transform frames before/after analysis
- **Exporters**: Output results in custom formats

## Plugin Types

### Analyzer Plugins

Analyzer plugins process individual frames and return analysis results.

```python
from src.plugins import AnalyzerPlugin
import numpy as np

class ColorHistogramAnalyzer(AnalyzerPlugin):
    """Analyze color distribution in frames."""
    
    name = "color_histogram"
    version = "1.0.0"
    description = "Extracts color histogram from frames"
    
    def initialize(self, config=None):
        super().initialize(config)
        self.num_bins = self._config.get("num_bins", 256)
    
    def analyze(self, frame: np.ndarray) -> dict:
        """
        Analyze frame color distribution.
        
        Args:
            frame: RGB numpy array (H, W, 3)
            
        Returns:
            Dictionary with histogram data
        """
        histograms = {}
        for i, color in enumerate(['red', 'green', 'blue']):
            hist, _ = np.histogram(
                frame[:, :, i], 
                bins=self.num_bins, 
                range=(0, 256)
            )
            histograms[color] = hist.tolist()
        
        # Calculate dominant color
        means = [frame[:, :, i].mean() for i in range(3)]
        dominant = ['red', 'green', 'blue'][np.argmax(means)]
        
        return {
            "histograms": histograms,
            "dominant_color": dominant,
            "color_means": {
                "red": means[0],
                "green": means[1],
                "blue": means[2],
            }
        }
```

### Processor Plugins

Processor plugins transform frames during the analysis pipeline.

```python
from src.plugins import ProcessorPlugin
import numpy as np
import cv2

class ResizeProcessor(ProcessorPlugin):
    """Resize frames to a standard size."""
    
    name = "resize_processor"
    version = "1.0.0"
    description = "Resizes frames to target dimensions"
    
    def initialize(self, config=None):
        super().initialize(config)
        self.target_width = self._config.get("width", 640)
        self.target_height = self._config.get("height", 480)
    
    def process(self, frame: np.ndarray, metadata=None) -> np.ndarray:
        """
        Resize frame to target dimensions.
        
        Args:
            frame: RGB numpy array
            metadata: Optional frame metadata
            
        Returns:
            Resized frame
        """
        return cv2.resize(
            frame, 
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA
        )
```

### Exporter Plugins

Exporter plugins handle custom output formats.

```python
from src.plugins import ExporterPlugin
import csv

class CSVExporter(ExporterPlugin):
    """Export results to CSV format."""
    
    name = "csv_exporter"
    version = "1.0.0"
    description = "Exports analysis results to CSV"
    
    def export(self, results: dict, output_path: str) -> None:
        """
        Export results to CSV file.
        
        Args:
            results: Analysis results dictionary
            output_path: Output file path
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'file_name', 'duration', 'resolution', 
                'scenes', 'objects', 'quality'
            ])
            
            # Write data
            tags = results.get('content_tags', {})
            writer.writerow([
                results.get('file_name', ''),
                results.get('duration_formatted', ''),
                results.get('resolution', ''),
                ', '.join(tags.get('scenes', [])),
                ', '.join(tags.get('objects', [])),
                ', '.join(tags.get('quality', [])),
            ])
    
    def get_supported_formats(self) -> list:
        return ['csv']
```

## Plugin Registration

### Manual Registration

```python
from src.plugins import PluginRegistry

# Create registry
registry = PluginRegistry()

# Register plugins
registry.register(ColorHistogramAnalyzer())
registry.register(ResizeProcessor())
registry.register(CSVExporter())

# Initialize with config
registry.initialize_all({
    "color_histogram": {"num_bins": 128},
    "resize_processor": {"width": 320, "height": 240},
})
```

### Auto-Discovery

Plugins can be auto-discovered from a directory:

```python
from src.plugins import PluginRegistry

registry = PluginRegistry()
loaded = registry.discover("/path/to/plugins")
print(f"Loaded {loaded} plugins")
```

### Entry Points

For installable packages, use setuptools entry points:

```python
# setup.py
setup(
    name="my-video-plugin",
    entry_points={
        "video_engine.plugins": [
            "my_analyzer = my_plugin:MyAnalyzer",
        ]
    }
)
```

## Using Plugins with VideoAnalyzer

```python
from src.analyzers import VideoAnalyzer
from src.plugins import PluginRegistry

# Set up plugins
registry = PluginRegistry()
registry.register(ColorHistogramAnalyzer())
registry.initialize_all()

# Get analyzer plugin
color_analyzer = registry.get("color_histogram")

# Use with VideoAnalyzer
analyzer = VideoAnalyzer(
    ml_analyzer=color_analyzer  # Use as ML analyzer
)

result = analyzer.analyze("video.mp4")
```

## Plugin Lifecycle

1. **Instantiation**: Plugin object is created
2. **Registration**: Plugin is registered with registry
3. **Initialization**: `initialize()` called with config
4. **Usage**: Plugin methods are called during analysis
5. **Cleanup**: `cleanup()` called when done

```python
class MyPlugin(AnalyzerPlugin):
    name = "my_plugin"
    
    def __init__(self):
        super().__init__()
        self.model = None
    
    def initialize(self, config=None):
        super().initialize(config)
        # Load model or resources
        self.model = load_model(self._config.get("model_path"))
    
    def analyze(self, frame):
        return self.model.predict(frame)
    
    def cleanup(self):
        # Release resources
        if self.model:
            self.model.close()
        super().cleanup()
```

## Best Practices

### Error Handling

```python
def analyze(self, frame):
    try:
        result = self._process(frame)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}
```

### Configuration Validation

```python
def initialize(self, config=None):
    super().initialize(config)
    
    # Validate required config
    if "model_path" not in self._config:
        raise ValueError("model_path is required")
    
    # Validate types
    threshold = self._config.get("threshold", 0.5)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
```

### Logging

```python
import logging

class MyPlugin(AnalyzerPlugin):
    name = "my_plugin"
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(f"video_engine.plugins.{self.name}")
    
    def analyze(self, frame):
        self.logger.debug(f"Analyzing frame: {frame.shape}")
        # ...
```

### Testing

```python
import pytest
import numpy as np
from my_plugin import ColorHistogramAnalyzer

def test_color_histogram():
    plugin = ColorHistogramAnalyzer()
    plugin.initialize({"num_bins": 64})
    
    # Create test frame
    frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    result = plugin.analyze(frame)
    
    assert "histograms" in result
    assert "dominant_color" in result
    assert result["dominant_color"] in ["red", "green", "blue"]
```

## Example: ML-Based Object Detector

```python
from src.plugins import AnalyzerPlugin
import numpy as np

class YOLODetector(AnalyzerPlugin):
    """Object detection using YOLO."""
    
    name = "yolo_detector"
    version = "1.0.0"
    description = "Detects objects using YOLO model"
    
    def initialize(self, config=None):
        super().initialize(config)
        
        # Import YOLO (requires ultralytics)
        from ultralytics import YOLO
        
        model_path = self._config.get("model", "yolov8n.pt")
        self.model = YOLO(model_path)
        self.confidence = self._config.get("confidence", 0.5)
    
    def analyze(self, frame: np.ndarray) -> dict:
        """Detect objects in frame."""
        results = self.model(frame, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                if box.conf >= self.confidence:
                    detections.append({
                        "class": r.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy.tolist()[0],
                    })
        
        return {
            "detected_objects": [d["class"] for d in detections],
            "detections": detections,
            "object_count": len(detections),
        }
    
    def cleanup(self):
        del self.model
        super().cleanup()
```

## Plugin Directory Structure

```
my_plugin_package/
├── __init__.py
├── analyzer.py      # Analyzer plugin
├── processor.py     # Processor plugin
├── exporter.py      # Exporter plugin
├── models/          # Pre-trained models
└── tests/
    └── test_plugins.py
```
