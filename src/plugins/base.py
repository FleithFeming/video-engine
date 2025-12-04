"""Base classes for Video Engine plugins."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BasePlugin(ABC):
    """Base class for all plugins."""
    
    name: str = "base_plugin"
    version: str = "1.0.0"
    description: str = "Base plugin class"
    
    def __init__(self):
        """Initialize the plugin."""
        self._initialized = False
        self._config: Dict[str, Any] = {}
    
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
        """
        self._config = config or {}
        self._initialized = True
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        self._initialized = False
    
    def get_info(self) -> Dict[str, str]:
        """Get plugin information."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
        }


class AnalyzerPlugin(BasePlugin):
    """
    Base class for frame analyzer plugins.
    
    Analyzer plugins process individual frames and return
    analysis results such as detected objects, scenes, or metrics.
    
    Example:
        class MyAnalyzer(AnalyzerPlugin):
            name = "my_analyzer"
            
            def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
                # Custom analysis logic
                return {"custom_metric": 0.95}
    """
    
    @abstractmethod
    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze a frame and return results.
        
        Args:
            frame: Frame as RGB numpy array (H, W, 3)
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    def analyze_batch(self, frames: list) -> list:
        """
        Analyze multiple frames.
        
        Default implementation processes frames sequentially.
        Override for batch-optimized processing.
        
        Args:
            frames: List of frames as numpy arrays
            
        Returns:
            List of analysis result dictionaries
        """
        return [self.analyze(frame) for frame in frames]


class ProcessorPlugin(BasePlugin):
    """
    Base class for frame processor plugins.
    
    Processor plugins transform frames before or after analysis,
    such as resizing, color correction, or augmentation.
    
    Example:
        class ResizeProcessor(ProcessorPlugin):
            name = "resize_processor"
            
            def process(self, frame, metadata):
                return cv2.resize(frame, (640, 480))
    """
    
    @abstractmethod
    def process(
        self,
        frame: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Process a frame and return the modified frame.
        
        Args:
            frame: Frame as RGB numpy array
            metadata: Optional metadata about the frame
            
        Returns:
            Processed frame as numpy array
        """
        pass


class ExporterPlugin(BasePlugin):
    """
    Base class for result exporter plugins.
    
    Exporter plugins handle output of analysis results
    in various formats such as JSON, CSV, XML, or custom formats.
    
    Example:
        class CSVExporter(ExporterPlugin):
            name = "csv_exporter"
            
            def export(self, results, output_path):
                # Export to CSV format
                pass
    """
    
    @abstractmethod
    def export(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Export analysis results to a file.
        
        Args:
            results: Analysis results dictionary
            output_path: Output file path
        """
        pass
    
    def get_supported_formats(self) -> list:
        """
        Get list of supported export formats.
        
        Returns:
            List of format strings (e.g., ['json', 'csv'])
        """
        return []
