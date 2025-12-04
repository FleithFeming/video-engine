"""Plugin registry for discovering and managing plugins."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

from .base import BasePlugin, AnalyzerPlugin, ProcessorPlugin, ExporterPlugin


class PluginRegistry:
    """
    Registry for discovering, loading, and managing plugins.
    
    The registry supports multiple discovery methods:
    - Manual registration via register()
    - Directory scanning via discover()
    - Entry points via discover_entry_points()
    
    Example:
        registry = PluginRegistry()
        registry.discover("/path/to/plugins")
        registry.register(MyCustomPlugin())
        
        analyzer = registry.get("my_analyzer")
        if analyzer:
            result = analyzer.analyze(frame)
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_types: Dict[str, Type[BasePlugin]] = {
            "analyzer": AnalyzerPlugin,
            "processor": ProcessorPlugin,
            "exporter": ExporterPlugin,
        }
    
    def register(self, plugin: BasePlugin) -> bool:
        """
        Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful, False if plugin with same name exists
        """
        if not isinstance(plugin, BasePlugin):
            raise TypeError(f"Plugin must be an instance of BasePlugin, got {type(plugin)}")
        
        if plugin.name in self._plugins:
            return False
        
        self._plugins[plugin.name] = plugin
        return True
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin by name.
        
        Args:
            name: Plugin name to unregister
            
        Returns:
            True if plugin was unregistered, False if not found
        """
        if name in self._plugins:
            plugin = self._plugins.pop(name)
            plugin.cleanup()
            return True
        return False
    
    def get(self, name: str) -> Optional[BasePlugin]:
        """
        Get a registered plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)
    
    def get_all(self, plugin_type: Optional[str] = None) -> List[BasePlugin]:
        """
        Get all registered plugins, optionally filtered by type.
        
        Args:
            plugin_type: Filter by type ('analyzer', 'processor', 'exporter')
            
        Returns:
            List of plugin instances
        """
        plugins = list(self._plugins.values())
        
        if plugin_type and plugin_type in self._plugin_types:
            base_class = self._plugin_types[plugin_type]
            plugins = [p for p in plugins if isinstance(p, base_class)]
        
        return plugins
    
    def get_analyzers(self) -> List[AnalyzerPlugin]:
        """Get all registered analyzer plugins."""
        return [p for p in self._plugins.values() if isinstance(p, AnalyzerPlugin)]
    
    def get_processors(self) -> List[ProcessorPlugin]:
        """Get all registered processor plugins."""
        return [p for p in self._plugins.values() if isinstance(p, ProcessorPlugin)]
    
    def get_exporters(self) -> List[ExporterPlugin]:
        """Get all registered exporter plugins."""
        return [p for p in self._plugins.values() if isinstance(p, ExporterPlugin)]
    
    def discover(self, plugin_dir: str) -> int:
        """
        Discover and load plugins from a directory.
        
        Scans the directory for Python files and loads any classes
        that inherit from BasePlugin.
        
        Args:
            plugin_dir: Path to plugin directory
            
        Returns:
            Number of plugins loaded
        """
        plugin_path = Path(plugin_dir)
        if not plugin_path.exists():
            return 0
        
        loaded = 0
        
        for py_file in plugin_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                module_name = f"video_engine_plugin_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    
                    # Find and register plugin classes
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, BasePlugin) and 
                            attr not in (BasePlugin, AnalyzerPlugin, ProcessorPlugin, ExporterPlugin)):
                            try:
                                plugin = attr()
                                if self.register(plugin):
                                    loaded += 1
                            except Exception:
                                pass  # Skip plugins that fail to instantiate
            except Exception:
                pass  # Skip files that fail to load
        
        return loaded
    
    def discover_entry_points(self, group: str = "video_engine.plugins") -> int:
        """
        Discover plugins from setuptools entry points.
        
        Args:
            group: Entry point group name
            
        Returns:
            Number of plugins loaded
        """
        loaded = 0
        
        try:
            from importlib.metadata import entry_points
            
            if sys.version_info >= (3, 10):
                eps = entry_points(group=group)
            else:
                eps = entry_points().get(group, [])
            
            for ep in eps:
                try:
                    plugin_class = ep.load()
                    if issubclass(plugin_class, BasePlugin):
                        plugin = plugin_class()
                        if self.register(plugin):
                            loaded += 1
                except Exception:
                    pass
        except ImportError:
            pass
        
        return loaded
    
    def initialize_all(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize all registered plugins.
        
        Args:
            config: Configuration dict with plugin names as keys
        """
        config = config or {}
        for name, plugin in self._plugins.items():
            plugin_config = config.get(name, {})
            plugin.initialize(plugin_config)
    
    def cleanup_all(self) -> None:
        """Clean up all registered plugins."""
        for plugin in self._plugins.values():
            plugin.cleanup()
    
    def list_plugins(self) -> List[Dict[str, str]]:
        """
        Get information about all registered plugins.
        
        Returns:
            List of plugin info dictionaries
        """
        return [plugin.get_info() for plugin in self._plugins.values()]
    
    def __len__(self) -> int:
        """Get number of registered plugins."""
        return len(self._plugins)
    
    def __contains__(self, name: str) -> bool:
        """Check if a plugin is registered."""
        return name in self._plugins


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry.
    
    Returns:
        Global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry
