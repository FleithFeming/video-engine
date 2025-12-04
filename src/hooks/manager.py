"""Hook manager for event-based extensibility."""

from typing import Callable, Dict, List, Any, Optional
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class Hook:
    """
    Represents a single hook point with registered handlers.
    
    Hooks allow external code to execute custom logic at specific
    points in the video analysis pipeline.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a hook.
        
        Args:
            name: Unique hook name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._handlers: List[Callable] = []
        self._priority_handlers: Dict[int, List[Callable]] = {}
    
    def register(
        self, 
        handler: Callable, 
        priority: int = 0
    ) -> None:
        """
        Register a handler for this hook.
        
        Args:
            handler: Callable to execute when hook fires
            priority: Execution priority (higher = earlier, default 0)
        """
        if priority not in self._priority_handlers:
            self._priority_handlers[priority] = []
        self._priority_handlers[priority].append(handler)
        self._rebuild_handlers()
    
    def unregister(self, handler: Callable) -> bool:
        """
        Unregister a handler.
        
        Args:
            handler: Handler to remove
            
        Returns:
            True if handler was removed, False if not found
        """
        for priority, handlers in self._priority_handlers.items():
            if handler in handlers:
                handlers.remove(handler)
                self._rebuild_handlers()
                return True
        return False
    
    def _rebuild_handlers(self) -> None:
        """Rebuild the flat handler list from priority dict."""
        self._handlers = []
        for priority in sorted(self._priority_handlers.keys(), reverse=True):
            self._handlers.extend(self._priority_handlers[priority])
    
    def execute(self, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Execute all registered handlers.
        
        Args:
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            List of results from all handlers
        """
        results = []
        for handler in self._handlers:
            try:
                result = handler(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook handler {handler.__name__} failed: {e}")
                results.append(None)
        return results
    
    def execute_until_result(self, *args: Any, **kwargs: Any) -> Optional[Any]:
        """
        Execute handlers until one returns a non-None result.
        
        Args:
            *args: Positional arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            First non-None result, or None if all handlers return None
        """
        for handler in self._handlers:
            try:
                result = handler(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                logger.error(f"Hook handler {handler.__name__} failed: {e}")
        return None
    
    def __len__(self) -> int:
        """Get number of registered handlers."""
        return len(self._handlers)
    
    def __bool__(self) -> bool:
        """Check if hook has any handlers."""
        return len(self._handlers) > 0


class HookManager:
    """
    Manager for registering and executing hooks throughout the application.
    
    The HookManager provides a centralized way to define extension points
    in the video analysis pipeline. External code can register handlers
    that will be called at specific points.
    
    Built-in hooks:
        - pre_analyze: Before video analysis starts
        - post_analyze: After video analysis completes
        - pre_frame: Before processing each frame
        - post_frame: After processing each frame
        - on_error: When an error occurs
        - on_scene_change: When a scene change is detected
    
    Example:
        hooks = HookManager()
        
        @hooks.register("post_analyze")
        def log_result(video_path, result):
            print(f"Analyzed {video_path}: {len(result['frames'])} frames")
        
        # Later, in the analyzer:
        hooks.execute("post_analyze", video_path=path, result=result)
    """
    
    # Built-in hook definitions
    BUILTIN_HOOKS = {
        "pre_analyze": "Called before video analysis starts. Args: video_path, config",
        "post_analyze": "Called after video analysis completes. Args: video_path, result",
        "pre_frame": "Called before processing each frame. Args: frame, timestamp",
        "post_frame": "Called after processing each frame. Args: frame, timestamp, result",
        "on_error": "Called when an error occurs. Args: error, context",
        "on_scene_change": "Called when a scene change is detected. Args: timestamp, prev_frame, new_frame",
        "on_progress": "Called to report progress. Args: current, total, message",
    }
    
    def __init__(self, include_builtin: bool = True):
        """
        Initialize the hook manager.
        
        Args:
            include_builtin: Whether to create built-in hooks
        """
        self._hooks: Dict[str, Hook] = {}
        
        if include_builtin:
            for name, description in self.BUILTIN_HOOKS.items():
                self.create_hook(name, description)
    
    def create_hook(self, name: str, description: str = "") -> Hook:
        """
        Create a new hook point.
        
        Args:
            name: Unique hook name
            description: Human-readable description
            
        Returns:
            The created Hook instance
            
        Raises:
            ValueError: If hook with name already exists
        """
        if name in self._hooks:
            raise ValueError(f"Hook '{name}' already exists")
        
        hook = Hook(name, description)
        self._hooks[name] = hook
        return hook
    
    def get_hook(self, name: str) -> Optional[Hook]:
        """
        Get a hook by name.
        
        Args:
            name: Hook name
            
        Returns:
            Hook instance or None if not found
        """
        return self._hooks.get(name)
    
    def register(
        self, 
        hook_name: str, 
        priority: int = 0
    ) -> Callable:
        """
        Decorator to register a handler for a hook.
        
        Args:
            hook_name: Name of the hook to register for
            priority: Execution priority (higher = earlier)
            
        Returns:
            Decorator function
            
        Example:
            @hooks.register("post_analyze")
            def my_handler(video_path, result):
                print(f"Done: {video_path}")
        """
        def decorator(func: Callable) -> Callable:
            self.add_handler(hook_name, func, priority)
            return func
        return decorator
    
    def add_handler(
        self, 
        hook_name: str, 
        handler: Callable,
        priority: int = 0
    ) -> None:
        """
        Add a handler to a hook.
        
        Args:
            hook_name: Name of the hook
            handler: Callable to execute
            priority: Execution priority (higher = earlier)
            
        Raises:
            KeyError: If hook doesn't exist
        """
        if hook_name not in self._hooks:
            raise KeyError(f"Hook '{hook_name}' does not exist")
        
        self._hooks[hook_name].register(handler, priority)
    
    def remove_handler(self, hook_name: str, handler: Callable) -> bool:
        """
        Remove a handler from a hook.
        
        Args:
            hook_name: Name of the hook
            handler: Handler to remove
            
        Returns:
            True if removed, False if not found
        """
        if hook_name not in self._hooks:
            return False
        
        return self._hooks[hook_name].unregister(handler)
    
    def execute(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Execute all handlers for a hook.
        
        Args:
            hook_name: Name of the hook to execute
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            List of results from handlers (empty if hook doesn't exist)
        """
        hook = self._hooks.get(hook_name)
        if hook:
            return hook.execute(*args, **kwargs)
        return []
    
    def execute_until_result(
        self, 
        hook_name: str, 
        *args: Any, 
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Execute handlers until one returns a non-None result.
        
        Args:
            hook_name: Name of the hook
            *args: Arguments to pass to handlers
            **kwargs: Keyword arguments to pass to handlers
            
        Returns:
            First non-None result, or None
        """
        hook = self._hooks.get(hook_name)
        if hook:
            return hook.execute_until_result(*args, **kwargs)
        return None
    
    def list_hooks(self) -> List[Dict[str, Any]]:
        """
        Get information about all hooks.
        
        Returns:
            List of hook info dictionaries
        """
        return [
            {
                "name": hook.name,
                "description": hook.description,
                "handler_count": len(hook),
            }
            for hook in self._hooks.values()
        ]
    
    def clear_hook(self, hook_name: str) -> None:
        """
        Remove all handlers from a hook.
        
        Args:
            hook_name: Name of the hook to clear
        """
        if hook_name in self._hooks:
            self._hooks[hook_name]._handlers.clear()
            self._hooks[hook_name]._priority_handlers.clear()
    
    def clear_all(self) -> None:
        """Remove all handlers from all hooks."""
        for hook in self._hooks.values():
            hook._handlers.clear()
            hook._priority_handlers.clear()
    
    def __contains__(self, hook_name: str) -> bool:
        """Check if a hook exists."""
        return hook_name in self._hooks
    
    def __len__(self) -> int:
        """Get number of hooks."""
        return len(self._hooks)


# Global hook manager instance
_global_hooks: Optional[HookManager] = None


def get_hooks() -> HookManager:
    """
    Get the global hook manager.
    
    Returns:
        Global HookManager instance
    """
    global _global_hooks
    if _global_hooks is None:
        _global_hooks = HookManager()
    return _global_hooks
