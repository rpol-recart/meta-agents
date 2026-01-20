"""
Cache implementation for the Project Analysis Tool.
"""

from collections import OrderedDict
from typing import Any, Optional


class LRUCache:
    """LRU (Least Recently Used) cache implementation."""
    
    def __init__(self, capacity: int = 100):
        """
        Initialize the LRU cache.
        
        Args:
            capacity: Maximum number of items to store
        """
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        
        # Move to end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any):
        """
        Put an item in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Move to end to show it was recently used
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # Remove least recently used item
            self.cache.popitem(last=False)
        
        self.cache[key] = value
    
    def clear(self):
        """Clear all items from the cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get the current size of the cache."""
        return len(self.cache)