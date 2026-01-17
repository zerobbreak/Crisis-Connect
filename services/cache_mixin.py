"""
Caching Mixin for Services
Provides Redis-based caching functionality
"""

from functools import wraps
from typing import Any, Optional, Callable
import json
import hashlib
import structlog

logger = structlog.get_logger(__name__)


class CacheMixin:
    """Mixin to add caching capabilities to services"""
    
    async def _cache_get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if hasattr(self, 'app') and self.app and hasattr(self.app.state, 'redis'):
                redis = self.app.state.redis
                if redis:
                    cached = await redis.get(key)
                    if cached:
                        return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get failed: {e}", key=key)
        return None
    
    async def _cache_set(self, key: str, value: Any, ttl: int = 300):
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        try:
            if hasattr(self, 'app') and self.app and hasattr(self.app.state, 'redis'):
                redis = self.app.state.redis
                if redis:
                    await redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.warning(f"Cache set failed: {e}", key=key)
    
    async def _cache_delete(self, key: str):
        """
        Delete value from cache
        
        Args:
            key: Cache key
        """
        try:
            if hasattr(self, 'app') and self.app and hasattr(self.app.state, 'redis'):
                redis = self.app.state.redis
                if redis:
                    await redis.delete(key)
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}", key=key)
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from function arguments
        
        Args:
            prefix: Key prefix (usually function name)
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MD5 hash of the key components
        """
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cached(self, ttl: int = 300, key_prefix: Optional[str] = None):
        """
        Decorator to cache function results
        
        Args:
            ttl: Time to live in seconds
            key_prefix: Optional custom key prefix
            
        Returns:
            Decorated function with caching
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                prefix = key_prefix or f"{self.__class__.__name__}.{func.__name__}"
                cache_key = self._generate_cache_key(prefix, *args[1:], **kwargs)  # Skip 'self'
                
                # Try cache first
                cached = await self._cache_get(cache_key)
                if cached is not None:
                    logger.debug(f"Cache hit: {prefix}", key=cache_key)
                    return cached
                
                # Execute function
                logger.debug(f"Cache miss: {prefix}", key=cache_key)
                result = await func(*args, **kwargs)
                
                # Cache result
                await self._cache_set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
