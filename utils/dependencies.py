from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.requests import Request
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from functools import wraps
import logging
import structlog
from config import settings
from utils.db import get_db

logger = structlog.get_logger(__name__)

# --- Security Dependencies ---
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    if not settings.api_key:
        return True  # No API key required in development
    
    if not credentials or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Redis Cache Setup ---
redis_client = None

async def get_redis():
    """Get Redis client"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
            await redis_client.ping()
        except Exception as e:
            logger.warning("Redis connection failed, caching disabled", error=str(e))
            redis_client = None
    return redis_client

# --- Rate Limiting ---
async def rate_limit(request: Request):
    """Rate limiting based on client IP"""
    if not settings.debug:
        redis_conn = await get_redis()
        if redis_conn:
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}"
            
            current_requests = await redis_conn.get(key)
            if current_requests is None:
                await redis_conn.setex(key, settings.rate_limit_window, 1)
            elif int(current_requests) >= settings.rate_limit_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            else:
                await redis_conn.incr(key)
    return True

# --- Caching Decorator ---
def cache_response(expire_seconds: int = 300):
    """Cache response decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            redis_conn = await get_redis()
            if not redis_conn:
                return await func(*args, **kwargs)
            
            # Create cache key from function name and arguments
            # Note: This simple hashing might need improvement for complex args
            cache_key = f"cache:{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get cached result
            cached = await redis_conn.get(cache_key)
            if cached:
                return JSONResponse(content=eval(cached))
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await redis_conn.setex(cache_key, expire_seconds, str(result))
            return result
        return wrapper
    return decorator
