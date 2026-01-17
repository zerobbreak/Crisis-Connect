"""
Base Service Class for Crisis Connect
Provides common functionality for all services
"""

from abc import ABC
from typing import Optional, Dict, Any, Callable
import structlog
from motor.motor_asyncio import AsyncIOMotorDatabase
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from datetime import datetime


class BaseService(ABC):
    """Base class for all services with common functionality"""
    
    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None, app=None):
        """
        Initialize base service
        
        Args:
            db: MongoDB database instance
            app: FastAPI app instance (alternative to db)
        """
        if db:
            self.db = db
        elif app and hasattr(app, 'state'):
            self.db = getattr(app.state, 'db', None)
        else:
            self.db = None
            
        self.app = app
        self.logger = structlog.get_logger(self.__class__.__name__)
        self._setup()
    
    def _setup(self):
        """Override for service-specific setup"""
        pass
    
    async def _handle_error(
        self, 
        error: Exception, 
        context: Dict[str, Any],
        reraise: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Centralized error handling with logging
        
        Args:
            error: The exception that occurred
            context: Additional context for logging
            reraise: Whether to reraise the exception
            
        Returns:
            Error dict if not reraising, None otherwise
        """
        error_info = {
            "error": str(error),
            "error_type": type(error).__name__,
            "timestamp": datetime.utcnow().isoformat(),
            **context
        }
        
        self.logger.error(
            f"{self.__class__.__name__} error",
            **error_info
        )
        
        if reraise:
            raise
        
        return {
            "success": False,
            "error": str(error),
            "error_type": type(error).__name__
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError))
    )
    async def _api_call_with_retry(
        self, 
        func: Callable, 
        *args, 
        **kwargs
    ) -> Any:
        """
        Execute API call with automatic retry logic
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from the function call
            
        Raises:
            httpx.TimeoutException: If all retries timeout
            httpx.HTTPError: If HTTP error persists after retries
        """
        try:
            return await func(*args, **kwargs)
        except httpx.TimeoutException as e:
            self.logger.warning(
                "API timeout, retrying...",
                function=func.__name__,
                attempt=kwargs.get('attempt', 1)
            )
            raise
        except httpx.HTTPError as e:
            self.logger.error(
                f"API HTTP error: {e}",
                function=func.__name__,
                status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            )
            raise
    
    async def _validate_db_connection(self) -> bool:
        """
        Validate database connection is available
        
        Returns:
            True if connected, False otherwise
        """
        if not self.db:
            self.logger.error("Database connection not available")
            return False
        
        try:
            await self.db.command('ping')
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def _log_operation(
        self, 
        operation: str, 
        details: Dict[str, Any],
        level: str = "info"
    ):
        """
        Log service operation with consistent format
        
        Args:
            operation: Name of the operation
            details: Operation details
            level: Log level (info, warning, error)
        """
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(
            f"{self.__class__.__name__}.{operation}",
            **details
        )
