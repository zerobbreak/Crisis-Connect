"""
Logging and monitoring middleware for Crisis Connect API
"""
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.routing import APIRoute
import structlog

logger = structlog.get_logger(__name__)


class LoggingMiddleware:
    """Middleware for request/response logging and monitoring"""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Generate request ID
            request_id = str(uuid.uuid4())[:8]
            
            # Log request start
            start_time = time.time()
            
            # Add request ID to scope
            scope["request_id"] = request_id
            
            # Wrap send to log response
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Log response
                    process_time = time.time() - start_time
                    
                    # Extract request info
                    method = scope.get("method", "UNKNOWN")
                    path = scope.get("path", "UNKNOWN")
                    query_string = scope.get("query_string", b"").decode()
                    
                    # Extract response info
                    status_code = message.get("status", 0)
                    
                    # Log the request/response
                    logger.info(
                        "HTTP Request",
                        request_id=request_id,
                        method=method,
                        path=path,
                        query_string=query_string,
                        status_code=status_code,
                        process_time_ms=round(process_time * 1000, 2)
                    )
                
                await send(message)
            
            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


class LoggingRoute(APIRoute):
    """Custom route class with enhanced logging"""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Log route-specific information
            request_id = getattr(request.scope, "request_id", "unknown")
            
            logger.info(
                "Route Handler",
                request_id=request_id,
                route_name=self.name,
                endpoint=self.endpoint.__name__,
                path=request.url.path
            )
            
            try:
                response = await original_route_handler(request)
                return response
            except Exception as e:
                logger.error(
                    "Route Handler Error",
                    request_id=request_id,
                    route_name=self.name,
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return custom_route_handler


def setup_logging_middleware(app):
    """Setup logging middleware for the application"""
    
    # Add logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Replace default route class
    app.router.route_class = LoggingRoute
    
    return app
