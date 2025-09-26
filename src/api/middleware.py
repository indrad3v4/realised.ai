"""
Essential middleware for Team-Realized API
Rate limiting, CORS, logging, and error handling
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timing
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {response.status_code} "
                f"processed in {process_time:.3f}s"
            )

            # Add timing header
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"Error processing request: {str(e)} "
                f"after {process_time:.3f}s"
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "detail": "An unexpected error occurred",
                    "timestamp": time.time()
                }
            )


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware"""

    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.requests = {}  # Simple in-memory storage

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Current time
        now = time.time()

        # Clean old entries
        self.requests = {
            ip: times for ip, times in self.requests.items()
            if any(t > now - self.period for t in times)
        }

        # Check rate limit
        if client_ip in self.requests:
            # Filter recent requests
            recent_requests = [
                t for t in self.requests[client_ip] 
                if t > now - self.period
            ]

            if len(recent_requests) >= self.calls:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "detail": f"Maximum {self.calls} requests per {self.period} seconds",
                        "retry_after": self.period
                    },
                    headers={"Retry-After": str(self.period)}
                )

            # Add current request
            recent_requests.append(now)
            self.requests[client_ip] = recent_requests
        else:
            # First request from this IP
            self.requests[client_ip] = [now]

        # Process request normally
        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, self.calls - len(self.requests.get(client_ip, [])))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add CSP for production
        if not settings.debug:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self' https:"
            )

        return response


def setup_middleware(app):
    """Setup all middleware for the FastAPI app"""

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Custom middleware (order matters - last added runs first)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Rate limiting (be careful not to be too restrictive during development)
    if not settings.debug:
        app.add_middleware(RateLimitingMiddleware, calls=100, period=60)
    else:
        app.add_middleware(RateLimitingMiddleware, calls=1000, period=60)  # More lenient for dev

    logger.info("Middleware setup completed")


# Exception handlers
def setup_exception_handlers(app):
    """Setup exception handlers"""

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not found",
                "detail": "The requested resource was not found",
                "path": request.url.path
            }
        )

    @app.exception_handler(422)
    async def validation_error_handler(request: Request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "detail": "Invalid request data",
                "errors": exc.errors() if hasattr(exc, 'errors') else str(exc)
            }
        )

    @app.exception_handler(429)
    async def rate_limit_handler(request: Request, exc):
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "detail": "Too many requests, please try again later",
                "retry_after": 60
            }
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        logger.error(f"Internal server error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred"
            }
        )

    logger.info("Exception handlers setup completed")


# Health check utilities
def create_health_check_info():
    """Create health check information"""
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "debug_mode": settings.debug,
        "features": {
            "ai_models": settings.ai_models_enabled,
            "cities_supported": len(settings.target_cities),
            "min_investment": f"{settings.min_investment_amount} EUR",
            "max_investment": f"{settings.max_investment_amount} EUR"
        }
    }


# Export middleware setup functions
__all__ = [
    'LoggingMiddleware',
    'RateLimitingMiddleware', 
    'SecurityHeadersMiddleware',
    'setup_middleware',
    'setup_exception_handlers',
    'create_health_check_info'
]
