# app/middleware/error_handler.py
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.exceptions import ConnectorException, RateLimitException, AuthenticationException

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Global middleware to catch and format connector-related errors."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except RateLimitException as e:
            logger.warning(f"[{e.platform}] Rate limit hit: {e.message}")
            headers = {"Retry-After": str(e.retry_after)} if e.retry_after else {}
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limit_exceeded", "message": e.message, "platform": e.platform, "retry_after": e.retry_after},
                headers=headers
            )
        except AuthenticationException as e:
            logger.error(f"[{e.platform}] Authentication error: {e.message}")
            return JSONResponse(
                status_code=401,
                content={"error": "authentication_failed", "message": e.message, "platform": e.platform}
            )
        except ConnectorException as e:
            logger.warning(f"[{e.platform}] Connector error: {e.message}")
            status = 503 if e.recoverable else 500
            return JSONResponse(
                status_code=status,
                content={"error": "connector_error", "message": e.message, "platform": e.platform, "recoverable": e.recoverable}
            )
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": "internal_server_error", "message": "An unexpected error occurred"}
            )