# app/middleware/metrics_middleware.py
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.utils.metrics import metrics_collector

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        await metrics_collector.track_request(response.status_code, process_time)
        return response