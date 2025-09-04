# app/utils/metrics.py
import time
from collections import Counter, deque
import asyncio

class MetricsCollector:
    def __init__(self):
        self.total_requests = Counter()
        self.error_requests = Counter()
        self.response_times = deque(maxlen=1000)
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def track_request(self, status_code: int, process_time: float):
        async with self._lock:
            self.total_requests['count'] += 1
            if status_code >= 400:
                self.error_requests['count'] += 1
            self.response_times.append(process_time)

    async def get_metrics(self) -> dict:
        async with self._lock:
            total = self.total_requests['count']
            errors = self.error_requests['count']
            avg_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            return {
                "uptime_seconds": time.time() - self.start_time,
                "total_requests": total,
                "error_requests": errors,
                "success_rate_percent": (total - errors) / total * 100 if total > 0 else 100,
                "avg_response_time_ms": avg_time * 1000,
            }

metrics_collector = MetricsCollector()