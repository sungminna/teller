"""
Rate Limiting Middleware
=======================

요청 제한 및 트래픽 제어
"""

import logging
import time
from typing import Dict

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting 미들웨어"""

    def __init__(self, app, requests_per_minute: int = 100):
        super().__init__(app)
        self.settings = get_settings()
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        """요청 처리"""
        client_ip = self._get_client_ip(request)
        current_time = time.time()

        # Clean old requests
        self._cleanup_old_requests(client_ip, current_time)

        # Check rate limit
        if not self._is_allowed(client_ip, current_time):
            return Response(
                content="Rate limit exceeded. Please try again later.",
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(current_time + 60)),
                },
            )

        # Record request
        self._record_request(client_ip, current_time)

        response = await call_next(request)

        # Add rate limit headers
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + 60))

        return response

    def _get_client_ip(self, request: Request) -> str:
        """클라이언트 IP 추출"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _cleanup_old_requests(self, client_ip: str, current_time: float):
        """오래된 요청 기록 정리"""
        if client_ip not in self.client_requests:
            return

        # Remove requests older than 1 minute
        self.client_requests[client_ip] = [
            req_time for req_time in self.client_requests[client_ip] if current_time - req_time < 60
        ]

    def _is_allowed(self, client_ip: str, current_time: float) -> bool:
        """요청 허용 여부 확인"""
        if client_ip not in self.client_requests:
            return True

        return len(self.client_requests[client_ip]) < self.requests_per_minute

    def _record_request(self, client_ip: str, current_time: float):
        """요청 기록"""
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []

        self.client_requests[client_ip].append(current_time)

    def _get_remaining_requests(self, client_ip: str) -> int:
        """남은 요청 수 반환"""
        if client_ip not in self.client_requests:
            return self.requests_per_minute

        used = len(self.client_requests[client_ip])
        return max(0, self.requests_per_minute - used)
