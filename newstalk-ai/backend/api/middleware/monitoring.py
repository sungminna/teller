"""
Monitoring Middleware
====================

성능 메트릭 수집 및 추적
"""

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """성능 모니터링 미들웨어"""
    
    def __init__(self, app):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        """요청 처리 및 메트릭 수집"""
        start_time = time.time()
        method = request.method
        path = request.url.path
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            response.headers["X-Process-Time"] = str(process_time)
            
            logger.info(
                f"Request: {method} {path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {method} {path} - "
                f"Error: {str(e)} - "
                f"Time: {process_time:.3f}s"
            )
            raise


def track_ai_processing(func):
    """AI 처리 추적 데코레이터"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            process_time = time.time() - start_time
            logger.info(f"AI processing completed in {process_time:.3f}s")
            return result
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"AI processing failed after {process_time:.3f}s: {e}")
            raise
    return wrapper 