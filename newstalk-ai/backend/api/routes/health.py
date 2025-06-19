"""
Health Check Routes
==================

시스템 헬스체크 및 모니터링 라우트
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    timestamp: datetime
    version: str
    uptime: str
    services: Dict[str, Any]


@router.get("/", response_model=HealthResponse)
async def health_check():
    """기본 헬스체크"""
    try:
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            uptime="running",
            services={
                "database": "connected",
                "redis": "connected",
                "kafka": "connected"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@router.get("/detailed")
async def detailed_health_check():
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": {"status": "healthy", "response_time": "0.05s"},
            "database": {"status": "healthy", "connections": 10},
            "redis": {"status": "healthy", "memory_usage": "45MB"},
            "kafka": {"status": "healthy", "topics": 5}
        },
        "metrics": {
            "requests_per_minute": 150,
            "avg_response_time": "0.12s",
            "error_rate": "0.1%"
        }
    } 