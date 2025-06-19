"""
Authentication Middleware
========================

JWT 기반 인증 및 권한 관리
"""

import logging
import time
from typing import Optional

import jwt
from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)
security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """JWT 인증 미들웨어"""

    def __init__(self, app, skip_paths: Optional[list] = None):
        super().__init__(app)
        self.settings = get_settings()
        self.skip_paths = skip_paths or [
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/metrics",
            "/favicon.ico",
        ]

    async def dispatch(self, request: Request, call_next):
        """요청 처리"""
        start_time = time.time()

        # Skip authentication for certain paths
        path = request.url.path
        if any(path.startswith(skip_path) for skip_path in self.skip_paths):
            response = await call_next(request)
            return response

        # Extract token
        try:
            token = await self._extract_token(request)
            if token:
                payload = self._verify_token(token)
                request.state.user = payload
            else:
                request.state.user = None

        except HTTPException:
            return Response(content="Unauthorized", status_code=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            return Response(
                content="Internal Server Error", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        response = await call_next(request)

        # Add timing header
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response

    async def _extract_token(self, request: Request) -> Optional[str]:
        """토큰 추출"""
        # Bearer token from Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Token from query parameter (for WebSocket)
        token = request.query_params.get("token")
        if token:
            return token

        return None

    def _verify_token(self, token: str) -> dict:
        """토큰 검증"""
        try:
            payload = jwt.decode(token, self.settings.api.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    """액세스 토큰 생성"""
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = time.time() + expires_delta
    else:
        expire = time.time() + (settings.api.access_token_expire_minutes * 60)

    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, settings.api.secret_key, algorithm="HS256")
    return encoded_jwt


def get_current_user(request: Request) -> Optional[dict]:
    """현재 사용자 정보 반환"""
    return getattr(request.state, "user", None)


def require_auth(request: Request) -> dict:
    """인증 필수 데코레이터"""
    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return user


# Additional imports and classes needed
from pydantic import BaseModel
from fastapi import Depends


class TokenData(BaseModel):
    """JWT 토큰 데이터"""
    user_id: str
    username: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[dict] = None


def get_current_user_optional(request: Request) -> Optional[dict]:
    """선택적 인증 (토큰이 없어도 됨)"""
    return get_current_user(request)
