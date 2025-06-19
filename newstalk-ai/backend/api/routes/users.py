"""
Users Routes
============

사용자 관리, 인증, 프로필 라우트
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..middleware.auth import TokenData, get_current_user
from ...shared.models.user import User, UserProfile, UserSettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["users"])


class UserCreateRequest(BaseModel):
    """사용자 생성 요청"""
    username: str
    email: str
    password: str


class UserUpdateRequest(BaseModel):
    """사용자 정보 수정 요청"""
    username: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None


@router.post("/register")
async def register_user(user_data: UserCreateRequest):
    """사용자 등록"""
    try:
        # Mock user registration
        return {
            "message": "User registered successfully",
            "user_id": "mock_user_123",
            "username": user_data.username,
            "email": user_data.email
        }
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(status_code=400, detail="Registration failed")


@router.get("/profile")
async def get_user_profile(request: Request, current_user: TokenData = Depends(get_current_user)):
    """사용자 프로필 조회"""
    try:
        # Mock profile data
        return {
            "user_id": current_user.user_id,
            "username": current_user.username,
            "email": current_user.email,
            "preferences": current_user.preferences or {},
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True
        }
    except Exception as e:
        logger.error(f"Profile fetch failed: {e}")
        raise HTTPException(status_code=404, detail="Profile not found")


@router.put("/profile")
async def update_user_profile(
    request: Request, 
    update_data: UserUpdateRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """사용자 프로필 수정"""
    try:
        # Mock profile update
        return {
            "message": "Profile updated successfully",
            "user_id": current_user.user_id,
            "updated_fields": list(update_data.dict(exclude_none=True).keys())
        }
    except Exception as e:
        logger.error(f"Profile update failed: {e}")
        raise HTTPException(status_code=400, detail="Update failed")


@router.get("/settings")
async def get_user_settings(request: Request, current_user: TokenData = Depends(get_current_user)):
    """사용자 설정 조회"""
    try:
        # Mock settings
        return {
            "user_id": current_user.user_id,
            "notifications_enabled": True,
            "email_notifications": True,
            "push_notifications": True,
            "voice_speed": 1.0,
            "theme": "light",
            "language": "ko"
        }
    except Exception as e:
        logger.error(f"Settings fetch failed: {e}")
        raise HTTPException(status_code=404, detail="Settings not found") 