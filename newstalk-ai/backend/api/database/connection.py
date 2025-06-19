"""
Database Connection
==================

SQLAlchemy 기반 데이터베이스 연결 관리
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# Global variables
_engine = None
_async_session = None


def get_engine():
    """데이터베이스 엔진 반환"""
    global _engine
    if _engine is None:
        settings = get_settings()
        database_url = settings.get_database_url()

        # SQLite의 경우 특별 처리
        if "sqlite" in database_url:
            _engine = create_async_engine(
                database_url,
                echo=settings.debug,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
            )
        else:
            _engine = create_async_engine(
                database_url,
                echo=settings.debug,
                pool_size=settings.database.min_pool_size,
                max_overflow=settings.database.max_pool_size - settings.database.min_pool_size,
                pool_timeout=settings.database.pool_timeout,
                pool_recycle=settings.database.pool_recycle,
            )

        logger.info(
            f"Database engine created for: {database_url.split('@')[-1] if '@' in database_url else database_url}"
        )

    return _engine


def get_session_factory():
    """세션 팩토리 반환"""
    global _async_session
    if _async_session is None:
        engine = get_engine()
        _async_session = sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False,
        )
    return _async_session


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """데이터베이스 세션 의존성"""
    async_session = get_session_factory()
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """테이블 생성 (개발용)"""
    try:
        get_engine()
        # 실제 모델들이 정의되면 여기서 테이블 생성
        # async with engine.begin() as conn:
        #     await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise


async def close_db():
    """데이터베이스 연결 종료"""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database connection closed")
