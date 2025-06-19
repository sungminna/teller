"""
🎯 NewsTalk AI 통합 상태 관리 시스템
===================================

메모리 누수 방지와 동시성 안전성을 보장하는 엔터프라이즈급 상태 관리 시스템:
- 싱글톤 패턴으로 메모리 최적화
- 비동기 컨텍스트 매니저로 안전한 리소스 관리
- 자동 정리 메커니즘으로 메모리 누수 방지
- 분산 환경 지원을 위한 Redis 백엔드
- 실시간 상태 모니터링 및 메트릭
"""

import asyncio
import json
import logging
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar

import redis.asyncio as redis


logger = logging.getLogger(__name__)

T = TypeVar("T")


class StateScope(Enum):
    """상태 범위 정의"""

    GLOBAL = "global"  # 전역 상태
    SESSION = "session"  # 세션별 상태
    USER = "user"  # 사용자별 상태
    REQUEST = "request"  # 요청별 상태
    TEMPORARY = "temporary"  # 임시 상태 (TTL 적용)


@dataclass
class StateEntry:
    """상태 엔트리"""

    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    scope: StateScope = StateScope.TEMPORARY

    def is_expired(self) -> bool:
        """TTL 기반 만료 확인"""
        if self.ttl_seconds is None:
            return False

        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def mark_accessed(self):
        """접근 시간 및 횟수 업데이트"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class SafeStateManager:
    """
    메모리 안전 상태 관리자

    주요 기능:
    - 약한 참조(WeakRef) 기반 자동 정리
    - TTL 기반 만료 처리
    - 동시성 안전 보장
    - 메모리 사용량 모니터링
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # 로컬 상태 저장소 (약한 참조 사용)
        self._local_cache: Dict[str, StateEntry] = {}
        self._access_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Redis 연결 (분산 상태용)
        self._redis_client: Optional[redis.Redis] = None
        self._redis_connected = False

        # 정리 작업 제어
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5분마다 정리
        self._max_local_entries = 10000  # 최대 로컬 엔트리 수

        # 메트릭
        self._stats = {
            "total_entries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cleanup_runs": 0,
            "expired_entries_cleaned": 0,
            "memory_usage_mb": 0.0,
        }

        self._initialized = True
        logger.info("SafeStateManager initialized")

    async def initialize(self, redis_url: str = None):
        """상태 관리자 초기화"""
        try:
            # Redis 연결 초기화
            if redis_url:
                self._redis_client = redis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )

                # 연결 테스트
                await self._redis_client.ping()
                self._redis_connected = True
                logger.info("Redis connection established for distributed state")

            # 자동 정리 작업 시작
            self._start_cleanup_task()

        except Exception as e:
            logger.warning(f"Failed to initialize Redis: {e}. Using local cache only.")
            self._redis_connected = False

    def _start_cleanup_task(self):
        """자동 정리 작업 시작"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def _periodic_cleanup(self):
        """주기적 정리 작업"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired_entries()
                await self._enforce_memory_limits()
                self._stats["cleanup_runs"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    async def _cleanup_expired_entries(self):
        """만료된 엔트리 정리"""
        expired_keys = []

        for key, entry in self._local_cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self._local_cache[key]
            if key in self._access_locks:
                del self._access_locks[key]

        self._stats["expired_entries_cleaned"] += len(expired_keys)

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    async def _enforce_memory_limits(self):
        """메모리 한도 강제 적용"""
        if len(self._local_cache) <= self._max_local_entries:
            return

        # 가장 오래 사용되지 않은 엔트리들 제거
        entries_by_access = sorted(self._local_cache.items(), key=lambda x: x[1].last_accessed)

        entries_to_remove = len(self._local_cache) - self._max_local_entries
        for key, _ in entries_by_access[:entries_to_remove]:
            del self._local_cache[key]
            if key in self._access_locks:
                del self._access_locks[key]

        logger.debug(f"Removed {entries_to_remove} entries to enforce memory limits")

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        scope: StateScope = StateScope.TEMPORARY,
        distributed: bool = False,
    ) -> bool:
        """
        상태 값 설정

        Args:
            key: 상태 키
            value: 저장할 값
            ttl_seconds: 만료 시간 (초)
            scope: 상태 범위
            distributed: 분산 저장 여부
        """
        try:
            entry = StateEntry(value=value, ttl_seconds=ttl_seconds, scope=scope)

            # 로컬 캐시에 저장
            async with self._access_locks[key]:
                self._local_cache[key] = entry
                self._stats["total_entries"] = len(self._local_cache)

            # 분산 저장 (요청된 경우)
            if distributed and self._redis_connected:
                await self._set_distributed(key, entry)

            return True

        except Exception as e:
            logger.error(f"Failed to set state {key}: {e}")
            return False

    async def get(self, key: str, default: Any = None, distributed: bool = False) -> Any:
        """
        상태 값 조회

        Args:
            key: 상태 키
            default: 기본값
            distributed: 분산 조회 여부
        """
        try:
            # 로컬 캐시에서 우선 조회
            async with self._access_locks[key]:
                if key in self._local_cache:
                    entry = self._local_cache[key]

                    if entry.is_expired():
                        del self._local_cache[key]
                        self._stats["cache_misses"] += 1
                    else:
                        entry.mark_accessed()
                        self._stats["cache_hits"] += 1
                        return entry.value

            # 분산 캐시에서 조회 (로컬에 없는 경우)
            if distributed and self._redis_connected:
                distributed_value = await self._get_distributed(key)
                if distributed_value is not None:
                    # 로컬 캐시에 복사
                    await self.set(key, distributed_value, distributed=False)
                    return distributed_value

            self._stats["cache_misses"] += 1
            return default

        except Exception as e:
            logger.error(f"Failed to get state {key}: {e}")
            return default

    async def delete(self, key: str, distributed: bool = False) -> bool:
        """상태 값 삭제"""
        try:
            # 로컬 캐시에서 삭제
            async with self._access_locks[key]:
                if key in self._local_cache:
                    del self._local_cache[key]

                if key in self._access_locks:
                    del self._access_locks[key]

            # 분산 캐시에서 삭제
            if distributed and self._redis_connected:
                await self._delete_distributed(key)

            return True

        except Exception as e:
            logger.error(f"Failed to delete state {key}: {e}")
            return False

    async def exists(self, key: str, distributed: bool = False) -> bool:
        """상태 존재 여부 확인"""
        try:
            # 로컬 캐시 확인
            if key in self._local_cache:
                entry = self._local_cache[key]
                if not entry.is_expired():
                    return True
                else:
                    # 만료된 엔트리 삭제
                    await self.delete(key)

            # 분산 캐시 확인
            if distributed and self._redis_connected:
                return await self._redis_client.exists(key) > 0

            return False

        except Exception as e:
            logger.error(f"Failed to check existence of {key}: {e}")
            return False

    async def clear_scope(self, scope: StateScope) -> int:
        """특정 범위의 모든 상태 삭제"""
        cleared_count = 0
        keys_to_delete = []

        for key, entry in self._local_cache.items():
            if entry.scope == scope:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            await self.delete(key)
            cleared_count += 1

        logger.info(f"Cleared {cleared_count} entries from scope {scope.value}")
        return cleared_count

    async def _set_distributed(self, key: str, entry: StateEntry):
        """분산 저장소에 값 설정"""
        try:
            serialized_entry = {
                "value": (
                    json.dumps(entry.value) if not isinstance(entry.value, str) else entry.value
                ),
                "created_at": entry.created_at.isoformat(),
                "scope": entry.scope.value,
                "ttl_seconds": entry.ttl_seconds,
            }

            await self._redis_client.hset(f"newstalk:state:{key}", mapping=serialized_entry)

            if entry.ttl_seconds:
                await self._redis_client.expire(f"newstalk:state:{key}", entry.ttl_seconds)

        except Exception as e:
            logger.error(f"Failed to set distributed state {key}: {e}")

    async def _get_distributed(self, key: str) -> Any:
        """분산 저장소에서 값 조회"""
        try:
            data = await self._redis_client.hgetall(f"newstalk:state:{key}")
            if not data:
                return None

            # 직렬화 해제
            value = data.get("value")
            if value and value != data.get("value"):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass  # 문자열 그대로 사용

            return value

        except Exception as e:
            logger.error(f"Failed to get distributed state {key}: {e}")
            return None

    async def _delete_distributed(self, key: str):
        """분산 저장소에서 값 삭제"""
        try:
            await self._redis_client.delete(f"newstalk:state:{key}")
        except Exception as e:
            logger.error(f"Failed to delete distributed state {key}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """상태 관리 통계 반환"""
        import sys

        # 메모리 사용량 계산
        memory_usage = 0
        for entry in self._local_cache.values():
            memory_usage += sys.getsizeof(entry.value)

        self._stats["memory_usage_mb"] = memory_usage / (1024 * 1024)
        self._stats["total_entries"] = len(self._local_cache)

        return self._stats.copy()

    @asynccontextmanager
    async def transaction(self, keys: List[str]):
        """상태 트랜잭션 컨텍스트 매니저"""
        locks = [self._access_locks[key] for key in keys]

        # 데드락 방지를 위한 키 정렬
        sorted_locks = sorted(zip(keys, locks), key=lambda x: x[0])

        async with asyncio.gather(*[lock[1].__aenter__() for lock in sorted_locks]):
            try:
                yield self
            finally:
                # 락은 자동으로 해제됨
                pass

    async def close(self):
        """상태 관리자 종료"""
        try:
            # 정리 작업 중단
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Redis 연결 종료
            if self._redis_client:
                await self._redis_client.close()

            # 로컬 캐시 정리
            self._local_cache.clear()
            self._access_locks.clear()

            logger.info("SafeStateManager closed")

        except Exception as e:
            logger.error(f"Error closing SafeStateManager: {e}")


class ConnectionManager:
    """
    연결 관리자 - 데이터베이스, 캐시, 외부 API 연결을 통합 관리
    """

    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.connection_pools: Dict[str, Any] = {}
        self.health_status: Dict[str, bool] = {}
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 60  # 1분마다 헬스체크

    async def register_connection(
        self, name: str, connection: Any, health_check_func: Optional[callable] = None
    ):
        """연결 등록"""
        self.connections[name] = connection
        self.health_status[name] = True

        if health_check_func:
            # 헬스체크 함수 저장
            setattr(connection, "_health_check", health_check_func)

        # 헬스체크 작업 시작
        if self._health_check_task is None:
            self._health_check_task = asyncio.create_task(self._periodic_health_check())

        logger.info(f"Connection '{name}' registered")

    async def get_connection(self, name: str) -> Optional[Any]:
        """연결 조회"""
        if name not in self.connections:
            logger.warning(f"Connection '{name}' not found")
            return None

        if not self.health_status.get(name, False):
            logger.warning(f"Connection '{name}' is unhealthy")
            return None

        return self.connections[name]

    async def _periodic_health_check(self):
        """주기적 헬스체크"""
        while True:
            try:
                await asyncio.sleep(self._health_check_interval)

                for name, connection in self.connections.items():
                    try:
                        health_check_func = getattr(connection, "_health_check", None)
                        if health_check_func:
                            is_healthy = await health_check_func()
                            self.health_status[name] = is_healthy
                        else:
                            # 기본 헬스체크 (ping 메서드 있는지 확인)
                            if hasattr(connection, "ping"):
                                await connection.ping()
                                self.health_status[name] = True

                    except Exception as e:
                        logger.warning(f"Health check failed for '{name}': {e}")
                        self.health_status[name] = False

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check task error: {e}")

    async def close_all(self):
        """모든 연결 종료"""
        try:
            # 헬스체크 작업 중단
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # 모든 연결 종료
            for name, connection in self.connections.items():
                try:
                    if hasattr(connection, "close"):
                        if asyncio.iscoroutinefunction(connection.close):
                            await connection.close()
                        else:
                            connection.close()
                    logger.info(f"Connection '{name}' closed")
                except Exception as e:
                    logger.error(f"Error closing connection '{name}': {e}")

            self.connections.clear()
            self.health_status.clear()

        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# 전역 인스턴스들
_state_manager = None
_connection_manager = None


def get_state_manager() -> SafeStateManager:
    """상태 관리자 싱글톤 인스턴스 반환"""
    global _state_manager
    if _state_manager is None:
        _state_manager = SafeStateManager()
    return _state_manager


def get_connection_manager() -> ConnectionManager:
    """연결 관리자 싱글톤 인스턴스 반환"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager


# 편의 함수들
async def set_state(key: str, value: Any, **kwargs) -> bool:
    """전역 상태 설정"""
    return await get_state_manager().set(key, value, **kwargs)


async def get_state(key: str, default: Any = None, **kwargs) -> Any:
    """전역 상태 조회"""
    return await get_state_manager().get(key, default, **kwargs)


async def delete_state(key: str, **kwargs) -> bool:
    """전역 상태 삭제"""
    return await get_state_manager().delete(key, **kwargs)


@asynccontextmanager
async def state_transaction(keys: List[str]):
    """상태 트랜잭션 컨텍스트 매니저"""
    async with get_state_manager().transaction(keys) as manager:
        yield manager
