"""
비동기 처리 유틸리티
Airflow와 FastAPI 간 호환성 문제 해결
"""

import asyncio
import concurrent.futures
import threading
import logging
from typing import Any, Coroutine, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')

def run_async_in_sync_context(coro: Coroutine[Any, Any, T]) -> T:
    """
    동기 컨텍스트에서 비동기 함수를 안전하게 실행
    
    Airflow 태스크에서 asyncio.run() 사용 시 발생하는 문제를 해결:
    - 이미 실행 중인 이벤트 루프가 있을 때 처리
    - 새로운 스레드에서 별도 이벤트 루프 생성
    
    Args:
        coro: 실행할 코루틴
    
    Returns:
        코루틴 실행 결과
    """
    try:
        # 현재 이벤트 루프 확인
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # 이미 실행 중인 루프가 있으면 새 스레드에서 실행
                return _run_in_new_thread(coro)
        except RuntimeError:
            # 실행 중인 루프가 없음
            pass
        
        # 새 이벤트 루프에서 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(coro)
        
    except Exception as e:
        logger.error(f"Error running async function in sync context: {e}")
        raise

def _run_in_new_thread(coro: Coroutine[Any, Any, T]) -> T:
    """
    새로운 스레드에서 코루틴 실행
    
    Args:
        coro: 실행할 코루틴
    
    Returns:
        코루틴 실행 결과
    """
    def run_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result()

def safe_async_run(coro: Coroutine[Any, Any, T], timeout: float = None) -> T:
    """
    안전한 비동기 함수 실행 (타임아웃 지원)
    
    Args:
        coro: 실행할 코루틴
        timeout: 타임아웃 시간 (초)
    
    Returns:
        코루틴 실행 결과
    
    Raises:
        asyncio.TimeoutError: 타임아웃 발생 시
    """
    async def _run_with_timeout():
        if timeout:
            return await asyncio.wait_for(coro, timeout=timeout)
        else:
            return await coro
    
    return run_async_in_sync_context(_run_with_timeout())

class AsyncTaskManager:
    """
    비동기 태스크 관리자
    메모리 누수 방지 및 리소스 정리 보장
    """
    
    def __init__(self):
        self.tasks = set()
        self._cleanup_interval = 60  # 1분마다 정리
    
    def create_task(self, coro: Coroutine) -> asyncio.Task:
        """
        태스크 생성 및 추적
        
        Args:
            coro: 실행할 코루틴
        
        Returns:
            생성된 태스크
        """
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task
    
    async def cancel_all(self):
        """모든 태스크 취소"""
        for task in self.tasks.copy():
            if not task.done():
                task.cancel()
        
        # 취소 완료 대기
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()
    
    def cleanup_completed(self):
        """완료된 태스크 정리"""
        completed = {task for task in self.tasks if task.done()}
        self.tasks -= completed
        
        logger.debug(f"Cleaned up {len(completed)} completed tasks")

# 전역 태스크 매니저
task_manager = AsyncTaskManager()

def with_task_cleanup(func):
    """
    태스크 정리를 보장하는 데코레이터
    
    Args:
        func: 데코레이트할 함수
    
    Returns:
        래핑된 함수
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 완료된 태스크 정리
            task_manager.cleanup_completed()
    
    return wrapper

async def batch_process_with_semaphore(
    items: list,
    process_func: callable,
    max_concurrent: int = 10,
    batch_size: int = None
) -> list:
    """
    세마포어를 사용한 배치 처리
    
    Args:
        items: 처리할 아이템 리스트
        process_func: 처리 함수 (async)
        max_concurrent: 최대 동시 처리 수
        batch_size: 배치 크기 (None이면 전체를 하나의 배치로)
    
    Returns:
        처리 결과 리스트
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _process_item(item):
        async with semaphore:
            return await process_func(item)
    
    if batch_size:
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [task_manager.create_task(_process_item(item)) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            results.extend(batch_results)
        return results
    else:
        tasks = [task_manager.create_task(_process_item(item)) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True) 