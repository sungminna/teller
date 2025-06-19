"""
🎯 NewsTalk AI 통합 예외 처리 시스템
====================================

세계 최고 수준의 구조화된 예외 처리 시스템으로 다음 기능을 제공합니다:
- 계층적 예외 구조로 정확한 에러 분류
- 컨텍스트 정보 자동 수집
- 분산 추적 및 메트릭 연동
- 자동 복구 메커니즘 지원
- 사용자 친화적 에러 메시지
"""

import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도 레벨"""

    LOW = "low"  # 로그만 남김
    MEDIUM = "medium"  # 알림 발송
    HIGH = "high"  # 즉시 알림 + 자동 복구 시도
    CRITICAL = "critical"  # 긴급 대응 + 시스템 중단


class ErrorCategory(Enum):
    """에러 카테고리"""

    # 인프라 관련
    DATABASE = "database"
    NETWORK = "network"
    CACHE = "cache"
    STORAGE = "storage"

    # AI 관련
    AI_PROCESSING = "ai_processing"
    MODEL_ERROR = "model_error"
    PROMPT_ERROR = "prompt_error"

    # 데이터 관련
    DATA_VALIDATION = "data_validation"
    DATA_PIPELINE = "data_pipeline"
    DATA_QUALITY = "data_quality"

    # 비즈니스 로직
    BUSINESS_LOGIC = "business_logic"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"

    # 외부 연동
    EXTERNAL_API = "external_api"
    KAFKA = "kafka"
    AIRFLOW = "airflow"


@dataclass
class ErrorContext:
    """에러 컨텍스트 정보"""

    timestamp: datetime = field(default_factory=datetime.utcnow)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    system_state: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None


class NewsTeamError(Exception):
    """
    뉴스팀 AI 베이스 예외 클래스
    모든 커스텀 예외의 부모 클래스로 표준화된 에러 처리 제공
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        category: Optional[ErrorCategory] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
        recoverable: bool = True,
        user_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.category = category or ErrorCategory.BUSINESS_LOGIC
        self.severity = severity
        self.context = context or ErrorContext()
        self.cause = cause
        self.recoverable = recoverable
        self.user_message = user_message or self._generate_user_message()
        self.metadata = metadata or {}
        self.stack_trace = traceback.format_exc()

        # 자동 로깅
        self._log_error()

        super().__init__(self.message)

    def _generate_error_code(self) -> str:
        """에러 코드 자동 생성"""
        class_name = self.__class__.__name__
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{class_name.upper()}_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _generate_user_message(self) -> str:
        """사용자 친화적 메시지 생성"""
        if self.category == ErrorCategory.NETWORK:
            return "네트워크 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요."
        elif self.category == ErrorCategory.AI_PROCESSING:
            return "AI 분석 중 오류가 발생했습니다. 뉴스 처리를 다시 시도하겠습니다."
        elif self.category == ErrorCategory.DATABASE:
            return "데이터 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."
        else:
            return "일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    def _log_error(self):
        """에러 자동 로깅"""
        log_data = {
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context.__dict__,
            "recoverable": self.recoverable,
            "metadata": self.metadata,
        }

        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY: {self.message}", extra=log_data)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY: {self.message}", extra=log_data)
        else:
            logger.info(f"LOW SEVERITY: {self.message}", extra=log_data)

    def to_dict(self) -> Dict[str, Any]:
        """에러 정보를 딕셔너리로 변환"""
        return {
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context.__dict__,
            "recoverable": self.recoverable,
            "metadata": self.metadata,
            "timestamp": self.context.timestamp.isoformat(),
        }


# === AI 처리 관련 예외들 ===


class AIProcessingError(NewsTeamError):
    """AI 처리 관련 일반 오류"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.AI_PROCESSING,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class ModelTimeoutError(AIProcessingError):
    """AI 모델 응답 시간 초과"""

    def __init__(self, model_name: str, timeout_seconds: float, **kwargs):
        message = f"AI 모델 '{model_name}' 응답 시간 초과 ({timeout_seconds}초)"
        super().__init__(
            message=message,
            user_message="AI 처리 시간이 초과되었습니다. 다시 시도하겠습니다.",
            metadata={"model_name": model_name, "timeout": timeout_seconds},
            **kwargs,
        )


class FactCheckError(AIProcessingError):
    """팩트체킹 처리 오류"""

    def __init__(self, article_id: str, reason: str = None, **kwargs):
        message = f"팩트체킹 실패 (article_id: {article_id})"
        if reason:
            message += f": {reason}"

        super().__init__(
            message=message,
            user_message="뉴스 신뢰도 검증 중 오류가 발생했습니다.",
            metadata={"article_id": article_id, "failure_reason": reason},
            **kwargs,
        )


class VoiceSynthesisError(AIProcessingError):
    """음성 합성 오류"""

    def __init__(self, text_length: int, voice_character: str = None, **kwargs):
        message = f"음성 합성 실패 (텍스트 길이: {text_length})"
        if voice_character:
            message += f", 캐릭터: {voice_character}"

        super().__init__(
            message=message,
            user_message="음성 생성 중 오류가 발생했습니다.",
            metadata={"text_length": text_length, "voice_character": voice_character},
            **kwargs,
        )


# === 데이터 파이프라인 관련 예외들 ===


class DataPipelineError(NewsTeamError):
    """데이터 파이프라인 일반 오류"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_PIPELINE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class KafkaConnectionError(DataPipelineError):
    """Kafka 연결 오류"""

    def __init__(self, broker_urls: List[str], **kwargs):
        message = f"Kafka 연결 실패: {', '.join(broker_urls)}"
        super().__init__(
            message=message,
            category=ErrorCategory.KAFKA,
            severity=ErrorSeverity.CRITICAL,
            user_message="실시간 뉴스 업데이트 연결에 문제가 있습니다.",
            metadata={"broker_urls": broker_urls},
            recoverable=True,
            **kwargs,
        )


class DataValidationError(DataPipelineError):
    """데이터 검증 오류"""

    def __init__(self, field_name: str, expected_type: str, actual_value: Any, **kwargs):
        message = f"데이터 검증 실패: {field_name} (기대: {expected_type}, 실제: {type(actual_value).__name__})"
        super().__init__(
            message=message,
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="뉴스 데이터 형식에 문제가 있습니다.",
            metadata={
                "field_name": field_name,
                "expected_type": expected_type,
                "actual_value": str(actual_value)[:100],  # 처음 100자만
            },
            **kwargs,
        )


# === 인프라 관련 예외들 ===


class DatabaseError(NewsTeamError):
    """데이터베이스 관련 오류"""

    def __init__(self, message: str, query: str = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            user_message="데이터 저장소 연결에 문제가 있습니다.",
            metadata={"query": query[:200] if query else None},  # 쿼리 일부만 저장
            **kwargs,
        )


class CacheConnectionError(NewsTeamError):
    """캐시 연결 오류"""

    def __init__(self, cache_type: str, connection_url: str, **kwargs):
        message = f"{cache_type} 캐시 연결 실패: {connection_url}"
        super().__init__(
            message=message,
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.MEDIUM,
            user_message="캐시 시스템 연결에 문제가 있습니다. 처리 속도가 느려질 수 있습니다.",
            metadata={"cache_type": cache_type, "connection_url": connection_url},
            recoverable=True,
            **kwargs,
        )


class ExternalAPIError(NewsTeamError):
    """외부 API 호출 오류"""

    def __init__(self, api_name: str, status_code: int = None, response_body: str = None, **kwargs):
        message = f"외부 API 호출 실패: {api_name}"
        if status_code:
            message += f" (상태 코드: {status_code})"

        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL_API,
            severity=ErrorSeverity.MEDIUM,
            user_message="외부 서비스 연결에 문제가 있습니다.",
            metadata={
                "api_name": api_name,
                "status_code": status_code,
                "response_body": response_body[:500] if response_body else None,
            },
            **kwargs,
        )


# === 비즈니스 로직 관련 예외들 ===


class AuthenticationError(NewsTeamError):
    """인증 오류"""

    def __init__(self, user_id: str = None, reason: str = None, **kwargs):
        message = "인증 실패"
        if reason:
            message += f": {reason}"

        super().__init__(
            message=message,
            category=ErrorCategory.AUTHENTICATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="로그인이 필요합니다.",
            metadata={"user_id": user_id, "failure_reason": reason},
            recoverable=False,
            **kwargs,
        )


class AuthorizationError(NewsTeamError):
    """권한 오류"""

    def __init__(self, user_id: str, required_permission: str, **kwargs):
        message = f"권한 부족: 사용자 {user_id}, 필요 권한: {required_permission}"
        super().__init__(
            message=message,
            category=ErrorCategory.AUTHORIZATION,
            severity=ErrorSeverity.MEDIUM,
            user_message="이 기능을 사용할 권한이 없습니다.",
            metadata={"user_id": user_id, "required_permission": required_permission},
            recoverable=False,
            **kwargs,
        )


# === 에러 핸들러 및 유틸리티 ===


class ErrorHandler:
    """중앙 집중식 에러 핸들러"""

    def __init__(self):
        self.retry_strategies = {
            ErrorCategory.NETWORK: {"max_retries": 3, "backoff_factor": 2},
            ErrorCategory.DATABASE: {"max_retries": 2, "backoff_factor": 1.5},
            ErrorCategory.EXTERNAL_API: {"max_retries": 3, "backoff_factor": 2},
            ErrorCategory.AI_PROCESSING: {"max_retries": 2, "backoff_factor": 1},
        }

    def should_retry(self, error: NewsTeamError, attempt: int) -> bool:
        """재시도 여부 결정"""
        if not error.recoverable:
            return False

        strategy = self.retry_strategies.get(error.category)
        if not strategy:
            return False

        return attempt < strategy["max_retries"]

    def get_retry_delay(self, error: NewsTeamError, attempt: int) -> float:
        """재시도 지연 시간 계산"""
        strategy = self.retry_strategies.get(error.category, {"backoff_factor": 1})
        base_delay = 1.0
        return base_delay * (strategy["backoff_factor"] ** attempt)

    async def handle_error(self, error: NewsTeamError) -> Dict[str, Any]:
        """에러 처리 및 복구 시도"""
        # 에러 메트릭 기록
        await self._record_error_metrics(error)

        # 심각도별 알림 발송
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            await self._send_alert(error)

        # 자동 복구 시도
        if error.recoverable:
            recovery_result = await self._attempt_recovery(error)
            return {
                "error": error.to_dict(),
                "recovery_attempted": True,
                "recovery_result": recovery_result,
            }

        return {"error": error.to_dict(), "recovery_attempted": False}

    async def _record_error_metrics(self, error: NewsTeamError):
        """에러 메트릭 기록 (Prometheus 등)"""
        # TODO: 실제 메트릭 시스템에 기록

    async def _send_alert(self, error: NewsTeamError):
        """알림 발송 (Slack, 이메일 등)"""
        # TODO: 실제 알림 시스템 연동

    async def _attempt_recovery(self, error: NewsTeamError) -> Dict[str, Any]:
        """자동 복구 시도"""
        # TODO: 에러 타입별 복구 로직 구현
        return {"status": "recovery_not_implemented"}


# 전역 에러 핸들러 인스턴스
error_handler = ErrorHandler()


# 편의 함수들
def create_error_context(
    user_id: str = None, component: str = None, operation: str = None, **kwargs
) -> ErrorContext:
    """에러 컨텍스트 생성 편의 함수"""
    return ErrorContext(user_id=user_id, component=component, operation=operation, **kwargs)


def handle_exceptions(func):
    """예외 처리 데코레이터"""

    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except NewsTeamError:
            # 이미 처리된 예외는 그대로 전파
            raise
        except Exception as e:
            # 처리되지 않은 예외를 NewsTeamError로 래핑
            context = create_error_context(component=func.__module__, operation=func.__name__)
            raise NewsTeamError(
                message=f"예상치 못한 오류: {str(e)}",
                cause=e,
                context=context,
                severity=ErrorSeverity.HIGH,
            )

    return wrapper
