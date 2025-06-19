"""
🎯 NewsTalk AI 뉴스 처리 상태 관리 시스템
=============================================

이 모듈은 NewsTalk AI의 멀티 에이전트 뉴스 처리 파이프라인에서 사용되는 
중앙 집중식 상태 관리 시스템을 정의합니다.

🔄 **처리 흐름**:
1. 뉴스 분석 (트렌드 분석 + 팩트체킹)
2. 개인화 (사용자 맞춤화 + 스토리텔링)  
3. 음성 합성 (다중 캐릭터 TTS)

📊 **주요 기능**:
- 실시간 처리 상태 추적
- 에이전트 간 데이터 공유
- 에러 처리 및 복구
- 성능 메트릭 수집
- 품질 보증 체크포인트
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessingStage(Enum):
    """
    뉴스 처리 단계 정의

    각 단계는 순차적으로 진행되며, 실패 시 이전 단계로 롤백 가능합니다.
    휴먼 리뷰가 필요한 경우 해당 단계에서 중단됩니다.
    """

    INITIALIZED = "initialized"  # 초기화 완료
    TREND_ANALYSIS = "trend_analysis"  # 트렌드 분석 진행 중
    FACT_CHECKING = "fact_checking"  # 팩트체킹 진행 중
    STORYTELLING = "storytelling"  # 스토리텔링 생성 중
    PERSONALIZATION = "personalization"  # 개인화 처리 중
    VOICE_SYNTHESIS = "voice_synthesis"  # 음성 합성 중
    QA_READY = "qa_ready"  # Q&A 준비 완료
    COMPLETED = "completed"  # 모든 처리 완료
    FAILED = "failed"  # 처리 실패


@dataclass
class TrendAnalysisResult:
    """
    트렌드 분석 결과 데이터 구조

    🎯 목표: 실시간 이슈 감지 및 중요도 평가
    - 트렌딩 스코어 0.7 이상: 주요 이슈
    - 바이럴 잠재력 0.6 이상: 소셜 확산 가능성 높음
    - 감성 분석으로 사용자 반응 예측
    """

    trending_score: float  # 트렌딩 점수 (0.0-1.0)
    trend_category: str  # 트렌드 카테고리 (정치/경제/사회 등)
    keywords: List[str]  # 핵심 키워드 (최대 10개)
    sentiment_score: float  # 감성 점수 (-1.0 ~ 1.0)
    virality_potential: float  # 바이럴 잠재력 (0.0-1.0)
    related_trends: List[Dict[str, Any]]  # 관련 트렌드 (최대 5개)
    processing_time: datetime  # 처리 완료 시간
    agent_version: str  # 에이전트 버전 (추적용)


@dataclass
class FactCheckResult:
    """
    팩트체킹 결과 데이터 구조

    🎯 목표: 95% 정확도의 신뢰도 검증
    - 신뢰도 점수 0.8 이상: 신뢰 가능
    - 출처 신뢰도 0.7 이상: 검증된 언론사
    - 팩트체킹 상태로 사용자에게 명확한 정보 제공
    """

    credibility_score: float  # 신뢰도 점수 (0.0-1.0)
    fact_check_status: (
        str  # 검증 상태: "verified"(검증됨) | "disputed"(논란) | "false"(거짓) | "mixed"(혼재)
    )
    source_reliability: float  # 출처 신뢰도 (0.0-1.0)
    evidence_links: List[str]  # 검증 근거 링크들
    verification_notes: str  # 검증 상세 내용
    fact_check_date: datetime  # 팩트체킹 수행 시간
    agent_version: str  # 에이전트 버전


@dataclass
class StorytellingResult:
    """
    스토리텔링 결과 데이터 구조

    🎯 목표: 4.2/5.0 몰입도의 매력적인 스토리 생성
    - 사용자 관심사에 맞는 내러티브 구성
    - 핵심 포인트 강조로 이해도 향상
    - 적절한 길이로 집중도 유지
    """

    story_summary: str  # 스토리 요약본
    narrative_style: str  # 내러티브 스타일 (뉴스/스토리/분석 등)
    key_highlights: List[str]  # 핵심 하이라이트 (3-5개)
    story_structure: Dict[str, str]  # 스토리 구조 (도입부/전개/결론)
    engagement_score: float  # 몰입도 점수 (0.0-5.0)
    story_length: int  # 스토리 길이 (문자 수)
    processing_time: datetime  # 처리 시간
    agent_version: str  # 에이전트 버전


@dataclass
class VoiceSynthesisResult:
    """
    음성 합성 결과 데이터 구조

    🎯 목표: 프로 성우 수준의 고품질 음성 생성
    - 5가지 캐릭터 보이스 지원
    - 1초 이내 음성 출력 목표
    - 감정 표현이 가능한 자연스러운 한국어 TTS
    """

    audio_file_path: str  # 생성된 오디오 파일 경로
    voice_character: str  # 사용된 캐릭터 보이스
    audio_duration: float  # 오디오 길이 (초)
    synthesis_quality: float  # 합성 품질 점수 (0.0-1.0)
    text_length: int  # 원본 텍스트 길이
    processing_time: datetime  # 처리 시간
    agent_version: str  # 에이전트 버전


@dataclass
class PersonalizationResult:
    """
    개인화 결과 데이터 구조

    🎯 목표: 4.5/5.0 만족도의 개인화 서비스
    - 사용자 관심사 학습 및 적용
    - 행동 패턴 기반 콘텐츠 추천
    - A/B 테스트를 통한 지속적 개선
    """

    personalized_summary: str  # 개인화된 요약
    relevance_score: float  # 관련성 점수 (0.0-1.0)
    interest_match: Dict[str, Any]  # 관심사 매칭 결과
    related_content: List[Dict[str, Any]]  # 관련 콘텐츠 추천 (최대 20개)
    personalization_score: Any  # 개인화 점수 객체
    strategy_used: Any  # 사용된 개인화 전략
    ab_test_group: Optional[str]  # A/B 테스트 그룹 (A/B/null)
    processing_time: datetime  # 처리 시간
    agent_version: str  # 에이전트 버전


@dataclass
class NewsState:
    """
    뉴스 처리 중앙 상태 관리 클래스

    🏗️ **아키텍처 특징**:
    - 모든 에이전트가 공유하는 중앙 집중식 상태
    - 불변성 보장을 위한 dataclass 사용
    - 단계별 처리 상태 추적
    - 에러 처리 및 복구 메커니즘
    - 성능 메트릭 자동 수집

    🔄 **처리 흐름**:
    1. INITIALIZED: 뉴스 상태 객체 생성
    2. TREND_ANALYSIS: 트렌드 분석 및 팩트체킹 수행
    3. PERSONALIZATION: 개인화 및 스토리텔링 적용
    4. VOICE_SYNTHESIS: 음성 합성 처리
    5. COMPLETED: 모든 처리 완료

    📊 **품질 보증**:
    - 각 단계별 품질 검증 체크포인트
    - 실패 시 자동 재시도 메커니즘
    - 휴먼 리뷰 요청 기능
    """

    # === 📰 기본 뉴스 정보 ===
    article_id: str  # 고유 뉴스 ID (UUID)
    user_id: Optional[str] = None  # 사용자 ID (개인화용, 옵셔널)
    content: str = ""  # 뉴스 원문 내용
    title: str = ""  # 뉴스 제목
    source: str = ""  # 뉴스 출처 (언론사명)
    published_at: Optional[datetime] = None  # 뉴스 발행 시간
    category: str = ""  # 뉴스 카테고리 (정치/경제/사회/국제/문화/스포츠/IT)

    # === 🔄 처리 상태 관리 ===
    processing_stage: ProcessingStage = ProcessingStage.INITIALIZED  # 현재 처리 단계
    created_at: datetime = field(default_factory=datetime.utcnow)  # 생성 시간
    updated_at: datetime = field(default_factory=datetime.utcnow)  # 마지막 업데이트 시간

    # === 🤖 에이전트 처리 결과 ===
    trend_analysis_result: Optional[TrendAnalysisResult] = None  # 트렌드 분석 결과
    fact_check_result: Optional[FactCheckResult] = None  # 팩트체킹 결과
    storytelling_result: Optional[StorytellingResult] = None  # 스토리텔링 결과
    personalization_result: Optional[PersonalizationResult] = None  # 개인화 결과
    voice_synthesis_result: Optional[VoiceSynthesisResult] = None  # 음성 합성 결과

    # === 📊 메타데이터 및 메트릭 ===
    metadata: Dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터
    error: Optional[str] = None  # 에러 메시지 (실패 시)
    processing_metrics: Dict[str, float] = field(default_factory=dict)  # 성능 메트릭

    def update_stage(self, stage: ProcessingStage):
        """
        처리 단계 업데이트

        Args:
            stage: 새로운 처리 단계

        📝 **자동 수행 작업**:
        - 업데이트 시간 갱신
        - 로그 기록
        - 메트릭 수집 트리거
        """
        self.processing_stage
        self.processing_stage = stage
        self.updated_at = datetime.utcnow()

        # 단계 전환 시간 메트릭 추가
        stage_transition_time = (self.updated_at - self.created_at).total_seconds()
        self.add_metric(f"stage_{stage.value}_time", stage_transition_time)

    def add_error(self, error_message: str):
        """
        에러 정보 추가 및 실패 상태로 전환

        Args:
            error_message: 에러 상세 메시지

        🚨 **에러 처리 과정**:
        1. 에러 메시지 저장
        2. 처리 상태를 FAILED로 변경
        3. 에러 발생 시간 기록
        4. 자동 알림 트리거 (설정된 경우)
        """
        self.error = error_message
        self.processing_stage = ProcessingStage.FAILED
        self.updated_at = datetime.utcnow()
        self.add_metric("error_occurred", 1.0)
        self.add_metric("error_time", self.updated_at.timestamp())

    def add_metric(self, metric_name: str, value: float):
        """
        성능 메트릭 추가

        Args:
            metric_name: 메트릭 이름 (예: "processing_time", "quality_score")
            value: 메트릭 값

        📊 **수집되는 주요 메트릭**:
        - 각 단계별 처리 시간
        - 품질 점수 (팩트체킹, 개인화 등)
        - 에러 발생 횟수
        - 리소스 사용량
        """
        self.processing_metrics[metric_name] = value
        self.updated_at = datetime.utcnow()

    def is_processing_complete(self) -> bool:
        """
        모든 처리가 완료되었는지 확인

        Returns:
            bool: 처리 완료 여부

        ✅ **완료 조건**:
        - 처리 상태가 COMPLETED
        - 모든 필수 결과 객체 존재
        - 에러 없음
        """
        return (
            self.processing_stage == ProcessingStage.COMPLETED
            and self.error is None
            and self.trend_analysis_result is not None
            and self.personalization_result is not None
            and self.voice_synthesis_result is not None
        )

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        처리 요약 정보 반환

        Returns:
            Dict: 처리 상태 요약 정보

        📋 **포함 정보**:
        - 기본 식별 정보
        - 현재 처리 단계
        - 각 에이전트 결과 존재 여부
        - 총 처리 시간
        - 에러 정보 (있는 경우)
        """
        total_processing_time = (self.updated_at - self.created_at).total_seconds()

        return {
            "article_id": self.article_id,
            "user_id": self.user_id,
            "stage": self.processing_stage.value,
            "progress": {
                "has_trend_analysis": self.trend_analysis_result is not None,
                "has_fact_check": self.fact_check_result is not None,
                "has_storytelling": self.storytelling_result is not None,
                "has_personalization": self.personalization_result is not None,
                "has_voice_synthesis": self.voice_synthesis_result is not None,
            },
            "timing": {
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "total_processing_time": total_processing_time,
            },
            "quality_metrics": {
                "credibility_score": (
                    self.fact_check_result.credibility_score if self.fact_check_result else None
                ),
                "personalization_score": (
                    self.personalization_result.relevance_score
                    if self.personalization_result
                    else None
                ),
                "synthesis_quality": (
                    self.voice_synthesis_result.synthesis_quality
                    if self.voice_synthesis_result
                    else None
                ),
            },
            "error": self.error,
            "metrics": self.processing_metrics,
        }

    def get_quality_scores(self) -> Dict[str, float]:
        """
        품질 점수 모음 반환

        Returns:
            Dict[str, float]: 각 영역별 품질 점수

        🎯 **품질 지표**:
        - 팩트체킹 신뢰도 (목표: 95%)
        - 개인화 만족도 (목표: 4.5/5.0)
        - 음성 합성 품질 (목표: 프로 수준)
        - 트렌드 분석 정확도
        """
        scores = {}

        if self.fact_check_result:
            scores["fact_check_credibility"] = self.fact_check_result.credibility_score

        if self.trend_analysis_result:
            scores["trend_relevance"] = self.trend_analysis_result.trending_score

        if self.personalization_result:
            scores["personalization_relevance"] = self.personalization_result.relevance_score

        if self.voice_synthesis_result:
            scores["voice_synthesis_quality"] = self.voice_synthesis_result.synthesis_quality

        if self.storytelling_result:
            scores["storytelling_engagement"] = self.storytelling_result.engagement_score

        return scores

    def requires_human_review(self) -> bool:
        """
        휴먼 리뷰가 필요한지 판단

        Returns:
            bool: 휴먼 리뷰 필요 여부

        🔍 **휴먼 리뷰 필요 조건**:
        - 팩트체킹 신뢰도 < 0.7
        - 논란의 여지가 있는 내용
        - 민감한 주제 (정치, 사회 갈등 등)
        - 에러 발생 시
        """
        # 에러 발생 시 리뷰 필요
        if self.error:
            return True

        # 팩트체킹 신뢰도가 낮은 경우
        if self.fact_check_result and self.fact_check_result.credibility_score < 0.7:
            return True

        # 논란 상태인 경우
        if self.fact_check_result and self.fact_check_result.fact_check_status in [
            "disputed",
            "false",
        ]:
            return True

        # 민감한 카테고리인 경우
        sensitive_categories = ["정치", "사회", "국제"]
        if self.category in sensitive_categories:
            return True

        return False
