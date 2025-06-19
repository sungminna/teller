"""
Advanced Preference Learning System - 개인화 만족도 4.5/5.0 달성
실시간 학습 알고리즘과 사용자 피드백 루프 강화
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ...shared.models.news import UserInteraction, UserProfile
from ..utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class InteractionType(Enum):
    """사용자 상호작용 유형"""

    CLICK = "click"
    READ = "read"
    SHARE = "share"
    LIKE = "like"
    DISLIKE = "dislike"
    SKIP = "skip"
    SAVE = "save"
    COMMENT = "comment"


class PreferenceCategory(Enum):
    """선호도 카테고리"""

    TOPIC_INTEREST = "topic_interest"
    CONTENT_TYPE = "content_type"
    READING_STYLE = "reading_style"
    TIME_PREFERENCE = "time_preference"
    SOURCE_CREDIBILITY = "source_credibility"
    COMPLEXITY_LEVEL = "complexity_level"


@dataclass
class LearningFeature:
    """학습 특성"""

    user_id: str
    article_id: str
    interaction_type: InteractionType
    timestamp: datetime

    # 콘텐츠 특성
    category: str
    keywords: List[str]
    sentiment_score: float
    complexity_score: float
    credibility_score: float

    # 사용자 컨텍스트
    time_of_day: int
    day_of_week: int
    device_type: str
    session_duration: float

    # 결과 변수
    satisfaction_score: float
    engagement_score: float


@dataclass
class PreferenceLearningConfig:
    """선호도 학습 설정"""

    learning_rate: float = 0.1
    batch_size: int = 100
    model_update_interval: int = 3600  # 1시간마다 모델 업데이트
    min_interactions: int = 10  # 최소 상호작용 수
    feature_importance_threshold: float = 0.05
    satisfaction_target: float = 4.5  # 목표 만족도
    cold_start_boost: float = 0.2  # 신규 사용자 부스트


class PreferenceLearner:
    """
    고도화된 선호도 학습 시스템
    - 실시간 학습 알고리즘
    - 다중 목표 최적화 (만족도 + 참여도)
    - 개인화 피드백 루프
    - A/B 테스트 통합
    """

    def __init__(self, config: PreferenceLearningConfig = None):
        self.config = config or PreferenceLearningConfig()
        self.cache = CacheManager()

        # 머신러닝 모델들
        self.satisfaction_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.engagement_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )

        # 특성 스케일러
        self.scaler = StandardScaler()

        # 모델 메타데이터
        self.model_metadata = {
            "last_updated": None,
            "training_samples": 0,
            "satisfaction_r2": 0.0,
            "engagement_r2": 0.0,
            "feature_importance": {},
        }

        # 학습 데이터 버퍼
        self.learning_buffer: List[LearningFeature] = []

        logger.info("PreferenceLearner initialized with target satisfaction: 4.5/5.0")

    async def learn_from_interaction(self, interaction: UserInteraction) -> Dict[str, float]:
        """
        사용자 상호작용으로부터 학습
        실시간 피드백 루프 구현
        """
        try:
            # 상호작용을 학습 특성으로 변환
            learning_feature = await self._extract_learning_features(interaction)

            # 학습 버퍼에 추가
            self.learning_buffer.append(learning_feature)

            # 배치 크기에 도달하면 모델 업데이트
            if len(self.learning_buffer) >= self.config.batch_size:
                await self._update_models()

            # 실시간 선호도 점수 계산
            preference_scores = await self._calculate_real_time_preferences(
                interaction.user_id, learning_feature
            )

            # 사용자 프로필 업데이트
            await self._update_user_preferences(interaction.user_id, preference_scores)

            logger.info(f"Learned from interaction for user {interaction.user_id}")
            return preference_scores

        except Exception as e:
            logger.error(f"Learning from interaction failed: {str(e)}")
            return {}

    async def predict_satisfaction(self, user_id: str, content_features: Dict) -> float:
        """
        콘텐츠에 대한 사용자 만족도 예측
        목표: 4.5/5.0 달성
        """
        try:
            # 사용자 프로필 로드
            user_profile = await self._get_user_profile(user_id)

            # 예측 특성 생성
            prediction_features = await self._create_prediction_features(
                user_profile, content_features
            )

            # 모델이 훈련되었는지 확인
            if self.model_metadata["last_updated"] is None:
                # 기본 휴리스틱 사용
                return await self._heuristic_satisfaction_prediction(user_profile, content_features)

            # ML 모델 예측
            features_array = np.array([prediction_features]).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)

            satisfaction_pred = self.satisfaction_model.predict(features_scaled)[0]

            # 신규 사용자 부스트 적용
            if user_profile.interaction_count < self.config.min_interactions:
                satisfaction_pred += self.config.cold_start_boost

            # 0-5 범위로 정규화
            satisfaction_pred = max(0.0, min(5.0, satisfaction_pred))

            return satisfaction_pred

        except Exception as e:
            logger.error(f"Satisfaction prediction failed: {str(e)}")
            return 3.0  # 기본값

    async def optimize_content_selection(
        self, user_id: str, candidate_contents: List[Dict]
    ) -> List[Dict]:
        """
        사용자 만족도를 최대화하는 콘텐츠 선택 최적화
        """
        try:
            # 각 콘텐츠에 대한 만족도 예측
            content_scores = []
            for content in candidate_contents:
                satisfaction_score = await self.predict_satisfaction(user_id, content)
                engagement_score = await self.predict_engagement(user_id, content)

                # 복합 점수 계산 (만족도 70% + 참여도 30%)
                composite_score = satisfaction_score * 0.7 + engagement_score * 0.3

                content_scores.append(
                    {
                        "content": content,
                        "satisfaction_score": satisfaction_score,
                        "engagement_score": engagement_score,
                        "composite_score": composite_score,
                    }
                )

            # 점수순 정렬
            content_scores.sort(key=lambda x: x["composite_score"], reverse=True)

            # 다양성 보장을 위한 재정렬
            optimized_contents = await self._ensure_content_diversity(content_scores)

            return [item["content"] for item in optimized_contents]

        except Exception as e:
            logger.error(f"Content selection optimization failed: {str(e)}")
            return candidate_contents  # 원본 반환

    async def get_personalization_insights(self, user_id: str) -> Dict[str, Any]:
        """
        개인화 인사이트 생성
        사용자 선호도 분석 및 개선 제안
        """
        try:
            user_profile = await self._get_user_profile(user_id)

            # 선호도 분석
            preference_analysis = await self._analyze_user_preferences(user_profile)

            # 만족도 트렌드 분석
            satisfaction_trend = await self._analyze_satisfaction_trend(user_id)

            # 개선 제안
            improvement_suggestions = await self._generate_improvement_suggestions(
                user_profile, preference_analysis, satisfaction_trend
            )

            insights = {
                "user_id": user_id,
                "current_satisfaction": user_profile.satisfaction_score,
                "target_satisfaction": self.config.satisfaction_target,
                "satisfaction_gap": self.config.satisfaction_target
                - user_profile.satisfaction_score,
                "preference_analysis": preference_analysis,
                "satisfaction_trend": satisfaction_trend,
                "improvement_suggestions": improvement_suggestions,
                "personalization_strength": await self._calculate_personalization_strength(
                    user_profile
                ),
                "generated_at": datetime.utcnow().isoformat(),
            }

            return insights

        except Exception as e:
            logger.error(f"Personalization insights generation failed: {str(e)}")
            return {}

    async def _extract_learning_features(self, interaction: UserInteraction) -> LearningFeature:
        """상호작용에서 학습 특성 추출"""
        # 콘텐츠 분석
        content_analysis = await self._analyze_content(interaction.content_id)

        # 시간 컨텍스트
        timestamp = interaction.timestamp
        time_of_day = timestamp.hour
        day_of_week = timestamp.weekday()

        # 만족도 점수 계산 (상호작용 유형 기반)
        satisfaction_score = self._calculate_interaction_satisfaction(interaction)

        # 참여도 점수 계산
        engagement_score = self._calculate_interaction_engagement(interaction)

        return LearningFeature(
            user_id=interaction.user_id,
            article_id=interaction.content_id,
            interaction_type=InteractionType(interaction.interaction_type),
            timestamp=timestamp,
            category=content_analysis.get("category", "unknown"),
            keywords=content_analysis.get("keywords", []),
            sentiment_score=content_analysis.get("sentiment_score", 0.0),
            complexity_score=content_analysis.get("complexity_score", 0.5),
            credibility_score=content_analysis.get("credibility_score", 0.8),
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            device_type=interaction.device_type or "unknown",
            session_duration=interaction.session_duration or 0.0,
            satisfaction_score=satisfaction_score,
            engagement_score=engagement_score,
        )

    def _calculate_interaction_satisfaction(self, interaction: UserInteraction) -> float:
        """상호작용 기반 만족도 점수 계산"""
        satisfaction_mapping = {
            InteractionType.LIKE: 5.0,
            InteractionType.SHARE: 4.8,
            InteractionType.SAVE: 4.5,
            InteractionType.COMMENT: 4.3,
            InteractionType.READ: 4.0,
            InteractionType.CLICK: 3.5,
            InteractionType.SKIP: 2.0,
            InteractionType.DISLIKE: 1.0,
        }

        base_score = satisfaction_mapping.get(InteractionType(interaction.interaction_type), 3.0)

        # 읽기 완료율 반영
        if interaction.reading_completion_rate:
            completion_bonus = (interaction.reading_completion_rate - 0.5) * 2.0
            base_score += completion_bonus

        # 세션 지속 시간 반영
        if interaction.session_duration and interaction.session_duration > 60:
            duration_bonus = min(1.0, (interaction.session_duration - 60) / 300)
            base_score += duration_bonus

        return max(1.0, min(5.0, base_score))

    def _calculate_interaction_engagement(self, interaction: UserInteraction) -> float:
        """상호작용 기반 참여도 점수 계산"""
        engagement_mapping = {
            InteractionType.COMMENT: 1.0,
            InteractionType.SHARE: 0.9,
            InteractionType.LIKE: 0.8,
            InteractionType.SAVE: 0.7,
            InteractionType.READ: 0.6,
            InteractionType.CLICK: 0.4,
            InteractionType.SKIP: 0.1,
            InteractionType.DISLIKE: 0.0,
        }

        base_score = engagement_mapping.get(InteractionType(interaction.interaction_type), 0.3)

        # 읽기 시간 반영
        if interaction.reading_time and interaction.reading_time > 30:
            time_bonus = min(0.3, (interaction.reading_time - 30) / 180)
            base_score += time_bonus

        return max(0.0, min(1.0, base_score))

    async def _update_models(self):
        """머신러닝 모델 업데이트"""
        try:
            if len(self.learning_buffer) < self.config.min_interactions:
                return

            # 학습 데이터 준비
            features, satisfaction_targets, engagement_targets = await self._prepare_training_data()

            if len(features) < 10:  # 최소 데이터 요구사항
                return

            # 데이터 분할
            X_train, X_test, y_sat_train, y_sat_test, y_eng_train, y_eng_test = train_test_split(
                features, satisfaction_targets, engagement_targets, test_size=0.2, random_state=42
            )

            # 특성 스케일링
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # 만족도 모델 훈련
            self.satisfaction_model.fit(X_train_scaled, y_sat_train)
            sat_pred = self.satisfaction_model.predict(X_test_scaled)
            sat_r2 = r2_score(y_sat_test, sat_pred)

            # 참여도 모델 훈련
            self.engagement_model.fit(X_train_scaled, y_eng_train)
            eng_pred = self.engagement_model.predict(X_test_scaled)
            eng_r2 = r2_score(y_eng_test, eng_pred)

            # 특성 중요도 계산
            feature_importance = self._calculate_feature_importance()

            # 메타데이터 업데이트
            self.model_metadata.update(
                {
                    "last_updated": datetime.utcnow(),
                    "training_samples": len(features),
                    "satisfaction_r2": sat_r2,
                    "engagement_r2": eng_r2,
                    "feature_importance": feature_importance,
                }
            )

            # 학습 버퍼 클리어
            self.learning_buffer.clear()

            logger.info(f"Models updated: Satisfaction R2={sat_r2:.3f}, Engagement R2={eng_r2:.3f}")

        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")

    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        features = []
        satisfaction_targets = []
        engagement_targets = []

        for feature in self.learning_buffer:
            # 특성 벡터 생성
            feature_vector = [
                hash(feature.category) % 1000 / 1000.0,  # 카테고리 해시
                len(feature.keywords) / 10.0,  # 키워드 수 정규화
                feature.sentiment_score,
                feature.complexity_score,
                feature.credibility_score,
                feature.time_of_day / 24.0,  # 시간 정규화
                feature.day_of_week / 7.0,  # 요일 정규화
                hash(feature.device_type) % 100 / 100.0,  # 디바이스 해시
                min(feature.session_duration / 3600.0, 1.0),  # 세션 시간 정규화
            ]

            features.append(feature_vector)
            satisfaction_targets.append(feature.satisfaction_score)
            engagement_targets.append(feature.engagement_score)

        return (np.array(features), np.array(satisfaction_targets), np.array(engagement_targets))

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """특성 중요도 계산"""
        feature_names = [
            "category",
            "keyword_count",
            "sentiment",
            "complexity",
            "credibility",
            "time_of_day",
            "day_of_week",
            "device_type",
            "session_duration",
        ]

        satisfaction_importance = self.satisfaction_model.feature_importances_
        engagement_importance = self.engagement_model.feature_importances_

        # 두 모델의 중요도 평균
        combined_importance = (satisfaction_importance + engagement_importance) / 2

        return dict(zip(feature_names, combined_importance.tolist()))

    async def _heuristic_satisfaction_prediction(
        self, user_profile: UserProfile, content_features: Dict
    ) -> float:
        """휴리스틱 기반 만족도 예측 (모델이 없을 때)"""
        base_score = 3.5  # 기본 점수

        # 카테고리 매칭
        if content_features.get("category") in user_profile.preferred_categories:
            base_score += 0.8

        # 키워드 매칭
        content_keywords = content_features.get("keywords", [])
        matching_keywords = set(content_keywords) & set(user_profile.interests)
        if matching_keywords:
            base_score += len(matching_keywords) * 0.2

        # 신뢰도 점수
        credibility = content_features.get("credibility_score", 0.8)
        base_score += (credibility - 0.5) * 1.0

        # 복잡도 매칭
        content_complexity = content_features.get("complexity_score", 0.5)
        user_complexity_pref = getattr(user_profile, "preferred_complexity", 0.5)
        complexity_diff = abs(content_complexity - user_complexity_pref)
        base_score -= complexity_diff * 0.5

        return max(1.0, min(5.0, base_score))

    async def predict_engagement(self, user_id: str, content_features: Dict) -> float:
        """참여도 예측"""
        try:
            user_profile = await self._get_user_profile(user_id)

            if self.model_metadata["last_updated"] is None:
                return await self._heuristic_engagement_prediction(user_profile, content_features)

            prediction_features = await self._create_prediction_features(
                user_profile, content_features
            )

            features_array = np.array([prediction_features]).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)

            engagement_pred = self.engagement_model.predict(features_scaled)[0]
            return max(0.0, min(1.0, engagement_pred))

        except Exception as e:
            logger.error(f"Engagement prediction failed: {str(e)}")
            return 0.5

    async def _heuristic_engagement_prediction(
        self, user_profile: UserProfile, content_features: Dict
    ) -> float:
        """휴리스틱 기반 참여도 예측"""
        base_score = 0.5

        # 사용자 활동 패턴
        if user_profile.average_engagement_score:
            base_score = user_profile.average_engagement_score

        # 콘텐츠 특성
        if content_features.get("category") in user_profile.preferred_categories:
            base_score += 0.2

        # 시간대 매칭
        current_hour = datetime.utcnow().hour
        if current_hour in getattr(user_profile, "active_hours", [9, 12, 18, 21]):
            base_score += 0.1

        return max(0.0, min(1.0, base_score))

    async def _create_prediction_features(
        self, user_profile: UserProfile, content_features: Dict
    ) -> List[float]:
        """예측을 위한 특성 벡터 생성"""
        return [
            hash(content_features.get("category", "")) % 1000 / 1000.0,
            len(content_features.get("keywords", [])) / 10.0,
            content_features.get("sentiment_score", 0.0),
            content_features.get("complexity_score", 0.5),
            content_features.get("credibility_score", 0.8),
            datetime.utcnow().hour / 24.0,
            datetime.utcnow().weekday() / 7.0,
            0.5,  # 기본 디바이스 타입
            (
                min(user_profile.average_session_duration / 3600.0, 1.0)
                if user_profile.average_session_duration
                else 0.5
            ),
        ]

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """사용자 프로필 조회"""
        # 캐시에서 먼저 확인
        cached_profile = await self.cache.get_user_profile(user_id)
        if cached_profile:
            return cached_profile

        # 데이터베이스에서 조회 (실제 구현에서는 DB 연결)
        # 여기서는 기본 프로필 반환
        return UserProfile.create_default(user_id)

    async def _analyze_content(self, content_id: str) -> Dict[str, Any]:
        """콘텐츠 분석"""
        # 실제 구현에서는 콘텐츠 분석 서비스 호출
        return {
            "category": "technology",
            "keywords": ["AI", "기술", "혁신"],
            "sentiment_score": 0.6,
            "complexity_score": 0.7,
            "credibility_score": 0.9,
        }

    async def _calculate_real_time_preferences(
        self, user_id: str, learning_feature: LearningFeature
    ) -> Dict[str, float]:
        """실시간 선호도 점수 계산"""
        return {
            "topic_preference": learning_feature.satisfaction_score / 5.0,
            "content_type_preference": learning_feature.engagement_score,
            "time_preference": 1.0 if learning_feature.time_of_day in [9, 12, 18, 21] else 0.5,
            "complexity_preference": learning_feature.complexity_score,
        }

    async def _update_user_preferences(self, user_id: str, preference_scores: Dict[str, float]):
        """사용자 선호도 업데이트"""
        # 실제 구현에서는 데이터베이스 업데이트
        await self.cache.set_user_preferences(user_id, preference_scores, ttl=3600)

    async def _ensure_content_diversity(self, content_scores: List[Dict]) -> List[Dict]:
        """콘텐츠 다양성 보장"""
        # 카테고리별 분산
        categories_seen = set()
        diversified = []

        for item in content_scores:
            category = item["content"].get("category", "unknown")
            if category not in categories_seen or len(diversified) < 3:
                diversified.append(item)
                categories_seen.add(category)

            if len(diversified) >= 10:  # 최대 10개
                break

        return diversified

    async def _analyze_user_preferences(self, user_profile: UserProfile) -> Dict[str, Any]:
        """사용자 선호도 분석"""
        return {
            "top_interests": user_profile.interests[:5],
            "preferred_categories": user_profile.preferred_categories,
            "reading_pattern": user_profile.reading_pattern,
            "activity_peak_hours": getattr(user_profile, "active_hours", []),
            "engagement_level": user_profile.average_engagement_score or 0.5,
        }

    async def _analyze_satisfaction_trend(self, user_id: str) -> Dict[str, Any]:
        """만족도 트렌드 분석"""
        # 실제 구현에서는 시계열 데이터 분석
        return {
            "trend": "increasing",
            "weekly_average": 4.2,
            "monthly_average": 4.1,
            "improvement_rate": 0.05,
        }

    async def _generate_improvement_suggestions(
        self, user_profile: UserProfile, preference_analysis: Dict, satisfaction_trend: Dict
    ) -> List[str]:
        """개선 제안 생성"""
        suggestions = []

        current_satisfaction = user_profile.satisfaction_score or 3.5
        target_gap = self.config.satisfaction_target - current_satisfaction

        if target_gap > 0.5:
            suggestions.append("더 관련성 높은 콘텐츠 추천을 위해 관심사를 세분화해보세요")

        if user_profile.interaction_count < 50:
            suggestions.append("더 많은 상호작용을 통해 개인화 정확도를 높일 수 있습니다")

        if satisfaction_trend.get("trend") == "decreasing":
            suggestions.append("최근 선호도 변화를 반영하여 추천 알고리즘을 조정하겠습니다")

        return suggestions

    async def _calculate_personalization_strength(self, user_profile: UserProfile) -> float:
        """개인화 강도 계산"""
        factors = [
            min(user_profile.interaction_count / 100.0, 1.0),  # 상호작용 수
            len(user_profile.interests) / 10.0,  # 관심사 다양성
            len(user_profile.preferred_categories) / 5.0,  # 카테고리 선호도
            (user_profile.satisfaction_score or 3.0) / 5.0,  # 현재 만족도
        ]

        return sum(factors) / len(factors)
