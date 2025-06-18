"""
🎯 Personalization Agent - 사용자 개인화, 추천, 스토리텔링 통합 전문 에이전트 (Stage 3)
- 사용자 프로필 기반 콘텐츠 필터링
- 개인화 만족도 4.5/5.0 달성 목표  
- 맞춤형 요약 및 추천 시스템
- A/B 테스트 기반 지속적 개선
- 4.2/5.0 몰입도 목표의 스토리텔링 시스템
"""
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langfuse import Langfuse

from ..state.news_state import NewsState, PersonalizationResult
from ..tools.user_profiler import UserProfiler
from ..tools.content_recommender import ContentRecommender
from ..tools.preference_learner import PreferenceLearner
from ..utils.cache_manager import CacheManager
from ...shared.models.news import NewsArticle, UserProfile, PersonalizationScore
from ...shared.config.settings import settings

logger = logging.getLogger(__name__)

class PersonalizationStrategy(Enum):
    """개인화 전략"""
    INTEREST_BASED = "interest_based"
    BEHAVIOR_BASED = "behavior_based"
    HYBRID = "hybrid"
    COLLABORATIVE = "collaborative"

@dataclass
class PersonalizationConfig:
    """개인화 설정"""
    target_satisfaction: float = 4.5  # 목표 만족도 4.5/5.0
    min_personalization_score: float = 0.8
    max_recommendations: int = 20
    learning_rate: float = 0.1
    cold_start_threshold: int = 5  # 신규 사용자 임계값
    enable_ab_testing: bool = True
    strategy: PersonalizationStrategy = PersonalizationStrategy.HYBRID

class PersonalizationAgent:
    """
    개인화 전문 에이전트
    - 사용자 프로필 기반 콘텐츠 필터링
    - 개인화 만족도 4.5/5.0 달성 목표
    - 맞춤형 요약 및 추천 시스템
    - A/B 테스트 기반 지속적 개선
    """
    
    def __init__(self, config: PersonalizationConfig = None):
        self.config = config or PersonalizationConfig()
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.3,  # 개인화를 위해 약간의 창의성 허용
            max_tokens=2000,
            api_key=settings.OPENAI_API_KEY
        )
        
        # 전문 도구들 초기화
        self.user_profiler = UserProfiler()
        self.content_recommender = ContentRecommender()
        self.preference_learner = PreferenceLearner(
            learning_rate=self.config.learning_rate
        )
        
        # 캐시 및 추적 시스템
        self.cache = CacheManager()
        self.langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST
        )
        
        logger.info(f"Personalization Agent initialized with target satisfaction: {self.config.target_satisfaction}")
    
    async def personalize_content(self, state: NewsState) -> NewsState:
        """
        콘텐츠 개인화 메인 프로세스
        4.5/5.0 만족도 목표로 개인화 수행
        """
        try:
            trace = self.langfuse.trace(
                name="content_personalization",
                input={
                    "user_id": state.user_id,
                    "article_id": state.article_id,
                    "content_length": len(state.content)
                }
            )
            
            logger.info(f"Starting personalization for user {state.user_id}, article {state.article_id}")
            
            # 1. 사용자 프로필 로드 및 업데이트
            user_profile = await self._get_or_create_user_profile(state.user_id, trace)
            
            # 2. 개인화 전략 결정
            strategy = await self._determine_personalization_strategy(user_profile, trace)
            
            # 3. 병렬 개인화 분석
            tasks = [
                self._analyze_user_interests(state.content, user_profile, trace),
                self._generate_personalized_summary(state.content, user_profile, trace),
                self._calculate_relevance_score(state.content, user_profile, trace),
                self._recommend_related_content(state.content, user_profile, trace)
            ]
            
            interest_analysis, personalized_summary, relevance_score, related_content = await asyncio.gather(*tasks)
            
            # 4. A/B 테스트 그룹 결정 (활성화된 경우)
            ab_group = None
            if self.config.enable_ab_testing:
                ab_group = await self._assign_ab_test_group(state.user_id, trace)
            
            # 5. 개인화 점수 계산
            personalization_score = await self._calculate_personalization_score(
                interest_analysis, relevance_score, user_profile
            )
            
            # 6. 결과 취합
            personalization_result = PersonalizationResult(
                personalized_summary=personalized_summary,
                relevance_score=relevance_score,
                interest_match=interest_analysis,
                related_content=related_content,
                personalization_score=personalization_score,
                strategy_used=strategy,
                ab_test_group=ab_group,
                processing_time=datetime.utcnow(),
                agent_version="personalization_v1.0"
            )
            
            # 7. 품질 검증
            if personalization_score.overall_score < self.config.min_personalization_score:
                logger.warning(f"Low personalization score {personalization_score.overall_score} for user {state.user_id}")
                # 개인화 전략 조정
                personalization_result = await self._adjust_personalization_strategy(
                    personalization_result, user_profile, trace
                )
            
            state.personalization_result = personalization_result
            state.processing_stage = "personalization_complete"
            
            # 8. 사용자 프로필 업데이트 (학습)
            await self._update_user_profile(user_profile, state, personalization_result)
            
            # Langfuse 추적
            trace.update(
                output={
                    "personalization_score": personalization_score.overall_score,
                    "strategy": strategy.value,
                    "ab_group": ab_group
                }
            )
            
            logger.info(f"Personalization completed for user {state.user_id} with score {personalization_score.overall_score:.2f}")
            return state
            
        except Exception as e:
            logger.error(f"Personalization failed for user {state.user_id}: {str(e)}")
            state.error = f"Personalization error: {str(e)}"
            state.processing_stage = "personalization_failed"
            return state
    
    async def _get_or_create_user_profile(self, user_id: str, trace) -> UserProfile:
        """사용자 프로필 로드 또는 생성"""
        span = trace.span(name="user_profile_management")
        
        try:
            # 캐시에서 프로필 확인
            cached_profile = await self.cache.get_user_profile(user_id)
            if cached_profile:
                return cached_profile
            
            # 데이터베이스에서 프로필 로드
            profile = await self.user_profiler.get_user_profile(user_id)
            
            if not profile:
                # 신규 사용자 프로필 생성
                profile = await self.user_profiler.create_user_profile(user_id)
                logger.info(f"Created new user profile for {user_id}")
            
            # 캐시에 저장
            await self.cache.set_user_profile(user_id, profile, ttl=3600)
            
            span.update(output={"profile_loaded": True, "is_new_user": profile.interaction_count < self.config.cold_start_threshold})
            return profile
            
        except Exception as e:
            logger.error(f"Failed to load user profile for {user_id}: {str(e)}")
            # 기본 프로필 반환
            return UserProfile.create_default(user_id)
    
    async def _determine_personalization_strategy(self, user_profile: UserProfile, trace) -> PersonalizationStrategy:
        """개인화 전략 결정"""
        span = trace.span(name="strategy_determination")
        
        # 신규 사용자 처리
        if user_profile.interaction_count < self.config.cold_start_threshold:
            strategy = PersonalizationStrategy.INTEREST_BASED
            span.update(output={"strategy": strategy.value, "reason": "cold_start"})
            return strategy
        
        # 기존 사용자 - 하이브리드 전략
        if user_profile.satisfaction_score >= 4.0:
            strategy = PersonalizationStrategy.HYBRID
        elif user_profile.click_through_rate > 0.1:
            strategy = PersonalizationStrategy.BEHAVIOR_BASED
        else:
            strategy = PersonalizationStrategy.COLLABORATIVE
        
        span.update(output={
            "strategy": strategy.value,
            "satisfaction_score": user_profile.satisfaction_score,
            "ctr": user_profile.click_through_rate
        })
        return strategy
    
    async def _analyze_user_interests(self, content: str, user_profile: UserProfile, trace) -> Dict:
        """사용자 관심사 분석"""
        span = trace.span(name="interest_analysis")
        
        system_prompt = f"""
        당신은 개인화 전문가입니다. 사용자의 관심사와 뉴스 콘텐츠의 매칭도를 분석해주세요.
        
        사용자 프로필:
        - 주요 관심사: {', '.join(user_profile.interests)}
        - 선호 카테고리: {', '.join(user_profile.preferred_categories)}
        - 읽기 패턴: {user_profile.reading_pattern}
        - 만족도 점수: {user_profile.satisfaction_score}/5.0
        
        다음 기준으로 분석해주세요:
        1. 관심사 일치도 (0-100)
        2. 카테고리 적합성 (0-100)
        3. 콘텐츠 복잡도 적합성 (0-100)
        4. 개인화 추천 사유
        """
        
        human_prompt = f"""
        뉴스 콘텐츠:
        {content[:1500]}...
        
        위 콘텐츠를 분석하여 다음 JSON 형식으로 응답해주세요:
        {{
            "interest_match": 85,
            "category_fit": 90,
            "complexity_fit": 75,
            "overall_relevance": 83,
            "personalization_reason": "사용자의 기술 관심사와 높은 일치도",
            "engagement_prediction": 0.85,
            "reading_time_estimate": 180
        }}
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            result = json.loads(response.content)
            span.update(output=result)
            return result
            
        except Exception as e:
            logger.error(f"Interest analysis failed: {str(e)}")
            return {
                "interest_match": 50,
                "category_fit": 50,
                "complexity_fit": 50,
                "overall_relevance": 50,
                "personalization_reason": "기본 분석",
                "engagement_prediction": 0.5,
                "reading_time_estimate": 120
            }
    
    async def _generate_personalized_summary(self, content: str, user_profile: UserProfile, trace) -> str:
        """개인화된 요약 생성"""
        span = trace.span(name="personalized_summary")
        
        # 사용자 선호도에 따른 요약 스타일 결정
        summary_style = self._determine_summary_style(user_profile)
        
        system_prompt = f"""
        사용자 맞춤형 뉴스 요약을 생성해주세요.
        
        사용자 특성:
        - 선호 길이: {user_profile.preferred_summary_length}
        - 관심 분야: {', '.join(user_profile.interests)}
        - 읽기 수준: {user_profile.reading_level}
        - 요약 스타일: {summary_style}
        
        요약 지침:
        1. 사용자 관심사에 중점을 둔 요약
        2. 적절한 길이와 복잡도
        3. 개인화된 관점 제공
        4. 행동 유도 요소 포함
        """
        
        human_prompt = f"""
        다음 뉴스를 사용자 맞춤형으로 요약해주세요:
        
        {content}
        
        개인화된 요약 (한국어, {user_profile.preferred_summary_length}자 내외):
        """
        
        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
            
            personalized_summary = response.content.strip()
            
            span.update(output={
                "summary_length": len(personalized_summary),
                "style": summary_style
            })
            
            return personalized_summary
            
        except Exception as e:
            logger.error(f"Personalized summary generation failed: {str(e)}")
            return content[:200] + "..."  # 기본 요약
    
    async def _calculate_relevance_score(self, content: str, user_profile: UserProfile, trace) -> float:
        """콘텐츠 관련성 점수 계산"""
        span = trace.span(name="relevance_calculation")
        
        try:
            # 키워드 매칭 점수
            keyword_score = await self._calculate_keyword_match_score(content, user_profile)
            
            # 카테고리 매칭 점수
            category_score = await self._calculate_category_match_score(content, user_profile)
            
            # 시간적 관련성 점수
            temporal_score = await self._calculate_temporal_relevance_score(content, user_profile)
            
            # 사용자 행동 기반 점수
            behavior_score = await self._calculate_behavior_based_score(content, user_profile)
            
            # 가중 평균 계산
            weights = {
                'keyword': 0.3,
                'category': 0.25,
                'temporal': 0.2,
                'behavior': 0.25
            }
            
            relevance_score = (
                keyword_score * weights['keyword'] +
                category_score * weights['category'] +
                temporal_score * weights['temporal'] +
                behavior_score * weights['behavior']
            )
            
            span.update(output={
                "keyword_score": keyword_score,
                "category_score": category_score,
                "temporal_score": temporal_score,
                "behavior_score": behavior_score,
                "final_score": relevance_score
            })
            
            return min(1.0, max(0.0, relevance_score))
            
        except Exception as e:
            logger.error(f"Relevance score calculation failed: {str(e)}")
            return 0.5  # 기본 점수
    
    async def _recommend_related_content(self, content: str, user_profile: UserProfile, trace) -> List[Dict]:
        """관련 콘텐츠 추천"""
        span = trace.span(name="content_recommendation")
        
        try:
            # 콘텐츠 임베딩 생성
            content_embedding = await self.content_recommender.generate_embedding(content)
            
            # 사용자 선호도 기반 유사 콘텐츠 검색
            similar_content = await self.content_recommender.find_similar_content(
                content_embedding,
                user_profile,
                limit=self.config.max_recommendations
            )
            
            # 추천 점수 계산 및 정렬
            recommendations = []
            for item in similar_content:
                recommendation_score = await self._calculate_recommendation_score(
                    item, user_profile
                )
                
                recommendations.append({
                    "id": item.id,
                    "title": item.title,
                    "summary": item.summary,
                    "category": item.category,
                    "score": recommendation_score,
                    "reason": await self._generate_recommendation_reason(item, user_profile)
                })
            
            # 점수순 정렬
            recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            span.update(output={
                "recommendations_count": len(recommendations),
                "avg_score": np.mean([r['score'] for r in recommendations]) if recommendations else 0
            })
            
            return recommendations[:10]  # 상위 10개 반환
            
        except Exception as e:
            logger.error(f"Content recommendation failed: {str(e)}")
            return []
    
    async def _calculate_personalization_score(self, interest_analysis: Dict, relevance_score: float, user_profile: UserProfile) -> PersonalizationScore:
        """개선된 개인화 점수 계산 - 콜드 스타트 문제 해결 및 적응적 가중치"""
        try:
            import numpy as np
            
            # 1. 정규화된 점수 계산
            interest_score = np.clip(interest_analysis.get('overall_relevance', 50) / 100.0, 0, 1)
            engagement_prediction = np.clip(interest_analysis.get('engagement_prediction', 0.5), 0, 1)
            relevance_score = np.clip(relevance_score, 0, 1)
            
            # 2. 사용자 경험 수준에 따른 적응적 가중치
            interaction_count = user_profile.interaction_count
            if interaction_count < 5:  # 콜드 스타트 단계
                weights = [0.2, 0.5, 0.2, 0.1]  # 콘텐츠 품질 중심
                logger.info(f"Cold start user {user_profile.user_id}: using content-quality weights")
            elif interaction_count < 50:  # 학습 단계
                weights = [0.3, 0.4, 0.2, 0.1]
                logger.info(f"Learning user {user_profile.user_id}: using balanced weights")
            else:  # 성숙 단계
                weights = [0.4, 0.3, 0.2, 0.1]  # 개인화 중심
                logger.info(f"Mature user {user_profile.user_id}: using personalization weights")
            
            # 3. 시간 감쇠 적용
            time_decay = self._calculate_time_decay(user_profile.last_interaction)
            
            # 4. 최종 점수 계산
            overall_score = (
                interest_score * weights[0] +
                relevance_score * weights[1] +
                engagement_prediction * weights[2] +
                time_decay * weights[3]
            )
            
            # 5. 신뢰도 계산 (상호작용 수에 따라 증가)
            confidence = min(1.0, interaction_count / 100.0)
            
            logger.info(f"Personalization score calculated: {overall_score:.3f} (confidence: {confidence:.3f})")
            
            return PersonalizationScore(
                overall_score=overall_score,
                interest_match=interest_score,
                relevance_score=relevance_score,
                engagement_prediction=engagement_prediction,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Personalization score calculation failed: {str(e)}")
            # 안전한 기본값 반환
            return PersonalizationScore(
                overall_score=0.6,  # 중립적 점수
                interest_match=0.5,
                relevance_score=0.5,
                engagement_prediction=0.5,
                confidence=0.3  # 낮은 신뢰도
            )
    
    def _calculate_time_decay(self, last_interaction: Optional[datetime]) -> float:
        """시간 감쇠 계산 - 최근 활동일수록 높은 가중치"""
        try:
            if not last_interaction:
                return 0.5  # 기본값
            
            hours_since_last = (datetime.utcnow() - last_interaction).total_seconds() / 3600
            
            if hours_since_last < 1:
                return 1.0  # 1시간 이내
            elif hours_since_last < 24:
                return 0.8  # 24시간 이내
            elif hours_since_last < 168:  # 1주일
                return 0.6
            elif hours_since_last < 720:  # 1개월
                return 0.4
            else:
                return 0.2  # 1개월 이상
                
        except Exception:
            return 0.5
    
    def _determine_summary_style(self, user_profile: UserProfile) -> str:
        """사용자 프로필 기반 요약 스타일 결정"""
        if user_profile.reading_level == "advanced":
            return "detailed_analytical"
        elif user_profile.reading_level == "beginner":
            return "simple_conversational"
        else:
            return "balanced_informative"
    
    async def _calculate_keyword_match_score(self, content: str, user_profile: UserProfile) -> float:
        """키워드 매칭 점수 계산"""
        try:
            content_lower = content.lower()
            matches = 0
            total_keywords = len(user_profile.interests)
            
            if total_keywords == 0:
                return 0.5
            
            for interest in user_profile.interests:
                if interest.lower() in content_lower:
                    matches += 1
            
            return matches / total_keywords
            
        except Exception:
            return 0.5
    
    async def _calculate_category_match_score(self, content: str, user_profile: UserProfile) -> float:
        """카테고리 매칭 점수 계산"""
        # 실제 구현에서는 콘텐츠의 카테고리를 분석하여 매칭
        # 여기서는 간단한 키워드 기반 매칭으로 구현
        try:
            if not user_profile.preferred_categories:
                return 0.5
            
            content_lower = content.lower()
            category_keywords = {
                'technology': ['기술', '테크', '인공지능', 'ai', '소프트웨어'],
                'business': ['비즈니스', '경제', '기업', '투자', '시장'],
                'politics': ['정치', '정부', '선거', '정책', '국회'],
                'sports': ['스포츠', '축구', '야구', '올림픽', '경기'],
                'entertainment': ['연예', '영화', '음악', '드라마', '예술']
            }
            
            matches = 0
            for category in user_profile.preferred_categories:
                if category in category_keywords:
                    keywords = category_keywords[category]
                    if any(keyword in content_lower for keyword in keywords):
                        matches += 1
            
            return matches / len(user_profile.preferred_categories)
            
        except Exception:
            return 0.5
    
    async def _calculate_temporal_relevance_score(self, content: str, user_profile: UserProfile) -> float:
        """시간적 관련성 점수 계산"""
        try:
            # 사용자의 활동 시간대와 뉴스 발행 시간 고려
            current_hour = datetime.utcnow().hour
            user_active_hours = user_profile.active_hours or [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            
            if current_hour in user_active_hours:
                return 1.0
            else:
                # 활동 시간대와의 거리에 따라 점수 감소
                min_distance = min(abs(current_hour - hour) for hour in user_active_hours)
                return max(0.1, 1.0 - (min_distance / 12.0))
                
        except Exception:
            return 0.7
    
    async def _calculate_behavior_based_score(self, content: str, user_profile: UserProfile) -> float:
        """사용자 행동 기반 점수 계산"""
        try:
            # 클릭률, 읽기 완료율, 공유율 등을 고려
            ctr = user_profile.click_through_rate or 0.1
            completion_rate = user_profile.reading_completion_rate or 0.7
            engagement_score = user_profile.average_engagement_score or 0.6
            
            # 가중 평균
            behavior_score = (ctr * 0.4 + completion_rate * 0.4 + engagement_score * 0.2)
            return min(1.0, max(0.1, behavior_score))
            
        except Exception:
            return 0.5
    
    async def _calculate_recommendation_score(self, item, user_profile: UserProfile) -> float:
        """추천 점수 계산"""
        try:
            # 유사도, 사용자 선호도, 신선도 등을 종합
            similarity_score = getattr(item, 'similarity_score', 0.5)
            freshness_score = self._calculate_freshness_score(item.published_at)
            preference_score = await self._calculate_preference_score(item, user_profile)
            
            return (similarity_score * 0.4 + preference_score * 0.4 + freshness_score * 0.2)
            
        except Exception:
            return 0.5
    
    def _calculate_freshness_score(self, published_at) -> float:
        """신선도 점수 계산"""
        try:
            if not published_at:
                return 0.5
            
            hours_old = (datetime.utcnow() - published_at).total_seconds() / 3600
            
            if hours_old < 1:
                return 1.0
            elif hours_old < 6:
                return 0.8
            elif hours_old < 24:
                return 0.6
            elif hours_old < 72:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5
    
    async def _calculate_preference_score(self, item, user_profile: UserProfile) -> float:
        """선호도 점수 계산"""
        try:
            # 카테고리 선호도
            category_match = 1.0 if item.category in user_profile.preferred_categories else 0.3
            
            # 키워드 매칭
            keyword_matches = sum(1 for interest in user_profile.interests 
                                if interest.lower() in item.title.lower() or 
                                   interest.lower() in item.summary.lower())
            keyword_score = min(1.0, keyword_matches / max(1, len(user_profile.interests)))
            
            return (category_match * 0.6 + keyword_score * 0.4)
            
        except Exception:
            return 0.5
    
    async def _generate_recommendation_reason(self, item, user_profile: UserProfile) -> str:
        """추천 이유 생성"""
        try:
            reasons = []
            
            if item.category in user_profile.preferred_categories:
                reasons.append(f"선호 카테고리 '{item.category}' 매칭")
            
            matching_interests = [interest for interest in user_profile.interests 
                                if interest.lower() in item.title.lower()]
            if matching_interests:
                reasons.append(f"관심사 '{', '.join(matching_interests)}' 관련")
            
            if not reasons:
                reasons.append("사용자 패턴 기반 추천")
            
            return "; ".join(reasons)
            
        except Exception:
            return "개인화 추천"
    
    async def _assign_ab_test_group(self, user_id: str, trace) -> Optional[str]:
        """A/B 테스트 그룹 할당"""
        try:
            # 사용자 ID 기반 해시로 일관된 그룹 할당
            import hashlib
            hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            group = "A" if hash_value % 2 == 0 else "B"
            
            trace.span("ab_test_assignment").update(output={"group": group})
            return group
            
        except Exception:
            return None
    
    async def _adjust_personalization_strategy(self, result: PersonalizationResult, user_profile: UserProfile, trace) -> PersonalizationResult:
        """개인화 전략 조정"""
        span = trace.span(name="strategy_adjustment")
        
        try:
            # 낮은 점수의 원인 분석
            if result.relevance_score < 0.5:
                # 관련성 개선
                result.personalized_summary = await self._enhance_summary_relevance(
                    result.personalized_summary, user_profile
                )
            
            if result.interest_match.get('overall_relevance', 0) < 50:
                # 관심사 매칭 개선
                result.related_content = await self._find_better_matches(user_profile)
            
            # 조정된 점수 재계산
            adjusted_score = await self._calculate_personalization_score(
                result.interest_match, result.relevance_score, user_profile
            )
            result.personalization_score = adjusted_score
            
            span.update(output={"adjusted_score": adjusted_score.overall_score})
            return result
            
        except Exception as e:
            logger.error(f"Strategy adjustment failed: {str(e)}")
            return result
    
    async def _enhance_summary_relevance(self, summary: str, user_profile: UserProfile) -> str:
        """요약 관련성 향상"""
        try:
            # 사용자 관심사를 더 강조한 요약 재생성
            enhanced_prompt = f"""
            다음 요약을 사용자의 관심사({', '.join(user_profile.interests)})에 더 초점을 맞춰 개선해주세요:
            
            기존 요약: {summary}
            
            개선된 요약:
            """
            
            response = await self.llm.ainvoke([HumanMessage(content=enhanced_prompt)])
            return response.content.strip()
            
        except Exception:
            return summary
    
    async def _find_better_matches(self, user_profile: UserProfile) -> List[Dict]:
        """더 나은 매칭 콘텐츠 검색"""
        try:
            # 사용자 관심사에 더 특화된 콘텐츠 검색
            better_matches = await self.content_recommender.find_content_by_interests(
                user_profile.interests,
                limit=5
            )
            
            return [
                {
                    "id": item.id,
                    "title": item.title,
                    "summary": item.summary,
                    "score": 0.8,  # 높은 기본 점수
                    "reason": "관심사 특화 추천"
                }
                for item in better_matches
            ]
            
        except Exception:
            return []
    
    async def _update_user_profile(self, user_profile: UserProfile, state: NewsState, result: PersonalizationResult):
        """사용자 프로필 업데이트 (학습)"""
        try:
            # 상호작용 카운트 증가
            user_profile.interaction_count += 1
            
            # 개인화 만족도 업데이트 (가중 평균)
            new_satisfaction = result.personalization_score.overall_score * 5.0  # 5점 척도로 변환
            if user_profile.satisfaction_score:
                user_profile.satisfaction_score = (
                    user_profile.satisfaction_score * 0.9 + new_satisfaction * 0.1
                )
            else:
                user_profile.satisfaction_score = new_satisfaction
            
            # 프로필 저장
            await self.user_profiler.update_user_profile(user_profile)
            
            # 캐시 업데이트
            await self.cache.set_user_profile(state.user_id, user_profile, ttl=3600)
            
            logger.info(f"Updated user profile for {state.user_id}, satisfaction: {user_profile.satisfaction_score:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {str(e)}")

    async def get_personalized_feed_optimized(self, user_id: str, limit: int = 20) -> List[Dict]:
        """메모리 효율적인 개인화 피드 생성"""
        try:
            # 1. 사용자 프로필 캐싱
            user_profile = await self.cache.get_user_profile(user_id)
            if not user_profile:
                user_profile = await self._get_or_create_user_profile(user_id, None)
                await self.cache.set_user_profile(user_id, user_profile, ttl=3600)
            
            # 2. 스트리밍 방식으로 처리 (메모리 효율성)
            async def article_scorer():
                """배치별 기사 점수 계산 제너레이터"""
                batch_size = 50
                async for article_batch in self._stream_articles_by_relevance(user_profile, batch_size):
                    scored_batch = []
                    for article in article_batch:
                        score = await self._calculate_score_cached(article, user_profile)
                        scored_batch.append((article, score))
                    
                    # 배치별 정렬 후 상위 항목만 유지
                    scored_batch.sort(key=lambda x: x[1], reverse=True)
                    yield scored_batch[:limit * 2]  # 여유분 포함
            
            # 3. 힙을 사용한 Top-K 선택 (메모리 효율적)
            import heapq
            top_articles = []
            
            async for scored_batch in article_scorer():
                for article, score in scored_batch:
                    if len(top_articles) < limit:
                        heapq.heappush(top_articles, (score, article))
                    elif score > top_articles[0][0]:
                        heapq.heapreplace(top_articles, (score, article))
            
            # 최종 정렬 및 반환
            result = [article for score, article in sorted(top_articles, reverse=True)]
            
            logger.info(f"Optimized personalized feed generated for user {user_id}: {len(result)} articles")
            return result
            
        except Exception as e:
            logger.error(f"Optimized personalized feed generation failed: {e}")
            return []
    
    async def _stream_articles_by_relevance(self, user_profile: UserProfile, batch_size: int = 50):
        """관련성 기반 기사 스트리밍"""
        try:
            # 사용자 관심사 기반 사전 필터링
            interest_keywords = [interest.lower() for interest in user_profile.interests]
            preferred_categories = user_profile.preferred_categories or []
            
            # 배치별 데이터베이스 조회
            offset = 0
            while True:
                query = """
                SELECT id, title, content, category, published_at, quality_score
                FROM news_articles 
                WHERE published_at > NOW() - INTERVAL '7 days'
                  AND status = 'published'
                  AND quality_score >= 0.6
                  AND (
                    category = ANY($1) OR 
                    LOWER(title) ~ ANY($2) OR 
                    LOWER(content) ~ ANY($3)
                  )
                ORDER BY published_at DESC, quality_score DESC
                LIMIT $4 OFFSET $5;
                """
                
                # 키워드 패턴 생성
                keyword_patterns = [f".*{keyword}.*" for keyword in interest_keywords[:10]]  # 상위 10개만
                
                async with self.db_pool.acquire() as conn:
                    articles = await conn.fetch(
                        query, 
                        preferred_categories,
                        keyword_patterns,
                        keyword_patterns,
                        batch_size, 
                        offset
                    )
                
                if not articles:
                    break
                
                yield articles
                offset += batch_size
                
                # 메모리 사용량 제한 (최대 1000개 기사)
                if offset >= 1000:
                    break
                    
        except Exception as e:
            logger.error(f"Article streaming failed: {e}")
            yield []
    
    async def _calculate_score_cached(self, article: Dict, user_profile: UserProfile) -> float:
        """캐시된 점수 계산"""
        try:
            # 캐시 키 생성
            cache_key = f"score:{user_profile.user_id}:{article['id']}"
            
            # 캐시에서 점수 확인
            cached_score = await self.cache.get(cache_key)
            if cached_score is not None:
                return float(cached_score)
            
            # 점수 계산
            interest_analysis = await self._analyze_user_interests(
                article['content'], user_profile, None
            )
            relevance_score = await self._calculate_relevance_score(
                article['content'], user_profile, None
            )
            
            personalization_score = await self._calculate_personalization_score(
                interest_analysis, relevance_score, user_profile
            )
            
            final_score = personalization_score.overall_score
            
            # 캐시에 저장 (1시간 TTL)
            await self.cache.set(cache_key, final_score, ttl=3600)
            
            return final_score
            
        except Exception as e:
            logger.error(f"Cached score calculation failed: {e}")
            return 0.5
    
    def optimize_memory_usage(self):
        """메모리 사용량 최적화"""
        try:
            import gc
            import psutil
            import os
            
            # 현재 메모리 사용량 확인
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 가비지 컬렉션 강제 실행
            collected = gc.collect()
            
            # 메모리 사용량 재확인
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = memory_before - memory_after
            
            logger.info(f"Memory optimization: {collected} objects collected, "
                       f"{memory_freed:.2f}MB freed, "
                       f"current usage: {memory_after:.2f}MB")
            
            return {
                "objects_collected": collected,
                "memory_freed_mb": memory_freed,
                "current_memory_mb": memory_after
            }
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return None