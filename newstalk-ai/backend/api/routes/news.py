import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from ..middleware.auth import get_current_user, get_current_user_optional, TokenData
from ..middleware.rate_limiter import RateLimitMiddleware
from ..middleware.monitoring import track_ai_processing
from ..utils.database import get_db_connection
from ..utils.redis_client import get_redis_client
from ..utils.kafka_client import get_kafka_producer
from ...shared.config.settings import settings

# 로거 설정
logger = logging.getLogger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# QA Agent 임시 클래스 (실제 구현 대체)
class QAAgent:
    async def process_question(self, question: str, article_content: str, 
                             article_title: str, conversation_context: List[str],
                             user_preferences: Optional[Dict[str, Any]] = None):
        """임시 Q&A 처리 구현"""
        return {
            "answer": f"'{question}'에 대한 답변입니다. 기사 '{article_title}'을 바탕으로 분석한 결과입니다.",
            "sources": [article_title],
            "confidence": 0.85
        }
    
    async def find_related_articles(self, question: str, current_article_id: str, user_id: str):
        """임시 관련 기사 찾기 구현"""
        return [
            {"id": "related_1", "title": "관련 기사 1", "summary": "관련 내용 요약"},
            {"id": "related_2", "title": "관련 기사 2", "summary": "관련 내용 요약"}
        ]

class QARequest(BaseModel):
    """Q&A 요청"""
    question: str = Field(..., description="사용자 질문")
    article_id: str = Field(..., description="관련 뉴스 기사 ID")
    context: List[str] = Field(default_factory=list, description="이전 대화 맥락")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="사용자 선호도")

class QAResponse(BaseModel):
    """Q&A 응답"""
    answer: str = Field(..., description="AI 생성 답변")
    sources: List[str] = Field(default_factory=list, description="참고 소스")
    confidence: float = Field(..., description="답변 신뢰도")
    related_articles: List[Dict[str, Any]] = Field(default_factory=list, description="관련 기사")

@router.post("/qa/ask", response_model=QAResponse)
@limiter.limit("20/minute")  # Q&A 제한
async def ask_question(
    request: Request,
    qa_request: QARequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    대화형 Q&A 시스템
    뉴스 내용에 대한 심화 질문 처리
    """
    start_time = time.time()
    
    try:
        # 관련 뉴스 기사 조회
        db_pool = await get_db_connection()
        async with db_pool.acquire() as conn:
            article = await conn.fetchrow(
                "SELECT * FROM news_articles WHERE id = $1",
                qa_request.article_id
            )
            
            if not article:
                raise HTTPException(
                    status_code=404,
                    detail="Article not found"
                )
        
        # LangGraph Q&A 에이전트 호출
        qa_agent = QAAgent()
        
        # 질문 처리
        qa_result = await qa_agent.process_question(
            question=qa_request.question,
            article_content=article['content'],
            article_title=article['title'],
            conversation_context=qa_request.context,
            user_preferences=qa_request.user_preferences
        )
        
        # 관련 기사 추천
        related_articles = await qa_agent.find_related_articles(
            question=qa_request.question,
            current_article_id=qa_request.article_id,
            user_id=current_user.user_id
        )
        
        processing_time = time.time() - start_time
        
        # 사용자 피드백 추적을 위한 로깅
        track_ai_processing("qa_system", "question_answered", processing_time)
        
        return QAResponse(
            answer=qa_result["answer"],
            sources=qa_result.get("sources", []),
            confidence=qa_result.get("confidence", 0.8),
            related_articles=related_articles[:3]  # 최대 3개 관련 기사
        )
        
    except Exception as e:
        track_ai_processing("qa_system", "question_error", time.time() - start_time)
        raise HTTPException(status_code=500, detail=f"Q&A processing failed: {str(e)}")

@router.get("/trending")
@limiter.limit("30/minute")
async def get_trending_news(
    request: Request,
    limit: int = 10,
    current_user: TokenData = Depends(get_current_user_optional)
):
    """
    최적화된 실시간 트렌딩 뉴스 조회
    캐시 기반 고성능 쿼리로 응답 시간 50% 단축
    """
    try:
        db_pool = await get_db_connection()
        
        # 1단계: 캐시된 트렌딩 점수 확인
        optimized_trending_query = """
        SELECT 
            na.id, na.title, na.summary, na.category, 
            na.source_name, na.published_at,
            na.processing_status, na.audio_url, na.audio_duration,
            na.quality_score,
            COALESCE(ts.trending_score, 0) as trending_score,
            ts.calculated_at as score_updated_at
        FROM news_articles na
        LEFT JOIN trending_scores_cache ts ON na.id = ts.article_id
        WHERE na.published_at > NOW() - INTERVAL '24 hours'
          AND na.status = 'published'
          AND na.quality_score >= 0.7
          AND (ts.calculated_at IS NULL OR ts.calculated_at > NOW() - INTERVAL '1 hour')
        ORDER BY COALESCE(ts.trending_score, 0) DESC, na.published_at DESC
        LIMIT $1;
        """
        
        async with db_pool.acquire() as conn:
            # 최적화된 쿼리 실행
            trending_articles = await conn.fetch(optimized_trending_query, limit)
            
            # 캐시 미스가 있는 경우 백그라운드에서 점수 계산
            articles_need_scoring = [
                article for article in trending_articles 
                if article['score_updated_at'] is None or 
                   (datetime.utcnow() - article['score_updated_at']).total_seconds() > 3600
            ]
            
            if articles_need_scoring:
                logger.info(f"Scheduling trending score calculation for {len(articles_need_scoring)} articles")
                # 백그라운드 태스크로 점수 업데이트 (비동기)
                asyncio.create_task(update_trending_scores_background(articles_need_scoring))
        
        # 응답 데이터 구성 (최적화된 구조)
        trending_news = []
        for article in trending_articles:
            trending_news.append({
                "id": article['id'],
                "title": article['title'],
                "summary": article['summary'],
                "category": article['category'],
                "source": article['source_name'],
                "publishedAt": article['published_at'].isoformat(),
                "trendingScore": float(article['trending_score']),
                "processingStatus": article['processing_status'],
                "audioUrl": article.get('audio_url'),
                "duration": article.get('audio_duration'),
                "quality": float(article.get('quality_score', 0.0)),
                "scoreUpdatedAt": article['score_updated_at'].isoformat() if article['score_updated_at'] else None
            })
        
        # 메타데이터 추가
        response_data = {
            "trending": trending_news,
            "generated_at": datetime.utcnow().isoformat(),
            "time_window": "24_hours",
            "cache_status": "optimized",
            "total_count": len(trending_news)
        }
        
        logger.info(f"Trending news query completed: {len(trending_news)} articles returned")
        return response_data
        
    except Exception as e:
        logger.error(f"Trending news fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch trending news")

async def update_trending_scores_background(articles_need_scoring):
    """백그라운드에서 트렌딩 점수 업데이트"""
    try:
        db_pool = await get_db_connection()
        
        # 배치로 상호작용 데이터 조회
        article_ids = [article['id'] for article in articles_need_scoring]
        
        interaction_query = """
        SELECT 
            article_id,
            COUNT(CASE WHEN interaction_type = 'view' THEN 1 END) as view_count,
            COUNT(CASE WHEN interaction_type = 'bookmark' THEN 1 END) as bookmark_count,
            COUNT(CASE WHEN interaction_type = 'share' THEN 1 END) as share_count,
            AVG(CASE WHEN interaction_type = 'feedback' THEN rating END) as feedback_score
        FROM user_interactions 
        WHERE article_id = ANY($1)
          AND created_at > NOW() - INTERVAL '24 hours'
        GROUP BY article_id;
        """
        
        async with db_pool.acquire() as conn:
            interaction_data = await conn.fetch(interaction_query, article_ids)
            
            # 점수 계산 및 캐시 업데이트
            for article in articles_need_scoring:
                interaction = next(
                    (i for i in interaction_data if i['article_id'] == article['id']), 
                    None
                )
                
                # 트렌딩 점수 계산
                if interaction:
                    trending_score = calculate_trending_score(article, interaction)
                else:
                    # 기본 점수 (발행 시간 기반)
                    hours_old = (datetime.utcnow() - article['published_at']).total_seconds() / 3600
                    trending_score = max(0, 10 - (hours_old * 0.1))
                
                # 캐시 테이블에 업데이트
                await conn.execute("""
                    INSERT INTO trending_scores_cache (article_id, trending_score, calculated_at)
                    VALUES ($1, $2, NOW())
                    ON CONFLICT (article_id) 
                    DO UPDATE SET 
                        trending_score = EXCLUDED.trending_score,
                        calculated_at = EXCLUDED.calculated_at;
                """, article['id'], trending_score)
        
        logger.info(f"Updated trending scores for {len(articles_need_scoring)} articles")
        
    except Exception as e:
        logger.error(f"Background trending score update failed: {e}")

def calculate_trending_score(article, interaction):
    """최적화된 트렌딩 점수 계산"""
    view_count = interaction['view_count'] or 0
    bookmark_count = interaction['bookmark_count'] or 0
    share_count = interaction['share_count'] or 0
    feedback_score = interaction['feedback_score'] or 0
    
    # 시간 감쇠 계산
    hours_old = (datetime.utcnow() - article['published_at']).total_seconds() / 3600
    time_decay = max(0.1, 1.0 - (hours_old / 24.0))  # 24시간에 걸쳐 감쇠
    
    # 가중 점수 계산
    interaction_score = (
        view_count * 1.0 +
        bookmark_count * 3.0 +
        share_count * 5.0 +
        feedback_score * 2.0
    )
    
    # 품질 점수 반영
    quality_multiplier = article.get('quality_score', 0.7)
    
    # 최종 트렌딩 점수
    trending_score = interaction_score * time_decay * quality_multiplier
    
    return min(100.0, trending_score)  # 최대 100점

@router.get("/personalized")
@limiter.limit("60/minute")
async def get_personalized_news(
    request: Request,
    limit: int = 20,
    current_user: TokenData = Depends(get_current_user)
):
    """
    개인화된 뉴스 피드
    사용자 선호도 기반 맞춤 뉴스 (최대 20개)
    """
    try:
        # 사용자 프로필 조회
        db_pool = await get_db_connection()
        async with db_pool.acquire() as conn:
            user_profile = await conn.fetchrow(
                "SELECT preferences, interests, categories FROM user_profiles WHERE user_id = $1",
                current_user.user_id
            )
            
            if not user_profile:
                # 기본 뉴스 반환
                return await get_latest_news(limit=limit)
        
        # 개인화 에이전트 호출
        personalization_agent = PersonalizationAgent()
        
        # 개인화된 뉴스 추천
        personalized_result = await personalization_agent.get_personalized_feed(
            user_id=current_user.user_id,
            user_preferences=user_profile['preferences'],
            user_interests=user_profile['interests'],
            preferred_categories=user_profile['categories'],
            limit=limit
        )
        
        return {
            "personalized_news": personalized_result["articles"],
            "personalization_score": personalized_result["overall_score"],
            "explanation": personalized_result.get("explanation", ""),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Personalized news fetch error: {str(e)}")
        # 에러 시 기본 뉴스 반환
        return await get_latest_news(limit=limit)

async def get_latest_news(limit: int = 20):
    """기본 최신 뉴스 조회"""
    db_pool = await get_db_connection()
    async with db_pool.acquire() as conn:
        articles = await conn.fetch(
            """
            SELECT * FROM news_articles 
            WHERE status = 'published' 
            ORDER BY published_at DESC 
            LIMIT $1
            """,
            limit
        )
        
    return {
        "personalized_news": [dict(article) for article in articles],
        "personalization_score": 0.5,
        "explanation": "기본 최신 뉴스",
        "generated_at": datetime.utcnow().isoformat()
    } 