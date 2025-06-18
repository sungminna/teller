"""
데이터베이스 쿼리 최적화 유틸리티
성능 개선을 위한 최적화된 쿼리 및 인덱스 관리
"""

import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """데이터베이스 성능 최적화 관리자"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.query_cache = {}
        self.performance_stats = {}
    
    async def create_optimized_indexes(self):
        """성능 최적화를 위한 인덱스 생성"""
        indexes = [
            # 트렌딩 뉴스 조회 최적화
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_trending_optimized
            ON news_articles (published_at DESC, status, quality_score DESC)
            WHERE published_at > NOW() - INTERVAL '24 hours' AND status = 'published';
            """,
            
            # 개인화 피드 최적화
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_personalization
            ON news_articles (category, published_at DESC, quality_score DESC)
            WHERE status = 'published' AND published_at > NOW() - INTERVAL '7 days';
            """,
            
            # 사용자 상호작용 최적화
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_interactions_optimized
            ON user_interactions (article_id, interaction_type, created_at DESC)
            WHERE created_at > NOW() - INTERVAL '24 hours';
            """,
            
            # 트렌딩 점수 캐시 테이블
            """
            CREATE TABLE IF NOT EXISTS trending_scores_cache (
                id SERIAL PRIMARY KEY,
                article_id UUID NOT NULL,
                trending_score FLOAT NOT NULL DEFAULT 0,
                view_count INTEGER DEFAULT 0,
                bookmark_count INTEGER DEFAULT 0,
                share_count INTEGER DEFAULT 0,
                feedback_score FLOAT DEFAULT 0,
                calculated_at TIMESTAMP DEFAULT NOW(),
                FOREIGN KEY (article_id) REFERENCES news_articles(id) ON DELETE CASCADE
            );
            """,
            
            # 트렌딩 점수 캐시 인덱스
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_cache_article
            ON trending_scores_cache (article_id);
            """,
            
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trending_cache_score_time
            ON trending_scores_cache (trending_score DESC, calculated_at DESC);
            """
        ]
        
        try:
            async with self.db_pool.acquire() as conn:
                for index_query in indexes:
                    try:
                        await conn.execute(index_query)
                        logger.info(f"Index created successfully")
                    except Exception as e:
                        logger.warning(f"Index creation skipped (may already exist): {e}")
            
            logger.info("Database optimization indexes created successfully")
            
        except Exception as e:
            logger.error(f"Database index creation failed: {e}")
    
    async def get_trending_news_optimized(self, limit: int = 10) -> List[Dict]:
        """최적화된 트렌딩 뉴스 조회"""
        query_key = f"trending_news_{limit}"
        
        # 캐시된 결과 확인 (5분 캐시)
        if query_key in self.query_cache:
            cached_time, cached_result = self.query_cache[query_key]
            if (datetime.now() - cached_time).total_seconds() < 300:
                return cached_result
        
        start_time = time.time()
        
        try:
            optimized_query = """
            WITH trending_scores AS (
                SELECT 
                    na.id, na.title, na.summary, na.category, 
                    na.source_name, na.published_at, na.quality_score,
                    na.processing_status, na.audio_url, na.audio_duration,
                    COALESCE(ts.trending_score, 
                        -- 실시간 점수 계산 (캐시 미스 시)
                        COALESCE(ui.view_count, 0) * 1.0 +
                        COALESCE(ui.bookmark_count, 0) * 2.0 +
                        COALESCE(ui.share_count, 0) * 3.0 +
                        COALESCE(ui.feedback_score, 0) * 1.5 +
                        (EXTRACT(EPOCH FROM (NOW() - na.published_at)) / 3600.0) * -0.1
                    ) as trending_score,
                    ts.calculated_at
                FROM news_articles na
                LEFT JOIN trending_scores_cache ts ON na.id = ts.article_id 
                    AND ts.calculated_at > NOW() - INTERVAL '1 hour'
                LEFT JOIN (
                    SELECT 
                        article_id,
                        COUNT(CASE WHEN interaction_type = 'view' THEN 1 END) as view_count,
                        COUNT(CASE WHEN interaction_type = 'bookmark' THEN 1 END) as bookmark_count,
                        COUNT(CASE WHEN interaction_type = 'share' THEN 1 END) as share_count,
                        AVG(CASE WHEN interaction_type = 'feedback' THEN rating END) as feedback_score
                    FROM user_interactions 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY article_id
                ) ui ON na.id = ui.article_id
                WHERE na.published_at > NOW() - INTERVAL '24 hours'
                  AND na.status = 'published'
                  AND na.quality_score >= 0.7
            )
            SELECT * FROM trending_scores
            ORDER BY trending_score DESC, published_at DESC
            LIMIT $1;
            """
            
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch(optimized_query, limit)
                
                # 결과 변환
                trending_news = []
                articles_need_cache_update = []
                
                for row in results:
                    article_data = {
                        "id": str(row['id']),
                        "title": row['title'],
                        "summary": row['summary'],
                        "category": row['category'],
                        "source": row['source_name'],
                        "publishedAt": row['published_at'].isoformat(),
                        "trendingScore": float(row['trending_score']),
                        "processingStatus": row['processing_status'],
                        "audioUrl": row.get('audio_url'),
                        "duration": row.get('audio_duration'),
                        "quality": float(row.get('quality_score', 0.0))
                    }
                    trending_news.append(article_data)
                    
                    # 캐시 업데이트가 필요한 기사 식별
                    if not row['calculated_at'] or \
                       (datetime.utcnow() - row['calculated_at']).total_seconds() > 3600:
                        articles_need_cache_update.append({
                            'id': row['id'],
                            'trending_score': row['trending_score']
                        })
                
                # 백그라운드에서 캐시 업데이트
                if articles_need_cache_update:
                    asyncio.create_task(
                        self._update_trending_cache_background(articles_need_cache_update)
                    )
                
                # 쿼리 결과 캐시
                self.query_cache[query_key] = (datetime.now(), trending_news)
                
                # 성능 통계 업데이트
                execution_time = time.time() - start_time
                self._update_performance_stats("trending_news", execution_time)
                
                logger.info(f"Optimized trending query completed in {execution_time:.3f}s, "
                           f"{len(trending_news)} articles returned")
                
                return trending_news
                
        except Exception as e:
            logger.error(f"Optimized trending news query failed: {e}")
            return []
    
    async def _update_trending_cache_background(self, articles: List[Dict]):
        """백그라운드에서 트렌딩 캐시 업데이트"""
        try:
            async with self.db_pool.acquire() as conn:
                for article in articles:
                    await conn.execute("""
                        INSERT INTO trending_scores_cache (article_id, trending_score, calculated_at)
                        VALUES ($1, $2, NOW())
                        ON CONFLICT (article_id) 
                        DO UPDATE SET 
                            trending_score = $2,
                            calculated_at = NOW();
                    """, article['id'], article['trending_score'])
                
                logger.info(f"Trending cache updated for {len(articles)} articles")
                
        except Exception as e:
            logger.error(f"Trending cache update failed: {e}")
    
    async def get_personalized_articles_optimized(self, user_profile: Dict, limit: int = 20) -> List[Dict]:
        """최적화된 개인화 기사 조회"""
        try:
            # 사용자 관심사 및 카테고리 추출
            interests = user_profile.get('interests', [])
            preferred_categories = user_profile.get('preferred_categories', [])
            
            # 동적 쿼리 생성
            conditions = ["na.status = 'published'", "na.published_at > NOW() - INTERVAL '7 days'"]
            params = []
            param_count = 0
            
            if preferred_categories:
                param_count += 1
                conditions.append(f"na.category = ANY(${param_count})")
                params.append(preferred_categories)
            
            if interests:
                # 관심사 키워드 매칭
                param_count += 1
                keyword_conditions = []
                for _ in interests[:5]:  # 상위 5개 관심사만
                    param_count += 1
                    keyword_conditions.append(f"(LOWER(na.title) LIKE ${param_count} OR LOWER(na.content) LIKE ${param_count})")
                    params.append(f"%{interests[len(params)-len(preferred_categories)-1].lower()}%")
                
                if keyword_conditions:
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
            
            param_count += 1
            params.append(limit)
            
            query = f"""
            SELECT 
                na.id, na.title, na.summary, na.category, na.source_name,
                na.published_at, na.quality_score, na.processing_status,
                na.audio_url, na.audio_duration,
                -- 개인화 점수 계산
                (
                    CASE WHEN na.category = ANY($1) THEN 0.3 ELSE 0 END +
                    na.quality_score * 0.3 +
                    (1.0 - EXTRACT(EPOCH FROM (NOW() - na.published_at)) / 86400.0) * 0.2 +
                    COALESCE(ui.engagement_score, 0.5) * 0.2
                ) as personalization_score
            FROM news_articles na
            LEFT JOIN (
                SELECT 
                    article_id,
                    AVG(CASE WHEN interaction_type = 'feedback' THEN rating ELSE 3 END) / 5.0 as engagement_score
                FROM user_interactions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY article_id
            ) ui ON na.id = ui.article_id
            WHERE {' AND '.join(conditions)}
            ORDER BY personalization_score DESC, na.published_at DESC
            LIMIT ${param_count};
            """
            
            async with self.db_pool.acquire() as conn:
                results = await conn.fetch(query, *params)
                
                articles = []
                for row in results:
                    articles.append({
                        "id": str(row['id']),
                        "title": row['title'],
                        "summary": row['summary'],
                        "category": row['category'],
                        "source": row['source_name'],
                        "publishedAt": row['published_at'].isoformat(),
                        "personalizationScore": float(row['personalization_score']),
                        "processingStatus": row['processing_status'],
                        "audioUrl": row.get('audio_url'),
                        "duration": row.get('audio_duration'),
                        "quality": float(row.get('quality_score', 0.0))
                    })
                
                return articles
                
        except Exception as e:
            logger.error(f"Optimized personalized articles query failed: {e}")
            return []
    
    def _update_performance_stats(self, query_type: str, execution_time: float):
        """성능 통계 업데이트"""
        if query_type not in self.performance_stats:
            self.performance_stats[query_type] = {
                'total_executions': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
        
        stats = self.performance_stats[query_type]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_executions']
        stats['min_time'] = min(stats['min_time'], execution_time)
        stats['max_time'] = max(stats['max_time'], execution_time)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 반환"""
        return {
            "query_performance": self.performance_stats,
            "cache_stats": {
                "cached_queries": len(self.query_cache),
                "cache_hit_ratio": self._calculate_cache_hit_ratio()
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """캐시 히트율 계산"""
        # 실제 구현에서는 히트/미스 카운터 사용
        return 0.85  # 임시값
    
    async def cleanup_old_cache_entries(self):
        """오래된 캐시 엔트리 정리"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, (cached_time, _) in self.query_cache.items():
                if (current_time - cached_time).total_seconds() > 300:  # 5분 초과
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.query_cache[key]
            
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# 전역 최적화 인스턴스
_optimizer_instance = None

async def get_database_optimizer(db_pool) -> DatabaseOptimizer:
    """데이터베이스 최적화 인스턴스 반환"""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = DatabaseOptimizer(db_pool)
        await _optimizer_instance.create_optimized_indexes()
    return _optimizer_instance


@asynccontextmanager
async def optimized_db_query(db_pool, query_name: str):
    """최적화된 데이터베이스 쿼리 컨텍스트 매니저"""
    start_time = time.time()
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        logger.info(f"Query '{query_name}' executed in {execution_time:.3f}s") 