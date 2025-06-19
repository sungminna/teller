"""
📰 News Aggregation Service
===========================

뉴스 수집 및 집계 서비스
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """뉴스 아이템 데이터 클래스"""
    
    id: str
    title: str
    content: str
    source: str
    category: str
    published_at: datetime
    url: str
    author: Optional[str] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "source": self.source,
            "category": self.category,
            "published_at": self.published_at.isoformat(),
            "url": self.url,
            "author": self.author,
            "tags": self.tags or []
        }


class NewsAggregationService:
    """뉴스 집계 서비스"""
    
    def __init__(self):
        self.sources = {
            "reuters": "https://feeds.reuters.com/reuters/topNews",
            "bbc": "https://feeds.bbci.co.uk/news/rss.xml",
            "cnn": "http://rss.cnn.com/rss/edition.rss"
        }
        self.categories = ["technology", "politics", "business", "health", "sports"]
        
    async def collect_news(self, sources: Optional[List[str]] = None, 
                          category: Optional[str] = None,
                          limit: int = 50) -> List[NewsItem]:
        """
        뉴스 수집
        
        Args:
            sources: 수집할 소스 목록
            category: 카테고리 필터
            limit: 최대 수집 개수
            
        Returns:
            List[NewsItem]: 수집된 뉴스 목록
        """
        try:
            target_sources = sources or list(self.sources.keys())
            
            # 각 소스에서 뉴스 수집
            all_news = []
            for source in target_sources:
                news_items = await self._collect_from_source(source, category)
                all_news.extend(news_items)
            
            # 중복 제거 및 정렬
            unique_news = self._remove_duplicates(all_news)
            sorted_news = sorted(unique_news, key=lambda x: x.published_at, reverse=True)
            
            logger.info(f"Collected {len(sorted_news)} unique news items from {len(target_sources)} sources")
            return sorted_news[:limit]
            
        except Exception as e:
            logger.error(f"News collection failed: {str(e)}")
            return []
    
    async def _collect_from_source(self, source: str, category: Optional[str] = None) -> List[NewsItem]:
        """
        특정 소스에서 뉴스 수집
        
        Args:
            source: 소스 이름
            category: 카테고리 필터
            
        Returns:
            List[NewsItem]: 수집된 뉴스 목록
        """
        # 실제 구현에서는 RSS 파싱이나 API 호출
        # 현재는 모의 데이터 생성
        
        await asyncio.sleep(0.5)  # API 호출 시뮬레이션
        
        mock_news = []
        for i in range(10):  # 소스당 10개 뉴스
            news_item = NewsItem(
                id=f"{source}_{i}_{int(datetime.now().timestamp())}",
                title=f"Breaking: {source.upper()} News Story {i+1}",
                content=f"This is a sample news content from {source}. " * 10,
                source=source,
                category=category or self._get_random_category(),
                published_at=datetime.now() - timedelta(hours=i),
                url=f"https://{source}.com/news/{i}",
                author=f"{source.title()} Reporter",
                tags=[category] if category else ["general", "news"]
            )
            mock_news.append(news_item)
        
        return mock_news
    
    def _get_random_category(self) -> str:
        """랜덤 카테고리 반환"""
        import random
        return random.choice(self.categories)
    
    def _remove_duplicates(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """
        중복 뉴스 제거
        
        Args:
            news_items: 뉴스 아이템 목록
            
        Returns:
            List[NewsItem]: 중복이 제거된 뉴스 목록
        """
        seen_titles = set()
        unique_news = []
        
        for item in news_items:
            # 제목의 유사성으로 중복 판단 (간단한 구현)
            title_words = set(item.title.lower().split())
            
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.lower().split())
                # 70% 이상 단어가 겹치면 중복으로 판단
                overlap = len(title_words & seen_words) / len(title_words | seen_words)
                if overlap > 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(item.title)
                unique_news.append(item)
        
        return unique_news
    
    async def aggregate_by_category(self, news_items: List[NewsItem]) -> Dict[str, List[NewsItem]]:
        """
        카테고리별 뉴스 집계
        
        Args:
            news_items: 뉴스 아이템 목록
            
        Returns:
            Dict[str, List[NewsItem]]: 카테고리별 뉴스 딕셔너리
        """
        categorized = {}
        
        for item in news_items:
            category = item.category
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(item)
        
        # 각 카테고리별로 시간순 정렬
        for category in categorized:
            categorized[category].sort(key=lambda x: x.published_at, reverse=True)
        
        return categorized
    
    async def get_trending_topics(self, news_items: List[NewsItem], 
                                 time_window_hours: int = 24) -> List[Dict[str, Any]]:
        """
        트렌딩 토픽 분석
        
        Args:
            news_items: 뉴스 아이템 목록
            time_window_hours: 분석 시간 윈도우 (시간)
            
        Returns:
            List[Dict[str, Any]]: 트렌딩 토픽 목록
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_news = [item for item in news_items if item.published_at >= cutoff_time]
        
        # 키워드 빈도 분석
        keyword_count = {}
        for item in recent_news:
            # 제목과 태그에서 키워드 추출
            words = item.title.lower().split()
            tags = item.tags or []
            
            for word in words + tags:
                if len(word) > 3:  # 3글자 이상 단어만
                    keyword_count[word] = keyword_count.get(word, 0) + 1
        
        # 상위 트렌딩 토픽 선별
        trending = []
        for keyword, count in sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:10]:
            trending.append({
                "keyword": keyword,
                "count": count,
                "trend_score": count / len(recent_news) if recent_news else 0
            })
        
        return trending
    
    async def get_news_summary(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """
        뉴스 요약 통계
        
        Args:
            news_items: 뉴스 아이템 목록
            
        Returns:
            Dict[str, Any]: 뉴스 요약 정보
        """
        if not news_items:
            return {
                "total_articles": 0,
                "sources": [],
                "categories": {},
                "latest_update": None,
                "time_range": None
            }
        
        # 기본 통계
        total_articles = len(news_items)
        sources = list(set(item.source for item in news_items))
        
        # 카테고리별 개수
        categories = {}
        for item in news_items:
            categories[item.category] = categories.get(item.category, 0) + 1
        
        # 시간 범위
        latest_update = max(item.published_at for item in news_items)
        earliest_update = min(item.published_at for item in news_items)
        
        return {
            "total_articles": total_articles,
            "sources": sources,
            "source_count": len(sources),
            "categories": categories,
            "category_count": len(categories),
            "latest_update": latest_update.isoformat(),
            "earliest_update": earliest_update.isoformat(),
            "time_range_hours": (latest_update - earliest_update).total_seconds() / 3600
        }
    
    async def filter_by_keywords(self, news_items: List[NewsItem], 
                                keywords: List[str],
                                match_all: bool = False) -> List[NewsItem]:
        """
        키워드로 뉴스 필터링
        
        Args:
            news_items: 뉴스 아이템 목록
            keywords: 검색 키워드 목록
            match_all: 모든 키워드가 포함되어야 하는지 여부
            
        Returns:
            List[NewsItem]: 필터링된 뉴스 목록
        """
        filtered_news = []
        
        for item in news_items:
            content_text = (item.title + " " + item.content).lower()
            
            if match_all:
                # 모든 키워드가 포함되어야 함
                if all(keyword.lower() in content_text for keyword in keywords):
                    filtered_news.append(item)
            else:
                # 하나 이상의 키워드가 포함되면 됨
                if any(keyword.lower() in content_text for keyword in keywords):
                    filtered_news.append(item)
        
        return filtered_news
    
    async def get_source_reliability(self, source: str) -> Dict[str, Any]:
        """
        소스 신뢰도 평가
        
        Args:
            source: 소스 이름
            
        Returns:
            Dict[str, Any]: 소스 신뢰도 정보
        """
        # 기본 신뢰도 데이터 (실제로는 외부 DB나 API에서 가져옴)
        reliability_data = {
            "reuters": {"score": 0.95, "grade": "A+", "bias": "minimal"},
            "bbc": {"score": 0.90, "grade": "A", "bias": "minimal"},
            "cnn": {"score": 0.85, "grade": "A-", "bias": "slight_left"},
            "fox": {"score": 0.80, "grade": "B+", "bias": "slight_right"},
            "default": {"score": 0.70, "grade": "B", "bias": "unknown"}
        }
        
        return reliability_data.get(source.lower(), reliability_data["default"])
    
    async def schedule_collection(self, interval_minutes: int = 30) -> None:
        """
        정기적인 뉴스 수집 스케줄링
        
        Args:
            interval_minutes: 수집 간격 (분)
        """
        logger.info(f"Starting scheduled news collection every {interval_minutes} minutes")
        
        while True:
            try:
                news_items = await self.collect_news()
                logger.info(f"Scheduled collection completed: {len(news_items)} articles")
                
                # 수집된 뉴스 처리 (실제로는 데이터베이스 저장 등)
                await self._process_collected_news(news_items)
                
            except Exception as e:
                logger.error(f"Scheduled collection failed: {str(e)}")
            
            # 다음 수집까지 대기
            await asyncio.sleep(interval_minutes * 60)
    
    async def _process_collected_news(self, news_items: List[NewsItem]) -> None:
        """
        수집된 뉴스 처리
        
        Args:
            news_items: 처리할 뉴스 목록
        """
        # 실제 구현에서는 데이터베이스 저장, 캐시 업데이트 등
        logger.info(f"Processing {len(news_items)} collected news items")
        
        # 간단한 처리 시뮬레이션
        for item in news_items:
            # 뉴스 품질 검증, 중복 체크, 카테고리 분류 등
            pass 