"""
📊 Content Quality Analyzer Service
===================================

뉴스 콘텐츠의 품질을 분석하는 서비스
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """품질 메트릭 데이터 클래스"""
    
    overall_score: float
    readability_score: float
    factual_accuracy: float
    bias_score: float
    source_credibility: float
    freshness_score: float
    engagement_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "overall_score": self.overall_score,
            "readability_score": self.readability_score,
            "factual_accuracy": self.factual_accuracy,
            "bias_score": self.bias_score,
            "source_credibility": self.source_credibility,
            "freshness_score": self.freshness_score,
            "engagement_potential": self.engagement_potential
        }


class ContentQualityAnalyzer:
    """콘텐츠 품질 분석기"""
    
    def __init__(self):
        self.quality_threshold = 0.7
        self.bias_threshold = 0.3
        self.credible_sources = {
            "reuters.com": 0.95,
            "bbc.com": 0.90,
            "cnn.com": 0.85,
            "nytimes.com": 0.90,
            "washingtonpost.com": 0.88
        }
        
    async def analyze_content_quality(self, content: Dict[str, Any]) -> QualityMetrics:
        """
        콘텐츠 품질 분석
        
        Args:
            content: 분석할 콘텐츠 데이터
            
        Returns:
            QualityMetrics: 품질 메트릭 결과
        """
        try:
            # 기본 품질 점수 계산
            readability = self._calculate_readability(content.get("content", ""))
            factual_accuracy = await self._assess_factual_accuracy(content)
            bias_score = self._analyze_bias(content.get("content", ""))
            source_credibility = self._evaluate_source_credibility(content.get("source", ""))
            freshness = self._calculate_freshness(content.get("published_at"))
            engagement = self._predict_engagement(content)
            
            # 전체 점수 계산 (가중 평균)
            overall_score = (
                readability * 0.15 +
                factual_accuracy * 0.25 +
                (1 - bias_score) * 0.20 +  # 편향성이 낮을수록 좋음
                source_credibility * 0.20 +
                freshness * 0.10 +
                engagement * 0.10
            )
            
            metrics = QualityMetrics(
                overall_score=overall_score,
                readability_score=readability,
                factual_accuracy=factual_accuracy,
                bias_score=bias_score,
                source_credibility=source_credibility,
                freshness_score=freshness,
                engagement_potential=engagement
            )
            
            logger.info(f"Content quality analysis completed. Overall score: {overall_score:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Content quality analysis failed: {str(e)}")
            # 기본값 반환
            return QualityMetrics(
                overall_score=0.5,
                readability_score=0.5,
                factual_accuracy=0.5,
                bias_score=0.5,
                source_credibility=0.5,
                freshness_score=0.5,
                engagement_potential=0.5
            )
    
    def _calculate_readability(self, content: str) -> float:
        """
        가독성 점수 계산
        
        Args:
            content: 콘텐츠 텍스트
            
        Returns:
            float: 가독성 점수 (0-1)
        """
        if not content:
            return 0.0
            
        # 간단한 가독성 메트릭
        words = content.split()
        sentences = content.split('.')
        
        if len(sentences) == 0:
            return 0.0
            
        avg_words_per_sentence = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # 가독성 점수 계산 (간단한 휴리스틱)
        readability = max(0, min(1, 1 - (avg_words_per_sentence - 15) / 20))
        readability *= max(0, min(1, 1 - (avg_word_length - 5) / 5))
        
        return readability
    
    async def _assess_factual_accuracy(self, content: Dict[str, Any]) -> float:
        """
        팩트 정확성 평가
        
        Args:
            content: 콘텐츠 데이터
            
        Returns:
            float: 팩트 정확성 점수 (0-1)
        """
        # 실제 구현에서는 외부 팩트체킹 API 사용
        # 현재는 모의 점수 반환
        
        text = content.get("content", "")
        
        # 기본적인 팩트체킹 휴리스틱
        fact_indicators = [
            "according to", "statistics show", "research indicates",
            "study found", "data reveals", "official report"
        ]
        
        fact_score = sum(1 for indicator in fact_indicators if indicator in text.lower())
        max_score = len(fact_indicators)
        
        # 정규화 및 기본 점수 추가
        normalized_score = (fact_score / max_score) * 0.3 + 0.7
        
        return min(1.0, normalized_score)
    
    def _analyze_bias(self, content: str) -> float:
        """
        편향성 분석
        
        Args:
            content: 콘텐츠 텍스트
            
        Returns:
            float: 편향성 점수 (0-1, 높을수록 편향적)
        """
        if not content:
            return 0.5
            
        # 편향성 지표 단어들
        bias_indicators = [
            "obviously", "clearly", "undoubtedly", "without question",
            "everyone knows", "it's obvious", "definitely", "absolutely"
        ]
        
        emotional_words = [
            "shocking", "outrageous", "incredible", "unbelievable",
            "devastating", "amazing", "terrible", "wonderful"
        ]
        
        content_lower = content.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in content_lower)
        emotional_count = sum(1 for word in emotional_words if word in content_lower)
        
        total_words = len(content.split())
        if total_words == 0:
            return 0.5
            
        bias_ratio = (bias_count + emotional_count) / total_words
        
        # 편향성 점수 계산 (0-1, 높을수록 편향적)
        bias_score = min(1.0, bias_ratio * 10)
        
        return bias_score
    
    def _evaluate_source_credibility(self, source: str) -> float:
        """
        출처 신뢰도 평가
        
        Args:
            source: 뉴스 출처
            
        Returns:
            float: 신뢰도 점수 (0-1)
        """
        if not source:
            return 0.5
            
        # 알려진 신뢰할 수 있는 출처 확인
        for domain, credibility in self.credible_sources.items():
            if domain in source.lower():
                return credibility
                
        # 기본 신뢰도 점수
        return 0.6
    
    def _calculate_freshness(self, published_at: Optional[str]) -> float:
        """
        뉴스 신선도 계산
        
        Args:
            published_at: 발행 시간
            
        Returns:
            float: 신선도 점수 (0-1)
        """
        if not published_at:
            return 0.5
            
        try:
            # 발행 시간 파싱 (간단한 구현)
            if isinstance(published_at, str):
                # 실제 구현에서는 더 정교한 날짜 파싱 필요
                published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                published_date = published_at
                
            now = datetime.now()
            time_diff = (now - published_date).total_seconds()
            
            # 24시간 이내면 1.0, 일주일 후면 0.0
            hours_diff = time_diff / 3600
            freshness = max(0, min(1, 1 - hours_diff / (24 * 7)))
            
            return freshness
            
        except Exception:
            return 0.5
    
    def _predict_engagement(self, content: Dict[str, Any]) -> float:
        """
        참여도 예측
        
        Args:
            content: 콘텐츠 데이터
            
        Returns:
            float: 참여도 점수 (0-1)
        """
        text = content.get("content", "")
        title = content.get("title", "")
        
        # 참여도 높은 키워드들
        engagement_keywords = [
            "breaking", "exclusive", "revealed", "secret", "shocking",
            "new study", "research", "discovery", "trend", "viral"
        ]
        
        combined_text = (title + " " + text).lower()
        engagement_score = sum(1 for keyword in engagement_keywords if keyword in combined_text)
        
        # 제목 길이도 고려 (너무 길거나 짧으면 참여도 낮음)
        title_length_score = 1.0
        if title:
            title_words = len(title.split())
            if title_words < 5 or title_words > 15:
                title_length_score = 0.7
                
        # 정규화
        normalized_engagement = min(1.0, (engagement_score / len(engagement_keywords)) * 0.8 + 0.2)
        
        return normalized_engagement * title_length_score
    
    def is_quality_content(self, metrics: QualityMetrics) -> bool:
        """
        콘텐츠가 품질 기준을 만족하는지 확인
        
        Args:
            metrics: 품질 메트릭
            
        Returns:
            bool: 품질 기준 만족 여부
        """
        return (
            metrics.overall_score >= self.quality_threshold and
            metrics.bias_score <= self.bias_threshold and
            metrics.source_credibility >= 0.6
        )
    
    def get_quality_summary(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """
        품질 분석 요약 정보 반환
        
        Args:
            metrics: 품질 메트릭
            
        Returns:
            Dict[str, Any]: 품질 요약 정보
        """
        return {
            "overall_grade": self._get_grade(metrics.overall_score),
            "is_quality_content": self.is_quality_content(metrics),
            "strengths": self._identify_strengths(metrics),
            "weaknesses": self._identify_weaknesses(metrics),
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _get_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B"
        elif score >= 0.6:
            return "C"
        else:
            return "D"
    
    def _identify_strengths(self, metrics: QualityMetrics) -> List[str]:
        """강점 식별"""
        strengths = []
        
        if metrics.readability_score >= 0.8:
            strengths.append("높은 가독성")
        if metrics.factual_accuracy >= 0.8:
            strengths.append("높은 팩트 정확성")
        if metrics.bias_score <= 0.3:
            strengths.append("낮은 편향성")
        if metrics.source_credibility >= 0.8:
            strengths.append("신뢰할 수 있는 출처")
        if metrics.freshness_score >= 0.8:
            strengths.append("최신 정보")
        if metrics.engagement_potential >= 0.8:
            strengths.append("높은 참여도 잠재력")
            
        return strengths
    
    def _identify_weaknesses(self, metrics: QualityMetrics) -> List[str]:
        """약점 식별"""
        weaknesses = []
        
        if metrics.readability_score < 0.5:
            weaknesses.append("낮은 가독성")
        if metrics.factual_accuracy < 0.6:
            weaknesses.append("팩트 정확성 부족")
        if metrics.bias_score > 0.7:
            weaknesses.append("높은 편향성")
        if metrics.source_credibility < 0.5:
            weaknesses.append("출처 신뢰성 부족")
        if metrics.freshness_score < 0.3:
            weaknesses.append("오래된 정보")
        if metrics.engagement_potential < 0.4:
            weaknesses.append("낮은 참여도")
            
        return weaknesses
    
    def _generate_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        if metrics.readability_score < 0.6:
            recommendations.append("문장 길이를 줄이고 복잡한 용어를 간단히 설명하세요")
        if metrics.factual_accuracy < 0.7:
            recommendations.append("더 많은 팩트와 데이터를 포함하세요")
        if metrics.bias_score > 0.5:
            recommendations.append("중립적인 표현을 사용하고 감정적 언어를 줄이세요")
        if metrics.source_credibility < 0.6:
            recommendations.append("더 신뢰할 수 있는 출처를 인용하세요")
        if metrics.engagement_potential < 0.5:
            recommendations.append("더 흥미로운 제목과 내용을 작성하세요")
            
        return recommendations 