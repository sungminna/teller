"""
🔍 Fact Verification Service
============================

뉴스 콘텐츠의 팩트를 검증하는 서비스
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class FactCheckResult:
    """팩트체크 결과 데이터 클래스"""
    
    claim: str
    verdict: str  # "TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIED"
    confidence: float
    sources: List[str]
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "sources": self.sources,
            "explanation": self.explanation
        }


class FactVerificationService:
    """팩트 검증 서비스"""
    
    def __init__(self):
        self.confidence_threshold = 0.7
        self.known_facts = {
            "earth is round": {"verdict": "TRUE", "confidence": 1.0},
            "water boils at 100 celsius": {"verdict": "TRUE", "confidence": 1.0},
            "humans can fly without assistance": {"verdict": "FALSE", "confidence": 1.0}
        }
        
    async def verify_facts(self, content: str) -> List[FactCheckResult]:
        """
        콘텐츠의 팩트를 검증
        
        Args:
            content: 검증할 콘텐츠
            
        Returns:
            List[FactCheckResult]: 팩트체크 결과 목록
        """
        try:
            # 클레임 추출
            claims = self._extract_claims(content)
            
            # 각 클레임 검증
            results = []
            for claim in claims:
                result = await self._verify_single_claim(claim)
                results.append(result)
                
            logger.info(f"Fact verification completed for {len(claims)} claims")
            return results
            
        except Exception as e:
            logger.error(f"Fact verification failed: {str(e)}")
            return []
    
    def _extract_claims(self, content: str) -> List[str]:
        """
        콘텐츠에서 팩트체크 가능한 클레임 추출
        
        Args:
            content: 콘텐츠 텍스트
            
        Returns:
            List[str]: 추출된 클레임 목록
        """
        # 간단한 클레임 추출 로직
        sentences = content.split('.')
        
        # 팩트성 문장 식별 키워드
        fact_indicators = [
            "according to", "statistics show", "research indicates",
            "study found", "data reveals", "reports that", "confirmed that"
        ]
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # 너무 짧은 문장 제외
                # 팩트성 지표가 포함된 문장 우선 선택
                if any(indicator in sentence.lower() for indicator in fact_indicators):
                    claims.append(sentence)
                elif len(claims) < 3:  # 최대 3개까지 추가 클레임 포함
                    claims.append(sentence)
                    
        return claims[:5]  # 최대 5개 클레임
    
    async def _verify_single_claim(self, claim: str) -> FactCheckResult:
        """
        단일 클레임 검증
        
        Args:
            claim: 검증할 클레임
            
        Returns:
            FactCheckResult: 팩트체크 결과
        """
        # 시뮬레이션된 검증 프로세스
        await asyncio.sleep(0.1)  # API 호출 시뮬레이션
        
        # 알려진 팩트 확인
        claim_lower = claim.lower()
        for known_fact, fact_data in self.known_facts.items():
            if known_fact in claim_lower:
                return FactCheckResult(
                    claim=claim,
                    verdict=fact_data["verdict"],
                    confidence=fact_data["confidence"],
                    sources=["Known Fact Database"],
                    explanation=f"This claim matches a verified fact in our database."
                )
        
        # 기본 휴리스틱 기반 검증
        verdict, confidence = self._heuristic_verification(claim)
        
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            sources=self._get_mock_sources(),
            explanation=self._generate_explanation(claim, verdict, confidence)
        )
    
    def _heuristic_verification(self, claim: str) -> tuple[str, float]:
        """
        휴리스틱 기반 검증
        
        Args:
            claim: 검증할 클레임
            
        Returns:
            tuple[str, float]: (verdict, confidence)
        """
        claim_lower = claim.lower()
        
        # 신뢰할 수 있는 지표들
        reliable_indicators = [
            "according to official", "government data", "peer-reviewed study",
            "scientific research", "official statistics", "verified by"
        ]
        
        # 의심스러운 지표들
        suspicious_indicators = [
            "some say", "it is believed", "rumors suggest",
            "unconfirmed reports", "allegedly", "supposedly"
        ]
        
        # 과장된 표현들
        exaggerated_indicators = [
            "100% effective", "completely safe", "never fails",
            "always works", "miracle", "revolutionary breakthrough"
        ]
        
        # 점수 계산
        reliability_score = sum(1 for indicator in reliable_indicators if indicator in claim_lower)
        suspicion_score = sum(1 for indicator in suspicious_indicators if indicator in claim_lower)
        exaggeration_score = sum(1 for indicator in exaggerated_indicators if indicator in claim_lower)
        
        # 숫자나 통계가 포함되어 있는지 확인
        has_numbers = any(char.isdigit() for char in claim)
        
        # 판정 로직
        if reliability_score > 0 and suspicion_score == 0:
            if has_numbers:
                return "TRUE", 0.8
            else:
                return "PARTIALLY_TRUE", 0.7
        elif suspicion_score > 0 or exaggeration_score > 0:
            return "UNVERIFIED", 0.4
        elif has_numbers and len(claim.split()) > 10:
            return "PARTIALLY_TRUE", 0.6
        else:
            return "UNVERIFIED", 0.5
    
    def _get_mock_sources(self) -> List[str]:
        """모의 소스 목록 반환"""
        return [
            "Reuters Fact Check Database",
            "Associated Press Verification",
            "Scientific Journal Database",
            "Government Statistics Portal"
        ]
    
    def _generate_explanation(self, claim: str, verdict: str, confidence: float) -> str:
        """
        검증 결과 설명 생성
        
        Args:
            claim: 원본 클레임
            verdict: 판정 결과
            confidence: 신뢰도
            
        Returns:
            str: 설명 텍스트
        """
        explanations = {
            "TRUE": f"This claim has been verified with {confidence:.1%} confidence based on reliable sources and factual indicators.",
            "FALSE": f"This claim contradicts verified information with {confidence:.1%} confidence.",
            "PARTIALLY_TRUE": f"This claim contains some accurate elements but may lack complete context or precision. Confidence: {confidence:.1%}.",
            "UNVERIFIED": f"Insufficient reliable evidence to verify this claim. Confidence in assessment: {confidence:.1%}."
        }
        
        return explanations.get(verdict, "Unable to determine the veracity of this claim.")
    
    def get_overall_credibility(self, results: List[FactCheckResult]) -> Dict[str, Any]:
        """
        전체 콘텐츠의 신뢰도 평가
        
        Args:
            results: 팩트체크 결과 목록
            
        Returns:
            Dict[str, Any]: 전체 신뢰도 평가
        """
        if not results:
            return {
                "overall_credibility": 0.5,
                "grade": "C",
                "summary": "No verifiable claims found",
                "recommendations": ["Add more factual content with verifiable sources"]
            }
        
        # 점수 계산
        total_score = 0
        for result in results:
            if result.verdict == "TRUE":
                total_score += result.confidence
            elif result.verdict == "PARTIALLY_TRUE":
                total_score += result.confidence * 0.7
            elif result.verdict == "FALSE":
                total_score -= result.confidence * 0.5
            # UNVERIFIED는 중립적으로 처리
        
        overall_credibility = max(0, min(1, total_score / len(results)))
        
        # 등급 계산
        grade = self._calculate_grade(overall_credibility)
        
        # 요약 및 권장사항
        summary = self._generate_summary(results, overall_credibility)
        recommendations = self._generate_recommendations(results, overall_credibility)
        
        return {
            "overall_credibility": overall_credibility,
            "grade": grade,
            "summary": summary,
            "recommendations": recommendations,
            "total_claims_checked": len(results),
            "verified_claims": len([r for r in results if r.verdict == "TRUE"]),
            "false_claims": len([r for r in results if r.verdict == "FALSE"]),
            "unverified_claims": len([r for r in results if r.verdict == "UNVERIFIED"])
        }
    
    def _calculate_grade(self, credibility: float) -> str:
        """신뢰도 점수를 등급으로 변환"""
        if credibility >= 0.9:
            return "A+"
        elif credibility >= 0.8:
            return "A"
        elif credibility >= 0.7:
            return "B"
        elif credibility >= 0.6:
            return "C"
        else:
            return "D"
    
    def _generate_summary(self, results: List[FactCheckResult], credibility: float) -> str:
        """신뢰도 요약 생성"""
        if credibility >= 0.8:
            return "High credibility content with well-verified claims"
        elif credibility >= 0.6:
            return "Moderately credible content with some verified information"
        elif credibility >= 0.4:
            return "Mixed credibility with both verified and unverified claims"
        else:
            return "Low credibility content requiring additional verification"
    
    def _generate_recommendations(self, results: List[FactCheckResult], credibility: float) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        false_claims = [r for r in results if r.verdict == "FALSE"]
        unverified_claims = [r for r in results if r.verdict == "UNVERIFIED"]
        
        if false_claims:
            recommendations.append("Remove or correct false claims identified in the fact-check")
        
        if len(unverified_claims) > len(results) * 0.5:
            recommendations.append("Add more verifiable sources and citations")
        
        if credibility < 0.6:
            recommendations.append("Include more factual content from reliable sources")
            recommendations.append("Provide specific data and statistics to support claims")
        
        if not recommendations:
            recommendations.append("Continue maintaining high factual standards")
        
        return recommendations
    
    async def real_time_fact_check(self, claim: str) -> FactCheckResult:
        """
        실시간 팩트체크
        
        Args:
            claim: 실시간으로 체크할 클레임
            
        Returns:
            FactCheckResult: 팩트체크 결과
        """
        return await self._verify_single_claim(claim)
    
    def is_credible_content(self, results: List[FactCheckResult]) -> bool:
        """
        콘텐츠가 신뢰할 수 있는지 판단
        
        Args:
            results: 팩트체크 결과 목록
            
        Returns:
            bool: 신뢰성 여부
        """
        if not results:
            return False
            
        credibility = self.get_overall_credibility(results)
        return credibility["overall_credibility"] >= self.confidence_threshold 