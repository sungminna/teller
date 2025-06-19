"""
🔍 Fact Verification Utilities
===============================

API에서 사용하는 팩트 검증 유틸리티
"""

from backend.services.fact_verification_service import FactVerificationService, FactCheckResult
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# 전역 팩트 검증 서비스 인스턴스
_fact_service = FactVerificationService()


async def verify_content_facts(content: str) -> List[FactCheckResult]:
    """
    콘텐츠 팩트 검증
    
    Args:
        content: 검증할 콘텐츠
        
    Returns:
        List[FactCheckResult]: 팩트체크 결과 목록
    """
    return await _fact_service.verify_facts(content)


async def get_content_credibility(content: str) -> Dict[str, Any]:
    """
    콘텐츠 신뢰도 평가
    
    Args:
        content: 평가할 콘텐츠
        
    Returns:
        Dict[str, Any]: 신뢰도 평가 결과
    """
    fact_results = await _fact_service.verify_facts(content)
    return _fact_service.get_overall_credibility(fact_results)


async def real_time_fact_check(claim: str) -> FactCheckResult:
    """
    실시간 팩트체크
    
    Args:
        claim: 체크할 클레임
        
    Returns:
        FactCheckResult: 팩트체크 결과
    """
    return await _fact_service.real_time_fact_check(claim)


def is_credible_content(fact_results: List[FactCheckResult]) -> bool:
    """
    콘텐츠 신뢰성 판단
    
    Args:
        fact_results: 팩트체크 결과 목록
        
    Returns:
        bool: 신뢰성 여부
    """
    return _fact_service.is_credible_content(fact_results) 