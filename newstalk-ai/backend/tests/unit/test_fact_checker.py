"""
Unit Tests for Fact Checker Agent - Stage 9
Tests for 95% fact-checking accuracy target
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, List, Any
import psutil
import time

from langgraph.agents.fact_checker import FactCheckerAgent
from api.utils.fact_verification import FactVerificationService
from backend.shared.models.fact_check import FactCheckResult, ClaimVerification


@pytest.mark.unit
class TestFactCheckerAgent:
    """Test suite for FactCheckerAgent"""

    @pytest.mark.asyncio
    async def test_fact_checker_initialization(self, fact_checker_agent):
        """Test fact checker agent initialization"""
        assert fact_checker_agent is not None
        assert hasattr(fact_checker_agent, 'verify_claims')
        assert hasattr(fact_checker_agent, 'confidence_threshold')

    @pytest.mark.asyncio
    async def test_simple_fact_verification(self, fact_checker_agent, sample_fact_check_data):
        """Test basic fact verification functionality"""
        claims = sample_fact_check_data["claims"]
        
        for claim_data in claims:
            result = await fact_checker_agent.verify_claim(claim_data["claim"])
            
            assert isinstance(result, FactCheckResult)
            assert result.verdict == claim_data["expected_result"]
            assert result.confidence >= 0.85  # Minimum confidence threshold
            assert len(result.sources) > 0

    @pytest.mark.asyncio
    async def test_batch_fact_verification(self, fact_checker_agent):
        """Test batch processing of fact verification"""
        claims = [
            "The Earth is round",
            "Water boils at 100°C at sea level",
            "The Great Wall of China is visible from space",
            "Humans have 46 chromosomes",
            "Lightning never strikes the same place twice"
        ]
        
        results = await fact_checker_agent.verify_claims_batch(claims, batch_size=3)
        
        assert len(results) == len(claims)
        assert all(isinstance(result, FactCheckResult) for result in results)
        assert all(result.confidence >= 0.0 for result in results)

    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, fact_checker_agent):
        """Test confidence threshold filtering"""
        claim = "This is an ambiguous claim that might be hard to verify"
        
        result = await fact_checker_agent.verify_claim(claim, confidence_threshold=0.9)
        
        if result.confidence < 0.9:
            assert result.needs_human_review is True
        else:
            assert result.verdict is not None

    @pytest.mark.asyncio
    async def test_source_verification(self, fact_checker_agent):
        """Test source verification and credibility"""
        claim = "NASA has confirmed that water exists on Mars"
        
        result = await fact_checker_agent.verify_claim(claim)
        
        assert len(result.sources) > 0
        assert any("NASA" in source or "scientific" in source.lower() 
                  for source in result.sources)
        assert result.source_credibility_score >= 0.8

    @pytest.mark.asyncio
    async def test_fact_checking_accuracy_benchmark(self, fact_checker_agent):
        """Test fact-checking accuracy against comprehensive benchmark dataset"""
        # Expanded benchmark dataset for 95% accuracy target
        benchmark_claims = [
            # Scientific Facts (High Confidence Expected)
            {"claim": "The Pacific Ocean is the largest ocean", "truth": True},
            {"claim": "Mount Everest is the tallest mountain", "truth": True},
            {"claim": "Water boils at 100°C at sea level", "truth": True},
            {"claim": "The speed of light is approximately 299,792,458 m/s", "truth": True},
            {"claim": "DNA has a double helix structure", "truth": True},
            {"claim": "Humans have 46 chromosomes", "truth": True},
            {"claim": "The Earth's core is made entirely of iron", "truth": False},
            {"claim": "Bananas are berries", "truth": True},
            {"claim": "Water conducts electricity", "truth": True},
            {"claim": "The human brain uses only 10% of its capacity", "truth": False},
            
            # Historical Facts
            {"claim": "World War II ended in 1945", "truth": True},
            {"claim": "The Berlin Wall fell in 1989", "truth": True},
            {"claim": "Napoleon was born in Italy", "truth": False},
            {"claim": "The American Civil War ended in 1865", "truth": True},
            {"claim": "The Titanic sank in 1912", "truth": True},
            
            # Geographic Facts
            {"claim": "Australia is both a country and a continent", "truth": True},
            {"claim": "The Nile River flows through Egypt", "truth": True},
            {"claim": "Russia has coastlines on three oceans", "truth": True},
            {"claim": "The Amazon River is longer than the Nile", "truth": False},
            {"claim": "Mount Kilimanjaro is in Kenya", "truth": False},
            
            # Mathematical Facts
            {"claim": "Pi is approximately 3.14159", "truth": True},
            {"claim": "The square root of 144 is 12", "truth": True},
            {"claim": "Prime numbers are divisible only by 1 and themselves", "truth": True},
            {"claim": "Zero is a positive number", "truth": False},
            {"claim": "The sum of angles in a triangle is 180 degrees", "truth": True},
            
            # Common Misconceptions (Should be detected as false)
            {"claim": "The Great Wall of China is visible from space", "truth": False},
            {"claim": "Lightning never strikes the same place twice", "truth": False},
            {"claim": "Goldfish have a 3-second memory", "truth": False},
            {"claim": "You lose most of your body heat through your head", "truth": False},
            {"claim": "Cracking knuckles causes arthritis", "truth": False},
        ]
        
        correct_predictions = 0
        high_confidence_predictions = 0
        total_predictions = len(benchmark_claims)
        
        for claim_data in benchmark_claims:
            result = await fact_checker_agent.verify_claim(claim_data["claim"])
            
            if result.confidence >= 0.85:  # Only count high-confidence predictions
                high_confidence_predictions += 1
                if result.verdict == claim_data["truth"]:
                    correct_predictions += 1
        
        accuracy = correct_predictions / high_confidence_predictions if high_confidence_predictions > 0 else 0
        confidence_rate = high_confidence_predictions / total_predictions
        
        # Assert 95% accuracy target
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} below 95% target"
        assert confidence_rate >= 0.80, f"High confidence rate {confidence_rate:.2%} below 80% threshold"

    @pytest.mark.asyncio
    async def test_multilingual_fact_checking(self, fact_checker_agent):
        """Test fact-checking in multiple languages"""
        multilingual_claims = [
            {"claim": "La capitale de la France est Paris", "language": "fr", "truth": True},
            {"claim": "프랑스의 수도는 파리이다", "language": "ko", "truth": True},
            {"claim": "Die Hauptstadt von Deutschland ist Berlin", "language": "de", "truth": True},
            {"claim": "La capital de España es Madrid", "language": "es", "truth": True},
            {"claim": "東京是日本的首都", "language": "zh", "truth": True},
            {"claim": "Столица России - Москва", "language": "ru", "truth": True}
        ]
        
        correct_predictions = 0
        
        for claim_data in multilingual_claims:
            result = await fact_checker_agent.verify_claim(
                claim_data["claim"], 
                language=claim_data["language"]
            )
            
            if result.verdict == claim_data["truth"]:
                correct_predictions += 1
            assert result.confidence >= 0.75  # Slightly lower for non-English

        multilingual_accuracy = correct_predictions / len(multilingual_claims)
        assert multilingual_accuracy >= 0.85, f"Multilingual accuracy {multilingual_accuracy:.2%} below 85% threshold"

    @pytest.mark.asyncio
    async def test_real_time_fact_checking(self, fact_checker_agent):
        """Test real-time fact-checking performance"""
        claim = "The current president of the United States is Joe Biden"
        
        start_time = datetime.utcnow()
        result = await fact_checker_agent.verify_claim(claim, real_time=True)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Should complete within 5 seconds for real-time verification
        assert processing_time <= 5.0
        assert result.is_current_info
        assert result.last_updated is not None

    @pytest.mark.asyncio
    async def test_claim_extraction_from_article(self, fact_checker_agent, sample_news_article):
        """Test extracting verifiable claims from news articles"""
        article_content = sample_news_article["content"]
        
        claims = await fact_checker_agent.extract_claims(article_content)
        
        assert len(claims) > 0
        assert all(isinstance(claim, str) for claim in claims)
        assert all(len(claim.strip()) > 10 for claim in claims)  # Meaningful claims

    @pytest.mark.asyncio
    async def test_fact_check_caching(self, fact_checker_agent, test_redis_client):
        """Test fact-checking result caching"""
        claim = "The Earth is round"
        
        # First verification
        start_time = time.time()
        result1 = await fact_checker_agent.verify_claim(claim, use_cache=True)
        first_duration = time.time() - start_time
        
        # Second verification (should use cache)
        start_time = time.time()
        result2 = await fact_checker_agent.verify_claim(claim, use_cache=True)
        second_duration = time.time() - start_time
        
        assert result1.verdict == result2.verdict
        assert result1.confidence == result2.confidence
        assert result2.from_cache is True
        assert second_duration < first_duration * 0.5  # Cache should be significantly faster

    @pytest.mark.asyncio
    async def test_error_handling(self, fact_checker_agent):
        """Test error handling in fact verification"""
        # Test with empty claim
        result = await fact_checker_agent.verify_claim("")
        assert result.error is not None
        assert result.verdict is None
        
        # Test with very long claim
        long_claim = "This is a very long claim " * 100
        result = await fact_checker_agent.verify_claim(long_claim)
        assert result.error is not None or result.needs_human_review

        # Test with malformed input
        result = await fact_checker_agent.verify_claim(None)
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_bias_detection(self, fact_checker_agent):
        """Test bias detection in fact-checking sources"""
        biased_claims = [
            "Political party X is the best choice for the country",
            "Brand Y products are superior to all competitors",
            "Religion Z is the only true faith"
        ]
        
        for biased_claim in biased_claims:
            result = await fact_checker_agent.verify_claim(biased_claim)
            
            assert result.bias_detected is True
            assert result.confidence < 0.7  # Lower confidence for biased claims
            assert result.needs_human_review is True


@pytest.mark.unit
class TestFactVerificationService:
    """Test suite for FactVerificationService"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test fact verification service initialization"""
        service = FactVerificationService()
        assert service is not None
        assert hasattr(service, 'verify_against_sources')

    @pytest.mark.asyncio
    async def test_source_reliability_scoring(self):
        """Test source reliability scoring"""
        service = FactVerificationService()
        
        reliable_sources = ["reuters.com", "bbc.com", "nasa.gov", "who.int", "nature.com", "science.org"]
        unreliable_sources = ["fake-news-site.com", "conspiracy-blog.net", "unverified-source.xyz"]
        
        for source in reliable_sources:
            score = await service.get_source_reliability_score(source)
            assert score >= 0.8, f"Reliable source {source} scored {score}, expected >= 0.8"
        
        for source in unreliable_sources:
            score = await service.get_source_reliability_score(source)
            assert score <= 0.5, f"Unreliable source {source} scored {score}, expected <= 0.5"

    @pytest.mark.asyncio
    async def test_cross_reference_verification(self):
        """Test cross-reference verification across multiple sources"""
        service = FactVerificationService()
        
        claim = "Water boils at 100°C at sea level"
        sources = ["physics-textbook.com", "encyclopedia.com", "scientific-journal.org"]
        
        result = await service.cross_reference_verify(claim, sources)
        
        assert result.consensus_reached is True
        assert result.agreement_percentage >= 0.9
        assert result.verdict is True

    @pytest.mark.asyncio
    async def test_temporal_fact_verification(self):
        """Test verification of time-sensitive facts"""
        service = FactVerificationService()
        
        # Current fact
        current_claim = "The current year is 2024"
        result = await service.verify_temporal_fact(current_claim)
        assert result.is_current is True
        
        # Historical fact
        historical_claim = "World War II ended in 1945"
        result = await service.verify_temporal_fact(historical_claim)
        assert result.is_historical is True
        assert result.verdict is True

    @pytest.mark.asyncio
    async def test_numerical_fact_verification(self):
        """Test verification of numerical claims"""
        service = FactVerificationService()
        
        numerical_claims = [
            {"claim": "The speed of light is 299,792,458 m/s", "expected": True},
            {"claim": "There are 366 days in a regular year", "expected": False},
            {"claim": "The human body has 206 bones", "expected": True},
            {"claim": "Pi equals exactly 3.14", "expected": False},
            {"claim": "The freezing point of water is 0°C", "expected": True}
        ]
        
        for claim_data in numerical_claims:
            result = await service.verify_numerical_claim(claim_data["claim"])
            assert result.verdict == claim_data["expected"]
            assert result.numerical_accuracy_score >= 0.9


@pytest.mark.unit
class TestFactCheckResult:
    """Test suite for FactCheckResult model"""

    def test_fact_check_result_creation(self):
        """Test FactCheckResult model creation"""
        result = FactCheckResult(
            claim="Test claim",
            verdict=True,
            confidence=0.95,
            sources=["source1.com", "source2.org"],
            reasoning="Clear scientific consensus"
        )
        
        assert result.claim == "Test claim"
        assert result.verdict is True
        assert result.confidence == 0.95
        assert len(result.sources) == 2

    def test_fact_check_result_validation(self):
        """Test FactCheckResult validation"""
        # Test confidence range validation
        with pytest.raises(ValueError):
            FactCheckResult(
                claim="Test",
                verdict=True,
                confidence=1.5,  # Invalid confidence
                sources=[],
                reasoning=""
            )
        
        # Test required fields
        with pytest.raises(ValueError):
            FactCheckResult(
                claim="",  # Empty claim
                verdict=True,
                confidence=0.9,
                sources=[],
                reasoning=""
            )

    def test_fact_check_result_serialization(self):
        """Test FactCheckResult serialization"""
        result = FactCheckResult(
            claim="Test claim",
            verdict=True,
            confidence=0.95,
            sources=["source1.com"],
            reasoning="Test reasoning"
        )
        
        serialized = result.to_dict()
        
        assert isinstance(serialized, dict)
        assert serialized["claim"] == "Test claim"
        assert serialized["verdict"] is True
        assert serialized["confidence"] == 0.95

    def test_fact_check_quality_score(self):
        """Test fact check quality score calculation"""
        high_quality_result = FactCheckResult(
            claim="Well-established scientific fact",
            verdict=True,
            confidence=0.98,
            sources=["nasa.gov", "nature.com", "science.org"],
            reasoning="Multiple authoritative sources confirm",
            source_credibility_score=0.95
        )
        
        quality_score = high_quality_result.calculate_quality_score()
        assert quality_score >= 0.9
        
        low_quality_result = FactCheckResult(
            claim="Ambiguous claim",
            verdict=None,
            confidence=0.6,
            sources=["unknown-blog.com"],
            reasoning="Insufficient evidence",
            source_credibility_score=0.3
        )
        
        quality_score = low_quality_result.calculate_quality_score()
        assert quality_score <= 0.5


@pytest.mark.unit
class TestFactCheckingPerformance:
    """Test suite for fact-checking performance metrics"""

    @pytest.mark.asyncio
    async def test_concurrent_fact_checking(self, fact_checker_agent):
        """Test concurrent fact-checking performance"""
        claims = [
            "The Earth is round",
            "Water boils at 100°C",
            "The Pacific Ocean is the largest ocean",
            "Mount Everest is the tallest mountain",
            "Humans have 46 chromosomes"
        ]
        
        # Test concurrent processing
        start_time = time.time()
        tasks = [fact_checker_agent.verify_claim(claim) for claim in claims]
        results = await asyncio.gather(*tasks)
        concurrent_duration = time.time() - start_time
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        for claim in claims:
            result = await fact_checker_agent.verify_claim(claim)
            sequential_results.append(result)
        sequential_duration = time.time() - start_time
        
        # Concurrent should be faster
        assert concurrent_duration < sequential_duration * 0.8
        assert len(results) == len(claims)
        assert all(isinstance(result, FactCheckResult) for result in results)

    @pytest.mark.asyncio
    async def test_memory_usage_during_fact_checking(self, fact_checker_agent):
        """Test memory usage during fact-checking operations"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple claims
        claims = ["Test claim " + str(i) for i in range(50)]
        results = await fact_checker_agent.verify_claims_batch(claims, batch_size=10)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 50 claims)
        assert memory_increase < 100, f"Memory usage increased by {memory_increase:.2f}MB"
        assert len(results) == len(claims)

    @pytest.mark.asyncio
    async def test_fact_checking_throughput(self, fact_checker_agent):
        """Test fact-checking throughput"""
        claims = ["Simple factual claim " + str(i) for i in range(100)]
        
        start_time = datetime.utcnow()
        results = await fact_checker_agent.verify_claims_batch(claims, batch_size=10)
        end_time = datetime.utcnow()
        
        processing_time = (end_time - start_time).total_seconds()
        throughput = len(claims) / processing_time
        
        # Should achieve at least 10 claims per second
        assert throughput >= 10.0, f"Throughput {throughput:.2f} claims/sec below 10.0 threshold"
        assert len(results) == len(claims)

    @pytest.mark.asyncio
    async def test_accuracy_under_load(self, fact_checker_agent):
        """Test fact-checking accuracy under high load"""
        # High-confidence facts that should maintain accuracy under load
        high_confidence_claims = [
            {"claim": "Water freezes at 0°C", "truth": True},
            {"claim": "The Earth has two moons", "truth": False},
            {"claim": "Humans need oxygen to survive", "truth": True},
            {"claim": "The sun rises in the west", "truth": False},
            {"claim": "Paris is the capital of France", "truth": True}
        ] * 20  # 100 total claims
        
        # Process all claims concurrently to simulate load
        tasks = [
            fact_checker_agent.verify_claim(claim_data["claim"]) 
            for claim_data in high_confidence_claims
        ]
        results = await asyncio.gather(*tasks)
        
        # Calculate accuracy
        correct_predictions = 0
        high_confidence_count = 0
        
        for i, result in enumerate(results):
            expected_truth = high_confidence_claims[i]["truth"]
            if result.confidence >= 0.85:
                high_confidence_count += 1
                if result.verdict == expected_truth:
                    correct_predictions += 1
        
        accuracy = correct_predictions / high_confidence_count if high_confidence_count > 0 else 0
        
        # Accuracy should remain high even under load
        assert accuracy >= 0.95, f"Accuracy under load {accuracy:.2%} below 95% target"
        assert high_confidence_count / len(results) >= 0.80, "Too few high-confidence predictions under load" 