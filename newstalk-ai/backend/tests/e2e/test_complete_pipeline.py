"""
End-to-End Tests for Complete NewsTalk AI Pipeline - Stage 9
Tests for complete system functionality from news ingestion to user delivery
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from api.main import app
from backend.shared.models.user import User
from backend.shared.models.news_article import NewsArticle
from backend.shared.models.personalized_content import PersonalizedContent


@pytest.mark.e2e
class TestCompleteNewsTalkPipeline:
    """End-to-end tests for complete NewsTalk AI pipeline"""

    @pytest.mark.asyncio
    async def test_complete_user_journey(self, test_client, test_db_session, test_redis_client):
        """Test complete user journey from registration to content consumption"""
        
        # Step 1: User Registration
        user_data = {
            "email": "test@example.com",
            "password": "securepassword123",
            "preferences": {
                "categories": ["technology", "science"],
                "languages": ["en"],
                "voice_preference": "female",
                "reading_speed": "medium"
            }
        }
        
        response = await test_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        user_id = response.json()["user_id"]
        
        # Step 2: User Login
        login_data = {
            "email": "test@example.com",
            "password": "securepassword123"
        }
        
        response = await test_client.post("/api/v1/auth/login", json=login_data)
        assert response.status_code == 200
        access_token = response.json()["access_token"]
        
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Step 3: News Collection (Simulated)
        with patch('backend.services.news_collector.collect_news') as mock_collect:
            mock_articles = [
                {
                    "title": "AI Breakthrough in Medical Diagnosis",
                    "content": "Scientists have developed an AI system that can diagnose rare diseases with 98% accuracy.",
                    "url": "https://example.com/ai-medical-breakthrough",
                    "source": "Medical Journal",
                    "category": "technology",
                    "published_at": datetime.utcnow().isoformat()
                },
                {
                    "title": "New Quantum Computing Milestone",
                    "content": "Researchers achieve quantum supremacy in protein folding simulations.",
                    "url": "https://example.com/quantum-breakthrough",
                    "source": "Science Daily",
                    "category": "science",
                    "published_at": datetime.utcnow().isoformat()
                }
            ]
            mock_collect.return_value = mock_articles
            
            # Trigger news collection
            response = await test_client.post(
                "/api/v1/admin/collect-news",
                headers=headers,
                json={"force_refresh": True}
            )
            assert response.status_code == 200
        
        # Step 4: Content Analysis and Fact-Checking
        await asyncio.sleep(2)  # Allow processing time
        
        # Check processed articles
        response = await test_client.get("/api/v1/news/articles", headers=headers)
        assert response.status_code == 200
        articles = response.json()["articles"]
        assert len(articles) > 0
        
        # Verify fact-checking results
        for article in articles:
            assert "fact_check" in article
            assert article["fact_check"]["confidence"] >= 0.85
            assert "analysis" in article
            assert "sentiment" in article["analysis"]
            assert "keywords" in article["analysis"]
        
        # Step 5: Personalized Content Generation
        response = await test_client.get("/api/v1/news/personalized", headers=headers)
        assert response.status_code == 200
        personalized_content = response.json()["content"]
        assert len(personalized_content) > 0
        
        # Verify personalization
        for content in personalized_content:
            assert content["category"] in user_data["preferences"]["categories"]
            assert "relevance_score" in content
            assert content["relevance_score"] >= 0.7
        
        # Step 6: Voice Content Generation
        article_id = personalized_content[0]["id"]
        response = await test_client.post(
            f"/api/v1/voice/generate/{article_id}",
            headers=headers,
            json={"voice_settings": user_data["preferences"]["voice_preference"]}
        )
        assert response.status_code == 200
        voice_content = response.json()
        assert "audio_url" in voice_content
        assert "duration" in voice_content
        assert voice_content["quality_score"] >= 0.9
        
        # Step 7: Content Consumption Tracking
        response = await test_client.post(
            f"/api/v1/analytics/track",
            headers=headers,
            json={
                "event": "content_consumed",
                "article_id": article_id,
                "duration": 120,
                "completion_rate": 0.95
            }
        )
        assert response.status_code == 200
        
        # Step 8: User Feedback
        response = await test_client.post(
            f"/api/v1/feedback",
            headers=headers,
            json={
                "article_id": article_id,
                "rating": 5,
                "feedback_type": "quality",
                "comments": "Excellent content and voice quality"
            }
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_real_time_news_processing(self, test_client, test_kafka_producer, test_redis_client):
        """Test real-time news processing pipeline"""
        
        # Step 1: Simulate breaking news event
        breaking_news = {
            "id": "breaking_news_001",
            "title": "Major Scientific Discovery Announced",
            "content": "Scientists at MIT have announced a breakthrough in fusion energy technology.",
            "source": "MIT News",
            "category": "science",
            "priority": "high",
            "published_at": datetime.utcnow().isoformat(),
            "is_breaking": True
        }
        
        # Step 2: Publish to Kafka
        test_kafka_producer.send('news_articles', value=json.dumps(breaking_news))
        test_kafka_producer.flush()
        
        # Step 3: Wait for processing
        await asyncio.sleep(5)
        
        # Step 4: Verify article processed and available
        response = await test_client.get(f"/api/v1/news/article/{breaking_news['id']}")
        assert response.status_code == 200
        
        processed_article = response.json()
        assert processed_article["is_breaking"] is True
        assert "fact_check" in processed_article
        assert "analysis" in processed_article
        
        # Step 5: Verify real-time notifications sent
        # Check Redis for notification queue
        notifications = await test_redis_client.lrange("breaking_news_notifications", 0, -1)
        assert len(notifications) > 0
        
        notification = json.loads(notifications[0])
        assert notification["article_id"] == breaking_news["id"]
        assert notification["priority"] == "high"

    @pytest.mark.asyncio
    async def test_multilingual_content_processing(self, test_client, test_db_session):
        """Test multilingual content processing and delivery"""
        
        # Step 1: Create multilingual user
        user_data = {
            "email": "multilingual@example.com",
            "password": "password123",
            "preferences": {
                "categories": ["technology"],
                "languages": ["en", "ko", "es"],
                "voice_preference": "female"
            }
        }
        
        response = await test_client.post("/api/v1/auth/register", json=user_data)
        assert response.status_code == 201
        
        # Login
        login_response = await test_client.post("/api/v1/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}
        
        # Step 2: Process multilingual articles
        multilingual_articles = [
            {
                "title": "AI Technology Advances",
                "content": "Artificial intelligence continues to evolve rapidly.",
                "language": "en",
                "category": "technology"
            },
            {
                "title": "AI 기술 발전",
                "content": "인공지능 기술이 빠르게 발전하고 있습니다.",
                "language": "ko",
                "category": "technology"
            },
            {
                "title": "Avances en Tecnología IA",
                "content": "La inteligencia artificial continúa evolucionando rápidamente.",
                "language": "es",
                "category": "technology"
            }
        ]
        
        with patch('backend.services.news_collector.collect_news') as mock_collect:
            mock_collect.return_value = multilingual_articles
            
            response = await test_client.post("/api/v1/admin/collect-news", headers=headers)
            assert response.status_code == 200
        
        # Step 3: Get personalized content in multiple languages
        for language in ["en", "ko", "es"]:
            response = await test_client.get(
                f"/api/v1/news/personalized?language={language}",
                headers=headers
            )
            assert response.status_code == 200
            
            content = response.json()["content"]
            assert len(content) > 0
            assert all(article["language"] == language for article in content)

    @pytest.mark.asyncio
    async def test_quality_assurance_pipeline(self, test_client, quality_thresholds):
        """Test quality assurance throughout the pipeline"""
        
        # Step 1: Process articles with varying quality
        test_articles = [
            {
                "title": "High Quality Scientific Article",
                "content": "Well-researched article with multiple credible sources and clear scientific methodology.",
                "source": "Nature Journal",
                "category": "science",
                "expected_quality": "high"
            },
            {
                "title": "Medium Quality News",
                "content": "Standard news article with basic information.",
                "source": "General News",
                "category": "general",
                "expected_quality": "medium"
            },
            {
                "title": "Low Quality Content",
                "content": "Poorly written article with questionable claims.",
                "source": "Unknown Blog",
                "category": "opinion",
                "expected_quality": "low"
            }
        ]
        
        with patch('backend.services.news_collector.collect_news') as mock_collect:
            mock_collect.return_value = test_articles
            
            response = await test_client.post("/api/v1/admin/collect-news")
            assert response.status_code == 200
        
        # Step 2: Check quality filtering
        response = await test_client.get("/api/v1/news/articles?quality_filter=high")
        assert response.status_code == 200
        
        high_quality_articles = response.json()["articles"]
        for article in high_quality_articles:
            assert article["quality_score"] >= quality_thresholds["content_relevance"]
            assert article["fact_check"]["confidence"] >= quality_thresholds["fact_checking_accuracy"]
        
        # Step 3: Verify quality metrics
        response = await test_client.get("/api/v1/admin/quality-metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert metrics["fact_checking_accuracy"] >= quality_thresholds["fact_checking_accuracy"]
        assert metrics["voice_quality_average"] >= quality_thresholds["voice_quality_score"]
        assert metrics["user_satisfaction"] >= quality_thresholds["user_satisfaction"]

    @pytest.mark.asyncio
    async def test_ab_testing_system(self, test_client, test_db_session):
        """Test A/B testing system for content optimization"""
        
        # Step 1: Create test users for A/B testing
        test_users = []
        for i in range(10):
            user_data = {
                "email": f"testuser{i}@example.com",
                "password": "password123",
                "preferences": {
                    "categories": ["technology"],
                    "languages": ["en"]
                }
            }
            
            response = await test_client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == 201
            test_users.append(response.json())
        
        # Step 2: Set up A/B test
        ab_test_config = {
            "test_name": "personalization_algorithm_v2",
            "description": "Testing new personalization algorithm",
            "variants": [
                {"name": "control", "weight": 0.5},
                {"name": "treatment", "weight": 0.5}
            ],
            "metrics": ["engagement_rate", "user_satisfaction", "content_relevance"],
            "duration_days": 7
        }
        
        response = await test_client.post("/api/v1/admin/ab-tests", json=ab_test_config)
        assert response.status_code == 201
        test_id = response.json()["test_id"]
        
        # Step 3: Simulate user interactions
        engagement_data = {}
        
        for user in test_users:
            # Login user
            login_response = await test_client.post("/api/v1/auth/login", json={
                "email": user["email"],
                "password": "password123"
            })
            headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}
            
            # Get personalized content (will be assigned to A/B test variant)
            response = await test_client.get("/api/v1/news/personalized", headers=headers)
            assert response.status_code == 200
            
            content = response.json()
            variant = content.get("ab_test_variant")
            assert variant in ["control", "treatment"]
            
            # Track engagement
            engagement_data[user["user_id"]] = {
                "variant": variant,
                "engagement_rate": 0.7 + (0.1 if variant == "treatment" else 0),
                "satisfaction": 4.2 + (0.3 if variant == "treatment" else 0)
            }
        
        # Step 4: Analyze A/B test results
        response = await test_client.get(f"/api/v1/admin/ab-tests/{test_id}/results")
        assert response.status_code == 200
        
        results = response.json()
        assert "statistical_significance" in results
        assert "variant_performance" in results
        assert len(results["variant_performance"]) == 2

    @pytest.mark.asyncio
    async def test_system_resilience_and_recovery(self, test_client, test_kafka_producer):
        """Test system resilience and recovery from failures"""
        
        # Step 1: Test database failure recovery
        with patch('api.database.connection.get_db_session') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = await test_client.get("/api/v1/news/articles")
            # Should return cached content or graceful error
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                # Verify content served from cache
                assert "from_cache" in response.headers
        
        # Step 2: Test high load handling
        concurrent_requests = []
        for i in range(100):
            concurrent_requests.append(
                test_client.get("/api/v1/news/articles")
            )
        
        responses = await asyncio.gather(*concurrent_requests, return_exceptions=True)
        
        # Most requests should succeed
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        success_rate = len(successful_responses) / len(responses)
        assert success_rate >= 0.95  # 95% success rate under load

    @pytest.mark.asyncio
    async def test_user_satisfaction_monitoring(self, test_client, test_db_session):
        """Test continuous user satisfaction monitoring"""
        
        # Step 1: Create user and generate content
        user_data = {
            "email": "satisfaction@example.com",
            "password": "password123",
            "preferences": {"categories": ["technology"]}
        }
        
        response = await test_client.post("/api/v1/auth/register", json=user_data)
        user_id = response.json()["user_id"]
        
        login_response = await test_client.post("/api/v1/auth/login", json={
            "email": user_data["email"],
            "password": user_data["password"]
        })
        headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}
        
        # Step 2: Simulate user interactions with varying satisfaction
        interactions = [
            {"article_id": "article_1", "rating": 5, "engagement_time": 180},
            {"article_id": "article_2", "rating": 4, "engagement_time": 120},
            {"article_id": "article_3", "rating": 3, "engagement_time": 60},
            {"article_id": "article_4", "rating": 5, "engagement_time": 200},
        ]
        
        for interaction in interactions:
            # Submit feedback
            await test_client.post("/api/v1/feedback", headers=headers, json={
                "article_id": interaction["article_id"],
                "rating": interaction["rating"],
                "feedback_type": "quality"
            })
            
            # Track engagement
            await test_client.post("/api/v1/analytics/track", headers=headers, json={
                "event": "content_consumed",
                "article_id": interaction["article_id"],
                "duration": interaction["engagement_time"]
            })
        
        # Step 3: Check satisfaction metrics
        response = await test_client.get("/api/v1/admin/satisfaction-metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert metrics["average_rating"] >= 4.0
        assert metrics["user_retention_rate"] >= 0.8
        assert "satisfaction_trends" in metrics

    @pytest.mark.asyncio
    async def test_content_freshness_and_relevance(self, test_client, test_redis_client):
        """Test content freshness and relevance over time"""
        
        # Step 1: Create articles with different timestamps
        current_time = datetime.utcnow()
        articles = [
            {
                "id": "fresh_article",
                "title": "Latest AI Development",
                "content": "Recent breakthrough in AI technology",
                "published_at": current_time.isoformat(),
                "category": "technology"
            },
            {
                "id": "old_article",
                "title": "Old Tech News",
                "content": "Outdated technology information",
                "published_at": (current_time - timedelta(days=30)).isoformat(),
                "category": "technology"
            }
        ]
        
        # Step 2: Process articles
        with patch('backend.services.news_collector.collect_news') as mock_collect:
            mock_collect.return_value = articles
            
            response = await test_client.post("/api/v1/admin/collect-news")
            assert response.status_code == 200
        
        # Step 3: Check content ranking by freshness
        response = await test_client.get("/api/v1/news/articles?sort=freshness")
        assert response.status_code == 200
        
        sorted_articles = response.json()["articles"]
        assert len(sorted_articles) >= 2
        
        # Fresh article should rank higher
        fresh_article = next(a for a in sorted_articles if a["id"] == "fresh_article")
        old_article = next(a for a in sorted_articles if a["id"] == "old_article")
        
        fresh_index = sorted_articles.index(fresh_article)
        old_index = sorted_articles.index(old_article)
        assert fresh_index < old_index
        
        # Step 4: Verify relevance scoring
        assert fresh_article["relevance_score"] > old_article["relevance_score"]
        assert fresh_article["freshness_score"] > old_article["freshness_score"]

    @pytest.mark.asyncio
    async def test_comprehensive_error_scenarios(self, test_client, test_kafka_producer):
        """Test comprehensive error scenarios and graceful degradation"""
        
        error_scenarios = [
            {
                "name": "malformed_article_data",
                "data": {"title": "", "content": None, "invalid_field": "test"},
                "expected_behavior": "reject_with_validation_error"
            },
            {
                "name": "extremely_long_content",
                "data": {
                    "title": "Long Article",
                    "content": "Very long content " * 10000,  # Extremely long
                    "source": "Test"
                },
                "expected_behavior": "truncate_or_reject"
            },
            {
                "name": "special_characters_content",
                "data": {
                    "title": "Special Characters: 🚀🎉💡",
                    "content": "Content with émojis and spëcial çharactërs",
                    "source": "Unicode Test"
                },
                "expected_behavior": "process_correctly"
            }
        ]
        
        for scenario in error_scenarios:
            test_kafka_producer.send('news_articles', value=json.dumps(scenario["data"]))
            test_kafka_producer.flush()
            
            await asyncio.sleep(2)
            
            # Check system response
            response = await test_client.get("/api/v1/admin/processing-status")
            assert response.status_code == 200
            
            status = response.json()
            if scenario["expected_behavior"] == "reject_with_validation_error":
                assert status["validation_errors"] > 0
            elif scenario["expected_behavior"] == "process_correctly":
                assert status["successful_processing"] > 0 