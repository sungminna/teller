"""
Performance and Load Testing for NewsTalk AI - Stage 9
Comprehensive performance testing to ensure system scalability
"""

import asyncio
import os
import statistics
import time
from unittest.mock import patch

import psutil
import pytest
from locust import HttpUser, between, task
from locust.env import Environment


@pytest.mark.performance
class TestSystemPerformance:
    """Performance test suite for NewsTalk AI system"""

    @pytest.mark.asyncio
    async def test_api_response_times(self, test_client, performance_test_config):
        """Test API response times under normal load"""

        endpoints = performance_test_config["endpoints_to_test"]
        target_response_time = performance_test_config["target_response_time"]

        response_times = {}

        for endpoint in endpoints:
            times = []

            # Test each endpoint 50 times
            for _ in range(50):
                start_time = time.time()

                try:
                    response = await test_client.get(endpoint)
                    end_time = time.time()

                    if response.status_code == 200:
                        response_time = end_time - start_time
                        times.append(response_time)
                except Exception as e:
                    print(f"Error testing {endpoint}: {e}")

            if times:
                response_times[endpoint] = {
                    "average": statistics.mean(times),
                    "median": statistics.median(times),
                    "p95": sorted(times)[int(len(times) * 0.95)],
                    "p99": sorted(times)[int(len(times) * 0.99)],
                }

        # Verify performance requirements
        for endpoint, metrics in response_times.items():
            assert (
                metrics["p95"] <= target_response_time
            ), f"{endpoint} P95 response time {metrics['p95']:.2f}s exceeds target {target_response_time}s"
            assert (
                metrics["average"] <= target_response_time * 0.7
            ), f"{endpoint} average response time {metrics['average']:.2f}s too high"

    @pytest.mark.asyncio
    async def test_concurrent_user_load(self, test_client, performance_test_config):
        """Test system performance under concurrent user load"""

        concurrent_users = performance_test_config["concurrent_users"]
        test_duration = performance_test_config["test_duration"]
        max_error_rate = performance_test_config["max_error_rate"]

        async def simulate_user_session():
            """Simulate a typical user session"""
            try:
                # Login
                login_response = await test_client.post(
                    "/api/v1/auth/login",
                    json={"email": "testuser@example.com", "password": "password123"},
                )

                if login_response.status_code != 200:
                    return {"success": False, "error": "login_failed"}

                headers = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

                # Get personalized content
                content_response = await test_client.get(
                    "/api/v1/news/personalized", headers=headers
                )
                if content_response.status_code != 200:
                    return {"success": False, "error": "content_fetch_failed"}

                # Track analytics
                await test_client.post(
                    "/api/v1/analytics/track",
                    headers=headers,
                    json={"event": "content_viewed", "article_id": "test_article", "duration": 60},
                )

                return {"success": True, "response_time": time.time()}

            except Exception as e:
                return {"success": False, "error": str(e)}

        # Run concurrent user sessions
        start_time = time.time()
        tasks = []

        for _ in range(concurrent_users):
            tasks.append(simulate_user_session())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        # Analyze results
        successful_sessions = [r for r in results if isinstance(r, dict) and r.get("success")]
        error_rate = 1 - (len(successful_sessions) / len(results))
        total_time = end_time - start_time

        # Verify performance requirements
        assert (
            error_rate <= max_error_rate
        ), f"Error rate {error_rate:.2%} exceeds maximum {max_error_rate:.2%}"
        assert (
            total_time <= test_duration * 2
        ), f"Test took {total_time:.2f}s, expected under {test_duration * 2}s"

    @pytest.mark.asyncio
    async def test_fact_checking_performance(self, fact_checker_agent, performance_test_config):
        """Test fact-checking system performance"""

        # Test claims of varying complexity
        test_claims = [
            "The Earth is round",  # Simple factual claim
            "Water boils at 100°C at sea level",  # Scientific fact
            "The current population of Tokyo is approximately 14 million people",  # Statistical claim
            "Artificial intelligence will revolutionize healthcare in the next decade",  # Complex prediction
            "The Great Wall of China was built entirely during the Ming Dynasty",  # Historical claim
        ]

        performance_metrics = []

        for claim in test_claims:
            start_time = time.time()

            result = await fact_checker_agent.verify_claim(claim)

            end_time = time.time()
            processing_time = end_time - start_time

            performance_metrics.append(
                {
                    "claim": claim,
                    "processing_time": processing_time,
                    "confidence": result.confidence,
                    "success": result.verdict is not None,
                }
            )

        # Verify performance requirements
        avg_processing_time = statistics.mean([m["processing_time"] for m in performance_metrics])
        assert (
            avg_processing_time <= 5.0
        ), f"Average fact-checking time {avg_processing_time:.2f}s exceeds 5s limit"

        success_rate = sum(1 for m in performance_metrics if m["success"]) / len(
            performance_metrics
        )
        assert (
            success_rate >= 0.9
        ), f"Fact-checking success rate {success_rate:.2%} below 90% requirement"

    @pytest.mark.asyncio
    async def test_voice_generation_performance(self, test_client):
        """Test voice generation performance"""

        # Test content of varying lengths
        test_contents = [
            "Short news summary.",  # ~20 characters
            "Medium length news article with several sentences and more detailed information about current events.",  # ~100 characters
            "Long form news article " * 20,  # ~500 characters
        ]

        performance_metrics = []

        for content in test_contents:
            start_time = time.time()

            with patch("backend.services.voice_service.generate_audio") as mock_voice:
                mock_voice.return_value = {
                    "audio_url": "https://example.com/audio.mp3",
                    "duration": len(content) * 0.1,  # Simulate processing time
                    "quality_score": 0.92,
                }

                response = await test_client.post(
                    "/api/v1/voice/generate",
                    json={
                        "text": content,
                        "voice_settings": {"voice": "female", "speed": "medium"},
                    },
                )

            end_time = time.time()
            processing_time = end_time - start_time

            performance_metrics.append(
                {
                    "content_length": len(content),
                    "processing_time": processing_time,
                    "success": response.status_code == 200,
                }
            )

        # Verify performance requirements
        for metric in performance_metrics:
            # Processing time should be reasonable relative to content length
            expected_time = metric["content_length"] * 0.01  # 10ms per character
            assert (
                metric["processing_time"] <= expected_time + 2.0
            ), f"Voice generation took {metric['processing_time']:.2f}s for {metric['content_length']} chars"

    @pytest.mark.asyncio
    async def test_database_performance(self, test_db_session):
        """Test database performance under load"""

        # Test database operations
        operations = [
            ("SELECT", "SELECT * FROM articles LIMIT 100"),
            (
                "INSERT",
                "INSERT INTO user_interactions (user_id, article_id, interaction_type) VALUES (1, 1, 'view')",
            ),
            ("UPDATE", "UPDATE articles SET view_count = view_count + 1 WHERE id = 1"),
            (
                "COMPLEX_JOIN",
                """
                SELECT a.*, u.preferences 
                FROM articles a 
                JOIN user_preferences u ON a.category = ANY(u.categories) 
                LIMIT 50
            """,
            ),
        ]

        performance_results = {}

        for op_name, query in operations:
            times = []

            for _ in range(10):  # Run each operation 10 times
                start_time = time.time()

                try:
                    result = await test_db_session.execute(query)
                    if op_name in ["SELECT", "COMPLEX_JOIN"]:
                        await result.fetchall()
                    else:
                        await test_db_session.commit()

                    end_time = time.time()
                    times.append(end_time - start_time)

                except Exception as e:
                    print(f"Database operation {op_name} failed: {e}")

            if times:
                performance_results[op_name] = {
                    "average": statistics.mean(times),
                    "max": max(times),
                }

        # Verify database performance requirements
        assert performance_results["SELECT"]["average"] <= 0.1, "SELECT queries too slow"
        assert performance_results["INSERT"]["average"] <= 0.05, "INSERT operations too slow"
        assert performance_results["COMPLEX_JOIN"]["average"] <= 0.5, "Complex joins too slow"

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, test_client):
        """Test memory usage under sustained load"""

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate sustained load
        tasks = []
        for i in range(100):
            tasks.append(test_client.get("/api/v1/news/articles"))

        await asyncio.gather(*tasks)

        # Check memory after load
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert (
            memory_increase <= 100
        ), f"Memory increased by {memory_increase:.1f}MB, should be under 100MB"

    @pytest.mark.asyncio
    async def test_cache_performance(self, test_redis_client):
        """Test caching system performance"""

        # Test cache operations
        cache_operations = []

        # Test SET operations
        for i in range(1000):
            start_time = time.time()
            await test_redis_client.set(f"test_key_{i}", f"test_value_{i}")
            end_time = time.time()
            cache_operations.append(("SET", end_time - start_time))

        # Test GET operations
        for i in range(1000):
            start_time = time.time()
            value = await test_redis_client.get(f"test_key_{i}")
            end_time = time.time()
            cache_operations.append(("GET", end_time - start_time))
            assert value is not None

        # Analyze cache performance
        set_times = [op[1] for op in cache_operations if op[0] == "SET"]
        get_times = [op[1] for op in cache_operations if op[0] == "GET"]

        avg_set_time = statistics.mean(set_times)
        avg_get_time = statistics.mean(get_times)

        # Cache operations should be very fast
        assert avg_set_time <= 0.001, f"Cache SET too slow: {avg_set_time:.4f}s"
        assert avg_get_time <= 0.001, f"Cache GET too slow: {avg_get_time:.4f}s"


class NewsApiUser(HttpUser):
    """Locust user class for load testing"""

    wait_time = between(1, 3)

    def on_start(self):
        """Login user at start"""
        response = self.client.post(
            "/api/v1/auth/login", json={"email": "loadtest@example.com", "password": "password123"}
        )

        if response.status_code == 200:
            self.token = response.json()["access_token"]
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}

    @task(3)
    def get_articles(self):
        """Get news articles"""
        self.client.get("/api/v1/news/articles", headers=self.headers)

    @task(2)
    def get_personalized_content(self):
        """Get personalized content"""
        self.client.get("/api/v1/news/personalized", headers=self.headers)

    @task(1)
    def track_analytics(self):
        """Track user analytics"""
        self.client.post(
            "/api/v1/analytics/track",
            headers=self.headers,
            json={"event": "content_viewed", "article_id": "test_article", "duration": 60},
        )

    @task(1)
    def generate_voice(self):
        """Generate voice content"""
        self.client.post(
            "/api/v1/voice/generate",
            headers=self.headers,
            json={
                "text": "Test content for voice generation",
                "voice_settings": {"voice": "female", "speed": "medium"},
            },
        )


@pytest.mark.performance
class TestLoadTesting:
    """Load testing using Locust"""

    def test_api_load_testing(self):
        """Run comprehensive load testing"""

        # Setup Locust environment
        env = Environment(user_classes=[NewsApiUser])
        env.create_local_runner()

        # Start load test
        env.runner.start(user_count=100, spawn_rate=10)

        # Run for 5 minutes
        time.sleep(300)

        # Stop load test
        env.runner.stop()

        # Analyze results
        stats = env.runner.stats

        # Verify performance requirements
        assert (
            stats.total.avg_response_time <= 2000
        ), f"Average response time {stats.total.avg_response_time}ms exceeds 2000ms"

        assert (
            stats.total.failure_count / stats.total.num_requests <= 0.05
        ), f"Error rate {stats.total.failure_count / stats.total.num_requests:.2%} exceeds 5%"

    def test_stress_testing(self):
        """Stress testing to find system limits"""

        user_counts = [50, 100, 200, 500, 1000]
        results = {}

        for user_count in user_counts:
            env = Environment(user_classes=[NewsApiUser])
            env.create_local_runner()

            # Start stress test
            env.runner.start(user_count=user_count, spawn_rate=50)

            # Run for 2 minutes
            time.sleep(120)

            # Collect metrics
            stats = env.runner.stats
            results[user_count] = {
                "avg_response_time": stats.total.avg_response_time,
                "error_rate": (
                    stats.total.failure_count / stats.total.num_requests
                    if stats.total.num_requests > 0
                    else 0
                ),
                "requests_per_second": stats.total.current_rps,
            }

            env.runner.stop()

            # Stop if error rate becomes too high
            if results[user_count]["error_rate"] > 0.1:
                break

        # Find maximum sustainable load
        max_users = max(
            [
                users
                for users, metrics in results.items()
                if metrics["error_rate"] <= 0.05 and metrics["avg_response_time"] <= 2000
            ]
        )

        assert (
            max_users >= 200
        ), f"System can only handle {max_users} concurrent users, target is 200+"


@pytest.mark.performance
class TestScalabilityTesting:
    """Scalability testing for different system components"""

    @pytest.mark.asyncio
    async def test_article_processing_scalability(self, test_kafka_producer):
        """Test article processing scalability"""

        batch_sizes = [10, 50, 100, 500, 1000]
        processing_times = {}

        for batch_size in batch_sizes:
            # Generate test articles
            articles = [
                {
                    "id": f"article_{i}",
                    "title": f"Test Article {i}",
                    "content": f"Content for article {i}" * 10,  # Longer content
                    "source": "Test Source",
                    "category": "technology",
                }
                for i in range(batch_size)
            ]

            # Measure processing time
            start_time = time.time()

            # Send articles to processing queue
            import json
            for article in articles:
                test_kafka_producer.send("news_articles", value=json.dumps(article))

            test_kafka_producer.flush()

            # Wait for processing to complete (simplified)
            await asyncio.sleep(batch_size * 0.1)  # Estimated processing time

            end_time = time.time()
            processing_time = end_time - start_time

            processing_times[batch_size] = {
                "total_time": processing_time,
                "time_per_article": processing_time / batch_size,
                "articles_per_second": batch_size / processing_time,
            }

        # Verify scalability - processing time should scale linearly
        for batch_size, metrics in processing_times.items():
            assert (
                metrics["time_per_article"] <= 1.0
            ), f"Processing time per article {metrics['time_per_article']:.2f}s too high for batch size {batch_size}"

            assert (
                metrics["articles_per_second"] >= 10
            ), f"Processing rate {metrics['articles_per_second']:.1f} articles/s too low for batch size {batch_size}"

    @pytest.mark.asyncio
    async def test_user_scaling(self, test_client):
        """Test system behavior with increasing user base"""

        user_counts = [100, 500, 1000, 5000]
        performance_metrics = {}

        for user_count in user_counts:
            # Simulate user base
            start_time = time.time()

            # Create concurrent requests simulating user activity
            tasks = []
            for i in range(min(user_count, 100)):  # Limit concurrent requests
                tasks.append(test_client.get("/api/v1/news/articles"))

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = time.time()

            # Analyze performance
            successful_responses = [
                r for r in responses if hasattr(r, "status_code") and r.status_code == 200
            ]

            performance_metrics[user_count] = {
                "response_time": end_time - start_time,
                "success_rate": len(successful_responses) / len(responses),
                "throughput": len(successful_responses) / (end_time - start_time),
            }

        # Verify scaling characteristics
        for user_count, metrics in performance_metrics.items():
            assert (
                metrics["success_rate"] >= 0.95
            ), f"Success rate {metrics['success_rate']:.2%} too low for {user_count} users"

            assert (
                metrics["throughput"] >= 10
            ), f"Throughput {metrics['throughput']:.1f} req/s too low for {user_count} users"
