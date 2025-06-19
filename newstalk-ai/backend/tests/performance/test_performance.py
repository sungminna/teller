"""
Performance Testing for NewsTalk AI - Stage 9
Comprehensive performance testing to ensure system scalability
"""

import asyncio
import os
import statistics
import time

import psutil
import pytest


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

    @pytest.mark.asyncio
    async def test_concurrent_user_load(self, test_client, performance_test_config):
        """Test system performance under concurrent user load"""

        concurrent_users = performance_test_config["concurrent_users"]
        max_error_rate = performance_test_config["max_error_rate"]

        async def simulate_user_session():
            """Simulate a typical user session"""
            try:
                # Get articles
                response = await test_client.get("/api/v1/news/articles")
                if response.status_code != 200:
                    return {"success": False, "error": "articles_fetch_failed"}

                return {"success": True}

            except Exception as e:
                return {"success": False, "error": str(e)}

        # Run concurrent user sessions
        tasks = []
        for _ in range(concurrent_users):
            tasks.append(simulate_user_session())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_sessions = [r for r in results if isinstance(r, dict) and r.get("success")]
        error_rate = 1 - (len(successful_sessions) / len(results))

        # Verify performance requirements
        assert (
            error_rate <= max_error_rate
        ), f"Error rate {error_rate:.2%} exceeds maximum {max_error_rate:.2%}"

    @pytest.mark.asyncio
    async def test_fact_checking_performance(self, fact_checker_agent):
        """Test fact-checking system performance"""

        test_claims = [
            "The Earth is round",
            "Water boils at 100°C at sea level",
            "The current population of Tokyo is approximately 14 million people",
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
                    "success": result.verdict is not None,
                }
            )

        # Verify performance requirements
        avg_processing_time = statistics.mean([m["processing_time"] for m in performance_metrics])
        assert (
            avg_processing_time <= 5.0
        ), f"Average fact-checking time {avg_processing_time:.2f}s exceeds 5s limit"

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
