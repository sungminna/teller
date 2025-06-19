"""
NewsTalk AI Testing Configuration - Stage 9
Comprehensive test fixtures and configuration for all test types
"""

import asyncio
import os
from datetime import datetime
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

# from api.database.connection import get_db_session  # TODO: Create database connection module

# Import application modules with proper paths
import sys
import os
sys.path.append(os.path.abspath('..'))

from httpx import AsyncClient
from kafka import KafkaConsumer, KafkaProducer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Import modules with error handling
try:
    from api.main import app
    from api.utils.redis_client import get_redis_client
    from langgraph.agents.personalization_agent import PersonalizationAgent
except ImportError as e:
    print(f"Import warning: {e}")
    # Create mock objects for testing
    app = MagicMock()
    get_redis_client = MagicMock()
    PersonalizationAgent = MagicMock


# Test Configuration
@pytest.fixture(scope="session")
def test_settings():
    """Test-specific settings configuration"""
    return {
        "ENVIRONMENT": "test",
        "DATABASE_URL": "sqlite+aiosqlite:///./test.db",
        "REDIS_URL": "redis://localhost:6379/15",  # Use test database
        "KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
        "KAFKA_TEST_TOPIC_PREFIX": "test_",
        "OPENAI_API_KEY": "test_key",
        "LANGFUSE_SECRET_KEY": "test_secret",
        "LANGFUSE_PUBLIC_KEY": "test_public",
        "ENABLE_FACT_CHECKING": True,
        "FACT_CHECKING_CONFIDENCE_THRESHOLD": 0.85,
        "VOICE_QUALITY_THRESHOLD": 0.9,
        "PIPELINE_TIMEOUT": 30,
        "TEST_DATA_PATH": "./tests/data",
    }


# Database Fixtures
@pytest_asyncio.fixture
async def test_db_engine(test_settings):
    """Create test database engine"""
    engine = create_async_engine(test_settings["DATABASE_URL"], echo=False, future=True)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async_session = sessionmaker(test_db_engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def test_redis_client(test_settings) -> AsyncGenerator[Redis, None]:
    """Create test Redis client"""
    redis_client = Redis.from_url(test_settings["REDIS_URL"])

    # Clear test database
    await redis_client.flushdb()

    yield redis_client

    # Cleanup
    await redis_client.flushdb()
    await redis_client.close()


# Kafka Fixtures
@pytest.fixture
def test_kafka_producer(test_settings):
    """Create test Kafka producer"""
    producer = KafkaProducer(
        bootstrap_servers=test_settings["KAFKA_BOOTSTRAP_SERVERS"],
        value_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
    )
    yield producer
    producer.close()


@pytest.fixture
def test_kafka_consumer(test_settings):
    """Create test Kafka consumer"""
    consumer = KafkaConsumer(
        bootstrap_servers=test_settings["KAFKA_BOOTSTRAP_SERVERS"],
        auto_offset_reset="earliest",
        consumer_timeout_ms=1000,
    )
    yield consumer
    consumer.close()


# HTTP Client Fixtures
@pytest_asyncio.fixture
async def test_client(test_db_session, test_redis_client) -> AsyncGenerator[AsyncClient, None]:
    """Create test HTTP client with dependency overrides"""

    # Override dependencies (import locally to avoid circular dependency)
    from api.database.connection import get_db_session

    app.dependency_overrides[get_db_session] = lambda: test_db_session
    app.dependency_overrides[get_redis_client] = lambda: test_redis_client

    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

    # Clear overrides
    app.dependency_overrides.clear()


# Mock Services
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    with patch("openai.AsyncOpenAI") as mock:
        # Mock chat completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test AI response"
        mock_response.usage.total_tokens = 100

        mock.return_value.chat.completions.create.return_value = mock_response
        yield mock


@pytest.fixture
def mock_langfuse_client():
    """Mock Langfuse client for testing"""
    with patch("langfuse.Langfuse") as mock:
        mock_client = MagicMock()
        mock_client.trace.return_value = MagicMock()
        mock_client.generation.return_value = MagicMock()
        mock.return_value = mock_client
        yield mock_client


# Agent Fixtures
@pytest_asyncio.fixture
async def fact_checker_agent(mock_openai_client, mock_langfuse_client):
    """Create fact checker agent for testing"""

    # Mock implementation
    class MockFactCheckerAgent:
        async def check_facts(self, content):
            return {"factual": True, "confidence": 0.9}

    yield MockFactCheckerAgent()


@pytest_asyncio.fixture
async def content_analyzer_agent(mock_openai_client, mock_langfuse_client):
    """Create content analyzer agent for testing"""

    # Mock implementation
    class MockContentAnalyzerAgent:
        async def analyze(self, content):
            return {"sentiment": "positive", "topics": ["technology"]}

    yield MockContentAnalyzerAgent()


@pytest_asyncio.fixture
async def personalization_agent(mock_openai_client, mock_langfuse_client):
    """Create personalization agent for testing"""
    agent = PersonalizationAgent()
    yield agent


# Test Data Fixtures
@pytest.fixture
def sample_news_article():
    """Sample news article for testing"""
    return {
        "id": "test_article_001",
        "title": "Breaking: New AI Technology Revolutionizes News Processing",
        "content": "A groundbreaking artificial intelligence system has been developed that can process and analyze news articles with unprecedented accuracy. The system, called NewsTalk AI, combines advanced language models with real-time fact-checking capabilities.",
        "url": "https://example.com/news/ai-revolution",
        "source": "Tech News Daily",
        "published_at": datetime.utcnow().isoformat(),
        "category": "technology",
        "language": "en",
        "metadata": {
            "word_count": 150,
            "reading_time": 60,
            "sentiment": "positive",
            "keywords": ["AI", "technology", "news", "processing"],
        },
    }


@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing"""
    return {
        "user_id": "test_user_001",
        "preferences": {
            "categories": ["technology", "science", "business"],
            "languages": ["en", "ko"],
            "reading_speed": "medium",
            "voice_preference": "female",
            "notification_frequency": "hourly",
        },
        "interaction_history": [
            {
                "article_id": "article_001",
                "action": "read",
                "timestamp": datetime.utcnow().isoformat(),
                "duration": 120,
            }
        ],
        "fact_checking_preferences": {
            "auto_verify": True,
            "show_confidence": True,
            "highlight_claims": True,
        },
    }


@pytest.fixture
def sample_fact_check_data():
    """Sample fact-checking test data"""
    return {
        "claims": [
            {
                "claim": "The Earth is round",
                "expected_result": True,
                "confidence": 0.99,
                "sources": ["NASA", "Scientific consensus"],
            },
            {
                "claim": "The Moon is made of cheese",
                "expected_result": False,
                "confidence": 0.99,
                "sources": ["Apollo missions", "Scientific analysis"],
            },
            {
                "claim": "Water boils at 100°C at sea level",
                "expected_result": True,
                "confidence": 0.98,
                "sources": ["Physics textbooks", "Scientific measurement"],
            },
        ]
    }


# Performance Test Fixtures
@pytest.fixture
def performance_test_config():
    """Configuration for performance testing"""
    return {
        "concurrent_users": 100,
        "test_duration": 60,  # seconds
        "ramp_up_time": 10,  # seconds
        "target_response_time": 2.0,  # seconds
        "max_error_rate": 0.05,  # 5%
        "endpoints_to_test": [
            "/api/v1/news/articles",
            "/api/v1/news/personalized",
            "/api/v1/fact-check",
            "/api/v1/voice/generate",
        ],
    }


# Quality Metrics Fixtures
@pytest.fixture
def quality_thresholds():
    """Quality thresholds for testing"""
    return {
        "fact_checking_accuracy": 0.95,
        "voice_quality_score": 0.9,
        "response_time_p95": 2.0,
        "availability": 0.999,
        "user_satisfaction": 4.5,
        "content_relevance": 0.85,
        "pipeline_success_rate": 0.98,
    }


# Test Environment Setup
@pytest.fixture(autouse=True)
def setup_test_environment(test_settings):
    """Setup test environment variables"""
    original_env = os.environ.copy()

    # Set test environment variables
    for key, value in test_settings.items():
        os.environ[key] = str(value)

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Test Data Directory
@pytest.fixture
def test_data_dir(test_settings):
    """Create and provide test data directory"""
    data_dir = test_settings["TEST_DATA_PATH"]
    os.makedirs(data_dir, exist_ok=True)

    # Create sample test files
    sample_files = {
        "sample_articles.json": [
            {
                "id": "article_001",
                "title": "Sample Article 1",
                "content": "This is a sample article for testing purposes.",
            }
        ],
        "fact_check_dataset.json": [{"claim": "Test claim", "verdict": True, "confidence": 0.95}],
    }

    import json

    for filename, content in sample_files.items():
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(content, f)

    yield data_dir


# Async Event Loop Configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test Markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "quality: marks tests as quality verification tests")
    config.addinivalue_line("markers", "mobile: marks tests as mobile app tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")


# Test Collection Hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "quality" in str(item.fspath):
            item.add_marker(pytest.mark.quality)
        elif "mobile" in str(item.fspath):
            item.add_marker(pytest.mark.mobile)


# Cleanup Fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Add any necessary cleanup logic here
