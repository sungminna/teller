"""
Enhanced Kafka Client Utility for Stage 6
Real-time streaming and comprehensive topic management
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# Global Kafka components
_kafka_producer: Optional[AIOKafkaProducer] = None
_kafka_admin: Optional[AIOKafkaAdminClient] = None
_active_consumers: Dict[str, AIOKafkaConsumer] = {}

# Stage 6: Topic Configuration
KAFKA_TOPICS = {
    "raw_news": "raw-news",
    "processed_news": "processed-news",
    "user_feedback": "user-feedback",
    "realtime_updates": "real-time-updates",
    "ai_processing": "ai-processing",
    "news_updates": "news-updates",  # Legacy
}


class KafkaStreamProcessor:
    """Real-time Kafka stream processor for Stage 6"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.running = False
        self.processors = {}

    async def start(self):
        """Start stream processing"""
        if self.running:
            return

        logger.info("🚀 Starting Kafka stream processor")
        self.running = True

        # Initialize topics
        await self.ensure_topics_exist()

        # Start processors
        await asyncio.gather(
            self._process_raw_news_stream(),
            self._process_user_feedback_stream(),
            self._process_realtime_updates_stream(),
            return_exceptions=True,
        )

    async def stop(self):
        """Stop stream processing"""
        self.running = False
        logger.info("⏹️ Stopping Kafka stream processor")

        # Close all consumers
        for consumer in _active_consumers.values():
            await consumer.stop()
        _active_consumers.clear()

    async def ensure_topics_exist(self):
        """Ensure all required topics exist"""
        admin = await get_kafka_admin()

        topics_to_create = [
            NewTopic(
                name=KAFKA_TOPICS["raw_news"],
                num_partitions=3,
                replication_factor=1,
                topic_configs={
                    "retention.ms": str(7 * 24 * 60 * 60 * 1000),  # 7 days
                    "compression.type": "gzip",
                },
            ),
            NewTopic(
                name=KAFKA_TOPICS["processed_news"],
                num_partitions=3,
                replication_factor=1,
                topic_configs={
                    "retention.ms": str(24 * 60 * 60 * 1000),  # 24 hours
                    "compression.type": "gzip",
                },
            ),
            NewTopic(
                name=KAFKA_TOPICS["user_feedback"],
                num_partitions=2,
                replication_factor=1,
                topic_configs={
                    "retention.ms": str(30 * 24 * 60 * 60 * 1000),  # 30 days
                },
            ),
            NewTopic(
                name=KAFKA_TOPICS["realtime_updates"],
                num_partitions=4,
                replication_factor=1,
                topic_configs={
                    "retention.ms": str(60 * 60 * 1000),  # 1 hour
                    "cleanup.policy": "delete",
                },
            ),
        ]

        try:
            await admin.create_topics(topics_to_create)
            logger.info("✅ Kafka topics created successfully")
        except TopicAlreadyExistsError:
            logger.info("ℹ️ Kafka topics already exist")
        except Exception as e:
            logger.error(f"❌ Failed to create topics: {e}")

    async def _process_raw_news_stream(self):
        """Process raw news stream"""
        consumer = await create_kafka_consumer([KAFKA_TOPICS["raw_news"]], "raw-news-processor")
        _active_consumers["raw_news"] = consumer

        logger.info("📰 Starting raw news stream processor")

        async for message in consumer:
            if not self.running:
                break

            try:
                news_data = message.value

                # Process raw news data
                processed_data = await self._transform_raw_news(news_data)

                # Publish to processed news topic
                await send_message(
                    KAFKA_TOPICS["processed_news"],
                    processed_data,
                    key=f"article_{news_data.get('id', uuid.uuid4())}",
                )

                # Send real-time update
                await self._send_realtime_update(
                    {
                        "type": "news_processed",
                        "article_id": news_data.get("id"),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            except Exception as e:
                logger.error(f"Error processing raw news: {e}")

    async def _process_user_feedback_stream(self):
        """Process user feedback stream"""
        consumer = await create_kafka_consumer(
            [KAFKA_TOPICS["user_feedback"]], "user-feedback-processor"
        )
        _active_consumers["user_feedback"] = consumer

        logger.info("👤 Starting user feedback stream processor")

        async for message in consumer:
            if not self.running:
                break

            try:
                feedback_data = message.value

                # Process feedback for personalization
                await self._process_user_feedback(feedback_data)

                # Update real-time user metrics
                await self._update_user_metrics(feedback_data)

            except Exception as e:
                logger.error(f"Error processing user feedback: {e}")

    async def _process_realtime_updates_stream(self):
        """Process real-time updates stream"""
        consumer = await create_kafka_consumer(
            [KAFKA_TOPICS["realtime_updates"]], "realtime-updates-processor"
        )
        _active_consumers["realtime_updates"] = consumer

        logger.info("⚡ Starting real-time updates stream processor")

        async for message in consumer:
            if not self.running:
                break

            try:
                update_data = message.value

                # Broadcast to connected clients via SSE
                await self._broadcast_realtime_update(update_data)

            except Exception as e:
                logger.error(f"Error processing real-time update: {e}")

    async def _transform_raw_news(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw news data for processing"""
        return {
            "id": raw_data.get("id"),
            "title": raw_data.get("title"),
            "content": raw_data.get("content"),
            "url": raw_data.get("url"),
            "category": raw_data.get("category"),
            "source": raw_data.get("source"),
            "published_at": raw_data.get("published_at"),
            "processed_at": datetime.utcnow().isoformat(),
            "processing_status": "ready_for_ai",
            "metadata": {
                "word_count": len(raw_data.get("content", "").split()),
                "language": raw_data.get("language", "en"),
                "quality_score": raw_data.get("quality_score", 0.0),
            },
        }

    async def _process_user_feedback(self, feedback_data: Dict[str, Any]):
        """Process user feedback for personalization"""
        # This would integrate with the personalization agent
        logger.info(f"Processing feedback from user {feedback_data.get('user_id')}")

    async def _update_user_metrics(self, feedback_data: Dict[str, Any]):
        """Update real-time user metrics"""
        from ..utils.redis_client import cache_hincrby

        user_id = feedback_data.get("user_id")
        if user_id:
            await cache_hincrby(f"user_metrics:{user_id}", "total_feedback", 1)
            await cache_hincrby(f"user_metrics:{user_id}", "daily_interactions", 1)

    async def _send_realtime_update(self, update_data: Dict[str, Any]):
        """Send real-time update"""
        await send_message(
            KAFKA_TOPICS["realtime_updates"],
            update_data,
            key=f"update_{datetime.utcnow().timestamp()}",
        )

    async def _broadcast_realtime_update(self, update_data: Dict[str, Any]):
        """Broadcast update to connected clients"""
        # This would integrate with SSE connections
        logger.info(f"Broadcasting update: {update_data.get('type')}")


# Enhanced Kafka client functions
async def get_kafka_producer() -> AIOKafkaProducer:
    """Get enhanced Kafka producer"""
    global _kafka_producer

    if _kafka_producer is None:
        _kafka_producer = await create_kafka_producer()

    return _kafka_producer


async def create_kafka_producer() -> AIOKafkaProducer:
    """Create enhanced Kafka producer with Stage 6 optimizations"""
    settings = get_settings()
    kafka_config = settings.get_kafka_config()

    try:
        producer = AIOKafkaProducer(
            bootstrap_servers=kafka_config["bootstrap_servers"],
            value_serializer=lambda v: (
                json.dumps(v, default=str).encode("utf-8") if not isinstance(v, bytes) else v
            ),
            key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
            compression_type=settings.kafka.producer_compression_type,
            batch_size=settings.kafka.producer_batch_size,
            linger_ms=settings.kafka.producer_linger_ms,
            max_request_size=settings.kafka.producer_max_request_size,
            retry_backoff_ms=100,
            request_timeout_ms=30000,
            acks=settings.kafka.producer_acks,
            enable_idempotence=True,
            retries=settings.kafka.producer_retries,
        )

        await producer.start()

        logger.info(f"✅ Enhanced Kafka producer created: {kafka_config['bootstrap_servers']}")
        return producer

    except Exception as e:
        logger.error(f"❌ Enhanced Kafka producer creation failed: {e}")
        raise


async def get_kafka_admin() -> AIOKafkaAdminClient:
    """Get Kafka admin client"""
    global _kafka_admin

    if _kafka_admin is None:
        settings = get_settings()
        kafka_config = settings.get_kafka_config()
        _kafka_admin = AIOKafkaAdminClient(bootstrap_servers=kafka_config["bootstrap_servers"])
        await _kafka_admin.start()

    return _kafka_admin


async def create_kafka_consumer(
    topics: List[str],
    group_id: str,
    auto_offset_reset: str = "latest",
    enable_auto_commit: bool = True,
) -> AIOKafkaConsumer:
    """Create enhanced Kafka consumer with Stage 6 optimizations"""
    settings = get_settings()
    kafka_config = settings.get_kafka_config()

    try:
        consumer = AIOKafkaConsumer(
            *topics,
            bootstrap_servers=kafka_config["bootstrap_servers"],
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")) if m else None,
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            max_poll_records=settings.kafka.consumer_max_poll_records,
            fetch_max_wait_ms=settings.kafka.consumer_fetch_max_wait_ms,
            fetch_min_bytes=1,
            fetch_max_bytes=52428800,  # 50MB
            session_timeout_ms=settings.kafka.consumer_session_timeout_ms,
            heartbeat_interval_ms=3000,
        )

        await consumer.start()

        logger.info(f"✅ Enhanced Kafka consumer created for topics: {topics} with servers: {kafka_config['bootstrap_servers']}")
        return consumer

    except Exception as e:
        logger.error(f"❌ Enhanced Kafka consumer creation failed: {e}")
        raise


# Enhanced message sending functions
async def send_message(
    topic: str,
    value: Any,
    key: Optional[str] = None,
    partition: Optional[int] = None,
    timestamp_ms: Optional[int] = None,
    headers: Optional[Dict[str, bytes]] = None,
) -> bool:
    """Enhanced message sending with better error handling"""
    try:
        producer = await get_kafka_producer()

        # Serialize value if it's not bytes
        if not isinstance(value, bytes):
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str).encode("utf-8")
            elif isinstance(value, str):
                value = value.encode("utf-8")

        # Send message
        await producer.send(
            topic=topic,
            value=value,
            key=key,
            partition=partition,
            timestamp_ms=timestamp_ms,
            headers=headers,
        )

        return True

    except Exception as e:
        logger.error(f"Kafka send error: {e}")
        return False


# Stage 6: Specialized publishing functions
async def publish_raw_news(news_data: Dict[str, Any]) -> bool:
    """Publish raw news data to raw-news topic"""
    return await send_message(
        KAFKA_TOPICS["raw_news"], news_data, key=f"news_{news_data.get('id', uuid.uuid4())}"
    )


async def publish_processed_news(processed_data: Dict[str, Any]) -> bool:
    """Publish processed news data"""
    return await send_message(
        KAFKA_TOPICS["processed_news"],
        processed_data,
        key=f"processed_{processed_data.get('id', uuid.uuid4())}",
    )


async def publish_user_feedback(user_id: int, feedback_data: Dict[str, Any]) -> bool:
    """Publish user feedback for personalization"""
    feedback_payload = {
        "user_id": user_id,
        "timestamp": datetime.utcnow().isoformat(),
        **feedback_data,
    }

    return await send_message(
        KAFKA_TOPICS["user_feedback"],
        feedback_payload,
        key=f"feedback_{user_id}_{datetime.utcnow().timestamp()}",
    )


async def publish_realtime_update(update_type: str, data: Dict[str, Any]) -> bool:
    """Publish real-time update"""
    update_payload = {"type": update_type, "timestamp": datetime.utcnow().isoformat(), "data": data}

    return await send_message(
        KAFKA_TOPICS["realtime_updates"],
        update_payload,
        key=f"update_{update_type}_{datetime.utcnow().timestamp()}",
    )


# Stream consumption utilities
async def consume_news_stream() -> AsyncGenerator[Dict[str, Any], None]:
    """Consume processed news stream"""
    consumer = await create_kafka_consumer([KAFKA_TOPICS["processed_news"]], "news-stream-consumer")

    try:
        async for message in consumer:
            yield message.value
    finally:
        await consumer.stop()


async def consume_realtime_updates() -> AsyncGenerator[Dict[str, Any], None]:
    """Consume real-time updates stream"""
    consumer = await create_kafka_consumer(
        [KAFKA_TOPICS["realtime_updates"]], "realtime-updates-consumer"
    )

    try:
        async for message in consumer:
            yield message.value
    finally:
        await consumer.stop()


# Cleanup functions
async def close_kafka_producer():
    """Close Kafka producer"""
    global _kafka_producer

    if _kafka_producer:
        await _kafka_producer.stop()
        _kafka_producer = None
        logger.info("✅ Enhanced Kafka producer closed")


async def close_kafka_admin():
    """Close Kafka admin client"""
    global _kafka_admin

    if _kafka_admin:
        await _kafka_admin.close()
        _kafka_admin = None
        logger.info("✅ Kafka admin client closed")


async def close_all_kafka_connections():
    """Close all Kafka connections"""
    await close_kafka_producer()
    await close_kafka_admin()

    for consumer in _active_consumers.values():
        await consumer.stop()
    _active_consumers.clear()

    logger.info("✅ All Kafka connections closed")


# Initialize stream processor
stream_processor = KafkaStreamProcessor()
