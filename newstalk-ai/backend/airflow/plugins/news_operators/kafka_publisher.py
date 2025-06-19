"""
Enhanced Kafka Publisher Operator for Stage 6
Real-time streaming with high-performance publishing
"""

import asyncio
import logging
import os

# Import enhanced Kafka client
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from airflow.configuration import conf
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from api.utils.kafka_client import (
    publish_processed_news,
    publish_raw_news,
    publish_realtime_update,
    publish_user_feedback,
    stream_processor,
)

logger = logging.getLogger(__name__)


class EnhancedKafkaPublisherOperator(BaseOperator):
    """Enhanced Kafka Publisher for Stage 6 real-time streaming"""

    template_fields = ["topic_name", "batch_size", "streaming_mode"]

    @apply_defaults
    def __init__(
        self,
        topic_name: str = "raw-news",
        batch_size: int = 50,
        enable_compression: bool = True,
        streaming_mode: bool = True,
        max_retries: int = 3,
        timeout_seconds: int = 120,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topic_name = topic_name
        self.batch_size = batch_size
        self.enable_compression = enable_compression
        self.streaming_mode = streaming_mode
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Database setup
        self.database_url = conf.get("core", "sql_alchemy_conn")
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute(self, context):
        """Execute enhanced Kafka publishing"""
        logger.info(f"Starting enhanced Kafka publishing to topic: {self.topic_name}")

        # Start stream processor if in streaming mode
        if self.streaming_mode:
            asyncio.run(self._ensure_stream_processor())

        # Get articles to publish
        articles = self._get_articles_to_publish()
        if not articles:
            logger.info("No articles to publish")
            return {"published": 0, "failed": 0}

        # Publish articles
        results = asyncio.run(self._publish_articles(articles))

        # Update article status
        self._update_article_status(articles, results)

        logger.info(f"Publishing completed: {results}")
        return results

    async def _ensure_stream_processor(self):
        """Ensure stream processor is running"""
        try:
            if not stream_processor.running:
                await stream_processor.start()
                logger.info("✅ Stream processor started")
        except Exception as e:
            logger.error(f"Failed to start stream processor: {e}")

    def _get_articles_to_publish(self) -> List[Dict[str, Any]]:
        """Get articles ready for publishing"""
        session = self.Session()
        try:
            # This would normally query from database
            # For now, return mock data for demonstration
            return [
                {
                    "id": f"article_{i}",
                    "title": f"Sample Article {i}",
                    "content": f"This is sample content for article {i}",
                    "url": f"https://example.com/article/{i}",
                    "category": "technology",
                    "source": "example.com",
                    "published_at": datetime.utcnow().isoformat(),
                    "quality_score": 0.8,
                }
                for i in range(1, min(self.batch_size + 1, 21))  # Max 20 for demo
            ]
        finally:
            session.close()

    async def _publish_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Publish articles to Kafka with enhanced performance"""
        results = {
            "total": len(articles),
            "published": 0,
            "failed": 0,
            "streaming_updates": 0,
            "processing_time": 0,
        }

        start_time = datetime.utcnow()

        # Determine publishing strategy based on topic
        if self.topic_name == "raw-news":
            results = await self._publish_raw_news(articles, results)
        elif self.topic_name == "processed-news":
            results = await self._publish_processed_news(articles, results)
        elif self.topic_name == "user-feedback":
            results = await self._publish_user_feedback(articles, results)
        else:
            # Default publishing
            results = await self._publish_generic(articles, results)

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        results["processing_time"] = processing_time

        # Send completion update
        if self.streaming_mode:
            await publish_realtime_update(
                "publishing_completed",
                {"topic": self.topic_name, "results": results, "processing_time": processing_time},
            )
            results["streaming_updates"] += 1

        return results

    async def _publish_raw_news(
        self, articles: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish raw news articles"""
        # Publish in parallel batches for better performance
        batch_size = 10
        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]

            # Create publishing tasks
            tasks = []
            for article in batch:
                tasks.append(publish_raw_news(article))

            # Execute batch
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results["failed"] += 1
                    logger.error(f"Failed to publish article {batch[j]['id']}: {result}")
                elif result:
                    results["published"] += 1

                    # Send real-time update for each published article
                    if self.streaming_mode:
                        await publish_realtime_update(
                            "article_published",
                            {
                                "article_id": batch[j]["id"],
                                "topic": self.topic_name,
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                        results["streaming_updates"] += 1
                else:
                    results["failed"] += 1

        return results

    async def _publish_processed_news(
        self, articles: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish processed news articles"""
        for article in articles:
            try:
                # Add processing metadata
                processed_article = {
                    **article,
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_stage": "validated",
                    "publisher_id": self.task_id,
                }

                success = await publish_processed_news(processed_article)

                if success:
                    results["published"] += 1

                    # Send processing completion update
                    if self.streaming_mode:
                        await publish_realtime_update(
                            "article_processed",
                            {
                                "article_id": article["id"],
                                "processing_stage": "validated",
                                "timestamp": datetime.utcnow().isoformat(),
                            },
                        )
                        results["streaming_updates"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to publish processed article {article['id']}: {e}")

        return results

    async def _publish_user_feedback(
        self, feedback_items: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish user feedback data"""
        for feedback in feedback_items:
            try:
                user_id = feedback.get("user_id", 1)  # Default user for demo
                feedback_data = {
                    "feedback_type": feedback.get("type", "rating"),
                    "article_id": feedback.get("article_id"),
                    "rating": feedback.get("rating", 5),
                    "comment": feedback.get("comment", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                }

                success = await publish_user_feedback(user_id, feedback_data)

                if success:
                    results["published"] += 1

                    # Send feedback update
                    if self.streaming_mode:
                        await publish_realtime_update(
                            "feedback_received",
                            {
                                "user_id": user_id,
                                "feedback_type": feedback_data["feedback_type"],
                                "article_id": feedback_data["article_id"],
                            },
                        )
                        results["streaming_updates"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to publish feedback: {e}")

        return results

    async def _publish_generic(
        self, items: List[Dict[str, Any]], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generic publishing for other topics"""
        from api.utils.kafka_client import send_message

        for item in items:
            try:
                success = await send_message(
                    topic=self.topic_name,
                    value=item,
                    key=f"{self.topic_name}_{item.get('id', datetime.utcnow().timestamp())}",
                )

                if success:
                    results["published"] += 1
                else:
                    results["failed"] += 1

            except Exception as e:
                results["failed"] += 1
                logger.error(f"Failed to publish item: {e}")

        return results

    def _update_article_status(self, articles: List[Dict[str, Any]], results: Dict[str, Any]):
        """Update article status in database"""
        session = self.Session()
        try:
            # This would normally update article status in database
            # For now, just log the update
            logger.info(f"Would update status for {results['published']} articles")

        except Exception as e:
            logger.error(f"Failed to update article status: {e}")
        finally:
            session.close()


class StreamingMetricsOperator(BaseOperator):
    """Operator for collecting and publishing streaming metrics"""

    @apply_defaults
    def __init__(self, metrics_interval: int = 60, *args, **kwargs):  # seconds
        super().__init__(*args, **kwargs)
        self.metrics_interval = metrics_interval

    def execute(self, context):
        """Collect and publish streaming metrics"""
        logger.info("Collecting streaming metrics")

        metrics = asyncio.run(self._collect_metrics())

        # Publish metrics
        asyncio.run(self._publish_metrics(metrics))

        return metrics

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect streaming performance metrics"""
        from api.utils.redis_client import cache_manager, timeseries

        # Get real-time stats
        pipeline_stats = await cache_manager.get_realtime_stats("pipeline_performance")
        throughput_stats = await cache_manager.get_realtime_stats("article_throughput")

        # Get TimeSeries data
        current_time = int(datetime.utcnow().timestamp() * 1000)
        one_hour_ago = current_time - (60 * 60 * 1000)

        pipeline_samples = await timeseries.get_range(
            "pipeline_duration", one_hour_ago, current_time
        )
        article_samples = await timeseries.get_range(
            "articles_processed", one_hour_ago, current_time
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_performance": pipeline_stats,
            "article_throughput": throughput_stats,
            "hourly_samples": {
                "pipeline_duration": len(pipeline_samples),
                "articles_processed": len(article_samples),
            },
            "streaming_health": {
                "processor_running": stream_processor.running,
                "active_consumers": len(stream_processor.processors),
            },
        }

    async def _publish_metrics(self, metrics: Dict[str, Any]):
        """Publish metrics to real-time updates"""
        await publish_realtime_update("streaming_metrics", metrics)


# Legacy operator for backward compatibility
class KafkaPublisherOperator(EnhancedKafkaPublisherOperator):
    """Legacy Kafka Publisher Operator - redirects to enhanced version"""

    def __init__(self, *args, **kwargs):
        # Map legacy parameters to new ones
        if "topic_name" in kwargs:
            kwargs["topic_name"] = kwargs.pop("topic_name")

        super().__init__(*args, **kwargs)
        logger.warning(
            "Using legacy KafkaPublisherOperator. Consider upgrading to EnhancedKafkaPublisherOperator"
        )


class KafkaTopicManagerOperator(BaseOperator):
    """
    Manage Kafka topics for news streaming
    """

    @apply_defaults
    def __init__(
        self,
        topic_configs: List[Dict[str, Any]],
        kafka_servers: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topic_configs = topic_configs
        self.kafka_servers = kafka_servers or ["localhost:9092"]

    def execute(self, context):
        """Create or update Kafka topics"""
        from kafka.admin import KafkaAdminClient
        from kafka.admin.config_resource import NewTopic

        admin_client = KafkaAdminClient(
            bootstrap_servers=self.kafka_servers, client_id="newstalk_admin"
        )

        try:
            # Get existing topics
            existing_topics = admin_client.list_topics()

            # Create new topics
            topics_to_create = []
            for config in self.topic_configs:
                topic_name = config["name"]
                if topic_name not in existing_topics:
                    new_topic = NewTopic(
                        name=topic_name,
                        num_partitions=config.get("partitions", 3),
                        replication_factor=config.get("replication_factor", 1),
                    )
                    topics_to_create.append(new_topic)

            if topics_to_create:
                result = admin_client.create_topics(topics_to_create)
                for topic, future in result.topic_names_to_futures.items():
                    try:
                        future.result()
                        logger.info(f"Topic {topic} created successfully")
                    except Exception as e:
                        logger.error(f"Failed to create topic {topic}: {str(e)}")

            return {
                "existing_topics": list(existing_topics),
                "created_topics": [t.name for t in topics_to_create],
            }

        finally:
            admin_client.close()
