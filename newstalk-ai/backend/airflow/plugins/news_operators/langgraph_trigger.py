"""
LangGraph Trigger Operator for NewsTeam AI
Triggers AI processing workflows
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

import httpx
from airflow.configuration import conf
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from shared.models.news import NewsArticle, ProcessingStatus
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class LangGraphTriggerOperator(BaseOperator):
    """Trigger LangGraph AI workflows for news analysis"""

    template_fields = ["workflow_type", "batch_size"]

    @apply_defaults
    def __init__(
        self,
        workflow_type: str = "news_analysis",
        langgraph_endpoint: str = "http://localhost:8000/api/v1/langgraph",
        batch_size: int = 20,
        timeout_seconds: int = 300,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workflow_type = workflow_type
        self.langgraph_endpoint = langgraph_endpoint
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds

        # Database setup
        self.database_url = conf.get("core", "sql_alchemy_conn")
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute(self, context):
        """Execute workflow triggering"""
        logger.info(f"Starting LangGraph workflow: {self.workflow_type}")

        # Get published articles
        articles = self._get_published_articles()
        if not articles:
            return {"triggered_workflows": 0}

        # Trigger workflows
        results = asyncio.run(self._trigger_workflows(articles))
        logger.info(f"Workflow completed: {results}")
        return results

    def _get_published_articles(self) -> List[NewsArticle]:
        """Get articles ready for AI processing"""
        session = self.Session()
        try:
            return (
                session.query(NewsArticle)
                .filter(NewsArticle.status == ProcessingStatus.PUBLISHED)
                .limit(self.batch_size)
                .all()
            )
        finally:
            session.close()

    async def _trigger_workflows(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Trigger workflows asynchronously"""
        results = {"total": len(articles), "successful": 0, "failed": 0, "workflow_ids": []}

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            tasks = [self._trigger_single_workflow(client, article) for article in articles]
            workflow_results = await asyncio.gather(*tasks, return_exceptions=True)

            session = self.Session()
            try:
                for i, result in enumerate(workflow_results):
                    article = articles[i]

                    if isinstance(result, Exception):
                        results["failed"] += 1
                        logger.error(f"Workflow failed for {article.id}: {result}")
                    else:
                        results["successful"] += 1
                        results["workflow_ids"].append(result["workflow_id"])

                        # Update article status
                        article.status = ProcessingStatus.AI_ANALYZED
                        article.metadata = article.metadata or {}
                        article.metadata["langgraph_workflow"] = {
                            "workflow_id": result["workflow_id"],
                            "triggered_at": datetime.utcnow().isoformat(),
                        }

                session.commit()
            finally:
                session.close()

        return results

    async def _trigger_single_workflow(
        self, client: httpx.AsyncClient, article: NewsArticle
    ) -> Dict[str, Any]:
        """Trigger single workflow"""
        payload = {
            "workflow_type": self.workflow_type,
            "article_data": {
                "id": article.id,
                "title": article.title,
                "content": article.content,
                "url": article.url,
                "category": article.category,
            },
            "config": {
                "enable_fact_checking": True,
                "enable_sentiment_analysis": True,
                "quality_threshold": 0.95,
            },
        }

        response = await client.post(f"{self.langgraph_endpoint}/trigger", json=payload)
        response.raise_for_status()
        return response.json()


class LangGraphStatusOperator(BaseOperator):
    """
    Check status of LangGraph workflows
    """

    @apply_defaults
    def __init__(
        self,
        langgraph_endpoint: str = "http://localhost:8000/api/v1/langgraph",
        check_hours: int = 24,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.langgraph_endpoint = langgraph_endpoint
        self.check_hours = check_hours

        # Database setup
        self.database_url = conf.get("core", "sql_alchemy_conn")
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)

    def execute(self, context):
        """Check workflow statuses"""
        logger.info("Checking LangGraph workflow statuses")

        # Get recent articles with workflows
        articles_with_workflows = self._get_articles_with_workflows()

        if not articles_with_workflows:
            return {"checked_workflows": 0}

        # Check workflow statuses
        status_results = self._check_workflow_statuses(articles_with_workflows)

        logger.info(f"Status check completed: {status_results}")
        return status_results

    def _get_articles_with_workflows(self) -> List[NewsArticle]:
        """Get articles with active workflows"""
        from datetime import timedelta

        session = self.Session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.check_hours)

            return (
                session.query(NewsArticle)
                .filter(
                    NewsArticle.status == ProcessingStatus.AI_ANALYZED,
                    NewsArticle.processed_at >= cutoff_time,
                )
                .all()
            )
        finally:
            session.close()

    def _check_workflow_statuses(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Check workflow statuses via API"""
        results = {
            "total_checked": len(articles),
            "completed": 0,
            "running": 0,
            "failed": 0,
            "unknown": 0,
        }

        with httpx.Client(timeout=30) as client:
            for article in articles:
                try:
                    workflow_info = article.metadata.get("langgraph_workflow", {})
                    workflow_id = workflow_info.get("workflow_id")

                    if not workflow_id:
                        results["unknown"] += 1
                        continue

                    # Check workflow status
                    response = client.get(f"{self.langgraph_endpoint}/status/{workflow_id}")

                    if response.status_code == 200:
                        status_data = response.json()
                        status = status_data.get("status", "unknown")

                        if status == "completed":
                            results["completed"] += 1
                        elif status == "running":
                            results["running"] += 1
                        elif status == "failed":
                            results["failed"] += 1
                        else:
                            results["unknown"] += 1
                    else:
                        results["unknown"] += 1

                except Exception as e:
                    logger.error(f"Status check failed for article {article.id}: {str(e)}")
                    results["unknown"] += 1

        return results
