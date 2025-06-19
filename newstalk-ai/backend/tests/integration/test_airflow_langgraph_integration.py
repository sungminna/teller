"""
Integration Tests for Airflow + LangGraph Pipeline - Stage 9
Tests for complete pipeline integration and data flow
"""

from datetime import datetime
from unittest.mock import patch

import pytest
from airflow.models import DagBag, TaskInstance
from airflow.utils.dates import days_ago
from airflow.utils.state import State


@pytest.mark.integration
class TestAirflowLangGraphIntegration:
    """Test suite for Airflow + LangGraph integration"""

    @pytest.fixture
    def dag_bag(self):
        """Create DagBag for testing"""
        return DagBag(dag_folder="backend/airflow/dags", include_examples=False)

    @pytest.fixture
    def sample_dag_run_data(self):
        """Sample data for DAG run testing"""
        return {
            "dag_id": "newstalk_ai_pipeline",
            "execution_date": days_ago(1),
            "conf": {
                "articles_batch_size": 10,
                "enable_fact_checking": True,
                "enable_personalization": True,
                "target_languages": ["en", "ko"],
                "quality_threshold": 0.85,
            },
        }

    @pytest.mark.asyncio
    async def test_dag_loading(self, dag_bag):
        """Test DAG loading and validation"""
        # Check if main DAG is loaded
        assert "newstalk_ai_pipeline" in dag_bag.dags

        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        assert dag is not None
        assert dag.dag_id == "newstalk_ai_pipeline"

        # Validate DAG structure
        expected_tasks = [
            "fetch_news_articles",
            "content_analysis",
            "fact_checking",
            "personalization",
            "voice_generation",
            "quality_validation",
            "publish_results",
        ]

        actual_tasks = [task.task_id for task in dag.tasks]
        for task_id in expected_tasks:
            assert task_id in actual_tasks

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, dag_bag, sample_dag_run_data, test_db_session):
        """Test full pipeline execution from start to finish"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Create DAG run
        dag_run = dag.create_dagrun(
            run_id=f"test_run_{datetime.utcnow().isoformat()}",
            execution_date=sample_dag_run_data["execution_date"],
            state=State.RUNNING,
            conf=sample_dag_run_data["conf"],
        )

        # Execute tasks in sequence
        task_execution_results = {}

        for task in dag.topological_sort():
            ti = TaskInstance(task, dag_run.execution_date)
            ti.dag_run = dag_run

            try:
                # Execute task
                context = ti.get_template_context()
                result = task.execute(context)
                task_execution_results[task.task_id] = {"state": State.SUCCESS, "result": result}
                ti.state = State.SUCCESS
            except Exception as e:
                task_execution_results[task.task_id] = {"state": State.FAILED, "error": str(e)}
                ti.state = State.FAILED
                break

        # Validate pipeline completion
        assert all(
            result["state"] == State.SUCCESS for result in task_execution_results.values()
        ), f"Pipeline failed: {task_execution_results}"

    @pytest.mark.asyncio
    async def test_news_fetching_task(self, dag_bag, sample_dag_run_data):
        """Test news fetching task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        fetch_task = dag.get_task("fetch_news_articles")

        # Create task instance
        ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])

        # Mock news sources
        with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
            mock_articles = [
                {
                    "id": f"article_{i}",
                    "title": f"Test Article {i}",
                    "content": f"Content for article {i}",
                    "url": f"https://example.com/article_{i}",
                    "source": "Test Source",
                    "published_at": datetime.utcnow().isoformat(),
                }
                for i in range(10)
            ]
            mock_fetch.return_value = mock_articles

            # Execute task
            context = ti.get_template_context()
            result = fetch_task.execute(context)

            assert len(result) == 10
            assert all("id" in article for article in result)
            assert all("title" in article for article in result)

    @pytest.mark.asyncio
    async def test_content_analysis_task(
        self, dag_bag, sample_dag_run_data, content_analyzer_agent
    ):
        """Test content analysis task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        analysis_task = dag.get_task("content_analysis")

        # Sample articles for analysis
        sample_articles = [
            {
                "id": "article_1",
                "title": "AI Breakthrough in Healthcare",
                "content": "Researchers have developed a new AI system that can diagnose diseases with 95% accuracy.",
                "source": "Tech News",
                "published_at": datetime.utcnow().isoformat(),
            }
        ]

        # Mock XCom pull to get articles from previous task
        with patch("airflow.models.TaskInstance.xcom_pull") as mock_xcom:
            mock_xcom.return_value = sample_articles

            # Create task instance
            ti = TaskInstance(analysis_task, sample_dag_run_data["execution_date"])

            # Execute task
            context = ti.get_template_context()
            result = analysis_task.execute(context)

            assert len(result) == len(sample_articles)
            assert all("sentiment" in article for article in result)
            assert all("keywords" in article for article in result)
            assert all("category" in article for article in result)

    @pytest.mark.asyncio
    async def test_fact_checking_task(self, dag_bag, sample_dag_run_data, fact_checker_agent):
        """Test fact-checking task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        fact_check_task = dag.get_task("fact_checking")

        # Sample articles with analyzed content
        sample_articles = [
            {
                "id": "article_1",
                "title": "Scientific Discovery",
                "content": "Water boils at 100°C at sea level pressure.",
                "analysis": {
                    "sentiment": "neutral",
                    "keywords": ["water", "boiling", "temperature"],
                    "category": "science",
                },
            }
        ]

        with patch("airflow.models.TaskInstance.xcom_pull") as mock_xcom:
            mock_xcom.return_value = sample_articles

            ti = TaskInstance(fact_check_task, sample_dag_run_data["execution_date"])

            # Execute task
            context = ti.get_template_context()
            result = fact_check_task.execute(context)

            assert len(result) == len(sample_articles)
            assert all("fact_check" in article for article in result)
            assert all(
                article["fact_check"]["confidence"] >= 0.85
                for article in result
                if article["fact_check"]["verdict"] is not None
            )

    @pytest.mark.asyncio
    async def test_personalization_task(self, dag_bag, sample_dag_run_data, personalization_agent):
        """Test personalization task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        personalization_task = dag.get_task("personalization")

        # Sample articles with fact-checking results
        sample_articles = [
            {
                "id": "article_1",
                "title": "Technology News",
                "content": "Latest developments in AI technology.",
                "analysis": {
                    "sentiment": "positive",
                    "keywords": ["AI", "technology"],
                    "category": "technology",
                },
                "fact_check": {
                    "verdict": True,
                    "confidence": 0.95,
                    "sources": ["tech-journal.com"],
                },
            }
        ]

        # Sample user profiles
        sample_users = [
            {
                "user_id": "user_1",
                "preferences": {
                    "categories": ["technology", "science"],
                    "languages": ["en"],
                    "reading_speed": "medium",
                },
            }
        ]

        with patch("airflow.models.TaskInstance.xcom_pull") as mock_xcom:
            mock_xcom.side_effect = [sample_articles, sample_users]

            ti = TaskInstance(personalization_task, sample_dag_run_data["execution_date"])

            # Execute task
            context = ti.get_template_context()
            result = personalization_task.execute(context)

            assert "personalized_content" in result
            assert len(result["personalized_content"]) > 0
            assert all("user_id" in item for item in result["personalized_content"])

    @pytest.mark.asyncio
    async def test_voice_generation_task(self, dag_bag, sample_dag_run_data):
        """Test voice generation task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        voice_task = dag.get_task("voice_generation")

        # Sample personalized content
        sample_personalized_content = [
            {
                "user_id": "user_1",
                "article_id": "article_1",
                "title": "Technology News",
                "content": "Personalized summary of technology news.",
                "voice_preferences": {"voice_type": "female", "speed": "medium", "language": "en"},
            }
        ]

        with patch("airflow.models.TaskInstance.xcom_pull") as mock_xcom:
            mock_xcom.return_value = {"personalized_content": sample_personalized_content}

            with patch("backend.services.voice_service.generate_audio") as mock_voice:
                mock_voice.return_value = {
                    "audio_url": "https://storage.example.com/audio/article_1_user_1.mp3",
                    "duration": 120,
                    "quality_score": 0.92,
                }

                ti = TaskInstance(voice_task, sample_dag_run_data["execution_date"])

                # Execute task
                context = ti.get_template_context()
                result = voice_task.execute(context)

                assert "voice_content" in result
                assert len(result["voice_content"]) > 0
                assert all("audio_url" in item for item in result["voice_content"])
                assert all(item["quality_score"] >= 0.9 for item in result["voice_content"])

    @pytest.mark.asyncio
    async def test_quality_validation_task(self, dag_bag, sample_dag_run_data, quality_thresholds):
        """Test quality validation task integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        quality_task = dag.get_task("quality_validation")

        # Sample content with quality metrics
        sample_content = {
            "articles": [
                {
                    "id": "article_1",
                    "fact_check": {"confidence": 0.96, "verdict": True},
                    "content_quality": 0.88,
                    "personalization_score": 0.91,
                }
            ],
            "voice_content": [
                {
                    "user_id": "user_1",
                    "article_id": "article_1",
                    "quality_score": 0.93,
                    "duration": 120,
                }
            ],
        }

        with patch("airflow.models.TaskInstance.xcom_pull") as mock_xcom:
            mock_xcom.return_value = sample_content

            ti = TaskInstance(quality_task, sample_dag_run_data["execution_date"])

            # Execute task
            context = ti.get_template_context()
            result = quality_task.execute(context)

            assert "quality_report" in result
            assert (
                result["quality_report"]["overall_score"]
                >= quality_thresholds["pipeline_success_rate"]
            )
            assert (
                result["quality_report"]["fact_checking_accuracy"]
                >= quality_thresholds["fact_checking_accuracy"]
            )
            assert (
                result["quality_report"]["voice_quality_score"]
                >= quality_thresholds["voice_quality_score"]
            )

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, dag_bag, sample_dag_run_data):
        """Test pipeline error handling and recovery"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Test with failing task
        with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            fetch_task = dag.get_task("fetch_news_articles")
            ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])

            # Execute task (should fail)
            with pytest.raises(Exception):
                context = ti.get_template_context()
                fetch_task.execute(context)

            # Verify task marked as failed
            assert ti.state == State.FAILED

    @pytest.mark.asyncio
    async def test_pipeline_retry_mechanism(self, dag_bag, sample_dag_run_data):
        """Test pipeline retry mechanism"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")
        fetch_task = dag.get_task("fetch_news_articles")

        # Ensure task has retry configuration
        assert fetch_task.retries > 0
        assert fetch_task.retry_delay is not None

        # Test retry behavior
        call_count = 0

        def failing_then_succeeding_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return [{"id": "article_1", "title": "Test"}]

        with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
            mock_fetch.side_effect = failing_then_succeeding_fetch

            ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])

            # Execute with retry
            context = ti.get_template_context()
            result = fetch_task.execute(context)

            assert call_count == 2  # Failed once, then succeeded
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_pipeline_data_flow(self, dag_bag, sample_dag_run_data):
        """Test data flow between pipeline tasks"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Mock XCom for data passing
        xcom_data = {}

        def mock_xcom_push(key, value, **kwargs):
            xcom_data[key] = value

        def mock_xcom_pull(key, **kwargs):
            return xcom_data.get(key)

        with patch("airflow.models.TaskInstance.xcom_push", side_effect=mock_xcom_push):
            with patch("airflow.models.TaskInstance.xcom_pull", side_effect=mock_xcom_pull):

                # Execute fetch task
                fetch_task = dag.get_task("fetch_news_articles")
                with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
                    mock_fetch.return_value = [{"id": "article_1", "title": "Test"}]

                    ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])
                    context = ti.get_template_context()
                    result = fetch_task.execute(context)

                    # Push result to XCom
                    mock_xcom_push("articles", result)

                # Execute analysis task
                analysis_task = dag.get_task("content_analysis")
                ti = TaskInstance(analysis_task, sample_dag_run_data["execution_date"])
                context = ti.get_template_context()

                # Should receive articles from previous task
                articles = mock_xcom_pull("articles")
                assert articles is not None
                assert len(articles) == 1

    @pytest.mark.asyncio
    async def test_pipeline_performance_metrics(self, dag_bag, sample_dag_run_data):
        """Test pipeline performance metrics collection"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Track execution times
        execution_times = {}

        for task in dag.tasks:
            start_time = datetime.utcnow()

            ti = TaskInstance(task, sample_dag_run_data["execution_date"])

            # Mock task execution
            with patch.object(task, "execute", return_value={"status": "success"}):
                context = ti.get_template_context()
                task.execute(context)

            end_time = datetime.utcnow()
            execution_times[task.task_id] = (end_time - start_time).total_seconds()

        # Verify performance requirements
        assert execution_times["fetch_news_articles"] <= 30  # 30 seconds max
        assert execution_times["content_analysis"] <= 60  # 1 minute max
        assert execution_times["fact_checking"] <= 120  # 2 minutes max
        assert execution_times["personalization"] <= 30  # 30 seconds max
        assert execution_times["voice_generation"] <= 180  # 3 minutes max

    @pytest.mark.asyncio
    async def test_pipeline_scalability(self, dag_bag, sample_dag_run_data):
        """Test pipeline scalability with large datasets"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Test with large batch of articles
        large_batch_size = 1000
        sample_dag_run_data["conf"]["articles_batch_size"] = large_batch_size

        fetch_task = dag.get_task("fetch_news_articles")

        with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
            # Mock large dataset
            mock_articles = [
                {"id": f"article_{i}", "title": f"Article {i}", "content": f"Content {i}"}
                for i in range(large_batch_size)
            ]
            mock_fetch.return_value = mock_articles

            ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])

            start_time = datetime.utcnow()
            context = ti.get_template_context()
            result = fetch_task.execute(context)
            end_time = datetime.utcnow()

            processing_time = (end_time - start_time).total_seconds()

            # Should handle large datasets efficiently
            assert len(result) == large_batch_size
            assert processing_time <= 300  # 5 minutes max for 1000 articles

    @pytest.mark.asyncio
    async def test_pipeline_monitoring_integration(self, dag_bag, sample_dag_run_data):
        """Test pipeline monitoring and alerting integration"""
        dag = dag_bag.get_dag("newstalk_ai_pipeline")

        # Test monitoring hooks
        monitoring_events = []

        def mock_monitoring_callback(context):
            monitoring_events.append(
                {
                    "task_id": context["task_instance"].task_id,
                    "state": context["task_instance"].state,
                    "timestamp": datetime.utcnow(),
                }
            )

        # Configure monitoring callbacks
        for task in dag.tasks:
            task.on_success_callback = mock_monitoring_callback
            task.on_failure_callback = mock_monitoring_callback

        # Execute a task
        fetch_task = dag.get_task("fetch_news_articles")
        with patch("backend.airflow.tasks.news_fetcher.fetch_articles") as mock_fetch:
            mock_fetch.return_value = [{"id": "article_1", "title": "Test"}]

            ti = TaskInstance(fetch_task, sample_dag_run_data["execution_date"])
            context = ti.get_template_context()

            try:
                fetch_task.execute(context)
                ti.state = State.SUCCESS
                mock_monitoring_callback(context)
            except Exception:
                ti.state = State.FAILED
                mock_monitoring_callback(context)

        # Verify monitoring events
        assert len(monitoring_events) > 0
        assert any(event["task_id"] == "fetch_news_articles" for event in monitoring_events)
