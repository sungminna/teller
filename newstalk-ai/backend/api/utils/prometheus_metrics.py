"""
Stage 7: Prometheus Metrics Integration
Comprehensive monitoring for Airflow, LangGraph, API, and business metrics
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
import json

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        multiprocess, values
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return self
        def labels(self, *args, **kwargs): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    Summary = Histogram
    Info = Counter
    CollectorRegistry = None
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"

from ...shared.config.settings import get_settings
from .redis_client import cache_manager

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Comprehensive Prometheus metrics for Stage 7 monitoring"""
    
    def __init__(self):
        self.settings = get_settings()
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics disabled")
            return
        
        # === AIRFLOW METRICS ===
        self.airflow_dag_duration = Histogram(
            'airflow_dag_duration_seconds',
            'Duration of DAG execution in seconds',
            ['dag_id', 'status'],
            registry=self.registry
        )
        
        self.airflow_dag_success_rate = Gauge(
            'airflow_dag_success_rate',
            'DAG success rate percentage',
            ['dag_id'],
            registry=self.registry
        )
        
        self.airflow_task_duration = Histogram(
            'airflow_task_duration_seconds',
            'Duration of task execution in seconds',
            ['dag_id', 'task_id', 'status'],
            registry=self.registry
        )
        
        self.airflow_queue_size = Gauge(
            'airflow_queue_size',
            'Number of tasks in queue',
            ['queue_name'],
            registry=self.registry
        )
        
        self.airflow_worker_health = Gauge(
            'airflow_worker_health',
            'Health status of Airflow workers',
            ['worker_id'],
            registry=self.registry
        )
        
        # === LANGGRAPH METRICS ===
        self.langgraph_agent_execution_time = Histogram(
            'langgraph_agent_execution_seconds',
            'Agent execution time in seconds',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.langgraph_agent_success_rate = Gauge(
            'langgraph_agent_success_rate',
            'Agent success rate percentage',
            ['agent_name'],
            registry=self.registry
        )
        
        self.langgraph_checkpoint_frequency = Counter(
            'langgraph_checkpoints_total',
            'Total number of checkpoints created',
            ['agent_name', 'checkpoint_type'],
            registry=self.registry
        )
        
        self.langgraph_graph_execution_time = Histogram(
            'langgraph_graph_execution_seconds',
            'Graph execution time in seconds',
            ['graph_name', 'status'],
            registry=self.registry
        )
        
        self.langgraph_memory_usage = Gauge(
            'langgraph_memory_usage_bytes',
            'Memory usage of LangGraph processes',
            ['process_id', 'agent_name'],
            registry=self.registry
        )
        
        # === API METRICS ===
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_request_count = Counter(
            'api_requests_total',
            'Total number of API requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.api_error_rate = Gauge(
            'api_error_rate',
            'API error rate percentage',
            ['endpoint'],
            registry=self.registry
        )
        
        self.api_throughput = Gauge(
            'api_throughput_requests_per_second',
            'API throughput in requests per second',
            ['endpoint'],
            registry=self.registry
        )
        
        self.api_concurrent_connections = Gauge(
            'api_concurrent_connections',
            'Number of concurrent API connections',
            ['connection_type'],
            registry=self.registry
        )
        
        # === BUSINESS METRICS ===
        self.user_satisfaction_score = Gauge(
            'user_satisfaction_score',
            'User satisfaction score (1-10)',
            ['user_segment'],
            registry=self.registry
        )
        
        self.content_quality_score = Gauge(
            'content_quality_score',
            'Content quality score (1-10)',
            ['content_type', 'source'],
            registry=self.registry
        )
        
        self.articles_processed = Counter(
            'articles_processed_total',
            'Total number of articles processed',
            ['source', 'status'],
            registry=self.registry
        )
        
        self.user_engagement_time = Histogram(
            'user_engagement_seconds',
            'User engagement time in seconds',
            ['content_type', 'user_segment'],
            registry=self.registry
        )
        
        self.revenue_metrics = Gauge(
            'revenue_daily_usd',
            'Daily revenue in USD',
            ['revenue_type'],
            registry=self.registry
        )
        
        # === LLM COST METRICS ===
        self.llm_cost_daily = Gauge(
            'llm_cost_daily_usd',
            'Daily LLM cost in USD',
            ['model'],
            registry=self.registry
        )
        
        self.llm_token_usage = Counter(
            'llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'token_type'],
            registry=self.registry
        )
        
        self.llm_request_latency = Histogram(
            'llm_request_duration_seconds',
            'LLM request latency in seconds',
            ['model', 'request_type'],
            registry=self.registry
        )
        
        # === SYSTEM METRICS ===
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            ['service'],
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            ['service'],
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_bytes',
            'System disk usage in bytes',
            ['service', 'mount_point'],
            registry=self.registry
        )
        
        self.redis_connections = Gauge(
            'redis_connections_active',
            'Number of active Redis connections',
            ['redis_instance'],
            registry=self.registry
        )
        
        self.kafka_lag = Gauge(
            'kafka_consumer_lag',
            'Kafka consumer lag',
            ['topic', 'consumer_group'],
            registry=self.registry
        )
        
        logger.info("✅ Prometheus metrics initialized successfully")
    
    # === AIRFLOW METRIC METHODS ===
    
    def record_dag_execution(self, dag_id: str, duration: float, status: str):
        """Record DAG execution metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.airflow_dag_duration.labels(dag_id=dag_id, status=status).observe(duration)
        logger.debug(f"Recorded DAG execution: {dag_id} - {duration:.2f}s - {status}")
    
    def update_dag_success_rate(self, dag_id: str, success_rate: float):
        """Update DAG success rate"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.airflow_dag_success_rate.labels(dag_id=dag_id).set(success_rate)
    
    def record_task_execution(self, dag_id: str, task_id: str, duration: float, status: str):
        """Record task execution metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.airflow_task_duration.labels(
            dag_id=dag_id, 
            task_id=task_id, 
            status=status
        ).observe(duration)
    
    def update_queue_size(self, queue_name: str, size: int):
        """Update queue size metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.airflow_queue_size.labels(queue_name=queue_name).set(size)
    
    def update_worker_health(self, worker_id: str, health_status: int):
        """Update worker health (1 = healthy, 0 = unhealthy)"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.airflow_worker_health.labels(worker_id=worker_id).set(health_status)
    
    # === LANGGRAPH METRIC METHODS ===
    
    def record_agent_execution(self, agent_name: str, duration: float, status: str):
        """Record agent execution metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.langgraph_agent_execution_time.labels(
            agent_name=agent_name, 
            status=status
        ).observe(duration)
    
    def update_agent_success_rate(self, agent_name: str, success_rate: float):
        """Update agent success rate"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.langgraph_agent_success_rate.labels(agent_name=agent_name).set(success_rate)
    
    def record_checkpoint(self, agent_name: str, checkpoint_type: str):
        """Record checkpoint creation"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.langgraph_checkpoint_frequency.labels(
            agent_name=agent_name, 
            checkpoint_type=checkpoint_type
        ).inc()
    
    def record_graph_execution(self, graph_name: str, duration: float, status: str):
        """Record graph execution metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.langgraph_graph_execution_time.labels(
            graph_name=graph_name, 
            status=status
        ).observe(duration)
    
    def update_memory_usage(self, process_id: str, agent_name: str, memory_bytes: int):
        """Update memory usage metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.langgraph_memory_usage.labels(
            process_id=process_id, 
            agent_name=agent_name
        ).set(memory_bytes)
    
    # === API METRIC METHODS ===
    
    def record_api_request(self, method: str, endpoint: str, duration: float, status_code: int):
        """Record API request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.api_request_duration.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).observe(duration)
        
        self.api_request_count.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
    
    def update_api_error_rate(self, endpoint: str, error_rate: float):
        """Update API error rate"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.api_error_rate.labels(endpoint=endpoint).set(error_rate)
    
    def update_api_throughput(self, endpoint: str, rps: float):
        """Update API throughput (requests per second)"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.api_throughput.labels(endpoint=endpoint).set(rps)
    
    def update_concurrent_connections(self, connection_type: str, count: int):
        """Update concurrent connections count"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.api_concurrent_connections.labels(connection_type=connection_type).set(count)
    
    # === BUSINESS METRIC METHODS ===
    
    def update_user_satisfaction(self, user_segment: str, score: float):
        """Update user satisfaction score"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.user_satisfaction_score.labels(user_segment=user_segment).set(score)
    
    def update_content_quality(self, content_type: str, source: str, score: float):
        """Update content quality score"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.content_quality_score.labels(
            content_type=content_type, 
            source=source
        ).set(score)
    
    def record_article_processed(self, source: str, status: str):
        """Record article processing"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.articles_processed.labels(source=source, status=status).inc()
    
    def record_user_engagement(self, content_type: str, user_segment: str, duration: float):
        """Record user engagement time"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.user_engagement_time.labels(
            content_type=content_type, 
            user_segment=user_segment
        ).observe(duration)
    
    def update_revenue(self, revenue_type: str, amount: float):
        """Update daily revenue metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.revenue_metrics.labels(revenue_type=revenue_type).set(amount)
    
    # === LLM COST METRIC METHODS ===
    
    def update_llm_cost(self, model: str, daily_cost: float):
        """Update daily LLM cost"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.llm_cost_daily.labels(model=model).set(daily_cost)
    
    def record_llm_tokens(self, model: str, token_type: str, count: int):
        """Record LLM token usage"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.llm_token_usage.labels(model=model, token_type=token_type).inc(count)
    
    def record_llm_latency(self, model: str, request_type: str, duration: float):
        """Record LLM request latency"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.llm_request_latency.labels(
            model=model, 
            request_type=request_type
        ).observe(duration)
    
    # === SYSTEM METRIC METHODS ===
    
    def update_system_metrics(self, service: str, cpu_percent: float, memory_bytes: int):
        """Update system resource metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.system_cpu_usage.labels(service=service).set(cpu_percent)
        self.system_memory_usage.labels(service=service).set(memory_bytes)
    
    def update_disk_usage(self, service: str, mount_point: str, usage_bytes: int):
        """Update disk usage metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.system_disk_usage.labels(
            service=service, 
            mount_point=mount_point
        ).set(usage_bytes)
    
    def update_redis_connections(self, redis_instance: str, count: int):
        """Update Redis connection count"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.redis_connections.labels(redis_instance=redis_instance).set(count)
    
    def update_kafka_lag(self, topic: str, consumer_group: str, lag: int):
        """Update Kafka consumer lag"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.kafka_lag.labels(topic=topic, consumer_group=consumer_group).set(lag)
    
    # === UTILITY METHODS ===
    
    def get_metrics_data(self) -> bytes:
        """Get Prometheus metrics data"""
        if not PROMETHEUS_AVAILABLE or not self.registry:
            return b"# Prometheus not available\n"
        
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get Prometheus content type"""
        return CONTENT_TYPE_LATEST
    
    async def collect_system_metrics(self):
        """Collect system metrics from various sources"""
        try:
            # This would be implemented to collect actual system metrics
            # For now, we'll use placeholder values
            
            # CPU and Memory (would use psutil in real implementation)
            self.update_system_metrics("api", 45.2, 1024 * 1024 * 512)  # 512MB
            self.update_system_metrics("airflow", 30.1, 1024 * 1024 * 1024)  # 1GB
            self.update_system_metrics("langgraph", 25.5, 1024 * 1024 * 256)  # 256MB
            
            # Redis connections (would query Redis INFO)
            self.update_redis_connections("main", 45)
            self.update_redis_connections("cache", 23)
            
            # Kafka lag (would query Kafka consumer groups)
            self.update_kafka_lag("raw-news", "news-processor", 12)
            self.update_kafka_lag("processed-news", "api-consumer", 5)
            
            logger.debug("System metrics collected")
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def collect_business_metrics(self):
        """Collect business metrics from Redis cache"""
        try:
            # Get cached business metrics
            user_satisfaction = await cache_manager.cache_get("metrics:user_satisfaction", default=8.5)
            content_quality = await cache_manager.cache_get("metrics:content_quality", default=7.8)
            
            self.update_user_satisfaction("premium", float(user_satisfaction))
            self.update_content_quality("news", "rss", float(content_quality))
            
            # Revenue metrics (placeholder)
            self.update_revenue("subscription", 1250.0)
            self.update_revenue("advertising", 340.0)
            
            logger.debug("Business metrics collected")
            
        except Exception as e:
            logger.error(f"Failed to collect business metrics: {e}")

# Global metrics instance
prometheus_metrics = PrometheusMetrics()

# Decorators for automatic metric collection

def track_api_request(endpoint: str = None):
    """Decorator to track API request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            start_time = time.time()
            status_code = 200
            
            try:
                # Execute the function
                result = await func(request, *args, **kwargs)
                
                # Extract status code if available
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                
                return result
                
            except Exception as e:
                status_code = 500
                raise
            
            finally:
                # Record metrics
                duration = time.time() - start_time
                method = getattr(request, 'method', 'GET')
                endpoint_name = endpoint or getattr(request, 'url', {}).get('path', '/unknown')
                
                prometheus_metrics.record_api_request(
                    method=method,
                    endpoint=endpoint_name,
                    duration=duration,
                    status_code=status_code
                )
        
        return wrapper
    return decorator

def track_dag_execution(dag_id: str):
    """Decorator to track DAG execution metrics"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
                
            except Exception as e:
                status = "failed"
                raise
            
            finally:
                # Record metrics
                duration = time.time() - start_time
                prometheus_metrics.record_dag_execution(dag_id, duration, status)
        
        return wrapper
    return decorator

def track_agent_execution(agent_name: str):
    """Decorator to track LangGraph agent execution metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                return result
                
            except Exception as e:
                status = "failed"
                raise
            
            finally:
                # Record metrics
                duration = time.time() - start_time
                prometheus_metrics.record_agent_execution(agent_name, duration, status)
        
        return wrapper
    return decorator

def track_llm_request(model: str, request_type: str = "completion"):
    """Decorator to track LLM request metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Extract token usage if available
                if isinstance(result, dict) and "usage" in result:
                    usage = result["usage"]
                    prometheus_metrics.record_llm_tokens(
                        model, "input", usage.get("prompt_tokens", 0)
                    )
                    prometheus_metrics.record_llm_tokens(
                        model, "output", usage.get("completion_tokens", 0)
                    )
                
                return result
                
            finally:
                # Record latency
                duration = time.time() - start_time
                prometheus_metrics.record_llm_latency(model, request_type, duration)
        
        return wrapper
    return decorator 