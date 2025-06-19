"""
Stage 7: Langfuse Integration for Comprehensive LLM Monitoring
Automatic tracking, prompt versioning, cost monitoring, and performance insights
"""

import logging
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Dict, List, Optional

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None


from ...shared.config.settings import get_settings
from .redis_client import cache_manager

logger = logging.getLogger(__name__)


class LangfuseManager:
    """Comprehensive Langfuse integration for Stage 7 monitoring"""

    def __init__(self):
        self.settings = get_settings()
        self.langfuse = None
        self.daily_cost_limit = 50.0  # $50 daily limit
        self.cost_alert_threshold = 0.8  # Alert at 80% of limit
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Langfuse client with error handling"""
        try:
            if not Langfuse:
                logger.warning("Langfuse not installed, monitoring disabled")
                return

            if not all(
                [
                    getattr(self.settings, "LANGFUSE_PUBLIC_KEY", None),
                    getattr(self.settings, "LANGFUSE_SECRET_KEY", None),
                    getattr(self.settings, "LANGFUSE_HOST", None),
                ]
            ):
                logger.warning("Langfuse credentials not configured, monitoring disabled")
                return

            self.langfuse = Langfuse(
                public_key=self.settings.LANGFUSE_PUBLIC_KEY,
                secret_key=self.settings.LANGFUSE_SECRET_KEY,
                host=self.settings.LANGFUSE_HOST,
            )

            logger.info("✅ Langfuse client initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Langfuse client: {e}")
            self.langfuse = None

    async def track_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        usage: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track LLM call with comprehensive monitoring"""
        if not self.langfuse:
            return {"tracked": False, "reason": "Langfuse not initialized"}

        try:
            # Generate IDs if not provided
            trace_id = trace_id or str(uuid.uuid4())
            generation_id = str(uuid.uuid4())

            # Calculate cost
            cost = await self._calculate_cost(model, usage)

            # Check daily cost limit
            daily_cost = await self._get_daily_cost()
            if daily_cost + cost > self.daily_cost_limit:
                await self._send_cost_alert(daily_cost + cost)
                logger.warning(f"Daily cost limit approaching: ${daily_cost + cost:.2f}")

            # Create generation tracking
            generation = self.langfuse.generation(
                id=generation_id,
                trace_id=trace_id,
                name=f"{model}_generation",
                model=model,
                input=prompt,
                output=response,
                usage=usage,
                metadata={
                    **(metadata or {}),
                    "cost_usd": cost,
                    "daily_cost_total": daily_cost + cost,
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": getattr(self.settings, "ENVIRONMENT", "development"),
                },
            )

            # Update daily cost tracking
            await self._update_daily_cost(cost)

            # Store performance metrics
            await self._store_performance_metrics(model, usage, cost)

            logger.info(f"LLM call tracked - Model: {model}, Cost: ${cost:.4f}")

            return {
                "tracked": True,
                "trace_id": trace_id,
                "generation_id": generation_id,
                "cost": cost,
                "daily_cost_total": daily_cost + cost,
            }

        except Exception as e:
            logger.error(f"Failed to track LLM call: {e}")
            return {"tracked": False, "reason": str(e)}

    async def track_agent_execution(
        self,
        agent_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track LangGraph agent execution"""
        if not self.langfuse:
            return ""

        try:
            trace_id = str(uuid.uuid4())

            # Create trace for agent execution
            trace = self.langfuse.trace(
                id=trace_id,
                name=f"agent_{agent_name}",
                input=input_data,
                output=output_data,
                metadata={
                    **(metadata or {}),
                    "agent_name": agent_name,
                    "execution_time_seconds": execution_time,
                    "success": success,
                    "timestamp": datetime.utcnow().isoformat(),
                    "environment": getattr(self.settings, "ENVIRONMENT", "development"),
                },
            )

            # Store agent performance metrics
            await self._store_agent_metrics(agent_name, execution_time, success)

            logger.info(f"Agent execution tracked - {agent_name}: {execution_time:.2f}s")

            return trace_id

        except Exception as e:
            logger.error(f"Failed to track agent execution: {e}")
            return ""

    async def track_pipeline_execution(
        self,
        pipeline_name: str,
        stages: List[Dict[str, Any]],
        total_time: float,
        success: bool,
        articles_processed: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Track complete pipeline execution"""
        if not self.langfuse:
            return ""

        try:
            trace_id = str(uuid.uuid4())

            # Create main pipeline trace
            trace = self.langfuse.trace(
                id=trace_id,
                name=f"pipeline_{pipeline_name}",
                input={"articles_count": articles_processed},
                output={"success": success, "total_time": total_time},
                metadata={
                    **(metadata or {}),
                    "pipeline_name": pipeline_name,
                    "total_time_seconds": total_time,
                    "articles_processed": articles_processed,
                    "success": success,
                    "stages_count": len(stages),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Create spans for each stage
            for i, stage in enumerate(stages):
                span_id = str(uuid.uuid4())
                span = self.langfuse.span(
                    id=span_id,
                    trace_id=trace_id,
                    name=stage.get("name", f"stage_{i}"),
                    input=stage.get("input", {}),
                    output=stage.get("output", {}),
                    metadata={
                        "stage_index": i,
                        "execution_time": stage.get("execution_time", 0),
                        "success": stage.get("success", True),
                    },
                )

            # Store pipeline metrics
            await self._store_pipeline_metrics(
                pipeline_name, total_time, success, articles_processed
            )

            logger.info(
                f"Pipeline execution tracked - {pipeline_name}: {total_time:.2f}s, {articles_processed} articles"
            )

            return trace_id

        except Exception as e:
            logger.error(f"Failed to track pipeline execution: {e}")
            return ""

    async def get_performance_insights(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get performance optimization insights"""
        try:
            # Get cached insights first
            cache_key = f"langfuse_insights_{time_range_hours}h"
            cached_insights = await cache_manager.cache_get(cache_key)
            if cached_insights:
                return cached_insights

            # Calculate insights from stored metrics
            insights = await self._calculate_performance_insights(time_range_hours)

            # Cache insights for 30 minutes
            await cache_manager.cache_set(cache_key, insights, expire=1800)

            return insights

        except Exception as e:
            logger.error(f"Failed to get performance insights: {e}")
            return {"error": str(e)}

    async def get_cost_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get cost analytics and projections"""
        try:
            # Get daily costs from Redis
            daily_costs = []
            for i in range(days):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                cost = await cache_manager.cache_get(f"daily_cost:{date}", default=0.0)
                daily_costs.append({"date": date, "cost": cost})

            # Calculate analytics
            total_cost = sum(day["cost"] for day in daily_costs)
            avg_daily_cost = total_cost / days if days > 0 else 0

            # Project monthly cost
            monthly_projection = avg_daily_cost * 30

            # Cost breakdown by model (from cached metrics)
            model_costs = await self._get_model_cost_breakdown(days)

            return {
                "daily_costs": daily_costs,
                "total_cost": total_cost,
                "average_daily_cost": avg_daily_cost,
                "monthly_projection": monthly_projection,
                "daily_limit": self.daily_cost_limit,
                "model_breakdown": model_costs,
                "cost_efficiency": await self._calculate_cost_efficiency(),
            }

        except Exception as e:
            logger.error(f"Failed to get cost analytics: {e}")
            return {"error": str(e)}

    async def manage_prompt_versions(
        self,
        prompt_name: str,
        prompt_template: str,
        version: Optional[str] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Manage prompt versions with Langfuse"""
        if not self.langfuse:
            return {"managed": False, "reason": "Langfuse not initialized"}

        try:
            # Create or update prompt
            prompt = self.langfuse.create_prompt(
                name=prompt_name,
                prompt=prompt_template,
                version=version,
                config={
                    "variables": variables or {},
                    "created_at": datetime.utcnow().isoformat(),
                    "environment": getattr(self.settings, "ENVIRONMENT", "development"),
                },
            )

            # Cache prompt for quick access
            cache_key = f"prompt:{prompt_name}:{version or 'latest'}"
            await cache_manager.cache_set(
                cache_key,
                {"template": prompt_template, "variables": variables, "version": version},
                expire=3600,
            )

            logger.info(f"Prompt version managed - {prompt_name} v{version}")

            return {
                "managed": True,
                "prompt_name": prompt_name,
                "version": version,
                "prompt_id": prompt.id if hasattr(prompt, "id") else None,
            }

        except Exception as e:
            logger.error(f"Failed to manage prompt version: {e}")
            return {"managed": False, "reason": str(e)}

    async def _calculate_cost(self, model: str, usage: Dict[str, Any]) -> float:
        """Calculate cost for LLM usage"""
        # Cost per 1K tokens (approximate rates)
        cost_rates = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }

        # Default to GPT-4 rates if model not found
        rates = cost_rates.get(model.lower(), cost_rates["gpt-4"])

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        input_cost = (input_tokens / 1000) * rates["input"]
        output_cost = (output_tokens / 1000) * rates["output"]

        return input_cost + output_cost

    async def _get_daily_cost(self) -> float:
        """Get current daily cost"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return await cache_manager.cache_get(f"daily_cost:{today}", default=0.0)

    async def _update_daily_cost(self, cost: float):
        """Update daily cost tracking"""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        current_cost = await self._get_daily_cost()
        new_cost = current_cost + cost

        # Store with 25-hour TTL to handle timezone issues
        await cache_manager.cache_set(f"daily_cost:{today}", new_cost, expire=90000)

        # Also update TimeSeries
        from .redis_client import timeseries

        await timeseries.add_sample("daily_llm_cost", value=cost)

    async def _send_cost_alert(self, total_cost: float):
        """Send cost alert when approaching limit"""
        try:
            alert_data = {
                "type": "cost_alert",
                "current_cost": total_cost,
                "daily_limit": self.daily_cost_limit,
                "percentage": (total_cost / self.daily_cost_limit) * 100,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Store alert for notification system
            await cache_manager.cache_set(
                f"cost_alert:{datetime.utcnow().timestamp()}", alert_data, expire=3600
            )

            logger.warning(f"Cost alert triggered: ${total_cost:.2f} / ${self.daily_cost_limit}")

        except Exception as e:
            logger.error(f"Failed to send cost alert: {e}")

    async def _store_performance_metrics(self, model: str, usage: Dict[str, Any], cost: float):
        """Store performance metrics in Redis"""
        try:
            # Update model usage statistics
            model_key = f"model_metrics:{model}"
            current_stats = await cache_manager.cache_hgetall(model_key)

            new_stats = {
                "total_calls": int(current_stats.get("total_calls", 0)) + 1,
                "total_tokens": int(current_stats.get("total_tokens", 0))
                + usage.get("total_tokens", 0),
                "total_cost": float(current_stats.get("total_cost", 0)) + cost,
                "last_used": datetime.utcnow().isoformat(),
            }

            for key, value in new_stats.items():
                await cache_manager.cache_hset(model_key, key, value)

            # Set TTL for cleanup (7 days)
            await cache_manager.cache_expire(model_key, 604800)

        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")

    async def _store_agent_metrics(self, agent_name: str, execution_time: float, success: bool):
        """Store agent performance metrics"""
        try:
            from .redis_client import timeseries

            # Update TimeSeries
            await timeseries.add_sample(f"agent_execution_time:{agent_name}", value=execution_time)
            await timeseries.add_sample(
                f"agent_success_rate:{agent_name}", value=1.0 if success else 0.0
            )

            # Update aggregated stats
            agent_key = f"agent_metrics:{agent_name}"
            current_stats = await cache_manager.cache_hgetall(agent_key)

            total_executions = int(current_stats.get("total_executions", 0)) + 1
            total_successes = int(current_stats.get("total_successes", 0)) + (1 if success else 0)
            total_time = float(current_stats.get("total_time", 0)) + execution_time

            new_stats = {
                "total_executions": total_executions,
                "total_successes": total_successes,
                "total_time": total_time,
                "success_rate": (total_successes / total_executions) * 100,
                "avg_execution_time": total_time / total_executions,
                "last_execution": datetime.utcnow().isoformat(),
            }

            for key, value in new_stats.items():
                await cache_manager.cache_hset(agent_key, key, value)

        except Exception as e:
            logger.error(f"Failed to store agent metrics: {e}")

    async def _store_pipeline_metrics(
        self, pipeline_name: str, total_time: float, success: bool, articles_processed: int
    ):
        """Store pipeline performance metrics"""
        try:
            from .redis_client import timeseries

            # Update TimeSeries
            await timeseries.add_sample(
                f"pipeline_execution_time:{pipeline_name}", value=total_time
            )
            await timeseries.add_sample(
                f"pipeline_articles_processed:{pipeline_name}", value=articles_processed
            )
            await timeseries.add_sample(
                f"pipeline_success_rate:{pipeline_name}", value=1.0 if success else 0.0
            )

            # Update aggregated stats
            pipeline_key = f"pipeline_metrics:{pipeline_name}"
            current_stats = await cache_manager.cache_hgetall(pipeline_key)

            total_executions = int(current_stats.get("total_executions", 0)) + 1
            total_successes = int(current_stats.get("total_successes", 0)) + (1 if success else 0)
            total_articles = int(current_stats.get("total_articles", 0)) + articles_processed

            new_stats = {
                "total_executions": total_executions,
                "total_successes": total_successes,
                "total_articles": total_articles,
                "success_rate": (total_successes / total_executions) * 100,
                "avg_articles_per_run": total_articles / total_executions,
                "last_execution": datetime.utcnow().isoformat(),
            }

            for key, value in new_stats.items():
                await cache_manager.cache_hset(pipeline_key, key, value)

        except Exception as e:
            logger.error(f"Failed to store pipeline metrics: {e}")

    async def _calculate_performance_insights(self, time_range_hours: int) -> Dict[str, Any]:
        """Calculate performance optimization insights"""
        try:
            insights = {
                "model_performance": await self._get_model_performance_insights(),
                "agent_performance": await self._get_agent_performance_insights(),
                "cost_optimization": await self._get_cost_optimization_insights(),
                "recommendations": [],
            }

            # Generate recommendations based on insights
            recommendations = []

            # Model recommendations
            if insights["model_performance"]:
                slowest_model = max(
                    insights["model_performance"].items(),
                    key=lambda x: x[1].get("avg_response_time", 0),
                )
                if slowest_model[1].get("avg_response_time", 0) > 5.0:
                    recommendations.append(
                        {
                            "type": "model_optimization",
                            "message": f"Consider switching from {slowest_model[0]} to a faster model for non-critical tasks",
                        }
                    )

            # Cost recommendations
            if insights["cost_optimization"].get("high_cost_models"):
                recommendations.append(
                    {
                        "type": "cost_optimization",
                        "message": "High-cost models detected. Consider using cheaper alternatives for simple tasks",
                    }
                )

            insights["recommendations"] = recommendations

            return insights

        except Exception as e:
            logger.error(f"Failed to calculate performance insights: {e}")
            return {"error": str(e)}

    async def _get_model_performance_insights(self) -> Dict[str, Any]:
        """Get model performance insights"""
        try:
            model_keys = await cache_manager.cache_keys("model_metrics:*")
            model_insights = {}

            for key in model_keys:
                model_name = key.split(":")[-1]
                stats = await cache_manager.cache_hgetall(key)

                if stats:
                    model_insights[model_name] = {
                        "total_calls": int(stats.get("total_calls", 0)),
                        "total_cost": float(stats.get("total_cost", 0)),
                        "avg_cost_per_call": float(stats.get("total_cost", 0))
                        / max(int(stats.get("total_calls", 1)), 1),
                        "cost_efficiency": self._calculate_model_efficiency(stats),
                    }

            return model_insights

        except Exception as e:
            logger.error(f"Failed to get model performance insights: {e}")
            return {}

    async def _get_agent_performance_insights(self) -> Dict[str, Any]:
        """Get agent performance insights"""
        try:
            agent_keys = await cache_manager.cache_keys("agent_metrics:*")
            agent_insights = {}

            for key in agent_keys:
                agent_name = key.split(":")[-1]
                stats = await cache_manager.cache_hgetall(key)

                if stats:
                    agent_insights[agent_name] = {
                        "success_rate": float(stats.get("success_rate", 0)),
                        "avg_execution_time": float(stats.get("avg_execution_time", 0)),
                        "total_executions": int(stats.get("total_executions", 0)),
                        "performance_grade": self._calculate_agent_grade(stats),
                    }

            return agent_insights

        except Exception as e:
            logger.error(f"Failed to get agent performance insights: {e}")
            return {}

    async def _get_cost_optimization_insights(self) -> Dict[str, Any]:
        """Get cost optimization insights"""
        try:
            # Get model costs
            model_costs = await self._get_model_cost_breakdown(7)

            # Identify high-cost models
            high_cost_models = [
                model
                for model, cost in model_costs.items()
                if cost > 10.0  # Models costing more than $10 in 7 days
            ]

            # Calculate cost trends
            daily_costs = []
            for i in range(7):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                cost = await cache_manager.cache_get(f"daily_cost:{date}", default=0.0)
                daily_costs.append(cost)

            # Calculate trend
            if len(daily_costs) >= 2:
                trend = "increasing" if daily_costs[0] > daily_costs[-1] else "decreasing"
            else:
                trend = "stable"

            return {
                "high_cost_models": high_cost_models,
                "cost_trend": trend,
                "potential_savings": await self._calculate_potential_savings(model_costs),
                "optimization_opportunities": await self._identify_optimization_opportunities(
                    model_costs
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cost optimization insights: {e}")
            return {}

    async def _get_model_cost_breakdown(self, days: int) -> Dict[str, float]:
        """Get cost breakdown by model"""
        try:
            model_keys = await cache_manager.cache_keys("model_metrics:*")
            model_costs = {}

            for key in model_keys:
                model_name = key.split(":")[-1]
                stats = await cache_manager.cache_hgetall(key)

                if stats:
                    model_costs[model_name] = float(stats.get("total_cost", 0))

            return model_costs

        except Exception as e:
            logger.error(f"Failed to get model cost breakdown: {e}")
            return {}

    def _calculate_model_efficiency(self, stats: Dict[str, Any]) -> str:
        """Calculate model efficiency grade"""
        total_calls = int(stats.get("total_calls", 0))
        total_cost = float(stats.get("total_cost", 0))

        if total_calls == 0:
            return "N/A"

        cost_per_call = total_cost / total_calls

        if cost_per_call < 0.01:
            return "Excellent"
        elif cost_per_call < 0.05:
            return "Good"
        elif cost_per_call < 0.1:
            return "Average"
        else:
            return "Poor"

    def _calculate_agent_grade(self, stats: Dict[str, Any]) -> str:
        """Calculate agent performance grade"""
        success_rate = float(stats.get("success_rate", 0))
        avg_time = float(stats.get("avg_execution_time", 0))

        # Grade based on success rate and execution time
        if success_rate >= 95 and avg_time < 30:
            return "A+"
        elif success_rate >= 90 and avg_time < 60:
            return "A"
        elif success_rate >= 85 and avg_time < 120:
            return "B"
        elif success_rate >= 75:
            return "C"
        else:
            return "D"

    async def _calculate_cost_efficiency(self) -> Dict[str, Any]:
        """Calculate overall cost efficiency metrics"""
        try:
            # Get total daily costs for the last 7 days
            total_cost = 0
            for i in range(7):
                date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
                cost = await cache_manager.cache_get(f"daily_cost:{date}", default=0.0)
                total_cost += cost

            # Get total articles processed (from pipeline metrics)
            pipeline_keys = await cache_manager.cache_keys("pipeline_metrics:*")
            total_articles = 0

            for key in pipeline_keys:
                stats = await cache_manager.cache_hgetall(key)
                total_articles += int(stats.get("total_articles", 0))

            cost_per_article = total_cost / max(total_articles, 1)

            return {
                "total_cost_7_days": total_cost,
                "total_articles_processed": total_articles,
                "cost_per_article": cost_per_article,
                "efficiency_rating": "Good" if cost_per_article < 0.1 else "Needs Improvement",
            }

        except Exception as e:
            logger.error(f"Failed to calculate cost efficiency: {e}")
            return {"error": str(e)}

    async def _calculate_potential_savings(self, model_costs: Dict[str, float]) -> float:
        """Calculate potential cost savings"""
        # Simple heuristic: if expensive models are used, suggest 20% savings
        sum(model_costs.values())
        expensive_models_cost = sum(
            cost
            for model, cost in model_costs.items()
            if "gpt-4" in model.lower() or "claude-3-opus" in model.lower()
        )

        # Potential savings by switching 50% of expensive model usage to cheaper alternatives
        potential_savings = expensive_models_cost * 0.5 * 0.7  # 70% cost reduction

        return potential_savings

    async def _identify_optimization_opportunities(
        self, model_costs: Dict[str, float]
    ) -> List[str]:
        """Identify optimization opportunities"""
        opportunities = []

        # Check for expensive model usage
        for model, cost in model_costs.items():
            if cost > 5.0:  # More than $5 in 7 days
                if "gpt-4" in model.lower():
                    opportunities.append(
                        f"Consider using GPT-3.5-turbo for simple tasks instead of {model}"
                    )
                elif "claude-3-opus" in model.lower():
                    opportunities.append(
                        f"Consider using Claude-3-Sonnet for most tasks instead of {model}"
                    )

        # Check for model diversity
        if len(model_costs) == 1:
            opportunities.append(
                "Consider using multiple models for different task types to optimize costs"
            )

        return opportunities


# Global Langfuse manager instance
langfuse_manager = LangfuseManager()


# Decorator for automatic LLM call tracking
def track_llm_call(model: str = None, metadata: Dict[str, Any] = None):
    """Decorator to automatically track LLM calls"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()

            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Extract tracking information from result
                if isinstance(result, dict) and "usage" in result:
                    execution_time = (datetime.utcnow() - start_time).total_seconds()

                    tracking_result = await langfuse_manager.track_llm_call(
                        model=model or result.get("model", "unknown"),
                        prompt=str(result.get("prompt", "")),
                        response=str(result.get("response", "")),
                        usage=result.get("usage", {}),
                        metadata={
                            **(metadata or {}),
                            "function_name": func.__name__,
                            "execution_time": execution_time,
                        },
                    )

                    # Add tracking info to result
                    result["langfuse_tracking"] = tracking_result

                return result

            except Exception as e:
                logger.error(f"Error in LLM call tracking: {e}")
                raise

        return wrapper

    return decorator


# Decorator for agent execution tracking
def track_agent_execution(agent_name: str, metadata: Dict[str, Any] = None):
    """Decorator to automatically track agent executions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            success = False

            try:
                # Execute the function
                result = await func(*args, **kwargs)
                success = True

                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Track the execution
                trace_id = await langfuse_manager.track_agent_execution(
                    agent_name=agent_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    output_data={"result": str(result)},
                    execution_time=execution_time,
                    success=success,
                    metadata=metadata,
                )

                # Add trace ID to result if it's a dict
                if isinstance(result, dict):
                    result["trace_id"] = trace_id

                return result

            except Exception as e:
                execution_time = (datetime.utcnow() - start_time).total_seconds()

                # Track failed execution
                await langfuse_manager.track_agent_execution(
                    agent_name=agent_name,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    output_data={"error": str(e)},
                    execution_time=execution_time,
                    success=False,
                    metadata=metadata,
                )

                raise

        return wrapper

    return decorator
