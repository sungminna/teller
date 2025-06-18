"""
Stage 7: Monitoring and Observability Routes
Comprehensive monitoring endpoints for Prometheus, Langfuse, and Slack integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from ..utils.langfuse_client import langfuse_manager
from ..utils.slack_notifier import slack_notifier, AlertSeverity, AlertCategory, check_thresholds_and_alert
from ..utils.redis_client import cache_manager
from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Pydantic models for API requests/responses
class AlertRequest(BaseModel):
    message: str
    severity: str = Field(default="info", description="Alert severity: info, warning, error, critical")
    category: str = Field(default="system", description="Alert category: system, performance, cost, business, security")
    channel: Optional[str] = Field(default=None, description="Slack channel override")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    playbook: Optional[str] = Field(default=None, description="Incident response playbook")

class MetricUpdate(BaseModel):
    metric_name: str
    value: float
    labels: Optional[Dict[str, str]] = Field(default=None)
    timestamp: Optional[datetime] = Field(default=None)

class ThresholdCheck(BaseModel):
    metric_name: str
    current_value: float
    threshold: float
    unit: str = ""
    service: Optional[str] = None

class MonitoringStatus(BaseModel):
    langfuse_enabled: bool
    slack_enabled: bool
    prometheus_enabled: bool
    redis_connected: bool
    last_health_check: datetime
    active_alerts: int
    daily_cost: float
    system_health: str

class AlertAnalytics(BaseModel):
    total_alerts: int
    alerts_by_severity: Dict[str, int]
    alerts_by_category: Dict[str, int]
    recent_alerts: List[Dict[str, Any]]
    top_alert_sources: List[Dict[str, Any]]

# === PROMETHEUS METRICS ENDPOINTS ===

@router.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get Prometheus metrics in standard format"""
    try:
        # Try to import prometheus metrics
        try:
            from ..utils.prometheus_metrics import prometheus_metrics
            metrics_data = prometheus_metrics.get_metrics_data()
            content_type = prometheus_metrics.get_content_type()
            
            return Response(
                content=metrics_data,
                media_type=content_type
            )
        except ImportError:
            # Fallback if prometheus_client not available
            logger.warning("Prometheus client not available")
            return "# Prometheus client not available\n"
    
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")

@router.post("/metrics/update")
async def update_metric(metric: MetricUpdate):
    """Update a specific metric value"""
    try:
        # Store metric in Redis for later collection
        metric_key = f"custom_metrics:{metric.metric_name}"
        metric_data = {
            "value": metric.value,
            "labels": metric.labels or {},
            "timestamp": (metric.timestamp or datetime.utcnow()).isoformat()
        }
        
        await cache_manager.cache_set(metric_key, metric_data, expire=3600)
        
        # Also try to update Prometheus metric if available
        try:
            from ..utils.prometheus_metrics import prometheus_metrics
            
            # Update business metrics based on metric name
            if "user_satisfaction" in metric.metric_name.lower():
                prometheus_metrics.update_user_satisfaction("general", metric.value)
            elif "content_quality" in metric.metric_name.lower():
                prometheus_metrics.update_content_quality("news", "api", metric.value)
            elif "articles_processed" in metric.metric_name.lower():
                prometheus_metrics.record_article_processed("api", "success")
            
        except ImportError:
            pass
        
        return {"status": "success", "metric": metric.metric_name, "value": metric.value}
        
    except Exception as e:
        logger.error(f"Error updating metric: {e}")
        raise HTTPException(status_code=500, detail="Failed to update metric")

@router.get("/metrics/business")
async def get_business_metrics():
    """Get business metrics summary"""
    try:
        # Collect business metrics from cache
        metrics = {}
        
        # User satisfaction
        user_satisfaction = await cache_manager.cache_get("metrics:user_satisfaction", default=8.5)
        metrics["user_satisfaction"] = float(user_satisfaction)
        
        # Content quality
        content_quality = await cache_manager.cache_get("metrics:content_quality", default=7.8)
        metrics["content_quality"] = float(content_quality)
        
        # Daily articles processed
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_articles = await cache_manager.cache_get(f"daily_articles:{today}", default=0)
        metrics["daily_articles_processed"] = int(daily_articles)
        
        # API uptime
        api_uptime = await cache_manager.cache_get("metrics:api_uptime", default=99.9)
        metrics["api_uptime_percent"] = float(api_uptime)
        
        # Daily cost
        daily_cost = await cache_manager.cache_get(f"daily_cost:{today}", default=0.0)
        metrics["daily_llm_cost_usd"] = float(daily_cost)
        
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting business metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get business metrics")

# === LANGFUSE ANALYTICS ENDPOINTS ===

@router.get("/langfuse/cost-analytics")
async def get_cost_analytics(days: int = 7):
    """Get LLM cost analytics from Langfuse"""
    try:
        analytics = await langfuse_manager.get_cost_analytics(days)
        
        if "error" in analytics:
            raise HTTPException(status_code=500, detail=analytics["error"])
        
        return {
            "status": "success",
            "analytics": analytics,
            "period_days": days
        }
        
    except Exception as e:
        logger.error(f"Error getting cost analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cost analytics")

@router.get("/langfuse/performance-insights")
async def get_performance_insights(time_range_hours: int = 24):
    """Get performance optimization insights from Langfuse"""
    try:
        insights = await langfuse_manager.get_performance_insights(time_range_hours)
        
        if "error" in insights:
            raise HTTPException(status_code=500, detail=insights["error"])
        
        return {
            "status": "success",
            "insights": insights,
            "time_range_hours": time_range_hours
        }
        
    except Exception as e:
        logger.error(f"Error getting performance insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance insights")

@router.post("/langfuse/track-llm-call")
async def track_llm_call_manually(
    model: str,
    prompt: str,
    response: str,
    usage: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
):
    """Manually track an LLM call"""
    try:
        result = await langfuse_manager.track_llm_call(
            model=model,
            prompt=prompt,
            response=response,
            usage=usage,
            metadata=metadata
        )
        
        return {
            "status": "success",
            "tracking_result": result
        }
        
    except Exception as e:
        logger.error(f"Error tracking LLM call: {e}")
        raise HTTPException(status_code=500, detail="Failed to track LLM call")

# === SLACK NOTIFICATION ENDPOINTS ===

@router.post("/alerts/send")
async def send_alert(alert: AlertRequest):
    """Send a custom alert to Slack"""
    try:
        # Validate severity and category
        try:
            severity = AlertSeverity(alert.severity.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {alert.severity}")
        
        try:
            category = AlertCategory(alert.category.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {alert.category}")
        
        # Send the alert
        success = await slack_notifier.send_alert(
            message=alert.message,
            severity=severity,
            category=category,
            channel=alert.channel,
            metadata=alert.metadata,
            playbook=alert.playbook
        )
        
        if success:
            return {"status": "success", "message": "Alert sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send alert")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to send alert")

@router.post("/alerts/cost")
async def send_cost_alert(current_cost: float, limit: float, model: Optional[str] = None):
    """Send a cost-related alert"""
    try:
        success = await slack_notifier.send_cost_alert(current_cost, limit, model)
        
        if success:
            return {"status": "success", "message": "Cost alert sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send cost alert")
        
    except Exception as e:
        logger.error(f"Error sending cost alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to send cost alert")

@router.post("/alerts/performance")
async def send_performance_alert(threshold_check: ThresholdCheck):
    """Send a performance-related alert"""
    try:
        success = await slack_notifier.send_performance_alert(
            metric_name=threshold_check.metric_name,
            current_value=threshold_check.current_value,
            threshold=threshold_check.threshold,
            unit=threshold_check.unit,
            service=threshold_check.service
        )
        
        if success:
            return {"status": "success", "message": "Performance alert sent successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send performance alert")
        
    except Exception as e:
        logger.error(f"Error sending performance alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to send performance alert")

@router.get("/alerts/analytics")
async def get_alert_analytics(days: int = 7) -> AlertAnalytics:
    """Get alert analytics and statistics"""
    try:
        # Get alert counts by severity
        alerts_by_severity = {}
        alerts_by_category = {}
        
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            
            # Count by severity
            for severity in ["info", "warning", "error", "critical"]:
                count = await cache_manager.cache_get(f"daily_alerts:{date}:{severity}", default=0)
                alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + int(count)
            
            # Count by category
            for category in ["system", "performance", "cost", "business", "security"]:
                count = await cache_manager.cache_get(f"daily_alerts:{date}:{category}", default=0)
                alerts_by_category[category] = alerts_by_category.get(category, 0) + int(count)
        
        # Get recent alerts
        recent_alerts = []
        alert_keys = await cache_manager.cache_keys("alert_analytics:*")
        
        # Sort by timestamp and get latest 10
        sorted_keys = sorted(alert_keys, key=lambda x: float(x.split(":")[-1]), reverse=True)[:10]
        
        for key in sorted_keys:
            alert_data = await cache_manager.cache_get(key)
            if alert_data:
                recent_alerts.append(alert_data)
        
        total_alerts = sum(alerts_by_severity.values())
        
        return AlertAnalytics(
            total_alerts=total_alerts,
            alerts_by_severity=alerts_by_severity,
            alerts_by_category=alerts_by_category,
            recent_alerts=recent_alerts,
            top_alert_sources=[]  # Could be implemented based on metadata
        )
        
    except Exception as e:
        logger.error(f"Error getting alert analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get alert analytics")

# === THRESHOLD MONITORING ENDPOINTS ===

@router.post("/thresholds/check")
async def check_thresholds(background_tasks: BackgroundTasks):
    """Trigger threshold checks and send alerts if needed"""
    try:
        # Run threshold checks in background
        background_tasks.add_task(check_thresholds_and_alert)
        
        return {"status": "success", "message": "Threshold checks initiated"}
        
    except Exception as e:
        logger.error(f"Error initiating threshold checks: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate threshold checks")

@router.get("/thresholds/current")
async def get_current_thresholds():
    """Get current threshold configurations"""
    try:
        return {
            "status": "success",
            "thresholds": slack_notifier.thresholds,
            "rate_limits": slack_notifier.rate_limits
        }
        
    except Exception as e:
        logger.error(f"Error getting thresholds: {e}")
        raise HTTPException(status_code=500, detail="Failed to get thresholds")

@router.put("/thresholds/update")
async def update_thresholds(thresholds: Dict[str, float]):
    """Update threshold configurations"""
    try:
        # Validate threshold names
        valid_thresholds = set(slack_notifier.thresholds.keys())
        invalid_thresholds = set(thresholds.keys()) - valid_thresholds
        
        if invalid_thresholds:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid threshold names: {list(invalid_thresholds)}"
            )
        
        # Update thresholds
        for threshold_name, value in thresholds.items():
            if value > 0:  # Basic validation
                slack_notifier.thresholds[threshold_name] = value
        
        return {
            "status": "success",
            "message": "Thresholds updated successfully",
            "updated_thresholds": thresholds
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating thresholds: {e}")
        raise HTTPException(status_code=500, detail="Failed to update thresholds")

# === SYSTEM HEALTH AND STATUS ENDPOINTS ===

@router.get("/health")
async def get_monitoring_health() -> MonitoringStatus:
    """Get overall monitoring system health"""
    try:
        # Check component availability
        langfuse_enabled = langfuse_manager.langfuse is not None
        slack_enabled = slack_notifier.webhook_url is not None
        
        # Check Redis connection
        redis_connected = True
        try:
            await cache_manager.cache_set("health_check", "ok", expire=60)
            await cache_manager.cache_get("health_check")
        except Exception:
            redis_connected = False
        
        # Get active alerts count
        today = datetime.utcnow().strftime("%Y-%m-%d")
        active_alerts = 0
        for severity in ["warning", "error", "critical"]:
            count = await cache_manager.cache_get(f"daily_alerts:{today}:{severity}", default=0)
            active_alerts += int(count)
        
        # Get daily cost
        daily_cost = await cache_manager.cache_get(f"daily_cost:{today}", default=0.0)
        
        # Determine system health
        if not redis_connected:
            system_health = "critical"
        elif active_alerts > 10:
            system_health = "degraded"
        elif active_alerts > 5:
            system_health = "warning"
        else:
            system_health = "healthy"
        
        return MonitoringStatus(
            langfuse_enabled=langfuse_enabled,
            slack_enabled=slack_enabled,
            prometheus_enabled=True,  # Always enabled (fallback available)
            redis_connected=redis_connected,
            last_health_check=datetime.utcnow(),
            active_alerts=active_alerts,
            daily_cost=float(daily_cost),
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring health")

@router.get("/status/dashboard")
async def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        # Get health status
        health = await get_monitoring_health()
        
        # Get recent metrics
        business_metrics = await get_business_metrics()
        
        # Get recent alerts
        alert_analytics = await get_alert_analytics(1)  # Last 24 hours
        
        # Get cost analytics
        cost_analytics = await get_cost_analytics(7)
        
        return {
            "status": "success",
            "dashboard": {
                "health": health,
                "business_metrics": business_metrics["metrics"],
                "recent_alerts": alert_analytics.recent_alerts[:5],  # Last 5 alerts
                "cost_summary": {
                    "daily_cost": cost_analytics["analytics"]["total_cost"] / 7,  # Avg daily
                    "monthly_projection": cost_analytics["analytics"]["monthly_projection"],
                    "daily_limit": cost_analytics["analytics"]["daily_limit"]
                },
                "alert_summary": {
                    "total_today": sum(alert_analytics.alerts_by_severity.values()),
                    "critical_today": alert_analytics.alerts_by_severity.get("critical", 0),
                    "error_today": alert_analytics.alerts_by_severity.get("error", 0)
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard data")

# === BACKGROUND MONITORING TASKS ===

@router.post("/tasks/start-monitoring")
async def start_background_monitoring(background_tasks: BackgroundTasks):
    """Start background monitoring tasks"""
    try:
        # Add periodic threshold checking
        background_tasks.add_task(periodic_threshold_monitoring)
        
        # Add periodic metrics collection
        background_tasks.add_task(periodic_metrics_collection)
        
        return {"status": "success", "message": "Background monitoring tasks started"}
        
    except Exception as e:
        logger.error(f"Error starting background monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to start background monitoring")

async def periodic_threshold_monitoring():
    """Periodic threshold monitoring task"""
    try:
        while True:
            await check_thresholds_and_alert()
            await asyncio.sleep(300)  # Check every 5 minutes
    except Exception as e:
        logger.error(f"Error in periodic threshold monitoring: {e}")

async def periodic_metrics_collection():
    """Periodic metrics collection task"""
    try:
        while True:
            # Collect business metrics
            try:
                from ..utils.prometheus_metrics import prometheus_metrics
                await prometheus_metrics.collect_business_metrics()
            except ImportError:
                pass
            
            await asyncio.sleep(60)  # Collect every minute
    except Exception as e:
        logger.error(f"Error in periodic metrics collection: {e}")

# === UTILITY ENDPOINTS ===

@router.post("/test/alert")
async def test_alert():
    """Send a test alert to verify Slack integration"""
    try:
        success = await slack_notifier.send_alert(
            message="🧪 Test alert from NewsTalk AI monitoring system",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM,
            metadata={"test": True, "timestamp": datetime.utcnow().isoformat()},
            suppress_rate_limit=True
        )
        
        if success:
            return {"status": "success", "message": "Test alert sent successfully"}
        else:
            return {"status": "warning", "message": "Test alert failed - check Slack configuration"}
        
    except Exception as e:
        logger.error(f"Error sending test alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test alert")

@router.get("/config")
async def get_monitoring_config():
    """Get current monitoring configuration"""
    try:
        settings = get_settings()
        
        config = {
            "langfuse": {
                "enabled": hasattr(settings, 'LANGFUSE_PUBLIC_KEY') and settings.LANGFUSE_PUBLIC_KEY is not None,
                "host": getattr(settings, 'LANGFUSE_HOST', None)
            },
            "slack": {
                "enabled": hasattr(settings, 'SLACK_WEBHOOK_URL') and settings.SLACK_WEBHOOK_URL is not None,
                "default_channel": getattr(settings, 'SLACK_DEFAULT_CHANNEL', '#alerts')
            },
            "redis": {
                "host": getattr(settings, 'REDIS_HOST', 'localhost'),
                "port": getattr(settings, 'REDIS_PORT', 6379)
            },
            "thresholds": slack_notifier.thresholds,
            "rate_limits": slack_notifier.rate_limits
        }
        
        return {"status": "success", "config": config}
        
    except Exception as e:
        logger.error(f"Error getting monitoring config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring config") 