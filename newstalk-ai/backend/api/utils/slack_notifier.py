"""
Stage 7: Slack Notification System
Real-time alerts, threshold-based notifications, and incident response integration
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import aiohttp

from ...shared.config.settings import get_settings
from .redis_client import cache_manager

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertCategory(Enum):
    """Alert categories"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    COST = "cost"
    BUSINESS = "business"
    SECURITY = "security"

class SlackNotifier:
    """Comprehensive Slack notification system for Stage 7"""
    
    def __init__(self):
        self.settings = get_settings()
        self.webhook_url = getattr(self.settings, 'SLACK_WEBHOOK_URL', None)
        self.bot_token = getattr(self.settings, 'SLACK_BOT_TOKEN', None)
        self.default_channel = getattr(self.settings, 'SLACK_DEFAULT_CHANNEL', '#alerts')
        self.session = None
        
        # Alert thresholds
        self.thresholds = {
            'api_response_time': 5.0,  # seconds
            'api_error_rate': 5.0,     # percentage
            'dag_failure_rate': 10.0,  # percentage
            'agent_failure_rate': 15.0, # percentage
            'daily_cost': 45.0,        # USD
            'queue_size': 100,         # tasks
            'cpu_usage': 80.0,         # percentage
            'memory_usage': 85.0,      # percentage
            'user_satisfaction': 7.0   # minimum score
        }
        
        # Rate limiting (prevent spam)
        self.rate_limits = {
            'same_alert': 300,     # 5 minutes
            'category_limit': 60,  # 1 minute
            'global_limit': 10     # 10 seconds
        }
        
        # Incident response playbooks
        self.playbooks = {
            'api_down': {
                'title': '🚨 API Service Down',
                'steps': [
                    'Check API health endpoint',
                    'Verify database connectivity',
                    'Check Redis cache status',
                    'Review recent deployments',
                    'Escalate to on-call engineer'
                ],
                'contacts': ['@dev-team', '@sre-team']
            },
            'high_cost': {
                'title': '💰 High LLM Cost Alert',
                'steps': [
                    'Review current LLM usage',
                    'Check for runaway processes',
                    'Verify cost optimization settings',
                    'Consider temporary rate limiting',
                    'Notify finance team'
                ],
                'contacts': ['@dev-team', '@finance']
            },
            'dag_failures': {
                'title': '⚠️ Multiple DAG Failures',
                'steps': [
                    'Check Airflow scheduler status',
                    'Review DAG logs for errors',
                    'Verify data source availability',
                    'Check resource constraints',
                    'Consider manual intervention'
                ],
                'contacts': ['@data-team', '@dev-team']
            },
            'performance_degradation': {
                'title': '📉 Performance Degradation',
                'steps': [
                    'Check system resource usage',
                    'Review recent code changes',
                    'Analyze database performance',
                    'Check external service status',
                    'Consider scaling resources'
                ],
                'contacts': ['@sre-team', '@dev-team']
            }
        }
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def send_alert(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        channel: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        playbook: Optional[str] = None,
        suppress_rate_limit: bool = False
    ) -> bool:
        """Send alert to Slack with rate limiting and formatting"""
        
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            # Check rate limiting
            if not suppress_rate_limit:
                if await self._is_rate_limited(message, category):
                    logger.debug(f"Alert rate limited: {message[:50]}...")
                    return False
            
            # Format the alert
            formatted_alert = await self._format_alert(
                message, severity, category, metadata, playbook
            )
            
            # Send to Slack
            success = await self._send_to_slack(formatted_alert, channel)
            
            if success:
                # Update rate limiting cache
                await self._update_rate_limit_cache(message, category)
                
                # Store alert for analytics
                await self._store_alert_analytics(
                    message, severity, category, metadata
                )
                
                logger.info(f"Slack alert sent: {severity.value} - {category.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def send_cost_alert(self, current_cost: float, limit: float, model: str = None):
        """Send cost-related alert"""
        percentage = (current_cost / limit) * 100
        
        if percentage >= 90:
            severity = AlertSeverity.CRITICAL
            emoji = "🚨"
        elif percentage >= 80:
            severity = AlertSeverity.ERROR
            emoji = "⚠️"
        else:
            severity = AlertSeverity.WARNING
            emoji = "💰"
        
        model_info = f" ({model})" if model else ""
        message = f"{emoji} LLM Cost Alert{model_info}: ${current_cost:.2f} / ${limit:.2f} ({percentage:.1f}%)"
        
        metadata = {
            "current_cost": current_cost,
            "cost_limit": limit,
            "percentage": percentage,
            "model": model,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_alert(
            message=message,
            severity=severity,
            category=AlertCategory.COST,
            metadata=metadata,
            playbook="high_cost" if percentage >= 80 else None
        )
    
    async def send_performance_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        unit: str = "",
        service: str = None
    ):
        """Send performance-related alert"""
        service_info = f" ({service})" if service else ""
        
        if current_value > threshold * 1.5:
            severity = AlertSeverity.CRITICAL
            emoji = "🚨"
        elif current_value > threshold * 1.2:
            severity = AlertSeverity.ERROR
            emoji = "⚠️"
        else:
            severity = AlertSeverity.WARNING
            emoji = "📈"
        
        message = f"{emoji} Performance Alert{service_info}: {metric_name} is {current_value:.2f}{unit} (threshold: {threshold:.2f}{unit})"
        
        metadata = {
            "metric_name": metric_name,
            "current_value": current_value,
            "threshold": threshold,
            "unit": unit,
            "service": service,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_alert(
            message=message,
            severity=severity,
            category=AlertCategory.PERFORMANCE,
            metadata=metadata,
            playbook="performance_degradation" if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else None
        )
    
    async def send_system_alert(
        self,
        component: str,
        status: str,
        details: str = None,
        severity: AlertSeverity = AlertSeverity.ERROR
    ):
        """Send system-related alert"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨"
        }
        
        emoji = emoji_map.get(severity, "ℹ️")
        message = f"{emoji} System Alert: {component} - {status}"
        
        if details:
            message += f"\nDetails: {details}"
        
        metadata = {
            "component": component,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        playbook = None
        if component.lower() == "api" and "down" in status.lower():
            playbook = "api_down"
        elif "failure" in status.lower():
            playbook = "dag_failures"
        
        return await self.send_alert(
            message=message,
            severity=severity,
            category=AlertCategory.SYSTEM,
            metadata=metadata,
            playbook=playbook
        )
    
    async def send_business_alert(
        self,
        metric_name: str,
        current_value: float,
        expected_value: float,
        impact: str = None
    ):
        """Send business-related alert"""
        deviation = abs(current_value - expected_value) / expected_value * 100
        
        if deviation > 20:
            severity = AlertSeverity.ERROR
            emoji = "📉"
        elif deviation > 10:
            severity = AlertSeverity.WARNING
            emoji = "⚠️"
        else:
            severity = AlertSeverity.INFO
            emoji = "📊"
        
        message = f"{emoji} Business Metric Alert: {metric_name} is {current_value:.2f} (expected: {expected_value:.2f}, deviation: {deviation:.1f}%)"
        
        if impact:
            message += f"\nImpact: {impact}"
        
        metadata = {
            "metric_name": metric_name,
            "current_value": current_value,
            "expected_value": expected_value,
            "deviation": deviation,
            "impact": impact,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.send_alert(
            message=message,
            severity=severity,
            category=AlertCategory.BUSINESS,
            metadata=metadata
        )
    
    async def send_daily_summary(self):
        """Send daily summary report"""
        try:
            # Collect daily metrics
            summary_data = await self._collect_daily_summary()
            
            # Format summary message
            message = await self._format_daily_summary(summary_data)
            
            return await self.send_alert(
                message=message,
                severity=AlertSeverity.INFO,
                category=AlertCategory.SYSTEM,
                channel="#daily-reports",
                suppress_rate_limit=True
            )
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
            return False
    
    async def _format_alert(
        self,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        metadata: Optional[Dict[str, Any]],
        playbook: Optional[str]
    ) -> Dict[str, Any]:
        """Format alert for Slack"""
        
        # Color coding based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",      # Green
            AlertSeverity.WARNING: "#ff9500",   # Orange
            AlertSeverity.ERROR: "#ff0000",     # Red
            AlertSeverity.CRITICAL: "#8B0000"   # Dark Red
        }
        
        color = color_map.get(severity, "#36a64f")
        
        # Build the attachment
        attachment = {
            "color": color,
            "title": f"{severity.value.upper()} - {category.value.upper()}",
            "text": message,
            "footer": "NewsTalk AI Monitoring",
            "ts": int(datetime.utcnow().timestamp())
        }
        
        # Add fields for metadata
        if metadata:
            fields = []
            for key, value in metadata.items():
                if key not in ['timestamp']:  # Skip timestamp as it's in footer
                    fields.append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
            
            if fields:
                attachment["fields"] = fields
        
        # Add playbook information
        if playbook and playbook in self.playbooks:
            playbook_info = self.playbooks[playbook]
            
            # Add playbook as a field
            attachment["fields"] = attachment.get("fields", [])
            attachment["fields"].append({
                "title": "📋 Incident Response Playbook",
                "value": playbook_info["title"],
                "short": False
            })
            
            # Add response steps
            steps_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(playbook_info["steps"])])
            attachment["fields"].append({
                "title": "🔧 Response Steps",
                "value": f"```{steps_text}```",
                "short": False
            })
            
            # Add contacts
            if playbook_info.get("contacts"):
                contacts_text = " ".join(playbook_info["contacts"])
                attachment["fields"].append({
                    "title": "👥 Contacts",
                    "value": contacts_text,
                    "short": False
                })
        
        return {
            "attachments": [attachment]
        }
    
    async def _send_to_slack(self, payload: Dict[str, Any], channel: Optional[str] = None) -> bool:
        """Send formatted payload to Slack"""
        try:
            # Add channel if specified
            if channel:
                payload["channel"] = channel
            elif self.default_channel:
                payload["channel"] = self.default_channel
            
            session = await self._get_session()
            
            async with session.post(
                self.webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    return True
                else:
                    logger.error(f"Slack API error: {response.status} - {await response.text()}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to send to Slack: {e}")
            return False
    
    async def _is_rate_limited(self, message: str, category: AlertCategory) -> bool:
        """Check if alert is rate limited"""
        try:
            current_time = datetime.utcnow().timestamp()
            
            # Check same alert rate limiting
            message_hash = str(hash(message))
            last_sent = await cache_manager.cache_get(f"alert_rate_limit:message:{message_hash}")
            
            if last_sent and (current_time - float(last_sent)) < self.rate_limits['same_alert']:
                return True
            
            # Check category rate limiting
            category_key = f"alert_rate_limit:category:{category.value}"
            last_category_alert = await cache_manager.cache_get(category_key)
            
            if last_category_alert and (current_time - float(last_category_alert)) < self.rate_limits['category_limit']:
                return True
            
            # Check global rate limiting
            last_global_alert = await cache_manager.cache_get("alert_rate_limit:global")
            
            if last_global_alert and (current_time - float(last_global_alert)) < self.rate_limits['global_limit']:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {e}")
            return False
    
    async def _update_rate_limit_cache(self, message: str, category: AlertCategory):
        """Update rate limiting cache"""
        try:
            current_time = datetime.utcnow().timestamp()
            
            # Update message rate limit
            message_hash = str(hash(message))
            await cache_manager.cache_set(
                f"alert_rate_limit:message:{message_hash}",
                current_time,
                expire=self.rate_limits['same_alert']
            )
            
            # Update category rate limit
            await cache_manager.cache_set(
                f"alert_rate_limit:category:{category.value}",
                current_time,
                expire=self.rate_limits['category_limit']
            )
            
            # Update global rate limit
            await cache_manager.cache_set(
                "alert_rate_limit:global",
                current_time,
                expire=self.rate_limits['global_limit']
            )
            
        except Exception as e:
            logger.error(f"Error updating rate limit cache: {e}")
    
    async def _store_alert_analytics(
        self,
        message: str,
        severity: AlertSeverity,
        category: AlertCategory,
        metadata: Optional[Dict[str, Any]]
    ):
        """Store alert for analytics"""
        try:
            alert_data = {
                "message": message,
                "severity": severity.value,
                "category": category.value,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store in Redis with expiration (30 days)
            alert_key = f"alert_analytics:{datetime.utcnow().timestamp()}"
            await cache_manager.cache_set(alert_key, alert_data, expire=2592000)
            
            # Update daily counters
            today = datetime.utcnow().strftime("%Y-%m-%d")
            await cache_manager.cache_incr(f"daily_alerts:{today}:{severity.value}")
            await cache_manager.cache_incr(f"daily_alerts:{today}:{category.value}")
            
        except Exception as e:
            logger.error(f"Error storing alert analytics: {e}")
    
    async def _collect_daily_summary(self) -> Dict[str, Any]:
        """Collect daily summary metrics"""
        try:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            # Get alert counts
            alert_counts = {}
            for severity in AlertSeverity:
                count = await cache_manager.cache_get(f"daily_alerts:{today}:{severity.value}", default=0)
                alert_counts[severity.value] = int(count)
            
            # Get category counts
            category_counts = {}
            for category in AlertCategory:
                count = await cache_manager.cache_get(f"daily_alerts:{today}:{category.value}", default=0)
                category_counts[category.value] = int(count)
            
            # Get key metrics from cache
            metrics = {
                "daily_cost": await cache_manager.cache_get(f"daily_cost:{today}", default=0.0),
                "articles_processed": await cache_manager.cache_get("metrics:daily_articles", default=0),
                "user_satisfaction": await cache_manager.cache_get("metrics:user_satisfaction", default=0.0),
                "api_uptime": await cache_manager.cache_get("metrics:api_uptime", default=100.0)
            }
            
            return {
                "date": today,
                "alert_counts": alert_counts,
                "category_counts": category_counts,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error collecting daily summary: {e}")
            return {}
    
    async def _format_daily_summary(self, summary_data: Dict[str, Any]) -> str:
        """Format daily summary message"""
        if not summary_data:
            return "📊 Daily Summary: No data available"
        
        date = summary_data.get("date", "Unknown")
        alert_counts = summary_data.get("alert_counts", {})
        metrics = summary_data.get("metrics", {})
        
        # Build summary message
        message = f"📊 **Daily Summary - {date}**\n\n"
        
        # Alert summary
        total_alerts = sum(alert_counts.values())
        message += f"🚨 **Alerts**: {total_alerts} total\n"
        
        if total_alerts > 0:
            for severity, count in alert_counts.items():
                if count > 0:
                    emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(severity, "•")
                    message += f"  {emoji} {severity.title()}: {count}\n"
        
        message += "\n"
        
        # Key metrics
        message += "📈 **Key Metrics**:\n"
        message += f"💰 Daily Cost: ${float(metrics.get('daily_cost', 0)):.2f}\n"
        message += f"📰 Articles Processed: {int(metrics.get('articles_processed', 0)):,}\n"
        message += f"😊 User Satisfaction: {float(metrics.get('user_satisfaction', 0)):.1f}/10\n"
        message += f"⚡ API Uptime: {float(metrics.get('api_uptime', 0)):.2f}%\n"
        
        # Health status
        if float(metrics.get('api_uptime', 0)) >= 99.9:
            message += "\n✅ **System Status**: Excellent"
        elif float(metrics.get('api_uptime', 0)) >= 99.0:
            message += "\n🟡 **System Status**: Good"
        else:
            message += "\n🔴 **System Status**: Needs Attention"
        
        return message

# Global Slack notifier instance
slack_notifier = SlackNotifier()

# Monitoring functions that can be called by other components
async def check_thresholds_and_alert():
    """Check various thresholds and send alerts if exceeded"""
    try:
        # Check API response time
        avg_response_time = await cache_manager.cache_get("metrics:api_avg_response_time", default=0.0)
        if float(avg_response_time) > slack_notifier.thresholds['api_response_time']:
            await slack_notifier.send_performance_alert(
                "API Response Time",
                float(avg_response_time),
                slack_notifier.thresholds['api_response_time'],
                "s",
                "API"
            )
        
        # Check error rate
        error_rate = await cache_manager.cache_get("metrics:api_error_rate", default=0.0)
        if float(error_rate) > slack_notifier.thresholds['api_error_rate']:
            await slack_notifier.send_performance_alert(
                "API Error Rate",
                float(error_rate),
                slack_notifier.thresholds['api_error_rate'],
                "%",
                "API"
            )
        
        # Check daily cost
        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily_cost = await cache_manager.cache_get(f"daily_cost:{today}", default=0.0)
        if float(daily_cost) > slack_notifier.thresholds['daily_cost']:
            await slack_notifier.send_cost_alert(
                float(daily_cost),
                slack_notifier.thresholds['daily_cost']
            )
        
        # Check user satisfaction
        user_satisfaction = await cache_manager.cache_get("metrics:user_satisfaction", default=10.0)
        if float(user_satisfaction) < slack_notifier.thresholds['user_satisfaction']:
            await slack_notifier.send_business_alert(
                "User Satisfaction",
                float(user_satisfaction),
                slack_notifier.thresholds['user_satisfaction'],
                "Low user satisfaction may impact retention"
            )
        
        logger.debug("Threshold checks completed")
        
    except Exception as e:
        logger.error(f"Error in threshold checking: {e}")

# Cleanup function
async def cleanup_slack_notifier():
    """Cleanup Slack notifier resources"""
    await slack_notifier.close() 