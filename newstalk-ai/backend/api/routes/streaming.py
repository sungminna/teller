"""
Stage 6: Real-time Streaming Routes
Server-Sent Events (SSE) and WebSocket endpoints for real-time updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import redis.asyncio as redis

from ..middleware.auth import get_current_user
from ..utils.kafka_client import consume_realtime_updates, publish_realtime_update
from ..utils.redis_client import cache_manager, get_redis_client
from ...shared.models.user import User
from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/streaming", tags=["streaming"])

# Global SSE connections manager
class SSEConnectionManager:
    """Manage Server-Sent Events connections"""
    
    def __init__(self):
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.user_connections: Dict[int, set] = {}
        self.settings = get_settings()
    
    async def connect(self, connection_id: str, user_id: int, request: Request):
        """Add new SSE connection"""
        self.connections[connection_id] = {
            'user_id': user_id,
            'connected_at': datetime.utcnow(),
            'last_heartbeat': datetime.utcnow(),
            'request': request
        }
        
        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(connection_id)
        
        logger.info(f"SSE connection established: {connection_id} for user {user_id}")
    
    async def disconnect(self, connection_id: str):
        """Remove SSE connection"""
        if connection_id in self.connections:
            user_id = self.connections[connection_id]['user_id']
            del self.connections[connection_id]
            
            if user_id in self.user_connections:
                self.user_connections[user_id].discard(connection_id)
                if not self.user_connections[user_id]:
                    del self.user_connections[user_id]
            
            logger.info(f"SSE connection closed: {connection_id}")
    
    async def send_to_user(self, user_id: int, event_type: str, data: Dict[str, Any]):
        """Send event to all connections for a specific user"""
        if user_id not in self.user_connections:
            return
        
        message = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in Redis for connection recovery
        await cache_manager.cache_set(
            f"sse_message:{user_id}:{datetime.utcnow().timestamp()}",
            message,
            expire=300  # 5 minutes
        )
    
    async def send_to_all(self, event_type: str, data: Dict[str, Any]):
        """Send event to all connected users"""
        message = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store in Redis for broadcasting
        await cache_manager.cache_set(
            f"sse_broadcast:{datetime.utcnow().timestamp()}",
            message,
            expire=300  # 5 minutes
        )
    
    async def cleanup_stale_connections(self):
        """Remove stale connections"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=5)
        stale_connections = []
        
        for conn_id, conn_info in self.connections.items():
            if conn_info['last_heartbeat'] < cutoff_time:
                stale_connections.append(conn_id)
        
        for conn_id in stale_connections:
            await self.disconnect(conn_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            'total_connections': len(self.connections),
            'unique_users': len(self.user_connections),
            'connections_per_user': {
                user_id: len(connections) 
                for user_id, connections in self.user_connections.items()
            }
        }

# Global connection manager
sse_manager = SSEConnectionManager()

@router.get("/events")
async def stream_events(
    request: Request,
    current_user: User = Depends(get_current_user)
):
    """
    Server-Sent Events endpoint for real-time updates
    """
    connection_id = f"sse_{current_user.id}_{datetime.utcnow().timestamp()}"
    
    async def event_stream():
        """Generate SSE events"""
        try:
            # Register connection
            await sse_manager.connect(connection_id, current_user.id, request)
            
            # Send connection established event
            yield {
                "event": "connected",
                "data": json.dumps({
                    "connection_id": connection_id,
                    "user_id": current_user.id,
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
            
            # Start consuming real-time updates
            last_heartbeat = datetime.utcnow()
            
            async for update in consume_user_updates(current_user.id):
                # Check if client is still connected
                if await request.is_disconnected():
                    break
                
                # Send heartbeat every 30 seconds
                if (datetime.utcnow() - last_heartbeat).seconds >= 30:
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    }
                    last_heartbeat = datetime.utcnow()
                
                # Send update
                if update:
                    yield {
                        "event": update.get('type', 'update'),
                        "data": json.dumps(update.get('data', {}))
                    }
        
        except Exception as e:
            logger.error(f"SSE stream error for user {current_user.id}: {e}")
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
        
        finally:
            # Clean up connection
            await sse_manager.disconnect(connection_id)
    
    return EventSourceResponse(event_stream())

async def consume_user_updates(user_id: int) -> AsyncGenerator[Dict[str, Any], None]:
    """Consume real-time updates for a specific user"""
    try:
        # Consume from Kafka real-time updates
        async for kafka_update in consume_realtime_updates():
            # Filter updates for this user
            if should_send_to_user(kafka_update, user_id):
                yield kafka_update
            
            # Also check for user-specific cached messages
            cached_messages = await get_cached_user_messages(user_id)
            for message in cached_messages:
                yield message
                
    except Exception as e:
        logger.error(f"Error consuming updates for user {user_id}: {e}")

def should_send_to_user(update: Dict[str, Any], user_id: int) -> bool:
    """Determine if update should be sent to specific user"""
    update_type = update.get('type', '')
    update_data = update.get('data', {})
    
    # Send all general updates
    if update_type in ['news_processed', 'ai_processing_completed', 'pipeline_metrics']:
        return True
    
    # Send user-specific updates
    if update_data.get('user_id') == user_id:
        return True
    
    # Send personalized content updates
    if update_type == 'personalized_content' and update_data.get('target_user_id') == user_id:
        return True
    
    return False

async def get_cached_user_messages(user_id: int) -> list:
    """Get cached messages for user"""
    try:
        redis_client = await get_redis_client()
        pattern = f"sse_message:{user_id}:*"
        
        keys = await redis_client.keys(pattern)
        messages = []
        
        for key in keys:
            message = await redis_client.get(key)
            if message:
                messages.append(json.loads(message))
                # Delete after reading
                await redis_client.delete(key)
        
        return messages
    except Exception as e:
        logger.error(f"Error getting cached messages: {e}")
        return []

@router.post("/notify")
async def send_notification(
    event_type: str,
    data: Dict[str, Any],
    user_id: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Send real-time notification
    """
    try:
        if user_id:
            # Send to specific user
            await sse_manager.send_to_user(user_id, event_type, data)
        else:
            # Broadcast to all users
            await sse_manager.send_to_all(event_type, data)
        
        # Also publish to Kafka for persistence
        await publish_realtime_update(event_type, {
            **data,
            'sender_user_id': current_user.id,
            'target_user_id': user_id
        })
        
        return {
            "success": True,
            "message": "Notification sent",
            "event_type": event_type,
            "target_user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to send notification")

@router.get("/status")
async def get_streaming_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get streaming service status
    """
    try:
        # Get connection stats
        connection_stats = sse_manager.get_connection_stats()
        
        # Get Kafka streaming status
        from ..utils.kafka_client import stream_processor
        kafka_status = {
            'running': stream_processor.running,
            'processors': len(stream_processor.processors) if hasattr(stream_processor, 'processors') else 0
        }
        
        # Get Redis cache stats
        redis_client = await get_redis_client()
        redis_info = await redis_client.info()
        
        return {
            "streaming_service": "healthy",
            "connections": connection_stats,
            "kafka_streaming": kafka_status,
            "redis_cache": {
                "connected_clients": redis_info.get('connected_clients', 0),
                "used_memory": redis_info.get('used_memory_human', '0B'),
                "keyspace_hits": redis_info.get('keyspace_hits', 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get streaming status")

@router.get("/metrics")
async def get_streaming_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get real-time streaming metrics
    """
    try:
        # Get real-time stats from Redis
        pipeline_stats = await cache_manager.get_realtime_stats('pipeline_performance')
        throughput_stats = await cache_manager.get_realtime_stats('article_throughput')
        
        # Get TimeSeries data
        from ..utils.redis_client import timeseries
        current_time = int(datetime.utcnow().timestamp() * 1000)
        one_hour_ago = current_time - (60 * 60 * 1000)
        
        pipeline_samples = await timeseries.get_range('pipeline_duration', one_hour_ago, current_time)
        article_samples = await timeseries.get_range('articles_processed', one_hour_ago, current_time)
        
        return {
            "real_time_stats": {
                "pipeline_performance": pipeline_stats,
                "article_throughput": throughput_stats
            },
            "time_series": {
                "pipeline_duration_samples": len(pipeline_samples),
                "articles_processed_samples": len(article_samples),
                "avg_pipeline_duration": sum(s[1] for s in pipeline_samples) / len(pipeline_samples) if pipeline_samples else 0,
                "total_articles_processed": sum(s[1] for s in article_samples)
            },
            "connection_metrics": sse_manager.get_connection_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting streaming metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get streaming metrics")

@router.post("/test-event")
async def send_test_event(
    event_type: str = "test",
    message: str = "Test message",
    current_user: User = Depends(get_current_user)
):
    """
    Send test event for debugging
    """
    try:
        test_data = {
            "message": message,
            "sender": current_user.username,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to current user
        await sse_manager.send_to_user(current_user.id, event_type, test_data)
        
        # Also publish to Kafka
        await publish_realtime_update(f"test_{event_type}", test_data)
        
        return {
            "success": True,
            "message": "Test event sent",
            "event_type": event_type,
            "data": test_data
        }
        
    except Exception as e:
        logger.error(f"Error sending test event: {e}")
        raise HTTPException(status_code=500, detail="Failed to send test event")

# Background task for connection cleanup
async def cleanup_connections():
    """Background task to cleanup stale connections"""
    while True:
        try:
            await sse_manager.cleanup_stale_connections()
            await asyncio.sleep(60)  # Run every minute
        except Exception as e:
            logger.error(f"Error in connection cleanup: {e}")
            await asyncio.sleep(60)

# Start background cleanup task
@router.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(cleanup_connections())

@router.get("/health")
async def streaming_health():
    """
    Health check endpoint for streaming service
    """
    try:
        # Check Redis connection
        redis_client = await get_redis_client()
        await redis_client.ping()
        
        # Check Kafka connection
        from ..utils.kafka_client import get_kafka_producer
        producer = await get_kafka_producer()
        
        return {
            "status": "healthy",
            "services": {
                "redis": "connected",
                "kafka": "connected",
                "sse_connections": len(sse_manager.connections)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Streaming health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        } 