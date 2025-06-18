"""
Enhanced Redis Client Utility for Stage 6
Advanced caching strategies and Redis TimeSeries support
"""

import asyncio
import json
import logging
from typing import Optional, Any, Dict, List, Union, Tuple
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.commands.timeseries import TimeSeries

from ...shared.config.settings import get_settings

logger = logging.getLogger(__name__)

# Global Redis clients
_redis_client: Optional[redis.Redis] = None
_redis_timeseries: Optional[redis.Redis] = None

class RedisCacheManager:
    """Advanced Redis cache manager for Stage 6"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
    async def get_client(self) -> redis.Redis:
        """Get Redis client instance"""
        return await get_redis_client()
    
    # Stage 6: User Session Management (24h TTL)
    async def set_user_session(self, user_id: int, session_data: Dict[str, Any]) -> bool:
        """Set user session with 24-hour TTL"""
        try:
            client = await self.get_client()
            session_key = f"user_session:{user_id}"
            
            session_payload = {
                'user_id': user_id,
                'created_at': datetime.utcnow().isoformat(),
                'last_activity': datetime.utcnow().isoformat(),
                **session_data
            }
            
            return await client.setex(
                session_key,
                self.settings.REDIS_USER_SESSION_TTL,
                json.dumps(session_payload, default=str)
            )
        except Exception as e:
            logger.error(f"Error setting user session: {e}")
            return False
    
    async def get_user_session(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        try:
            client = await self.get_client()
            session_key = f"user_session:{user_id}"
            
            session_data = await client.get(session_key)
            if session_data:
                return json.loads(session_data)
            return None
        except Exception as e:
            logger.error(f"Error getting user session: {e}")
            return None
    
    async def update_user_activity(self, user_id: int) -> bool:
        """Update user last activity timestamp"""
        try:
            client = await self.get_client()
            session_key = f"user_session:{user_id}"
            
            # Get current session
            session_data = await self.get_user_session(user_id)
            if session_data:
                session_data['last_activity'] = datetime.utcnow().isoformat()
                return await client.setex(
                    session_key,
                    self.settings.REDIS_USER_SESSION_TTL,
                    json.dumps(session_data, default=str)
                )
            return False
        except Exception as e:
            logger.error(f"Error updating user activity: {e}")
            return False
    
    # Stage 6: News Content Caching (6h TTL)
    async def cache_news_content(self, article_id: str, content_data: Dict[str, Any]) -> bool:
        """Cache news content with 6-hour TTL"""
        try:
            client = await self.get_client()
            content_key = f"news_content:{article_id}"
            
            content_payload = {
                'article_id': article_id,
                'cached_at': datetime.utcnow().isoformat(),
                **content_data
            }
            
            return await client.setex(
                content_key,
                self.settings.REDIS_NEWS_CONTENT_TTL,
                json.dumps(content_payload, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching news content: {e}")
            return False
    
    async def get_cached_news_content(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get cached news content"""
        try:
            client = await self.get_client()
            content_key = f"news_content:{article_id}"
            
            content_data = await client.get(content_key)
            if content_data:
                return json.loads(content_data)
            return None
        except Exception as e:
            logger.error(f"Error getting cached news content: {e}")
            return None
    
    async def cache_news_batch(self, articles: List[Dict[str, Any]]) -> int:
        """Cache multiple news articles in batch"""
        try:
            client = await self.get_client()
            pipeline = client.pipeline()
            
            cached_count = 0
            for article in articles:
                article_id = article.get('id')
                if article_id:
                    content_key = f"news_content:{article_id}"
                    content_payload = {
                        'article_id': article_id,
                        'cached_at': datetime.utcnow().isoformat(),
                        **article
                    }
                    
                    pipeline.setex(
                        content_key,
                        self.settings.REDIS_NEWS_CONTENT_TTL,
                        json.dumps(content_payload, default=str)
                    )
                    cached_count += 1
            
            await pipeline.execute()
            return cached_count
        except Exception as e:
            logger.error(f"Error batch caching news: {e}")
            return 0
    
    # Stage 6: Voice Files CDN Cache (Permanent)
    async def cache_voice_file_metadata(self, file_id: str, metadata: Dict[str, Any]) -> bool:
        """Cache voice file metadata permanently (CDN integration)"""
        try:
            client = await self.get_client()
            voice_key = f"voice_file:{file_id}"
            
            voice_payload = {
                'file_id': file_id,
                'cached_at': datetime.utcnow().isoformat(),
                'cdn_url': metadata.get('cdn_url'),
                'file_size': metadata.get('file_size'),
                'duration': metadata.get('duration'),
                'quality': metadata.get('quality'),
                **metadata
            }
            
            # Permanent cache (no TTL) for CDN files
            return await client.set(
                voice_key,
                json.dumps(voice_payload, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching voice file metadata: {e}")
            return False
    
    async def get_voice_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get voice file metadata from cache"""
        try:
            client = await self.get_client()
            voice_key = f"voice_file:{file_id}"
            
            metadata = await client.get(voice_key)
            if metadata:
                return json.loads(metadata)
            return None
        except Exception as e:
            logger.error(f"Error getting voice file metadata: {e}")
            return None
    
    # Stage 6: Real-time Statistics (5min TTL)
    async def update_realtime_stats(self, stat_type: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None) -> bool:
        """Update real-time statistics"""
        try:
            client = await self.get_client()
            
            # Update counter
            stat_key = f"realtime_stats:{stat_type}"
            if labels:
                label_str = ":".join([f"{k}={v}" for k, v in labels.items()])
                stat_key = f"{stat_key}:{label_str}"
            
            # Increment counter
            await client.hincrby(stat_key, "count", 1)
            await client.hincrbyfloat(stat_key, "value", value)
            await client.hset(stat_key, "last_updated", datetime.utcnow().isoformat())
            
            # Set TTL for cleanup
            await client.expire(stat_key, self.settings.REDIS_REALTIME_STATS_TTL)
            
            return True
        except Exception as e:
            logger.error(f"Error updating realtime stats: {e}")
            return False
    
    async def get_realtime_stats(self, stat_type: str, labels: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Get real-time statistics"""
        try:
            client = await self.get_client()
            
            stat_key = f"realtime_stats:{stat_type}"
            if labels:
                label_str = ":".join([f"{k}={v}" for k, v in labels.items()])
                stat_key = f"{stat_key}:{label_str}"
            
            stats = await client.hgetall(stat_key)
            if stats:
                return {
                    'count': int(stats.get('count', 0)),
                    'value': float(stats.get('value', 0.0)),
                    'last_updated': stats.get('last_updated')
                }
            return None
        except Exception as e:
            logger.error(f"Error getting realtime stats: {e}")
            return None

class RedisTimeSeries:
    """Redis TimeSeries for real-time metrics and analytics"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
    
    async def get_client(self) -> redis.Redis:
        """Get Redis TimeSeries client"""
        global _redis_timeseries
        
        if _redis_timeseries is None:
            _redis_timeseries = redis.Redis(
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD,
                db=1,  # Use different DB for TimeSeries
                decode_responses=True
            )
        
        return _redis_timeseries
    
    async def create_timeseries(self, key: str, retention_ms: int = 86400000, labels: Optional[Dict[str, str]] = None) -> bool:
        """Create a new time series"""
        try:
            client = await self.get_client()
            
            # Create time series with retention policy
            ts_args = [key]
            if retention_ms:
                ts_args.extend(['RETENTION', retention_ms])
            if labels:
                ts_args.append('LABELS')
                for k, v in labels.items():
                    ts_args.extend([k, v])
            
            await client.execute_command('TS.CREATE', *ts_args)
            return True
        except Exception as e:
            if "key already exists" not in str(e).lower():
                logger.error(f"Error creating timeseries: {e}")
            return False
    
    async def add_sample(self, key: str, timestamp: Optional[int] = None, value: float = 1.0) -> bool:
        """Add a sample to time series"""
        try:
            client = await self.get_client()
            
            if timestamp is None:
                timestamp = int(datetime.utcnow().timestamp() * 1000)
            
            await client.execute_command('TS.ADD', key, timestamp, value)
            return True
        except Exception as e:
            logger.error(f"Error adding timeseries sample: {e}")
            return False
    
    async def get_range(self, key: str, from_ts: int, to_ts: int) -> List[Tuple[int, float]]:
        """Get time series range"""
        try:
            client = await self.get_client()
            
            result = await client.execute_command('TS.RANGE', key, from_ts, to_ts)
            return [(int(ts), float(val)) for ts, val in result]
        except Exception as e:
            logger.error(f"Error getting timeseries range: {e}")
            return []
    
    async def increment_counter(self, key: str, timestamp: Optional[int] = None, value: float = 1.0) -> bool:
        """Increment a counter time series"""
        # Ensure time series exists
        await self.create_timeseries(key, labels={'type': 'counter'})
        return await self.add_sample(key, timestamp, value)

# Enhanced Redis client functions
async def get_redis_client() -> redis.Redis:
    """Get enhanced Redis client"""
    global _redis_client
    
    if _redis_client is None:
        _redis_client = await create_redis_client()
    
    return _redis_client

async def create_redis_client() -> redis.Redis:
    """Create enhanced Redis client with Stage 6 optimizations"""
    settings = get_settings()
    
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            db=0,
            decode_responses=True,
            max_connections=50,  # Increased for Stage 6
            retry_on_timeout=True,
            socket_timeout=10,  # Increased timeout
            socket_connect_timeout=10,
            health_check_interval=30
        )
        
        # Test connection
        await client.ping()
        
        logger.info(f"✅ Enhanced Redis client created: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        return client
        
    except Exception as e:
        logger.error(f"❌ Enhanced Redis client creation failed: {e}")
        raise

# Enhanced cache utility functions with Stage 6 strategies
async def cache_set(
    key: str, 
    value: Any, 
    expire: Optional[int] = None,
    serialize: bool = True
) -> bool:
    """Enhanced cache set with better serialization"""
    try:
        client = await get_redis_client()
        
        if serialize and not isinstance(value, (str, bytes, int, float)):
            value = json.dumps(value, default=str)
        
        if expire:
            return await client.setex(key, expire, value)
        else:
            return await client.set(key, value)
            
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False

async def cache_get(
    key: str,
    deserialize: bool = True,
    default: Any = None
) -> Any:
    """Enhanced cache get with better deserialization"""
    try:
        client = await get_redis_client()
        value = await client.get(key)
        
        if value is None:
            return default
        
        if deserialize and isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
        
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return default

async def cache_delete(key: str) -> bool:
    """Delete cache key"""
    try:
        client = await get_redis_client()
        return await client.delete(key) > 0
        
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return False

async def cache_exists(key: str) -> bool:
    """Check if cache key exists"""
    try:
        client = await get_redis_client()
        return await client.exists(key) > 0
        
    except Exception as e:
        logger.error(f"Cache exists error: {e}")
        return False

async def cache_expire(key: str, seconds: int) -> bool:
    """Set cache expiration"""
    try:
        client = await get_redis_client()
        return await client.expire(key, seconds)
        
    except Exception as e:
        logger.error(f"Cache expire error: {e}")
        return False

async def cache_ttl(key: str) -> int:
    """Get cache TTL"""
    try:
        client = await get_redis_client()
        return await client.ttl(key)
        
    except Exception as e:
        logger.error(f"Cache TTL error: {e}")
        return -1

# Hash operations
async def cache_hset(key: str, field: str, value: Any) -> bool:
    """Set hash field"""
    try:
        client = await get_redis_client()
        
        if not isinstance(value, (str, bytes, int, float)):
            value = json.dumps(value, default=str)
        
        return await client.hset(key, field, value) > 0
        
    except Exception as e:
        logger.error(f"Cache hset error: {e}")
        return False

async def cache_hget(key: str, field: str, deserialize: bool = True) -> Any:
    """Get hash field"""
    try:
        client = await get_redis_client()
        value = await client.hget(key, field)
        
        if value is None:
            return None
        
        if deserialize and isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
        
    except Exception as e:
        logger.error(f"Cache hget error: {e}")
        return None

async def cache_hgetall(key: str, deserialize: bool = True) -> Dict[str, Any]:
    """Get all hash fields"""
    try:
        client = await get_redis_client()
        hash_data = await client.hgetall(key)
        
        if not hash_data:
            return {}
        
        if deserialize:
            result = {}
            for field, value in hash_data.items():
                try:
                    result[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    result[field] = value
            return result
        
        return hash_data
        
    except Exception as e:
        logger.error(f"Cache hgetall error: {e}")
        return {}

async def cache_hincrby(key: str, field: str, amount: int = 1) -> int:
    """Increment hash field by amount"""
    try:
        client = await get_redis_client()
        return await client.hincrby(key, field, amount)
        
    except Exception as e:
        logger.error(f"Cache hincrby error: {e}")
        return 0

# List operations
async def cache_lpush(key: str, *values: Any) -> int:
    """Push values to left of list"""
    try:
        client = await get_redis_client()
        
        serialized_values = []
        for value in values:
            if not isinstance(value, (str, bytes, int, float)):
                serialized_values.append(json.dumps(value, default=str))
            else:
                serialized_values.append(value)
        
        return await client.lpush(key, *serialized_values)
        
    except Exception as e:
        logger.error(f"Cache lpush error: {e}")
        return 0

async def cache_rpush(key: str, *values: Any) -> int:
    """Push values to right of list"""
    try:
        client = await get_redis_client()
        
        serialized_values = []
        for value in values:
            if not isinstance(value, (str, bytes, int, float)):
                serialized_values.append(json.dumps(value, default=str))
            else:
                serialized_values.append(value)
        
        return await client.rpush(key, *serialized_values)
        
    except Exception as e:
        logger.error(f"Cache rpush error: {e}")
        return 0

async def cache_lpop(key: str, deserialize: bool = True) -> Any:
    """Pop value from left of list"""
    try:
        client = await get_redis_client()
        value = await client.lpop(key)
        
        if value is None:
            return None
        
        if deserialize and isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        
        return value
        
    except Exception as e:
        logger.error(f"Cache lpop error: {e}")
        return None

async def cache_lrange(
    key: str, 
    start: int = 0, 
    end: int = -1,
    deserialize: bool = True
) -> List[Any]:
    """Get range from list"""
    try:
        client = await get_redis_client()
        values = await client.lrange(key, start, end)
        
        if not values:
            return []
        
        if deserialize:
            result = []
            for value in values:
                try:
                    result.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    result.append(value)
            return result
        
        return values
        
    except Exception as e:
        logger.error(f"Cache lrange error: {e}")
        return []

async def cache_ltrim(key: str, start: int, end: int) -> bool:
    """Trim list to specified range"""
    try:
        client = await get_redis_client()
        return await client.ltrim(key, start, end)
        
    except Exception as e:
        logger.error(f"Cache ltrim error: {e}")
        return False

# Set operations
async def cache_sadd(key: str, *members: Any) -> int:
    """Add members to set"""
    try:
        client = await get_redis_client()
        
        serialized_members = []
        for member in members:
            if not isinstance(member, (str, bytes, int, float)):
                serialized_members.append(json.dumps(member, default=str))
            else:
                serialized_members.append(member)
        
        return await client.sadd(key, *serialized_members)
        
    except Exception as e:
        logger.error(f"Cache sadd error: {e}")
        return 0

async def cache_smembers(key: str, deserialize: bool = True) -> set:
    """Get all set members"""
    try:
        client = await get_redis_client()
        members = await client.smembers(key)
        
        if not members:
            return set()
        
        if deserialize:
            result = set()
            for member in members:
                try:
                    result.add(json.loads(member))
                except (json.JSONDecodeError, TypeError):
                    result.add(member)
            return result
        
        return members
        
    except Exception as e:
        logger.error(f"Cache smembers error: {e}")
        return set()

# Utility functions
async def cache_keys(pattern: str = "*") -> List[str]:
    """Get keys matching pattern"""
    try:
        client = await get_redis_client()
        return await client.keys(pattern)
        
    except Exception as e:
        logger.error(f"Cache keys error: {e}")
        return []

async def cache_flushdb() -> bool:
    """Flush current database"""
    try:
        client = await get_redis_client()
        return await client.flushdb()
        
    except Exception as e:
        logger.error(f"Cache flushdb error: {e}")
        return False

async def close_redis_client():
    """Close Redis client"""
    global _redis_client, _redis_timeseries
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
        logger.info("✅ Enhanced Redis client closed")
    
    if _redis_timeseries:
        await _redis_timeseries.close()
        _redis_timeseries = None
        logger.info("✅ Redis TimeSeries client closed")

# Initialize cache manager and timeseries
cache_manager = RedisCacheManager()
timeseries = RedisTimeSeries() 