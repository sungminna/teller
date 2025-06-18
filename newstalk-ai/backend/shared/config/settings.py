"""
NewsTalk AI 공통 설정 모듈
순환 의존성 해결을 위한 중앙 집중식 설정 관리
"""

import os
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """데이터베이스 설정"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="newstalk_ai", env="DB_NAME")
    username: str = Field(default="postgres", env="DB_USER")
    password: str = Field(default="", env="DB_PASSWORD")
    pool_size: int = Field(default=20, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=30, env="DB_MAX_OVERFLOW")
    
    @property
    def url(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RedisSettings(BaseSettings):
    """Redis 설정"""
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    database: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.database}"


class KafkaSettings(BaseSettings):
    """Kafka 설정"""
    bootstrap_servers: List[str] = Field(default=["localhost:9092"], env="KAFKA_BOOTSTRAP_SERVERS")
    security_protocol: str = Field(default="PLAINTEXT", env="KAFKA_SECURITY_PROTOCOL")
    sasl_mechanism: Optional[str] = Field(default=None, env="KAFKA_SASL_MECHANISM")
    sasl_username: Optional[str] = Field(default=None, env="KAFKA_SASL_USERNAME")
    sasl_password: Optional[str] = Field(default=None, env="KAFKA_SASL_PASSWORD")
    
    # Topic 설정
    raw_news_topic: str = Field(default="raw-news", env="KAFKA_RAW_NEWS_TOPIC")
    processed_news_topic: str = Field(default="processed-news", env="KAFKA_PROCESSED_NEWS_TOPIC")
    realtime_updates_topic: str = Field(default="realtime-updates", env="KAFKA_REALTIME_TOPIC")


class AISettings(BaseSettings):
    """AI 모델 설정"""
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    langfuse_public_key: str = Field(default="", env="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", env="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", env="LANGFUSE_HOST")
    
    # 🎯 LangGraph Checkpointing (Stage 3)
    postgres_checkpoint_url: str = Field(default="", env="POSTGRES_CHECKPOINT_URL")
    
    # 모델 설정
    default_model: str = Field(default="gpt-4", env="AI_DEFAULT_MODEL")
    max_tokens: int = Field(default=4000, env="AI_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="AI_TEMPERATURE")


class SecuritySettings(BaseSettings):
    """보안 설정"""
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")


class MonitoringSettings(BaseSettings):
    """모니터링 설정"""
    prometheus_port: int = Field(default=8000, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    alert_thresholds: Dict[str, float] = Field(default={
        "api_response_time": 2.0,
        "memory_usage": 80.0,
        "error_rate": 1.0
    })


class AppSettings(BaseSettings):
    """애플리케이션 설정"""
    app_name: str = Field(default="NewsTalk AI", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API 설정
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # CORS 설정
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")


class Settings(BaseSettings):
    """통합 설정 클래스"""
    
    # 하위 설정들
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    kafka: KafkaSettings = KafkaSettings()
    ai: AISettings = AISettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    app: AppSettings = AppSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 전역 설정 인스턴스 (싱글톤 패턴)
@lru_cache()
def get_settings() -> Settings:
    """설정 인스턴스 반환 (캐시됨)"""
    return Settings()


# 편의를 위한 전역 변수
settings = get_settings()


# 환경별 설정 검증
def validate_environment():
    """환경별 필수 설정 검증"""
    errors = []
    
    if settings.app.environment == "production":
        if not settings.ai.openai_api_key:
            errors.append("OpenAI API key is required in production")
        if settings.security.secret_key == "your-secret-key-here":
            errors.append("Secret key must be changed in production")
        if settings.app.debug:
            errors.append("Debug mode should be disabled in production")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")


# 설정 정보 출력 (보안 정보 제외)
def get_config_info() -> Dict[str, Any]:
    """설정 정보 반환 (보안 정보 제외)"""
    return {
        "app": {
            "name": settings.app.app_name,
            "version": settings.app.version,
            "environment": settings.app.environment,
            "debug": settings.app.debug
        },
        "database": {
            "host": settings.database.host,
            "port": settings.database.port,
            "database": settings.database.database
        },
        "redis": {
            "host": settings.redis.host,
            "port": settings.redis.port,
            "database": settings.redis.database
        },
        "kafka": {
            "bootstrap_servers": settings.kafka.bootstrap_servers,
            "topics": {
                "raw_news": settings.kafka.raw_news_topic,
                "processed_news": settings.kafka.processed_news_topic,
                "realtime_updates": settings.kafka.realtime_updates_topic
            }
        }
    }


# 초기화 시 검증 실행
if __name__ != "__main__":
    validate_environment()