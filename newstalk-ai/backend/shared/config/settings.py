"""
🎯 NewsTalk AI 통합 설정 시스템
===============================

순환 의존성 해결과 환경별 설정 관리를 위한 엔터프라이즈급 설정 시스템:
- 환경별 설정 자동 로드 (dev, staging, prod)
- 설정 검증 및 타입 안전성
- 동적 설정 업데이트 지원
- 보안 설정 암호화
- 설정 변경 추적 및 감사
- 페일오버 및 복구 설정
"""
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Type
from pathlib import Path
from enum import Enum
import yaml
from pydantic import BaseSettings, validator, Field
from pydantic.env_settings import SettingsSourceCallable
import redis
import asyncio
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class Environment(Enum):
    """환경 타입"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(Enum):
    """로그 레벨"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseSettings:
    """데이터베이스 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "newstalk_ai"
    username: str = "postgres"
    password: str = ""
    
    # 연결 풀 설정
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600
    
    # 성능 설정
    enable_query_cache: bool = True
    cache_ttl_seconds: int = 300
    slow_query_threshold: float = 1.0
    
    # 복제본 설정
    enable_read_replica: bool = False
    read_replica_urls: List[str] = field(default_factory=list)
    
    # SSL 설정
    ssl_mode: str = "prefer"
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None
    
    def get_database_url(self, for_replica: bool = False) -> str:
        """데이터베이스 URL 생성"""
        if for_replica and self.read_replica_urls:
            return self.read_replica_urls[0]  # 첫 번째 복제본 사용
        
        return (
            f"postgresql+asyncpg://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

@dataclass
class RedisSettings:
    """Redis 설정"""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    
    # 연결 풀 설정
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    
    # 클러스터 설정
    cluster_nodes: List[str] = field(default_factory=list)
    enable_cluster: bool = False
    
    # 센티넬 설정
    sentinel_hosts: List[str] = field(default_factory=list)
    sentinel_service: str = "mymaster"
    enable_sentinel: bool = False
    
    def get_redis_url(self) -> str:
        """Redis URL 생성"""
        auth_part = f":{self.password}@" if self.password else ""
        return f"redis://{auth_part}{self.host}:{self.port}/{self.database}"

@dataclass
class KafkaSettings:
    """Kafka 설정"""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    
    # 보안 설정
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    
    # 프로듀서 설정
    producer_acks: str = "all"
    producer_retries: int = 3
    producer_max_in_flight: int = 5
    producer_compression_type: str = "gzip"
    
    # 컨슈머 설정
    consumer_group_id: str = "newstalk-ai"
    consumer_auto_offset_reset: str = "latest"
    consumer_max_poll_records: int = 500
    
    # 토픽 설정
    topics: Dict[str, str] = field(default_factory=lambda: {
        "news_raw": "news.raw",
        "news_processed": "news.processed",
        "user_interactions": "user.interactions",
        "system_events": "system.events"
    })

@dataclass
class LangGraphSettings:
    """LangGraph 설정"""
    # PostgreSQL 체크포인팅
    postgres_url: str = ""
    checkpoint_namespace: str = "newstalk_ai"
    
    # OpenAI 설정
    openai_api_key: str = ""
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2000
    
    # Claude 설정
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-sonnet-20240229"
    
    # Langfuse 추적
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    
    # 에이전트 설정
    max_execution_time: int = 300  # 5분
    max_iterations: int = 10
    enable_debug: bool = False

@dataclass
class APISettings:
    """API 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # CORS 설정
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # 보안 설정
    secret_key: str = "your-secret-key-change-this"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    
    # 요청 제한
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # API 키 설정
    api_keys: Dict[str, str] = field(default_factory=dict)

@dataclass
class MonitoringSettings:
    """모니터링 설정"""
    # Prometheus 설정
    enable_prometheus: bool = True
    prometheus_port: int = 9090
    
    # 로깅 설정
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    enable_json_logging: bool = False
    
    # Slack 알림
    slack_webhook_url: Optional[str] = None
    slack_channel: str = "#alerts"
    enable_slack_alerts: bool = False
    
    # 메트릭 설정
    metrics_retention_days: int = 30
    enable_performance_tracking: bool = True
    
    # 헬스체크 설정
    health_check_interval: int = 60  # seconds
    health_check_timeout: int = 30   # seconds

@dataclass
class AISettings:
    """AI 처리 설정"""
    # 팩트체킹 설정
    fact_check_confidence_threshold: float = 0.7
    fact_check_timeout: int = 30
    enable_fact_checking: bool = True
    
    # 음성 합성 설정
    voice_synthesis_model: str = "azure"
    azure_speech_key: Optional[str] = None
    azure_speech_region: str = "koreacentral"
    
    # 감정 분석 설정
    sentiment_model: str = "huggingface"
    sentiment_threshold: float = 0.5
    
    # 개인화 설정
    personalization_window_days: int = 30
    min_interactions_for_personalization: int = 10
    
    # 성능 설정
    ai_request_timeout: int = 60
    ai_max_concurrent_requests: int = 10
    ai_retry_attempts: int = 3

class NewsTalkSettings(BaseSettings):
    """
    NewsTalk AI 통합 설정 클래스
    
    환경 변수, 설정 파일, 기본값을 통합하여 관리
    """
    
    # 환경 설정
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    
    # 서비스별 설정
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    kafka: KafkaSettings = field(default_factory=KafkaSettings)
    langgraph: LangGraphSettings = field(default_factory=LangGraphSettings)
    api: APISettings = field(default_factory=APISettings)
    monitoring: MonitoringSettings = field(default_factory=MonitoringSettings)
    ai: AISettings = field(default_factory=AISettings)
    
    # 설정 메타데이터
    config_file_path: Optional[str] = None
    config_last_updated: Optional[datetime] = None
    config_checksum: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = True
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,
                yaml_config_settings_source,
                env_settings,
                file_secret_settings,
            )
    
    def __post_init__(self):
        """초기화 후 설정 검증"""
        self._validate_settings()
        self._setup_logging()
        self._calculate_checksum()
    
    def _validate_settings(self):
        """설정 검증"""
        try:
            # 필수 설정 검증
            if self.environment == Environment.PRODUCTION:
                required_settings = [
                    (self.database.password, "Database password is required in production"),
                    (self.api.secret_key != "your-secret-key-change-this", "Secret key must be changed in production"),
                    (self.langgraph.openai_api_key, "OpenAI API key is required")
                ]
                
                for condition, message in required_settings:
                    if not condition:
                        raise ValueError(f"Production validation failed: {message}")
            
            # 포트 충돌 검증
            used_ports = {self.api.port, self.redis.port, self.database.port}
            if self.monitoring.enable_prometheus:
                used_ports.add(self.monitoring.prometheus_port)
            
            if len(used_ports) != len({self.api.port, self.redis.port, self.database.port, self.monitoring.prometheus_port}):
                logger.warning("Port conflicts detected in configuration")
            
            # URL 형식 검증
            if self.redis.host and not self._is_valid_host(self.redis.host):
                raise ValueError(f"Invalid Redis host: {self.redis.host}")
            
            if self.database.host and not self._is_valid_host(self.database.host):
                raise ValueError(f"Invalid database host: {self.database.host}")
            
            logger.info("Configuration validation completed successfully")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def _is_valid_host(self, host: str) -> bool:
        """호스트명 유효성 검증"""
        import re
        # IPv4, IPv6, 도메인명 검증
        ipv4_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        domain_pattern = r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        
        return (
            host == "localhost" or
            re.match(ipv4_pattern, host) or
            re.match(domain_pattern, host)
        )
    
    def _setup_logging(self):
        """로깅 설정"""
        log_level = getattr(logging, self.monitoring.log_level.value)
        
        # 기본 로거 설정
        logging.basicConfig(
            level=log_level,
            format=self.monitoring.log_format,
            filename=self.monitoring.log_file
        )
        
        # JSON 로깅 설정
        if self.monitoring.enable_json_logging:
            try:
                import pythonjsonlogger.jsonlogger
                json_handler = logging.StreamHandler()
                formatter = pythonjsonlogger.jsonlogger.JsonFormatter()
                json_handler.setFormatter(formatter)
                
                root_logger = logging.getLogger()
                root_logger.addHandler(json_handler)
                
            except ImportError:
                logger.warning("pythonjsonlogger not installed, falling back to standard logging")
    
    def _calculate_checksum(self):
        """설정 체크섬 계산"""
        try:
            config_str = self.json(sort_keys=True)
            self.config_checksum = hashlib.sha256(config_str.encode()).hexdigest()
            self.config_last_updated = datetime.utcnow()
        except Exception as e:
            logger.error(f"Failed to calculate config checksum: {e}")
    
    def get_database_url(self, for_replica: bool = False) -> str:
        """데이터베이스 URL 반환"""
        return self.database.get_database_url(for_replica)
    
    def get_redis_url(self) -> str:
        """Redis URL 반환"""
        return self.redis.get_redis_url()
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Kafka 클라이언트 설정 반환"""
        config = {
            "bootstrap_servers": self.kafka.bootstrap_servers,
            "security_protocol": self.kafka.security_protocol,
        }
        
        if self.kafka.sasl_mechanism:
            config.update({
                "sasl_mechanism": self.kafka.sasl_mechanism,
                "sasl_plain_username": self.kafka.sasl_username,
                "sasl_plain_password": self.kafka.sasl_password,
            })
        
        if self.kafka.ssl_cafile:
            config.update({
                "ssl_cafile": self.kafka.ssl_cafile,
                "ssl_certfile": self.kafka.ssl_certfile,
                "ssl_keyfile": self.kafka.ssl_keyfile,
            })
        
        return config
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """설정을 딕셔너리로 내보내기"""
        config_dict = self.dict()
        
        if not include_secrets:
            # 민감한 정보 마스킹
            sensitive_fields = [
                "database.password",
                "redis.password", 
                "kafka.sasl_password",
                "langgraph.openai_api_key",
                "langgraph.anthropic_api_key",
                "langgraph.langfuse_secret_key",
                "api.secret_key",
                "api.api_keys",
                "monitoring.slack_webhook_url",
                "ai.azure_speech_key"
            ]
            
            for field in sensitive_fields:
                keys = field.split(".")
                current = config_dict
                for key in keys[:-1]:
                    if key in current:
                        current = current[key]
                    else:
                        break
                else:
                    if keys[-1] in current and current[keys[-1]]:
                        current[keys[-1]] = "***MASKED***"
        
        return config_dict
    
    async def reload_from_file(self, config_path: str = None):
        """파일에서 설정 다시 로드"""
        try:
            if config_path:
                self.config_file_path = config_path
            
            if self.config_file_path and Path(self.config_file_path).exists():
                # 파일 로드 및 설정 업데이트
                new_settings = load_config_from_file(self.config_file_path)
                
                # 설정 병합
                for key, value in new_settings.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                # 검증 및 체크섬 재계산
                self._validate_settings()
                self._calculate_checksum()
                
                logger.info(f"Configuration reloaded from {self.config_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise
    
    def has_changed(self, other_checksum: str) -> bool:
        """설정 변경 여부 확인"""
        return self.config_checksum != other_checksum

def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """YAML 설정 파일 소스"""
    config_data = {}
    
    # 환경별 설정 파일 탐색
    env = os.getenv("ENVIRONMENT", "development").lower()
    config_files = [
        f"config.{env}.yaml",
        f"config.{env}.yml", 
        "config.yaml",
        "config.yml"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                config_data = load_config_from_file(str(config_path))
                logger.info(f"Loaded configuration from {config_path}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {config_path}: {e}")
                continue
    
    return config_data

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """파일에서 설정 로드"""
    config_path = Path(file_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {file_path}")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f) or {}
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                return {}
                
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {e}")
        raise

class ConfigManager:
    """
    설정 관리자
    
    동적 설정 업데이트, 분산 설정 관리, 설정 감사 등을 담당
    """
    
    def __init__(self, settings: NewsTalkSettings):
        self.settings = settings
        self._redis_client: Optional[redis.Redis] = None
        self._config_watchers: List[Callable] = []
        self._last_update_check = datetime.utcnow()
        
    async def initialize(self):
        """설정 관리자 초기화"""
        try:
            # Redis 연결 (분산 설정용)
            if self.settings.redis.host:
                self._redis_client = redis.from_url(
                    self.settings.get_redis_url(),
                    encoding="utf-8",
                    decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("Redis connection established for configuration management")
                
        except Exception as e:
            logger.warning(f"Failed to initialize Redis for config management: {e}")
    
    async def watch_config_changes(self, callback: Callable):
        """설정 변경 감시 콜백 등록"""
        self._config_watchers.append(callback)
    
    async def update_config(self, key: str, value: Any, persist: bool = True):
        """동적 설정 업데이트"""
        try:
            # 설정 업데이트
            keys = key.split(".")
            current = self.settings
            
            for k in keys[:-1]:
                current = getattr(current, k)
            
            setattr(current, keys[-1], value)
            
            # Redis에 저장 (분산 환경용)
            if persist and self._redis_client:
                await self._redis_client.set(
                    f"config:{key}",
                    json.dumps(value, default=str),
                    ex=3600  # 1시간 TTL
                )
            
            # 콜백 실행
            for callback in self._config_watchers:
                try:
                    await callback(key, value)
                except Exception as e:
                    logger.error(f"Config watcher callback failed: {e}")
            
            logger.info(f"Configuration updated: {key} = {value}")
            
        except Exception as e:
            logger.error(f"Failed to update configuration {key}: {e}")
            raise
    
    async def get_distributed_config(self, key: str) -> Optional[Any]:
        """분산 설정 조회"""
        if not self._redis_client:
            return None
        
        try:
            value = await self._redis_client.get(f"config:{key}")
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Failed to get distributed config {key}: {e}")
        
        return None
    
    async def sync_distributed_config(self):
        """분산 설정 동기화"""
        if not self._redis_client:
            return
        
        try:
            # Redis에서 모든 설정 키 조회
            keys = await self._redis_client.keys("config:*")
            
            for key in keys:
                config_key = key.replace("config:", "")
                value = await self.get_distributed_config(config_key)
                
                if value is not None:
                    await self.update_config(config_key, value, persist=False)
            
        except Exception as e:
            logger.error(f"Failed to sync distributed config: {e}")
    
    def get_config_audit_log(self) -> List[Dict[str, Any]]:
        """설정 변경 감사 로그 반환"""
        # 실제 구현에서는 데이터베이스나 파일에서 로드
        return [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "config_key": "example.key",
                "old_value": "old_value",
                "new_value": "new_value",
                "user": "system",
                "environment": self.settings.environment.value
            }
        ]

# 전역 설정 인스턴스
_settings: Optional[NewsTalkSettings] = None
_config_manager: Optional[ConfigManager] = None

def get_settings() -> NewsTalkSettings:
    """설정 싱글톤 인스턴스 반환"""
    global _settings
    if _settings is None:
        try:
            _settings = NewsTalkSettings()
            logger.info(f"Settings loaded for environment: {_settings.environment.value}")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            # 기본 설정으로 폴백
            _settings = NewsTalkSettings(environment=Environment.DEVELOPMENT)
    
    return _settings

async def get_config_manager() -> ConfigManager:
    """설정 관리자 싱글톤 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        settings = get_settings()
        _config_manager = ConfigManager(settings)
        await _config_manager.initialize()
    
    return _config_manager

def reload_settings(config_path: str = None):
    """설정 다시 로드"""
    global _settings, _config_manager
    _settings = None
    _config_manager = None
    
    if config_path:
        os.environ["CONFIG_FILE_PATH"] = config_path
    
    return get_settings()

# 편의 함수들
def get_database_url(for_replica: bool = False) -> str:
    """데이터베이스 URL 반환"""
    return get_settings().get_database_url(for_replica)

def get_redis_url() -> str:
    """Redis URL 반환"""
    return get_settings().get_redis_url()

def is_production() -> bool:
    """프로덕션 환경 여부"""
    return get_settings().is_production()

def is_development() -> bool:
    """개발 환경 여부"""
    return get_settings().is_development()

def get_log_level() -> str:
    """로그 레벨 반환"""
    return get_settings().monitoring.log_level.value