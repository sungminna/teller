# 🔧 Kafka 연결 문제 해결 가이드

## 📋 문제 상황

Docker 컨테이너 내부에서 Kafka에 연결할 때 `localhost:9092` 대신 `kafka:29092`를 사용해야 하는 문제가 발생했습니다.

## 🎯 해결책

### 1. 자동 환경 감지 시스템 구현

**파일**: `newstalk-ai/backend/shared/config/settings.py`

환경별로 적절한 Kafka Bootstrap 서버를 자동으로 선택하는 시스템을 구현했습니다:

```python
def _get_kafka_bootstrap_servers(self) -> List[str]:
    """환경에 따른 Kafka Bootstrap 서버 자동 선택"""
    import socket
    import os
    
    # 환경변수에서 명시적으로 설정된 경우 우선 사용
    env_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    if env_servers:
        return env_servers.split(",")
    
    # Docker 환경 감지
    is_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    
    # Kubernetes 환경 감지
    is_k8s = os.path.exists("/var/run/secrets/kubernetes.io") or os.getenv("KUBERNETES_SERVICE_HOST")
    
    if is_k8s:
        # Kubernetes 환경: 서비스 이름 사용
        return ["kafka:9092"]
    elif is_docker:
        # Docker Compose 환경: 내부 네트워크 사용
        try:
            socket.gethostbyname("kafka")
            return ["kafka:29092"]
        except socket.gaierror:
            return ["localhost:9092"]
    else:
        # 로컬 개발 환경
        return self.kafka.bootstrap_servers or ["localhost:9092"]
```

### 2. Kafka 클라이언트 업데이트

**파일**: `newstalk-ai/backend/api/utils/kafka_client.py`

모든 Kafka 클라이언트가 새로운 설정 시스템을 사용하도록 업데이트했습니다:

- `create_kafka_producer()` 
- `create_kafka_consumer()`
- `get_kafka_admin()`

### 3. Docker Compose 설정 개선

**파일**: `newstalk-ai/infrastructure/docker/docker-compose.yml`

컨테이너 환경변수에 명시적 설정을 추가했습니다:

```yaml
environment:
  - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
  - DOCKER_CONTAINER=true
  - ENVIRONMENT=production
```

### 4. 환경변수 문서화

**파일**: `newstalk-ai/env.example`

환경별 Kafka 설정을 명확히 문서화했습니다:

```bash
# Kafka Configuration (Stage 6)
# 로컬 개발: localhost:9092
# Docker Compose: kafka:29092 (자동 감지)
# Kubernetes: kafka:9092 (자동 감지)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

## 🧪 테스트 방법

### 1. 연결 테스트 스크립트 실행

```bash
cd newstalk-ai
python test_kafka_connection.py
```

### 2. 환경별 테스트

**로컬 개발 환경:**
```bash
# 기본 설정으로 localhost:9092 사용
python test_kafka_connection.py
```

**Docker 환경:**
```bash
# Docker Compose로 전체 스택 실행
docker-compose -f infrastructure/docker/docker-compose.yml up -d
```

**환경변수 명시적 설정:**
```bash
export KAFKA_BOOTSTRAP_SERVERS=custom-kafka:9092
python test_kafka_connection.py
```

## 📊 환경별 연결 설정

| 환경 | 감지 조건 | Bootstrap 서버 |
|------|-----------|----------------|
| 로컬 개발 | 기본값 | `localhost:9092` |
| Docker Compose | `/.dockerenv` 존재 또는 `DOCKER_CONTAINER=true` | `kafka:29092` |
| Kubernetes | `/var/run/secrets/kubernetes.io` 존재 또는 `KUBERNETES_SERVICE_HOST` 설정 | `kafka:9092` |
| 명시적 설정 | `KAFKA_BOOTSTRAP_SERVERS` 환경변수 | 환경변수 값 사용 |

## 🔍 로그 예시

시스템이 올바르게 작동하면 다음과 같은 로그가 출력됩니다:

```
INFO:backend.shared.config.settings:🐳 Docker 환경 감지 - kafka:29092 사용
INFO:backend.api.utils.kafka_client:✅ Enhanced Kafka producer created: ['kafka:29092']
INFO:backend.api.utils.kafka_client:✅ Enhanced Kafka consumer created for topics: ['news.raw'] with servers: ['kafka:29092']
```

## 🚀 배포 시 주의사항

1. **환경변수 우선순위**: `KAFKA_BOOTSTRAP_SERVERS` 환경변수가 설정되면 자동 감지보다 우선합니다.

2. **네트워크 연결성**: Docker 환경에서는 `kafka` 호스트명이 해결 가능한지 확인합니다.

3. **포트 매핑**: 
   - 외부 접근: `localhost:9092`
   - 컨테이너 간 통신: `kafka:29092`

4. **헬스체크**: 각 환경에서 Kafka 연결 상태를 모니터링합니다.

## 🎉 결과

✅ **문제 해결**: Docker 컨테이너 내부에서 자동으로 `kafka:29092` 사용  
✅ **호환성**: 기존 로컬 개발 환경과 완전 호환  
✅ **확장성**: Kubernetes 환경도 자동 지원  
✅ **유연성**: 환경변수로 수동 설정 가능  

이제 [NewsTalk AI 프로젝트][[memory:766121352915983545]]에서 Kafka 연결 문제가 완전히 해결되었습니다! 