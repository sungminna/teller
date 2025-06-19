#!/usr/bin/env python3
"""
🧪 Kafka 연결 설정 테스트 스크립트
===================================

환경별 Kafka Bootstrap 서버 자동 선택 기능을 테스트합니다.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# 백엔드 패키지 경로 추가
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.shared.config.settings import get_settings
from backend.api.utils.kafka_client import get_kafka_producer, get_kafka_admin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_kafka_connection():
    """Kafka 연결 테스트"""
    print("🧪 Kafka 연결 설정 테스트")
    print("=" * 50)
    
    # 설정 확인
    settings = get_settings()
    kafka_config = settings.get_kafka_config()
    
    print(f"📋 현재 환경 정보:")
    print(f"   - Docker 환경: {os.path.exists('/.dockerenv') or os.getenv('DOCKER_CONTAINER') == 'true'}")
    print(f"   - Kubernetes 환경: {os.path.exists('/var/run/secrets/kubernetes.io') or os.getenv('KUBERNETES_SERVICE_HOST')}")
    print(f"   - 환경변수 KAFKA_BOOTSTRAP_SERVERS: {os.getenv('KAFKA_BOOTSTRAP_SERVERS')}")
    print(f"   - 선택된 Bootstrap 서버: {kafka_config['bootstrap_servers']}")
    print()
    
    try:
        # Kafka Admin 클라이언트 테스트
        print("🔧 Kafka Admin 클라이언트 연결 테스트...")
        admin = await get_kafka_admin()
        
        # 메타데이터 조회 시도
        metadata = await admin.describe_cluster()
        print(f"✅ Kafka 클러스터 연결 성공!")
        print(f"   - 클러스터 ID: {metadata.cluster_id}")
        print(f"   - 브로커 수: {len(metadata.brokers)}")
        
        # 브로커 정보 출력
        for broker in metadata.brokers:
            print(f"   - 브로커: {broker.host}:{broker.port} (ID: {broker.node_id})")
        
        await admin.close()
        
    except Exception as e:
        print(f"❌ Kafka Admin 연결 실패: {e}")
        return False
    
    try:
        # Kafka Producer 테스트
        print("\n📤 Kafka Producer 연결 테스트...")
        producer = await get_kafka_producer()
        
        # 테스트 메시지 전송
        test_message = {
            "test": True,
            "message": "Kafka 연결 테스트",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        await producer.send("test-topic", value=test_message)
        print("✅ Kafka Producer 연결 및 메시지 전송 성공!")
        
        await producer.stop()
        
    except Exception as e:
        print(f"❌ Kafka Producer 연결 실패: {e}")
        return False
    
    print("\n🎉 모든 Kafka 연결 테스트 통과!")
    return True


async def test_environment_detection():
    """환경 감지 테스트"""
    print("\n🌍 환경 감지 테스트")
    print("=" * 50)
    
    settings = get_settings()
    
    # 다양한 환경변수 시나리오 테스트
    test_cases = [
        {
            "name": "환경변수 명시적 설정",
            "env_vars": {"KAFKA_BOOTSTRAP_SERVERS": "custom:9092"},
            "expected": ["custom:9092"]
        },
        {
            "name": "Docker 환경 시뮬레이션",
            "env_vars": {"DOCKER_CONTAINER": "true"},
            "expected": ["kafka:29092"]
        },
        {
            "name": "Kubernetes 환경 시뮬레이션", 
            "env_vars": {"KUBERNETES_SERVICE_HOST": "10.0.0.1"},
            "expected": ["kafka:9092"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 테스트: {test_case['name']}")
        
        # 환경변수 설정
        original_env = {}
        for key, value in test_case["env_vars"].items():
            original_env[key] = os.getenv(key)
            os.environ[key] = value
        
        try:
            # 새로운 설정 인스턴스 생성
            test_servers = settings._get_kafka_bootstrap_servers()
            print(f"   예상: {test_case['expected']}")
            print(f"   실제: {test_servers}")
            
            if test_servers == test_case["expected"]:
                print("   ✅ 통과")
            else:
                print("   ❌ 실패")
        
        finally:
            # 환경변수 복원
            for key in test_case["env_vars"]:
                if original_env[key] is not None:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)


if __name__ == "__main__":
    asyncio.run(test_kafka_connection())
    asyncio.run(test_environment_detection()) 