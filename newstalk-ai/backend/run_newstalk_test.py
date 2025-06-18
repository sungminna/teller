#!/usr/bin/env python3
"""
NewsTalk AI 시스템 실행 스크립트
"""
import asyncio
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graphs.news_processing_graph import NewsProcessingGraph

async def main():
    """메인 실행"""
    print("🚀 NewsTalk AI 시스템 시작")
    
    # 워크플로우 그래프 초기화
    graph = NewsProcessingGraph()
    
    # 시스템 정보 출력
    stats = graph.get_processing_stats()
    print(f"📊 시스템 정보: {stats}")
    
    # 테스트 뉴스 처리
    test_article = {
        "article_id": "demo_001",
        "title": "NewsTalk AI 시스템 데모",
        "content": "NewsTalk AI는 4개의 전문 에이전트로 구성된 혁신적인 뉴스 처리 시스템입니다. 트렌드 분석, 95% 정확도의 팩트체킹, 매력적인 스토리텔링, 그리고 다중 캐릭터 음성 합성을 통해 개인화된 뉴스 경험을 제공합니다."
    }
    
    print(f"\n📰 뉴스 처리 시작: {test_article['title']}")
    
    try:
        result = await graph.process_news(
            article_id=test_article["article_id"],
            content=test_article["content"],
            title=test_article["title"]
        )
        
        print(f"✅ 처리 완료: {result.processing_stage}")
        print(f"📊 메트릭: {result.metrics}")
        
    except Exception as e:
        print(f"❌ 처리 실패: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 