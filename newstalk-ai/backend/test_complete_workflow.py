#!/usr/bin/env python3
"""
🎯 NewsTalk AI 완전 통합 워크플로우 테스트 (Stage 3)
3개 전문 에이전트 순차 실행 테스트:
- 뉴스분석(+팩트체킹) → 개인화(+스토리텔링) → 음성합성
- 95% 팩트체킹 정확도, 4.5/5.0 개인화 만족도, 프로 성우 수준 음성 품질
"""
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.langgraph.graphs.news_processing_graph import NewsProcessingGraph
from backend.langgraph.state.news_state import NewsState, ProcessingStage

class NewsTalkAITester:
    """NewsTalk AI 통합 테스트 클래스"""
    
    def __init__(self):
        self.graph = NewsProcessingGraph()
        
    def create_test_news_samples(self) -> list[Dict[str, str]]:
        """테스트용 뉴스 샘플 생성"""
        return [
            {
                "article_id": "test_001",
                "title": "한국 AI 스타트업, 글로벌 시장 진출 가속화",
                "content": """
                한국의 AI 스타트업들이 글로벌 시장 진출을 가속화하고 있다. 
                업계에 따르면 올해 상반기 국내 AI 스타트업 투자 유치액이 작년 대비 40% 증가했으며,
                특히 자연어 처리와 컴퓨터 비전 분야에서 두각을 나타내고 있다.
                
                대표적인 사례로 NewsTalk AI는 실시간 뉴스 개인화 서비스로 주목받고 있으며,
                95% 이상의 팩트체킹 정확도와 다중 캐릭터 보이스 합성 기술을 통해
                차별화된 경쟁력을 확보했다고 발표했다.
                
                전문가들은 한국 AI 스타트업들의 기술력이 세계적 수준에 도달했다고 평가하며,
                향후 글로벌 시장에서의 성과를 기대한다고 밝혔다.
                """,
                "user_id": "test_user_001"
            },
            {
                "article_id": "test_002", 
                "title": "기후변화 대응, 신재생에너지 확산 필요성 대두",
                "content": """
                최근 발표된 IPCC 보고서에 따르면 지구 평균 기온이 산업화 이전 대비 1.1도 상승했으며,
                2030년까지 1.5도 상승할 가능성이 높다고 경고했다.
                
                이에 따라 각국 정부는 신재생에너지 확산을 위한 정책을 강화하고 있다.
                우리나라도 2030년까지 신재생에너지 비중을 30.2%까지 확대하겠다고 발표했다.
                
                태양광, 풍력 등 재생에너지 기술의 발전으로 발전 단가가 지속적으로 하락하고 있어
                경제성도 크게 개선되고 있는 상황이다.
                """,
                "user_id": "test_user_002"
            },
            {
                "article_id": "test_003",
                "title": "K-팝 산업, 메타버스와 AI 기술 접목으로 새로운 전환점",
                "content": """
                K-팝 산업이 메타버스와 AI 기술을 적극 도입하며 새로운 전환점을 맞고 있다.
                
                주요 기획사들은 가상 아이돌 육성, AI 작곡, 메타버스 콘서트 등을 통해
                팬들과의 새로운 소통 방식을 모색하고 있다. 
                
                업계 관계자는 "기술과 엔터테인먼트의 융합을 통해 K-팝의 글로벌 영향력이
                더욱 확대될 것"이라고 전망했다.
                
                특히 AI를 활용한 개인 맞춤형 콘텐츠 제작과 실시간 팬 상호작용 서비스가
                차세대 한류의 핵심 동력이 될 것으로 기대된다.
                """,
                "user_id": "test_user_003"  
            }
        ]
    
    async def test_single_workflow(self, sample: Dict[str, str]) -> Dict[str, Any]:
        """단일 워크플로우 테스트"""
        print(f"\n{'='*60}")
        print(f"🧪 테스트 시작: {sample['article_id']}")
        print(f"📰 제목: {sample['title']}")
        print(f"{'='*60}")
        
        start_time = datetime.utcnow()
        
        try:
            # 워크플로우 실행
            final_state = await self.graph.process_news(
                article_id=sample["article_id"],
                content=sample["content"],
                title=sample["title"],
                user_id=sample.get("user_id")
            )
            
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # 결과 분석
            test_result = self.analyze_test_result(final_state, processing_time)
            
            # 결과 출력
            self.print_test_result(sample["article_id"], test_result)
            
            return test_result
            
        except Exception as e:
            print(f"❌ 테스트 실패: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": (datetime.utcnow() - start_time).total_seconds()
            }
    
    def analyze_test_result(self, final_state: NewsState, processing_time: float) -> Dict[str, Any]:
        """테스트 결과 분석"""
        result = {
            "success": final_state.processing_stage == ProcessingStage.COMPLETED,
            "processing_time": processing_time,
            "processing_stage": final_state.processing_stage.value,
            "agents_completed": {},
            "quality_metrics": {},
            "errors": final_state.errors or []
        }
        
        # 🎯 3개 전문 에이전트 완료 상태 확인 (Stage 3)
        result["agents_completed"] = {
            "news_analysis": final_state.trend_analysis_result is not None,
            "fact_checking": final_state.fact_check_result is not None,
            "personalization": final_state.personalization_result is not None,
            "voice_synthesis": final_state.voice_synthesis_result is not None
        }
        
        # 🎯 품질 메트릭 수집 (Stage 3)
        if final_state.trend_analysis_result:
            result["quality_metrics"]["trend_score"] = final_state.trend_analysis_result.trending_score
            
        if final_state.fact_check_result:
            result["quality_metrics"]["credibility_score"] = final_state.fact_check_result.credibility_score
            
        if final_state.personalization_result:
            if hasattr(final_state.personalization_result, 'personalization_score'):
                result["quality_metrics"]["personalization_score"] = getattr(
                    final_state.personalization_result.personalization_score, 'overall_score', 0.0
                )
            
        if final_state.voice_synthesis_result:
            result["quality_metrics"]["synthesis_quality"] = final_state.voice_synthesis_result.synthesis_quality
        
        # 전체 품질 점수
        overall_quality = final_state.metrics.get("overall_quality_score", 0.0)
        result["quality_metrics"]["overall_quality"] = overall_quality
        
        return result
    
    def print_test_result(self, article_id: str, result: Dict[str, Any]):
        """테스트 결과 출력"""
        print(f"\n📊 테스트 결과: {article_id}")
        print(f"{'─'*50}")
        
        # 전체 성공 여부
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} 전체 상태: {'성공' if result['success'] else '실패'}")
        print(f"⏱️  처리 시간: {result['processing_time']:.2f}초")
        print(f"🏁 최종 단계: {result['processing_stage']}")
        
        # 🎯 3개 전문 에이전트별 완료 상태 (Stage 3)
        print(f"\n🤖 전문 에이전트별 완료 상태:")
        agents = result["agents_completed"]
        print(f"  🔍 뉴스분석(+팩트체킹): {'✅' if agents.get('news_analysis') and agents.get('fact_checking') else '❌'}")
        print(f"  👤 개인화(+스토리텔링): {'✅' if agents.get('personalization') else '❌'}")
        print(f"  🎵 음성합성: {'✅' if agents.get('voice_synthesis') else '❌'}")
        
        # 🎯 품질 메트릭 (Stage 3)
        quality = result["quality_metrics"]
        if quality:
            print(f"\n📏 품질 메트릭:")
            if "trend_score" in quality:
                print(f"  📈 트렌드 점수: {quality['trend_score']:.2f}")
            if "credibility_score" in quality:
                print(f"  🔍 신뢰도 점수: {quality['credibility_score']:.2f} (목표: 0.95)")
            if "personalization_score" in quality:
                print(f"  👤 개인화 점수: {quality['personalization_score']:.2f} (목표: 4.5)")
            if "synthesis_quality" in quality:
                print(f"  🎵 음성 품질: {quality['synthesis_quality']:.2f} (목표: 프로 수준)")
            if "overall_quality" in quality:
                print(f"  🏆 전체 품질: {quality['overall_quality']:.2f} (목표: 0.8)")
        
        # 에러 정보
        if result["errors"]:
            print(f"\n⚠️  에러 정보:")
            for error in result["errors"]:
                print(f"  • {error}")
        
        print(f"{'─'*50}")
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """포괄적 테스트 실행"""
        print(f"\n🚀 NewsTalk AI 완전 통합 테스트 시작")
        print(f"{'='*80}")
        print(f"🎯 테스트 목표 (Stage 3):")
        print(f"  • 3개 전문 에이전트 순차 실행 검증")
        print(f"  • 95% 팩트체킹 정확도 달성")
        print(f"  • 4.5/5.0 개인화 만족도 달성")
        print(f"  • 4.2/5.0 스토리텔링 몰입도 달성")
        print(f"  • 프로 성우 수준 음성 품질")
        print(f"  • 5분 이내 처리 완료")
        print(f"{'='*80}")
        
        samples = self.create_test_news_samples()
        all_results = []
        
        # 각 샘플 테스트
        for sample in samples:
            result = await self.test_single_workflow(sample)
            all_results.append(result)
        
        # 전체 결과 분석
        summary = self.generate_test_summary(all_results)
        self.print_test_summary(summary)
        
        return summary
    
    def generate_test_summary(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        """테스트 요약 생성"""
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r["success"])
        
        summary = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "average_processing_time": sum(r["processing_time"] for r in results) / total_tests if total_tests > 0 else 0,
            "agent_completion_rates": {},
            "quality_averages": {},
            "errors": []
        }
        
        # 🎯 3개 전문 에이전트별 완료율 (Stage 3)
        agent_names = ["news_analysis", "fact_checking", "personalization", "voice_synthesis"]
        for agent in agent_names:
            completed = sum(1 for r in results if r["agents_completed"].get(agent, False))
            summary["agent_completion_rates"][agent] = completed / total_tests if total_tests > 0 else 0
        
        # 품질 메트릭 평균
        quality_metrics = ["trend_score", "credibility_score", "personalization_score", "synthesis_quality", "overall_quality"]
        for metric in quality_metrics:
            values = [r["quality_metrics"].get(metric) for r in results if metric in r["quality_metrics"]]
            if values:
                summary["quality_averages"][metric] = sum(values) / len(values)
        
        # 에러 수집
        for result in results:
            summary["errors"].extend(result["errors"])
        
        return summary
    
    def print_test_summary(self, summary: Dict[str, Any]):
        """테스트 요약 출력"""
        print(f"\n🏆 전체 테스트 요약")
        print(f"{'='*80}")
        
        # 전체 성공률
        success_icon = "✅" if summary["success_rate"] >= 0.8 else "⚠️" if summary["success_rate"] >= 0.5 else "❌"
        print(f"{success_icon} 전체 성공률: {summary['success_rate']:.1%} ({summary['successful_tests']}/{summary['total_tests']})")
        print(f"⏱️  평균 처리 시간: {summary['average_processing_time']:.2f}초")
        
        # 🎯 3개 전문 에이전트별 완료율 (Stage 3)
        print(f"\n🤖 전문 에이전트별 완료율:")
        completion_rates = summary["agent_completion_rates"]
        print(f"  🔍 뉴스분석(+팩트체킹): {completion_rates.get('news_analysis', 0):.1%}")
        print(f"  👤 개인화(+스토리텔링): {completion_rates.get('personalization', 0):.1%}")
        print(f"  🎵 음성합성: {completion_rates.get('voice_synthesis', 0):.1%}")
        
        # 품질 메트릭 평균
        quality_avgs = summary["quality_averages"]
        if quality_avgs:
            print(f"\n📊 평균 품질 메트릭:")
            if "trend_score" in quality_avgs:
                print(f"  📈 트렌드 점수: {quality_avgs['trend_score']:.3f}")
            if "credibility_score" in quality_avgs:
                fact_check_icon = "🎯" if quality_avgs['credibility_score'] >= 0.95 else "⚠️"
                print(f"  {fact_check_icon} 신뢰도 점수: {quality_avgs['credibility_score']:.3f} (목표: 0.95)")
            if "personalization_score" in quality_avgs:
                personalization_icon = "🎯" if quality_avgs['personalization_score'] >= 4.5 else "⚠️"
                print(f"  {personalization_icon} 개인화 점수: {quality_avgs['personalization_score']:.3f} (목표: 4.5)")
            if "synthesis_quality" in quality_avgs:
                voice_icon = "🎯" if quality_avgs['synthesis_quality'] >= 0.8 else "⚠️"
                print(f"  {voice_icon} 음성 품질: {quality_avgs['synthesis_quality']:.3f} (목표: 0.8)")
            if "overall_quality" in quality_avgs:
                overall_icon = "🎯" if quality_avgs['overall_quality'] >= 0.8 else "⚠️"
                print(f"  {overall_icon} 전체 품질: {quality_avgs['overall_quality']:.3f} (목표: 0.8)")
        
        # 에러 요약
        if summary["errors"]:
            print(f"\n⚠️  에러 요약 ({len(summary['errors'])}개):")
            error_counts = {}
            for error in summary["errors"]:
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            for error_type, count in error_counts.items():
                print(f"  • {error_type}: {count}회")
        
        # 전체 평가
        print(f"\n🎯 종합 평가:")
        if summary["success_rate"] >= 0.9 and quality_avgs.get("overall_quality", 0) >= 0.8:
            print("  🏆 우수: NewsTalk AI 시스템이 안정적으로 작동하고 있습니다!")
        elif summary["success_rate"] >= 0.7:
            print("  ✅ 양호: 일부 개선이 필요하지만 전반적으로 잘 작동합니다.")
        else:
            print("  ⚠️  주의: 시스템 안정성 개선이 필요합니다.")
        
        print(f"{'='*80}")

async def main():
    """메인 실행 함수"""
    tester = NewsTalkAITester()
    
    try:
        # 시스템 정보 출력
        stats = tester.graph.get_processing_stats()
        print(f"🔧 시스템 정보:")
        print(f"  • 에이전트 수: {stats['agents_count']}개")
        print(f"  • 에이전트: {', '.join(stats['agents'])}")
        print(f"  • 품질 목표: {stats['quality_targets']}")
        
        # 통합 테스트 실행
        summary = await tester.run_comprehensive_test()
        
        # 종료 코드 결정
        exit_code = 0 if summary["success_rate"] >= 0.8 else 1
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 