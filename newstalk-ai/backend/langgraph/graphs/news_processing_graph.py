"""
🎯 NewsTalk AI 통합 워크플로우 - LangGraph StateGraph (Stage 3)
3개 전문 에이전트 순차 실행: 뉴스분석(+팩트체킹) → 개인화(+스토리텔링) → 음성합성
95% 팩트체킹 정확도, 4.5/5.0 개인화 만족도, 프로 성우 수준 음성 품질
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Literal
from uuid import uuid4

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres import PostgresCheckpointer
from langgraph.pregel import Pregel
from langfuse import Langfuse

from ..state.news_state import NewsState, ProcessingStage
from ..agents.analysis_agent import AnalysisAgent
from ..agents.personalization_agent import PersonalizationAgent
from ..agents.voice_synthesis_agent import VoiceSynthesisAgent
from ...shared.config.settings import settings

logger = logging.getLogger(__name__)

class NewsProcessingGraph:
    """
    🎯 NewsTalk AI 메인 워크플로우 그래프 (Stage 3)
    - 3개 전문 에이전트 순차 실행
    - News Analysis Agent: 95% 팩트체킹 정확도 목표
    - Personalization Agent: 4.5/5.0 개인화 만족도 + 스토리텔링
    - Voice Synthesis Agent: 프로 성우 수준 음성 품질
    - PostgreSQL 체크포인팅으로 안정성 보장
    - Human-in-the-loop 검토 지원
    """
    
    def __init__(self):
        # 🎯 3개 전문 에이전트 초기화 (Stage 3)
        self.analysis_agent = AnalysisAgent()                    # 뉴스분석 + 팩트체킹 통합
        self.personalization_agent = PersonalizationAgent()     # 개인화 + 스토리텔링 통합  
        self.voice_synthesis_agent = VoiceSynthesisAgent()       # 다중 캐릭터 음성합성
        
        # PostgreSQL 체크포인터 설정
        self.checkpointer = PostgresCheckpointer(
            connection_string=settings.ai.postgres_checkpoint_url,
            serde_kwargs={"allow_dangerous_deserialization": True}
        )
        
        # Langfuse 추적
        self.langfuse = Langfuse(
            public_key=settings.ai.langfuse_public_key,
            secret_key=settings.ai.langfuse_secret_key,
            host=settings.ai.langfuse_host
        )
        
        # StateGraph 구성
        self.graph = self._build_graph()
        
        logger.info("🎯 NewsTalk AI Processing Graph initialized with 3 specialized agents (Stage 3)")
    
    def _build_graph(self) -> Pregel:
        """LangGraph StateGraph 구축"""
        
        # StateGraph 생성
        workflow = StateGraph(NewsState)
        
        # 🎯 노드 추가 (3개 전문 에이전트 + 품질 검증) - Stage 3
        workflow.add_node("news_analysis", self._analysis_node)              # 뉴스분석 + 팩트체킹
        workflow.add_node("personalization", self._personalization_node)    # 개인화 + 스토리텔링
        workflow.add_node("voice_synthesis", self._voice_synthesis_node)     # 음성합성
        workflow.add_node("quality_validation", self._quality_validation_node)
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("completion", self._completion_node)
        
        # 워크플로우 시작점
        workflow.set_entry_point("news_analysis")
        
        # 🎯 엣지 정의 (3개 에이전트 순차 실행) - Stage 3
        workflow.add_conditional_edges(
            "news_analysis",
            self._should_proceed_after_analysis,
            {
                "continue": "personalization",
                "quality_check": "quality_validation",
                "human_review": "human_review",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "personalization", 
            self._should_proceed_after_personalization,
            {
                "continue": "voice_synthesis",
                "quality_check": "quality_validation",
                "human_review": "human_review",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "voice_synthesis",
            self._should_proceed_after_voice_synthesis,
            {
                "continue": "completion",
                "quality_check": "quality_validation",
                "human_review": "human_review", 
                "end": END
            }
        )
        
        # 품질 검증 후 라우팅
        workflow.add_conditional_edges(
            "quality_validation",
            self._route_after_quality_check,
            {
                "retry_analysis": "news_analysis",
                "retry_personalization": "personalization",
                "retry_voice": "voice_synthesis",
                "human_review": "human_review",
                "continue": "completion",
                "end": END
            }
        )
        
        # 휴먼 리뷰 후 라우팅
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human_review,
            {
                "approved": "completion",
                "retry": "news_analysis",
                "end": END
            }
        )
        
        # 완료 → 종료
        workflow.add_edge("completion", END)
        
        # 컴파일 (체크포인터 및 인터럽트 설정)
        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review"],  # 휴먼 리뷰 전 중단
            interrupt_after=["quality_validation"]  # 품질 검증 후 중단 가능
        )
    
    # === 🎯 3개 전문 에이전트 실행 노드들 (Stage 3) ===
    
    async def _analysis_node(self, state: NewsState) -> NewsState:
        """🎯 뉴스 분석 + 팩트체킹 에이전트 실행 (95% 정확도 목표)"""
        try:
            logger.info(f"🔍 Starting comprehensive news analysis for article {state.article_id}")
            state = await self.analysis_agent.analyze_trends(state)
            state.add_metric("analysis_completed", True)
            state.add_metric("fact_check_completed", True)
            return state
        except Exception as e:
            logger.error(f"News analysis node failed: {str(e)}")
            state.add_error(f"News analysis failed: {str(e)}")
            return state
    
    async def _personalization_node(self, state: NewsState) -> NewsState:
        """🎯 개인화 + 스토리텔링 에이전트 실행 (4.5/5.0 만족도 목표)"""
        try:
            logger.info(f"👤 Starting personalization + storytelling for article {state.article_id}")
            state = await self.personalization_agent.personalize_content(state)
            state.add_metric("personalization_completed", True)
            state.add_metric("storytelling_completed", True)
            return state
        except Exception as e:
            logger.error(f"Personalization node failed: {str(e)}")
            state.add_error(f"Personalization failed: {str(e)}")
            return state
    
    async def _voice_synthesis_node(self, state: NewsState) -> NewsState:
        """🎯 음성 합성 에이전트 실행 (프로 성우 수준 품질, 다중 캐릭터)"""
        try:
            logger.info(f"🎵 Starting voice synthesis for article {state.article_id}")
            state = await self.voice_synthesis_agent.synthesize_voice(state)
            state.add_metric("voice_synthesis_completed", True)
            return state
        except Exception as e:
            logger.error(f"Voice synthesis node failed: {str(e)}")
            state.add_error(f"Voice synthesis failed: {str(e)}")
            return state
    
    # === 품질 검증 및 라우팅 노드들 ===
    
    async def _quality_validation_node(self, state: NewsState) -> NewsState:
        """품질 검증 노드"""
        try:
            logger.info(f"Quality validation for article {state.article_id}")
            
            quality_score = 0.0
            validation_issues = []
            
            # 분석 결과 품질 체크
            if state.analysis_result:
                if state.analysis_result.trend_score >= 0.7:
                    quality_score += 0.25
                else:
                    validation_issues.append("Low trend analysis score")
            
            # 팩트체킹 품질 체크 (95% 목표)
            if state.fact_check_result:
                if state.fact_check_result.credibility_score >= 0.95:
                    quality_score += 0.25
                else:
                    validation_issues.append("Fact check accuracy below 95%")
            
            # 스토리텔링 품질 체크
            if state.storytelling_result:
                if state.storytelling_result.engagement_score >= 4.2:
                    quality_score += 0.25
                else:
                    validation_issues.append("Low storytelling engagement")
            
            # 음성 합성 품질 체크
            if state.voice_synthesis_result:
                if state.voice_synthesis_result.synthesis_quality >= 0.8:
                    quality_score += 0.25
                else:
                    validation_issues.append("Low voice synthesis quality")
            
            state.add_metric("overall_quality_score", quality_score)
            state.add_metric("validation_issues", validation_issues)
            
            return state
            
        except Exception as e:
            logger.error(f"Quality validation failed: {str(e)}")
            state.add_error(f"Quality validation error: {str(e)}")
            return state
    
    async def _human_review_node(self, state: NewsState) -> NewsState:
        """휴먼 리뷰 노드"""
        logger.info(f"Human review required for article {state.article_id}")
        state.update_stage(ProcessingStage.HUMAN_REVIEW)
        # 실제 구현에서는 여기서 대기하고 외부 승인을 받음
        return state
    
    async def _completion_node(self, state: NewsState) -> NewsState:
        """완료 노드"""
        try:
            logger.info(f"Completing processing for article {state.article_id}")
            state.update_stage(ProcessingStage.COMPLETED)
            state.completed_at = datetime.utcnow()
            
            total_time = (state.completed_at - state.created_at).total_seconds()
            state.add_metric("total_processing_time", total_time)
            
            return state
        except Exception as e:
            logger.error(f"Completion node failed: {str(e)}")
            state.add_error(f"Completion error: {str(e)}")
            return state
    
    # === 조건부 라우팅 함수들 ===
    
    def _should_proceed_after_analysis(self, state: NewsState) -> Literal["continue", "quality_check", "human_review", "end"]:
        """🎯 뉴스 분석 + 팩트체킹 후 다음 단계 결정"""
        if state.errors:
            return "end"
        
        if not state.trend_analysis_result:
            return "quality_check"
        
        # 95% 팩트체킹 정확도 기준 검증
        if hasattr(state, 'fact_check_result') and state.fact_check_result:
            if state.fact_check_result.credibility_score < 0.95:
                logger.warning(f"Low credibility score: {state.fact_check_result.credibility_score}")
                return "human_review"
        
        if state.trend_analysis_result.trending_score < 0.5:
            return "human_review"
        
        return "continue"
    
    def _should_proceed_after_personalization(self, state: NewsState) -> Literal["continue", "quality_check", "human_review", "end"]:
        """🎯 개인화 + 스토리텔링 후 다음 단계 결정 (4.5/5.0 만족도 목표)"""
        if state.errors:
            return "end"
        
        if not state.personalization_result:
            return "quality_check"
        
        # 4.5/5.0 개인화 만족도 목표 검증
        if hasattr(state.personalization_result, 'personalization_score') and \
           hasattr(state.personalization_result.personalization_score, 'overall_score'):
            if state.personalization_result.personalization_score.overall_score < 4.5:
                logger.warning(f"Low personalization score: {state.personalization_result.personalization_score.overall_score}")
                return "quality_check"
        
        return "continue"
    
    def _should_proceed_after_voice_synthesis(self, state: NewsState) -> Literal["continue", "quality_check", "human_review", "end"]:
        """음성 합성 후 다음 단계 결정"""
        if state.errors:
            return "end"
        
        if not state.voice_synthesis_result:
            return "quality_check"
        
        if state.voice_synthesis_result.synthesis_quality < 0.7:
            return "human_review"
        
        return "continue"
    
    def _route_after_quality_check(self, state: NewsState) -> Literal["retry_analysis", "retry_personalization", "retry_voice", "human_review", "continue", "end"]:
        """🎯 품질 검증 후 라우팅 (3개 에이전트)"""
        quality_score = state.metrics.get("overall_quality_score", 0.0)
        validation_issues = state.metrics.get("validation_issues", [])
        
        if quality_score >= 0.8:
            return "continue"
        
        if quality_score < 0.3:
            return "end"
        
        # 🎯 3개 전문 에이전트 재실행 분기
        if "Low news analysis score" in validation_issues or "Fact check accuracy below 95%" in validation_issues:
            return "retry_analysis"
        elif "Low personalization score" in validation_issues or "Low storytelling engagement" in validation_issues:
            return "retry_personalization"
        elif "Low voice synthesis quality" in validation_issues:
            return "retry_voice"
        
        return "human_review"
    
    def _route_after_human_review(self, state: NewsState) -> Literal["approved", "retry", "end"]:
        """휴먼 리뷰 후 라우팅"""
        # 실제 구현에서는 외부 승인 상태를 확인
        # 여기서는 시뮬레이션
        human_decision = state.metrics.get("human_review_decision", "approved")
        
        if human_decision == "approved":
            return "approved"
        elif human_decision == "retry":
            return "retry"
        else:
            return "end"
    
    # === 메인 실행 인터페이스 ===
    
    async def process_news(self, article_id: str, content: str, title: str = "", 
                          user_id: Optional[str] = None, config: Optional[Dict] = None) -> NewsState:
        """
        🎯 뉴스 처리 메인 진입점 (Stage 3)
        3개 전문 에이전트 순차 실행: 뉴스분석(+팩트체킹) → 개인화(+스토리텔링) → 음성합성
        """
        try:
            # 처리 세션 ID 생성
            processing_id = str(uuid4())
            
            # 초기 상태 생성
            initial_state = NewsState(
                article_id=article_id,
                user_id=user_id,
                processing_id=processing_id,
                title=title,
                content=content,
                processing_stage=ProcessingStage.INITIATED
            )
            
            # Langfuse 추적 시작
            trace = self.langfuse.trace(
                name="newstalk_ai_workflow",
                input={
                    "article_id": article_id,
                    "user_id": user_id,
                    "content_length": len(content),
                    "title": title
                }
            )
            
            logger.info(f"Starting NewsTalk AI processing for article {article_id}")
            
            # 그래프 설정
            graph_config = {
                "configurable": {
                    "thread_id": processing_id,
                    "checkpoint_ns": f"newstalk_{article_id}",
                    **(config or {})
                }
            }
            
            # 그래프 실행
            final_state = None
            async for state in self.graph.astream(initial_state, config=graph_config):
                final_state = state
                logger.debug(f"Processing stage: {state.processing_stage}")
            
            # 추적 완료
            trace.update(
                output={
                    "processing_stage": final_state.processing_stage.value,
                    "completed": final_state.processing_stage == ProcessingStage.COMPLETED,
                    "total_time": final_state.metrics.get("total_processing_time", 0),
                    "quality_score": final_state.metrics.get("overall_quality_score", 0),
                    "errors": len(final_state.errors) if final_state.errors else 0
                }
            )
            
            logger.info(f"NewsTalk AI processing completed for article {article_id} - Stage: {final_state.processing_stage}")
            return final_state
            
        except Exception as e:
            logger.error(f"NewsTalk AI processing failed for article {article_id}: {str(e)}")
            # 에러 상태 반환
            error_state = NewsState(
                article_id=article_id,
                processing_id=processing_id,
                processing_stage=ProcessingStage.FAILED
            )
            error_state.add_error(f"Workflow error: {str(e)}")
            return error_state

    def get_processing_stats(self) -> Dict:
        """🎯 처리 통계 반환 (Stage 3)"""
        return {
            "agents_count": 3,
            "agents": [
                "AnalysisAgent (News Analysis + Fact Checking)",
                "PersonalizationAgent (Personalization + Storytelling)", 
                "VoiceSynthesisAgent (Multi-Character Voice)"
            ],
            "quality_targets": {
                "fact_check_accuracy": "95%",
                "personalization_satisfaction": "4.5/5.0",
                "storytelling_engagement": "4.2/5.0",
                "voice_synthesis_quality": "Pro-level"
            },
            "workflow": "News Analysis → Personalization → Voice Synthesis",
            "checkpoint_enabled": True,
            "human_review_supported": True,
            "stage": "Stage 3"
        } 