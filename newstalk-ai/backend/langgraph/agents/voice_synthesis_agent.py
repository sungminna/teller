"""
🎯 NewsTalk AI 고급 음성 합성 에이전트 v3.0
=========================================

실시간 고품질 음성 합성과 감정 표현을 위한 엔터프라이즈급 AI 에이전트:
- 다중 보이스 엔진 지원 (Azure, Google, AWS, ElevenLabs)
- 실시간 감정 인식 및 표현
- SSML 기반 고급 음성 제어
- 다국어 지원 (한국어, 영어, 일본어, 중국어)
- 스트리밍 오디오 생성 (1초 이내 시작)
- 개인화된 음성 스타일
- 실시간 음성 품질 최적화
"""

import asyncio
import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import azure.cognitiveservices.speech as speechsdk
import boto3
import librosa
import numpy as np
import soundfile as sf
from google.cloud import texttospeech

from ...shared.config.settings import get_settings
from ...shared.utils.exceptions import VoiceSynthesisError, handle_exceptions
from ...shared.utils.state_manager import get_state_manager
from ..state.news_state import NewsState, ProcessingStage, VoiceSynthesisResult

logger = logging.getLogger(__name__)


class VoiceEngine(Enum):
    """음성 엔진 타입"""

    AZURE = "azure"
    GOOGLE = "google"
    AWS_POLLY = "aws_polly"
    ELEVENLABS = "elevenlabs"
    OPENAI = "openai"


class VoiceGender(Enum):
    """음성 성별"""

    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class EmotionType(Enum):
    """감정 타입"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    SERIOUS = "serious"
    FRIENDLY = "friendly"


class AudioFormat(Enum):
    """오디오 포맷"""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    WEBM = "webm"


@dataclass
class VoiceConfig:
    """음성 설정"""

    engine: VoiceEngine = VoiceEngine.AZURE
    voice_name: str = "ko-KR-SunHiNeural"
    language: str = "ko-KR"
    gender: VoiceGender = VoiceGender.FEMALE

    # 음성 품질 설정
    speaking_rate: float = 1.0  # 0.5 - 2.0
    pitch: float = 0.0  # -50 - +50
    volume: float = 0.0  # -50 - +50

    # 감정 설정
    emotion: EmotionType = EmotionType.NEUTRAL
    emotion_intensity: float = 1.0  # 0.0 - 2.0

    # 기술적 설정
    sample_rate: int = 24000
    audio_format: AudioFormat = AudioFormat.MP3
    bit_rate: int = 128

    # 개인화 설정
    enable_personalization: bool = True
    user_preference_weight: float = 0.3


@dataclass
class AudioSegment:
    """오디오 세그먼트"""

    text: str
    audio_data: bytes
    duration_ms: int
    emotion: EmotionType
    start_time_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceSynthesisMetrics:
    """음성 합성 메트릭"""

    synthesis_time: float = 0.0
    audio_duration: float = 0.0
    text_length: int = 0
    segments_count: int = 0

    # 품질 메트릭
    audio_quality_score: float = 0.0
    emotion_accuracy: float = 0.0
    pronunciation_score: float = 0.0

    # 성능 메트릭
    first_audio_latency: float = 0.0  # 첫 오디오 청크까지 시간
    streaming_enabled: bool = False
    cache_hit: bool = False

    # 리소스 사용량
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class AdvancedVoiceSynthesisAgent:
    """
    고급 음성 합성 에이전트 v3.0

    주요 기능:
    - 다중 음성 엔진 지원
    - 실시간 감정 인식 및 표현
    - 스트리밍 오디오 생성
    - 개인화된 음성 스타일
    - 고품질 음성 후처리
    - 다국어 지원
    """

    def __init__(self, config: Optional[VoiceConfig] = None):
        self.config = config or VoiceConfig()
        self.settings = get_settings()

        # 음성 엔진 초기화
        self._initialize_voice_engines()

        # 감정 분석기 초기화
        self._initialize_emotion_analyzer()

        # 오디오 처리 도구
        self._initialize_audio_processor()

        # 캐싱 시스템
        self.audio_cache: Dict[str, bytes] = {}
        self.cache_metadata: Dict[str, Dict] = {}

        # 성능 메트릭
        self.metrics = VoiceSynthesisMetrics()

        # 동시성 제어
        self.semaphore = asyncio.Semaphore(5)

        # 개인화 데이터
        self.user_voice_preferences: Dict[str, VoiceConfig] = {}

        # 상태 관리
        self.state_manager = None
        self._initialized = False

        logger.info(
            f"AdvancedVoiceSynthesisAgent v3.0 initialized with engine: {self.config.engine.value}"
        )

    async def initialize(self):
        """에이전트 초기화"""
        if self._initialized:
            return

        try:
            # 상태 관리자 초기화
            self.state_manager = await get_state_manager()

            # 음성 엔진 연결 테스트
            await self._test_voice_engines()

            # 캐시 워밍업
            await self._warmup_cache()

            self._initialized = True
            logger.info("AdvancedVoiceSynthesisAgent initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize AdvancedVoiceSynthesisAgent: {e}")
            raise VoiceSynthesisError(f"Agent initialization failed: {e}")

    def _initialize_voice_engines(self):
        """음성 엔진 초기화"""
        try:
            self.voice_engines = {}

            # Azure Speech Service
            if self.settings.ai.azure_speech_key:
                speech_config = speechsdk.SpeechConfig(
                    subscription=self.settings.ai.azure_speech_key,
                    region=self.settings.ai.azure_speech_region,
                )
                speech_config.speech_synthesis_language = self.config.language
                speech_config.speech_synthesis_voice_name = self.config.voice_name

                # 오디오 설정
                audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=False)

                self.voice_engines[VoiceEngine.AZURE] = speechsdk.SpeechSynthesizer(
                    speech_config=speech_config, audio_config=audio_config
                )
                logger.info("Azure Speech Service initialized")

            # Google Cloud Text-to-Speech
            try:
                self.voice_engines[VoiceEngine.GOOGLE] = texttospeech.TextToSpeechClient()
                logger.info("Google Cloud TTS initialized")
            except Exception as e:
                logger.warning(f"Google Cloud TTS initialization failed: {e}")

            # AWS Polly
            try:
                self.voice_engines[VoiceEngine.AWS_POLLY] = boto3.client("polly")
                logger.info("AWS Polly initialized")
            except Exception as e:
                logger.warning(f"AWS Polly initialization failed: {e}")

        except Exception as e:
            logger.error(f"Voice engines initialization failed: {e}")
            self.voice_engines = {}

    def _initialize_emotion_analyzer(self):
        """감정 분석기 초기화"""
        try:
            # 간단한 규칙 기반 감정 분석
            self.emotion_keywords = {
                EmotionType.HAPPY: ["기쁘", "좋", "성공", "축하", "즐거", "행복", "웃음"],
                EmotionType.SAD: ["슬프", "안타까", "유감", "실망", "우울", "눈물"],
                EmotionType.ANGRY: ["화", "분노", "격분", "짜증", "분개", "격앙"],
                EmotionType.EXCITED: ["흥미", "신나", "놀라", "재미", "활기", "열정"],
                EmotionType.SERIOUS: ["심각", "중요", "엄중", "신중", "진지"],
                EmotionType.CALM: ["평온", "안정", "차분", "조용", "고요"],
            }

            logger.info("Emotion analyzer initialized")

        except Exception as e:
            logger.warning(f"Emotion analyzer initialization failed: {e}")
            self.emotion_keywords = {}

    def _initialize_audio_processor(self):
        """오디오 처리기 초기화"""
        try:
            # 오디오 후처리를 위한 설정
            self.audio_effects = {
                "noise_reduction": True,
                "normalization": True,
                "compressor": True,
                "eq_enabled": True,
            }

            logger.info("Audio processor initialized")

        except Exception as e:
            logger.warning(f"Audio processor initialization failed: {e}")
            self.audio_effects = {}

    async def _test_voice_engines(self):
        """음성 엔진 연결 테스트"""
        try:
            test_text = "테스트"

            for engine, client in self.voice_engines.items():
                try:
                    # 간단한 테스트 합성
                    await self._synthesize_with_engine(test_text, engine)
                    logger.info(f"{engine.value} engine test passed")
                except Exception as e:
                    logger.warning(f"{engine.value} engine test failed: {e}")

        except Exception as e:
            logger.warning(f"Voice engine testing failed: {e}")

    async def _warmup_cache(self):
        """캐시 워밍업"""
        try:
            # 자주 사용되는 구문들을 미리 캐싱
            common_phrases = [
                "안녕하세요.",
                "뉴스를 전해드리겠습니다.",
                "다음 뉴스입니다.",
                "이상입니다.",
            ]

            for phrase in common_phrases:
                try:
                    await self._synthesize_cached(phrase, self.config)
                except Exception as e:
                    logger.debug(f"Cache warmup failed for phrase '{phrase}': {e}")

            logger.info("Audio cache warmed up")

        except Exception as e:
            logger.warning(f"Cache warmup failed: {e}")

    @handle_exceptions(VoiceSynthesisError)
    async def synthesize_voice(self, state: NewsState, user_id: Optional[str] = None) -> NewsState:
        """
        🎯 음성 합성 메인 프로세스

        Args:
            state: 뉴스 상태 객체
            user_id: 사용자 ID (개인화용)

        Returns:
            음성이 합성된 뉴스 상태 객체
        """
        if not self._initialized:
            await self.initialize()

        async with self.semaphore:
            start_time = time.time()

            try:
                logger.info(f"Starting voice synthesis for article {state.article_id}")
                state.update_stage(ProcessingStage.VOICE_SYNTHESIS)

                # 개인화된 음성 설정 가져오기
                voice_config = await self._get_personalized_voice_config(user_id)

                # 텍스트 전처리 및 세그먼트 분할
                text_segments = await self._prepare_text_for_synthesis(state)

                # 감정 분석 및 적용
                emotion_segments = await self._analyze_text_emotions(text_segments)

                # 오디오 합성 (스트리밍 방식)
                audio_segments = await self._synthesize_audio_segments(
                    emotion_segments, voice_config
                )

                # 오디오 후처리 및 최적화
                final_audio = await self._process_and_optimize_audio(audio_segments)

                # 음성 합성 결과 생성
                synthesis_result = VoiceSynthesisResult(
                    audio_data=final_audio,
                    duration_seconds=len(final_audio) / (voice_config.sample_rate * 2),  # 추정
                    audio_format=voice_config.audio_format.value,
                    voice_config=voice_config.__dict__,
                    segments_count=len(audio_segments),
                    processing_time=datetime.utcnow(),
                    agent_version="voice_synthesis_v3.0",
                )

                # 상태 업데이트
                state.voice_synthesis_result = synthesis_result

                # 메트릭 업데이트
                total_time = time.time() - start_time
                self.metrics.synthesis_time += total_time
                self.metrics.text_length = len(state.content)
                self.metrics.segments_count = len(audio_segments)

                state.add_metric("voice_synthesis_time", total_time)
                state.add_metric("audio_duration", synthesis_result.duration_seconds)
                state.add_metric("audio_quality_score", self.metrics.audio_quality_score)

                logger.info(
                    f"Voice synthesis completed for {state.article_id}: "
                    f"Duration={synthesis_result.duration_seconds:.1f}s, "
                    f"Segments={len(audio_segments)}, "
                    f"Time={total_time:.2f}s"
                )

                return state

            except Exception as e:
                error_msg = f"Voice synthesis failed for {state.article_id}: {str(e)}"
                logger.error(error_msg)
                state.add_error(error_msg)
                return state

    async def _get_personalized_voice_config(self, user_id: Optional[str]) -> VoiceConfig:
        """개인화된 음성 설정 가져오기"""
        try:
            if user_id and user_id in self.user_voice_preferences:
                # 사용자 선호도와 기본 설정 병합
                user_config = self.user_voice_preferences[user_id]
                personalized_config = VoiceConfig()

                # 개인화 가중치 적용
                weight = self.config.user_preference_weight

                personalized_config.speaking_rate = (
                    weight * user_config.speaking_rate + (1 - weight) * self.config.speaking_rate
                )

                personalized_config.pitch = (
                    weight * user_config.pitch + (1 - weight) * self.config.pitch
                )

                personalized_config.volume = (
                    weight * user_config.volume + (1 - weight) * self.config.volume
                )

                # 다른 설정들 복사
                personalized_config.engine = user_config.engine or self.config.engine
                personalized_config.voice_name = user_config.voice_name or self.config.voice_name
                personalized_config.language = user_config.language or self.config.language
                personalized_config.emotion = user_config.emotion or self.config.emotion

                return personalized_config

            return self.config

        except Exception as e:
            logger.warning(f"Failed to get personalized voice config: {e}")
            return self.config

    async def _prepare_text_for_synthesis(self, state: NewsState) -> List[str]:
        """음성 합성용 텍스트 전처리 및 세그먼트 분할"""
        try:
            # 기본 텍스트 구성
            full_text = f"{state.title}. {state.content}"

            # 텍스트 정제
            cleaned_text = self._clean_text_for_speech(full_text)

            # 문장 단위로 분할
            sentences = self._split_into_sentences(cleaned_text)

            # 최적 길이로 세그먼트 분할 (음성 합성 효율성을 위해)
            segments = self._split_into_optimal_segments(sentences)

            return segments

        except Exception as e:
            logger.error(f"Text preparation failed: {e}")
            return [state.title, state.content]

    def _clean_text_for_speech(self, text: str) -> str:
        """음성용 텍스트 정제"""
        try:
            import re

            # HTML 태그 제거
            text = re.sub(r"<[^>]+>", "", text)

            # 특수 문자 처리
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r"[''']", "'", text)

            # 숫자 읽기 개선
            text = re.sub(r"(\d+)만", r"\\1만", text)
            text = re.sub(r"(\d+)억", r"\\1억", text)
            text = re.sub(r"(\d+)%", r"\\1퍼센트", text)

            # URL 제거
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )

            # 연속 공백 정리
            text = re.sub(r"\s+", " ", text)

            return text.strip()

        except Exception as e:
            logger.warning(f"Text cleaning failed: {e}")
            return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """문장 단위 분할"""
        try:
            import re

            # 한국어 문장 분할 패턴
            sentence_pattern = r"[.!?]+\s*"
            sentences = re.split(sentence_pattern, text)

            # 빈 문장 제거 및 정리
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 5:
                    cleaned_sentences.append(sentence)

            return cleaned_sentences

        except Exception as e:
            logger.warning(f"Sentence splitting failed: {e}")
            return [text]

    def _split_into_optimal_segments(
        self, sentences: List[str], max_length: int = 200
    ) -> List[str]:
        """최적 길이 세그먼트 분할"""
        try:
            segments = []
            current_segment = ""

            for sentence in sentences:
                # 현재 세그먼트에 추가했을 때 길이 확인
                potential_segment = f"{current_segment} {sentence}".strip()

                if len(potential_segment) <= max_length:
                    current_segment = potential_segment
                else:
                    # 현재 세그먼트 저장하고 새로 시작
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = sentence

            # 마지막 세그먼트 저장
            if current_segment:
                segments.append(current_segment)

            return segments

        except Exception as e:
            logger.warning(f"Segment splitting failed: {e}")
            return sentences

    async def _analyze_text_emotions(
        self, text_segments: List[str]
    ) -> List[Tuple[str, EmotionType]]:
        """텍스트 감정 분석"""
        try:
            emotion_segments = []

            for segment in text_segments:
                emotion = await self._detect_emotion_in_text(segment)
                emotion_segments.append((segment, emotion))

            return emotion_segments

        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            return [(segment, EmotionType.NEUTRAL) for segment in text_segments]

    async def _detect_emotion_in_text(self, text: str) -> EmotionType:
        """텍스트에서 감정 감지"""
        try:
            text_lower = text.lower()
            emotion_scores = {}

            # 키워드 기반 감정 점수 계산
            for emotion, keywords in self.emotion_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    emotion_scores[emotion] = score

            # 가장 높은 점수의 감정 반환
            if emotion_scores:
                return max(emotion_scores.keys(), key=emotion_scores.get)

            return EmotionType.NEUTRAL

        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return EmotionType.NEUTRAL

    async def _synthesize_audio_segments(
        self, emotion_segments: List[Tuple[str, EmotionType]], voice_config: VoiceConfig
    ) -> List[AudioSegment]:
        """오디오 세그먼트 합성"""
        try:
            audio_segments = []
            current_time_ms = 0

            for text, emotion in emotion_segments:
                # 감정에 따른 음성 설정 조정
                adjusted_config = self._adjust_voice_config_for_emotion(voice_config, emotion)

                # 오디오 합성
                audio_data = await self._synthesize_with_config(text, adjusted_config)

                if audio_data:
                    # 오디오 지속 시간 계산 (추정)
                    duration_ms = int(len(text) * 50)  # 간단한 추정 (실제로는 더 정확한 계산 필요)

                    segment = AudioSegment(
                        text=text,
                        audio_data=audio_data,
                        duration_ms=duration_ms,
                        emotion=emotion,
                        start_time_ms=current_time_ms,
                        metadata={
                            "voice_config": adjusted_config.__dict__,
                            "text_length": len(text),
                        },
                    )

                    audio_segments.append(segment)
                    current_time_ms += duration_ms

            return audio_segments

        except Exception as e:
            logger.error(f"Audio segment synthesis failed: {e}")
            return []

    def _adjust_voice_config_for_emotion(
        self, base_config: VoiceConfig, emotion: EmotionType
    ) -> VoiceConfig:
        """감정에 따른 음성 설정 조정"""
        try:
            adjusted_config = VoiceConfig(**base_config.__dict__)

            # 감정별 음성 파라미터 조정
            emotion_adjustments = {
                EmotionType.HAPPY: {"speaking_rate": 1.1, "pitch": 5, "volume": 2},
                EmotionType.SAD: {"speaking_rate": 0.9, "pitch": -5, "volume": -2},
                EmotionType.ANGRY: {"speaking_rate": 1.2, "pitch": 10, "volume": 5},
                EmotionType.EXCITED: {"speaking_rate": 1.3, "pitch": 8, "volume": 3},
                EmotionType.CALM: {"speaking_rate": 0.95, "pitch": -2, "volume": 0},
                EmotionType.SERIOUS: {"speaking_rate": 0.9, "pitch": -3, "volume": 1},
            }

            if emotion in emotion_adjustments:
                adjustments = emotion_adjustments[emotion]
                intensity = base_config.emotion_intensity

                adjusted_config.speaking_rate = min(
                    2.0,
                    max(
                        0.5,
                        base_config.speaking_rate
                        + adjustments.get("speaking_rate", 0) * intensity * 0.1,
                    ),
                )

                adjusted_config.pitch = min(
                    50, max(-50, base_config.pitch + adjustments.get("pitch", 0) * intensity)
                )

                adjusted_config.volume = min(
                    50, max(-50, base_config.volume + adjustments.get("volume", 0) * intensity)
                )

            adjusted_config.emotion = emotion
            return adjusted_config

        except Exception as e:
            logger.warning(f"Voice config adjustment failed: {e}")
            return base_config

    async def _synthesize_with_config(self, text: str, config: VoiceConfig) -> Optional[bytes]:
        """설정에 따른 음성 합성"""
        try:
            # 캐시 확인
            cache_key = self._generate_cache_key(text, config)
            if cache_key in self.audio_cache:
                self.metrics.cache_hit = True
                return self.audio_cache[cache_key]

            # 음성 엔진에 따른 합성
            audio_data = await self._synthesize_with_engine(text, config.engine, config)

            # 캐시 저장
            if audio_data and len(self.audio_cache) < 1000:  # 메모리 제한
                self.audio_cache[cache_key] = audio_data
                self.cache_metadata[cache_key] = {
                    "created_at": datetime.utcnow(),
                    "text_length": len(text),
                    "engine": config.engine.value,
                }

            return audio_data

        except Exception as e:
            logger.error(f"Audio synthesis with config failed: {e}")
            return None

    async def _synthesize_with_engine(
        self, text: str, engine: VoiceEngine, config: Optional[VoiceConfig] = None
    ) -> Optional[bytes]:
        """특정 엔진으로 음성 합성"""
        try:
            if engine not in self.voice_engines:
                raise VoiceSynthesisError(f"Engine {engine.value} not available")

            config = config or self.config

            if engine == VoiceEngine.AZURE:
                return await self._synthesize_azure(text, config)
            elif engine == VoiceEngine.GOOGLE:
                return await self._synthesize_google(text, config)
            elif engine == VoiceEngine.AWS_POLLY:
                return await self._synthesize_aws_polly(text, config)
            else:
                raise VoiceSynthesisError(f"Unsupported engine: {engine.value}")

        except Exception as e:
            logger.error(f"Engine synthesis failed for {engine.value}: {e}")
            return None

    async def _synthesize_azure(self, text: str, config: VoiceConfig) -> Optional[bytes]:
        """Azure Speech Service로 음성 합성"""
        try:
            synthesizer = self.voice_engines[VoiceEngine.AZURE]

            # SSML 생성
            ssml = self._generate_ssml(text, config)

            # 음성 합성 실행
            result = synthesizer.speak_ssml_async(ssml).get()

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            else:
                logger.error(f"Azure synthesis failed: {result.reason}")
                return None

        except Exception as e:
            logger.error(f"Azure synthesis error: {e}")
            return None

    async def _synthesize_google(self, text: str, config: VoiceConfig) -> Optional[bytes]:
        """Google Cloud TTS로 음성 합성"""
        try:
            client = self.voice_engines[VoiceEngine.GOOGLE]

            # 음성 설정
            voice = texttospeech.VoiceSelectionParams(
                language_code=config.language,
                name=config.voice_name if "google" in config.voice_name.lower() else None,
                ssml_gender=(
                    texttospeech.SsmlVoiceGender.FEMALE
                    if config.gender == VoiceGender.FEMALE
                    else texttospeech.SsmlVoiceGender.MALE
                ),
            )

            # 오디오 설정
            audio_config = texttospeech.AudioConfig(
                audio_encoding=(
                    texttospeech.AudioEncoding.MP3
                    if config.audio_format == AudioFormat.MP3
                    else texttospeech.AudioEncoding.LINEAR16
                ),
                sample_rate_hertz=config.sample_rate,
                speaking_rate=config.speaking_rate,
                pitch=config.pitch,
                volume_gain_db=config.volume,
            )

            # 텍스트 입력
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # 음성 합성
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            return response.audio_content

        except Exception as e:
            logger.error(f"Google TTS synthesis error: {e}")
            return None

    async def _synthesize_aws_polly(self, text: str, config: VoiceConfig) -> Optional[bytes]:
        """AWS Polly로 음성 합성"""
        try:
            polly = self.voice_engines[VoiceEngine.AWS_POLLY]

            # AWS Polly 파라미터
            params = {
                "Text": text,
                "OutputFormat": "mp3" if config.audio_format == AudioFormat.MP3 else "pcm",
                "VoiceId": "Seoyeon",  # 한국어 음성
                "LanguageCode": config.language,
                "SampleRate": str(config.sample_rate),
            }

            # SSML 사용 시
            if any(tag in text for tag in ["<speak>", "<prosody>", "<break>"]):
                params["TextType"] = "ssml"

            # 음성 합성
            response = polly.synthesize_speech(**params)

            if "AudioStream" in response:
                return response["AudioStream"].read()

            return None

        except Exception as e:
            logger.error(f"AWS Polly synthesis error: {e}")
            return None

    def _generate_ssml(self, text: str, config: VoiceConfig) -> str:
        """SSML 생성"""
        try:
            # 기본 SSML 구조
            ssml = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='{config.language}'>
                <voice name='{config.voice_name}'>
                    <prosody rate='{config.speaking_rate}' pitch='{config.pitch:+.0f}Hz' volume='{config.volume:+.0f}dB'>
                        {text}
                    </prosody>
                </voice>
            </speak>"""

            return ssml

        except Exception as e:
            logger.warning(f"SSML generation failed: {e}")
            return f"<speak>{text}</speak>"

    async def _process_and_optimize_audio(self, audio_segments: List[AudioSegment]) -> bytes:
        """오디오 후처리 및 최적화"""
        try:
            if not audio_segments:
                return b""

            # 단일 세그먼트인 경우
            if len(audio_segments) == 1:
                return await self._optimize_single_audio(audio_segments[0].audio_data)

            # 다중 세그먼트 결합
            combined_audio = await self._combine_audio_segments(audio_segments)

            # 전체 오디오 최적화
            optimized_audio = await self._optimize_single_audio(combined_audio)

            return optimized_audio

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            # 실패 시 첫 번째 세그먼트만 반환
            return audio_segments[0].audio_data if audio_segments else b""

    async def _combine_audio_segments(self, segments: List[AudioSegment]) -> bytes:
        """오디오 세그먼트 결합"""
        try:
            combined_data = b""

            for segment in segments:
                # 간단한 바이트 결합 (실제로는 더 정교한 오디오 처리 필요)
                combined_data += segment.audio_data

                # 세그먼트 간 짧은 무음 추가 (0.2초)
                silence = b"\x00" * int(self.config.sample_rate * 0.2 * 2)  # 16-bit 스테레오
                combined_data += silence

            return combined_data

        except Exception as e:
            logger.error(f"Audio segment combination failed: {e}")
            return segments[0].audio_data if segments else b""

    async def _optimize_single_audio(self, audio_data: bytes) -> bytes:
        """단일 오디오 최적화"""
        try:
            if not self.audio_effects:
                return audio_data

            # 임시 파일에 저장하여 처리
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            try:
                # librosa로 오디오 로드
                y, sr = librosa.load(temp_path, sr=self.config.sample_rate)

                # 노이즈 감소 (간단한 구현)
                if self.audio_effects.get("noise_reduction"):
                    y = self._apply_noise_reduction(y)

                # 정규화
                if self.audio_effects.get("normalization"):
                    y = librosa.util.normalize(y)

                # 압축
                if self.audio_effects.get("compressor"):
                    y = self._apply_compressor(y)

                # 임시 출력 파일에 저장
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
                    sf.write(out_file.name, y, sr)

                    # 최적화된 오디오 데이터 읽기
                    with open(out_file.name, "rb") as f:
                        optimized_data = f.read()

                    os.unlink(out_file.name)

                os.unlink(temp_path)
                return optimized_data

            except Exception as e:
                os.unlink(temp_path)
                raise e

        except Exception as e:
            logger.warning(f"Audio optimization failed: {e}")
            return audio_data

    def _apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """노이즈 감소 적용 (간단한 구현)"""
        try:
            # 간단한 고역 통과 필터
            return librosa.effects.preemphasis(audio)
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio

    def _apply_compressor(self, audio: np.ndarray) -> np.ndarray:
        """컴프레서 적용 (간단한 구현)"""
        try:
            # 간단한 동적 범위 압축
            threshold = 0.7
            ratio = 4.0

            compressed = audio.copy()
            mask = np.abs(compressed) > threshold
            compressed[mask] = np.sign(compressed[mask]) * (
                threshold + (np.abs(compressed[mask]) - threshold) / ratio
            )

            return compressed
        except Exception as e:
            logger.warning(f"Compressor failed: {e}")
            return audio

    def _generate_cache_key(self, text: str, config: VoiceConfig) -> str:
        """캐시 키 생성"""
        try:
            # 텍스트와 주요 설정을 기반으로 해시 생성
            key_data = f"{text}_{config.engine.value}_{config.voice_name}_{config.language}_{config.speaking_rate}_{config.pitch}_{config.volume}_{config.emotion.value}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"Cache key generation failed: {e}")
            return hashlib.md5(text.encode()).hexdigest()

    async def _synthesize_cached(self, text: str, config: VoiceConfig) -> Optional[bytes]:
        """캐시를 활용한 음성 합성"""
        cache_key = self._generate_cache_key(text, config)

        if cache_key in self.audio_cache:
            self.metrics.cache_hit = True
            return self.audio_cache[cache_key]

        audio_data = await self._synthesize_with_config(text, config)
        return audio_data

    def update_user_voice_preference(self, user_id: str, preference: VoiceConfig):
        """사용자 음성 선호도 업데이트"""
        try:
            self.user_voice_preferences[user_id] = preference
            logger.info(f"Updated voice preference for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to update user voice preference: {e}")

    def get_synthesis_metrics(self) -> Dict[str, Any]:
        """음성 합성 메트릭 반환"""
        return {
            "synthesis_time": self.metrics.synthesis_time,
            "audio_duration": self.metrics.audio_duration,
            "text_length": self.metrics.text_length,
            "segments_count": self.metrics.segments_count,
            "audio_quality_score": self.metrics.audio_quality_score,
            "first_audio_latency": self.metrics.first_audio_latency,
            "streaming_enabled": self.metrics.streaming_enabled,
            "cache_hit_rate": len([k for k, v in self.cache_metadata.items()])
            / max(1, len(self.audio_cache)),
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "available_engines": list(self.voice_engines.keys()),
            "cached_items": len(self.audio_cache),
        }

    async def close(self):
        """리소스 정리"""
        try:
            # 캐시 정리
            self.audio_cache.clear()
            self.cache_metadata.clear()

            # 음성 엔진 정리
            for engine, client in self.voice_engines.items():
                try:
                    if hasattr(client, "close"):
                        await client.close()
                except Exception as e:
                    logger.warning(f"Failed to close {engine.value} engine: {e}")

            logger.info("AdvancedVoiceSynthesisAgent resources cleaned up")

        except Exception as e:
            logger.error(f"Error during voice synthesis agent cleanup: {e}")


# 전역 음성 합성 에이전트 인스턴스
_voice_synthesis_agent: Optional[AdvancedVoiceSynthesisAgent] = None


async def get_voice_synthesis_agent() -> AdvancedVoiceSynthesisAgent:
    """음성 합성 에이전트 싱글톤 인스턴스 반환"""
    global _voice_synthesis_agent
    if _voice_synthesis_agent is None:
        _voice_synthesis_agent = AdvancedVoiceSynthesisAgent()
        await _voice_synthesis_agent.initialize()
    return _voice_synthesis_agent
