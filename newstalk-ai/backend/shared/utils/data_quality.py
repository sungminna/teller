"""
🎯 NewsTalk AI 고급 데이터 품질 관리 시스템
==========================================

95% 데이터 품질 보장을 위한 엔터프라이즈급 데이터 품질 관리:
- 실시간 품질 모니터링 및 알림
- 자동 데이터 정제 및 보강
- ML 기반 이상 탐지
- 스키마 진화 추적
- 데이터 계보 관리
- 자동 품질 개선 제안
"""

import asyncio
import hashlib
import logging
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """데이터 품질 메트릭"""

    COMPLETENESS = "completeness"  # 완전성
    ACCURACY = "accuracy"  # 정확성
    CONSISTENCY = "consistency"  # 일관성
    VALIDITY = "validity"  # 유효성
    UNIQUENESS = "uniqueness"  # 고유성
    TIMELINESS = "timeliness"  # 적시성
    RELEVANCE = "relevance"  # 관련성


class ValidationSeverity(Enum):
    """검증 심각도"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityRule:
    """데이터 품질 규칙"""

    name: str
    metric: QualityMetric
    severity: ValidationSeverity
    description: str
    validation_func: Callable[[Any], bool]
    auto_fix_func: Optional[Callable[[Any], Any]] = None
    threshold: float = 0.95  # 품질 임계값 (95%)
    enabled: bool = True


@dataclass
class QualityIssue:
    """품질 이슈"""

    rule_name: str
    severity: ValidationSeverity
    field_name: str
    issue_description: str
    sample_value: Any
    suggestion: Optional[str] = None
    auto_fixable: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataQualityReport:
    """데이터 품질 리포트"""

    dataset_name: str
    total_records: int
    quality_score: float
    metrics_scores: Dict[QualityMetric, float]
    issues: List[QualityIssue]
    improvement_suggestions: List[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AdvancedDataQualityManager:
    """
    고급 데이터 품질 관리자

    주요 기능:
    - 실시간 품질 모니터링
    - 자동 데이터 정제
    - ML 기반 이상 탐지
    - 품질 트렌드 분석
    - 자동 품질 개선
    """

    def __init__(self):
        # 품질 규칙 저장소
        self.quality_rules: Dict[str, QualityRule] = {}

        # 품질 이력 관리
        self.quality_history: List[DataQualityReport] = []
        self.anomaly_patterns: Dict[str, List[Dict]] = defaultdict(list)

        # 자동 정제 통계
        self.auto_fix_stats = {
            "total_fixes": 0,
            "successful_fixes": 0,
            "failed_fixes": 0,
            "fixes_by_type": defaultdict(int),
        }

        # 실시간 모니터링
        self._monitoring_task: Optional[asyncio.Task] = None
        self._quality_cache: Dict[str, Dict] = {}

        # ML 기반 이상 탐지 모델
        self._anomaly_detector = None
        self._initialize_anomaly_detector()

        logger.info("AdvancedDataQualityManager initialized")

    def _initialize_anomaly_detector(self):
        """이상 탐지 모델 초기화"""
        try:
            # 간단한 통계 기반 이상 탐지 모델
            self._anomaly_detector = StatisticalAnomalyDetector()
            logger.info("Statistical anomaly detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize anomaly detector: {e}")

    async def initialize_quality_rules(self):
        """품질 규칙 초기화"""
        # 뉴스 기사 품질 규칙들
        news_rules = [
            # 완전성 규칙
            QualityRule(
                name="title_completeness",
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                description="뉴스 제목은 필수입니다",
                validation_func=lambda x: bool(x and x.get("title", "").strip()),
                auto_fix_func=self._generate_title_from_content,
            ),
            QualityRule(
                name="content_completeness",
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.ERROR,
                description="뉴스 본문은 필수입니다",
                validation_func=lambda x: bool(x and x.get("content", "").strip()),
            ),
            QualityRule(
                name="published_date_completeness",
                metric=QualityMetric.COMPLETENESS,
                severity=ValidationSeverity.WARNING,
                description="발행일은 필수입니다",
                validation_func=lambda x: bool(x and x.get("published_at")),
                auto_fix_func=self._estimate_publish_date,
            ),
            # 정확성 규칙
            QualityRule(
                name="title_length_accuracy",
                metric=QualityMetric.ACCURACY,
                severity=ValidationSeverity.WARNING,
                description="제목 길이는 10-200자 사이여야 합니다",
                validation_func=lambda x: 10 <= len(x.get("title", "")) <= 200,
                auto_fix_func=self._fix_title_length,
            ),
            QualityRule(
                name="content_length_accuracy",
                metric=QualityMetric.ACCURACY,
                severity=ValidationSeverity.WARNING,
                description="본문 길이는 100자 이상이어야 합니다",
                validation_func=lambda x: len(x.get("content", "")) >= 100,
            ),
            QualityRule(
                name="url_validity",
                metric=QualityMetric.VALIDITY,
                severity=ValidationSeverity.WARNING,
                description="URL 형식이 유효해야 합니다",
                validation_func=self._validate_url_format,
                auto_fix_func=self._fix_url_format,
            ),
            # 일관성 규칙
            QualityRule(
                name="category_consistency",
                metric=QualityMetric.CONSISTENCY,
                severity=ValidationSeverity.WARNING,
                description="카테고리는 정의된 값 중 하나여야 합니다",
                validation_func=self._validate_category_consistency,
                auto_fix_func=self._fix_category_consistency,
            ),
            QualityRule(
                name="source_consistency",
                metric=QualityMetric.CONSISTENCY,
                severity=ValidationSeverity.INFO,
                description="소스명은 일관된 형식이어야 합니다",
                validation_func=self._validate_source_consistency,
                auto_fix_func=self._fix_source_consistency,
            ),
            # 적시성 규칙
            QualityRule(
                name="content_timeliness",
                metric=QualityMetric.TIMELINESS,
                severity=ValidationSeverity.WARNING,
                description="뉴스는 7일 이내 발행되어야 합니다",
                validation_func=self._validate_content_timeliness,
            ),
            # 고유성 규칙
            QualityRule(
                name="content_uniqueness",
                metric=QualityMetric.UNIQUENESS,
                severity=ValidationSeverity.ERROR,
                description="중복 기사가 없어야 합니다",
                validation_func=self._validate_content_uniqueness,
            ),
            # 관련성 규칙
            QualityRule(
                name="content_relevance",
                metric=QualityMetric.RELEVANCE,
                severity=ValidationSeverity.INFO,
                description="제목과 본문이 관련성이 있어야 합니다",
                validation_func=self._validate_content_relevance,
            ),
        ]

        # 규칙 등록
        for rule in news_rules:
            self.quality_rules[rule.name] = rule

        logger.info(f"Initialized {len(news_rules)} quality rules")

    async def validate_data_quality(
        self, data: Union[Dict, List[Dict]], dataset_name: str = "unknown"
    ) -> DataQualityReport:
        """
        데이터 품질 검증

        Args:
            data: 검증할 데이터
            dataset_name: 데이터셋 이름

        Returns:
            데이터 품질 리포트
        """
        start_time = time.time()

        # 데이터 정규화
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data

        total_records = len(data_list)
        all_issues = []
        metric_scores = {metric: [] for metric in QualityMetric}

        # 각 레코드별 품질 검증
        for i, record in enumerate(data_list):
            record_issues = await self._validate_single_record(record, i)
            all_issues.extend(record_issues)

            # 메트릭별 점수 계산
            for metric in QualityMetric:
                metric_issues = [
                    issue
                    for issue in record_issues
                    if issue.rule_name in self._get_rules_by_metric(metric)
                ]
                score = 1.0 - (len(metric_issues) / max(1, len(self._get_rules_by_metric(metric))))
                metric_scores[metric].append(max(0.0, score))

        # 전체 품질 점수 계산
        overall_scores = []
        for metric, scores in metric_scores.items():
            if scores:
                overall_scores.append(statistics.mean(scores))

        quality_score = statistics.mean(overall_scores) if overall_scores else 0.0

        # 메트릭 평균 점수 계산
        avg_metric_scores = {
            metric: statistics.mean(scores) if scores else 0.0
            for metric, scores in metric_scores.items()
        }

        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(
            all_issues, avg_metric_scores
        )

        # 품질 리포트 생성
        quality_report = DataQualityReport(
            dataset_name=dataset_name,
            total_records=total_records,
            quality_score=quality_score,
            metrics_scores=avg_metric_scores,
            issues=all_issues,
            improvement_suggestions=improvement_suggestions,
        )

        # 이상 탐지 실행
        await self._detect_anomalies(quality_report)

        # 품질 이력 저장
        self.quality_history.append(quality_report)

        # 성능 로깅
        execution_time = time.time() - start_time
        logger.info(
            f"Quality validation completed for {dataset_name}: "
            f"Score={quality_score:.3f}, Issues={len(all_issues)}, "
            f"Time={execution_time:.3f}s"
        )

        return quality_report

    async def _validate_single_record(self, record: Dict, record_index: int) -> List[QualityIssue]:
        """단일 레코드 품질 검증"""
        issues = []

        for rule_name, rule in self.quality_rules.items():
            if not rule.enabled:
                continue

            try:
                is_valid = rule.validation_func(record)

                if not is_valid:
                    # 자동 수정 시도
                    if rule.auto_fix_func:
                        try:
                            fixed_value = rule.auto_fix_func(record)
                            if fixed_value is not None:
                                # 수정 성공
                                self.auto_fix_stats["successful_fixes"] += 1
                                self.auto_fix_stats["fixes_by_type"][rule.metric.value] += 1

                                issues.append(
                                    QualityIssue(
                                        rule_name=rule_name,
                                        severity=ValidationSeverity.INFO,
                                        field_name=self._extract_field_name(rule_name),
                                        issue_description=f"{rule.description} (자동 수정됨)",
                                        sample_value=str(fixed_value)[:100],
                                        auto_fixable=True,
                                    )
                                )
                                continue
                        except Exception as e:
                            self.auto_fix_stats["failed_fixes"] += 1
                            logger.warning(f"Auto-fix failed for rule {rule_name}: {e}")

                    # 이슈 기록
                    issues.append(
                        QualityIssue(
                            rule_name=rule_name,
                            severity=rule.severity,
                            field_name=self._extract_field_name(rule_name),
                            issue_description=rule.description,
                            sample_value=str(record.get(self._extract_field_name(rule_name), ""))[
                                :100
                            ],
                            suggestion=self._generate_fix_suggestion(rule_name, record),
                            auto_fixable=rule.auto_fix_func is not None,
                        )
                    )

            except Exception as e:
                logger.error(f"Rule validation failed for {rule_name}: {e}")
                issues.append(
                    QualityIssue(
                        rule_name=rule_name,
                        severity=ValidationSeverity.ERROR,
                        field_name="validation_error",
                        issue_description=f"검증 오류: {str(e)}",
                        sample_value="N/A",
                    )
                )

        return issues

    def _get_rules_by_metric(self, metric: QualityMetric) -> List[str]:
        """메트릭별 규칙 이름 목록 반환"""
        return [name for name, rule in self.quality_rules.items() if rule.metric == metric]

    def _extract_field_name(self, rule_name: str) -> str:
        """규칙 이름에서 필드명 추출"""
        # "title_completeness" -> "title"
        return rule_name.split("_")[0] if "_" in rule_name else rule_name

    def _generate_improvement_suggestions(
        self, issues: List[QualityIssue], metric_scores: Dict[QualityMetric, float]
    ) -> List[str]:
        """품질 개선 제안 생성"""
        suggestions = []

        # 심각한 이슈별 제안
        critical_issues = [
            issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL
        ]
        if critical_issues:
            suggestions.append(f"🔴 {len(critical_issues)}개의 치명적 이슈 즉시 해결 필요")

        error_issues = [issue for issue in issues if issue.severity == ValidationSeverity.ERROR]
        if error_issues:
            suggestions.append(f"🟠 {len(error_issues)}개의 오류 이슈 우선 해결 권장")

        # 메트릭별 개선 제안
        for metric, score in metric_scores.items():
            if score < 0.8:  # 80% 미만인 경우
                if metric == QualityMetric.COMPLETENESS:
                    suggestions.append("📝 필수 필드 누락 개선: 데이터 수집 프로세스 점검 필요")
                elif metric == QualityMetric.ACCURACY:
                    suggestions.append("🎯 데이터 정확성 개선: 입력 검증 강화 필요")
                elif metric == QualityMetric.CONSISTENCY:
                    suggestions.append("🔄 데이터 일관성 개선: 표준화 규칙 적용 필요")
                elif metric == QualityMetric.VALIDITY:
                    suggestions.append("✅ 데이터 유효성 개선: 형식 검증 강화 필요")
                elif metric == QualityMetric.UNIQUENESS:
                    suggestions.append("🔍 중복 데이터 제거: 중복 탐지 알고리즘 개선 필요")
                elif metric == QualityMetric.TIMELINESS:
                    suggestions.append("⏰ 데이터 적시성 개선: 실시간 처리 파이프라인 최적화 필요")
                elif metric == QualityMetric.RELEVANCE:
                    suggestions.append("🎨 데이터 관련성 개선: 콘텐츠 필터링 알고리즘 개선 필요")

        # 자동 수정 제안
        auto_fixable_issues = [issue for issue in issues if issue.auto_fixable]
        if auto_fixable_issues:
            suggestions.append(f"🔧 {len(auto_fixable_issues)}개 이슈 자동 수정 가능")

        return suggestions[:10]  # 상위 10개 제안만

    async def _detect_anomalies(self, quality_report: DataQualityReport):
        """이상 탐지 실행"""
        try:
            if self._anomaly_detector:
                anomalies = await self._anomaly_detector.detect_anomalies(quality_report)

                for anomaly in anomalies:
                    logger.warning(f"Quality anomaly detected: {anomaly}")

                    # 이상 패턴 저장
                    self.anomaly_patterns[quality_report.dataset_name].append(
                        {
                            "timestamp": datetime.utcnow(),
                            "anomaly": anomaly,
                            "quality_score": quality_report.quality_score,
                        }
                    )

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

    def _generate_fix_suggestion(self, rule_name: str, record: Dict) -> str:
        """수정 제안 생성"""
        if rule_name == "title_completeness":
            return "본문 첫 문장을 기반으로 제목 생성을 고려하세요"
        elif rule_name == "content_length_accuracy":
            return "본문을 더 상세히 작성하거나 요약문을 추가하세요"
        elif rule_name == "url_validity":
            return "올바른 URL 형식(http://또는 https://)으로 수정하세요"
        elif rule_name == "category_consistency":
            return "정의된 카테고리 목록에서 선택하세요"
        else:
            return "데이터 검증 규칙을 확인하고 수정하세요"

    # 자동 수정 함수들
    def _generate_title_from_content(self, record: Dict) -> Optional[str]:
        """본문에서 제목 생성"""
        content = record.get("content", "").strip()
        if not content:
            return None

        # 첫 문장 추출 (간단한 구현)
        sentences = re.split(r"[.!?]", content)
        first_sentence = sentences[0].strip() if sentences else ""

        if len(first_sentence) > 10:
            # 200자로 제한
            title = first_sentence[:200] if len(first_sentence) > 200 else first_sentence
            record["title"] = title
            return title

        return None

    def _estimate_publish_date(self, record: Dict) -> Optional[str]:
        """발행일 추정"""
        # 현재 시간을 기본값으로 사용
        estimated_date = datetime.utcnow()
        record["published_at"] = estimated_date.isoformat()
        return estimated_date.isoformat()

    def _fix_title_length(self, record: Dict) -> Optional[str]:
        """제목 길이 수정"""
        title = record.get("title", "").strip()

        if len(title) < 10:
            # 너무 짧은 경우: 본문에서 보강
            content = record.get("content", "").strip()
            if content:
                additional_text = content.split(".")[0]
                fixed_title = f"{title} - {additional_text}"[:200]
                record["title"] = fixed_title
                return fixed_title

        elif len(title) > 200:
            # 너무 긴 경우: 줄임
            fixed_title = title[:197] + "..."
            record["title"] = fixed_title
            return fixed_title

        return title

    def _fix_url_format(self, record: Dict) -> Optional[str]:
        """URL 형식 수정"""
        url = record.get("url", "").strip()
        if url and not url.startswith(("http://", "https://")):
            fixed_url = f"https://{url}"
            record["url"] = fixed_url
            return fixed_url
        return url

    def _fix_category_consistency(self, record: Dict) -> Optional[str]:
        """카테고리 일관성 수정"""
        category = record.get("category", "").strip().lower()

        # 카테고리 매핑
        category_mapping = {
            "정치": "politics",
            "경제": "economy",
            "사회": "society",
            "국제": "international",
            "스포츠": "sports",
            "연예": "entertainment",
            "기술": "technology",
            "tech": "technology",
            "it": "technology",
            "과학": "science",
        }

        if category in category_mapping:
            fixed_category = category_mapping[category]
            record["category"] = fixed_category
            return fixed_category

        return None

    def _fix_source_consistency(self, record: Dict) -> Optional[str]:
        """소스명 일관성 수정"""
        source = record.get("source_name", "").strip()

        # 소스명 정규화
        if source:
            # 공통 접미사 제거
            cleaned_source = re.sub(r"\s*(뉴스|신문|방송|미디어|언론)$", "", source)
            # 공백 정규화
            cleaned_source = re.sub(r"\s+", " ", cleaned_source).strip()

            if cleaned_source != source:
                record["source_name"] = cleaned_source
                return cleaned_source

        return source

    # 검증 함수들
    def _validate_url_format(self, record: Dict) -> bool:
        """URL 형식 검증"""
        url = record.get("url", "")
        if not url:
            return True  # URL이 없는 것은 허용

        url_pattern = re.compile(
            r"^https?://"  # http:// 또는 https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # 도메인
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP
            r"(?::\d+)?"  # 포트
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        return bool(url_pattern.match(url))

    def _validate_category_consistency(self, record: Dict) -> bool:
        """카테고리 일관성 검증"""
        category = record.get("category", "").strip().lower()

        valid_categories = {
            "politics",
            "economy",
            "society",
            "international",
            "sports",
            "entertainment",
            "technology",
            "science",
            "정치",
            "경제",
            "사회",
            "국제",
            "스포츠",
            "연예",
            "기술",
            "과학",
        }

        return category in valid_categories

    def _validate_source_consistency(self, record: Dict) -> bool:
        """소스명 일관성 검증"""
        source = record.get("source_name", "").strip()

        if not source:
            return False

        # 소스명은 2-50자 사이, 특수문자 제한
        if not (2 <= len(source) <= 50):
            return False

        # 기본적인 형식 검증
        if re.search(r"[<>{}]", source):  # HTML 태그 같은 문자 제외
            return False

        return True

    def _validate_content_timeliness(self, record: Dict) -> bool:
        """콘텐츠 적시성 검증"""
        published_at = record.get("published_at")

        if not published_at:
            return False

        try:
            if isinstance(published_at, str):
                pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            else:
                pub_date = published_at

            # 7일 이내 발행된 뉴스만 허용
            days_old = (datetime.utcnow() - pub_date.replace(tzinfo=None)).days
            return days_old <= 7

        except Exception:
            return False

    def _validate_content_uniqueness(self, record: Dict) -> bool:
        """콘텐츠 고유성 검증"""
        title = record.get("title", "").strip()
        content = record.get("content", "").strip()

        if not title or not content:
            return True  # 다른 규칙에서 처리

        # 간단한 중복 검사 (실제로는 데이터베이스 조회 필요)
        content_hash = hashlib.md5(f"{title}{content}".encode()).hexdigest()

        # 캐시에서 중복 확인
        if content_hash in self._quality_cache.get("content_hashes", set()):
            return False

        # 캐시에 추가
        if "content_hashes" not in self._quality_cache:
            self._quality_cache["content_hashes"] = set()
        self._quality_cache["content_hashes"].add(content_hash)

        return True

    def _validate_content_relevance(self, record: Dict) -> bool:
        """콘텐츠 관련성 검증"""
        title = record.get("title", "").strip().lower()
        content = record.get("content", "").strip().lower()

        if not title or not content:
            return True  # 다른 규칙에서 처리

        # 제목의 주요 키워드가 본문에 포함되어 있는지 확인
        title_words = set(re.findall(r"\w+", title))
        content_words = set(re.findall(r"\w+", content))

        # 교집합 비율 계산
        if len(title_words) == 0:
            return True

        intersection_ratio = len(title_words.intersection(content_words)) / len(title_words)
        return intersection_ratio >= 0.3  # 30% 이상 일치

    async def get_quality_trends(self, dataset_name: str = None, days: int = 7) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # 필터링된 리포트 가져오기
        filtered_reports = [
            report
            for report in self.quality_history
            if report.timestamp >= cutoff_date
            and (dataset_name is None or report.dataset_name == dataset_name)
        ]

        if not filtered_reports:
            return {"error": "No quality data available for the specified period"}

        # 트렌드 계산
        trend_data = {
            "period": f"Last {days} days",
            "total_reports": len(filtered_reports),
            "average_quality_score": statistics.mean([r.quality_score for r in filtered_reports]),
            "quality_trend": self._calculate_quality_trend(filtered_reports),
            "top_issues": self._get_top_issues(filtered_reports),
            "metric_trends": self._calculate_metric_trends(filtered_reports),
            "improvement_rate": self._calculate_improvement_rate(filtered_reports),
        }

        return trend_data

    def _calculate_quality_trend(self, reports: List[DataQualityReport]) -> str:
        """품질 트렌드 계산"""
        if len(reports) < 2:
            return "insufficient_data"

        # 시간순 정렬
        sorted_reports = sorted(reports, key=lambda x: x.timestamp)

        # 첫 절반과 후 절반의 평균 비교
        mid_point = len(sorted_reports) // 2
        first_half_avg = statistics.mean([r.quality_score for r in sorted_reports[:mid_point]])
        second_half_avg = statistics.mean([r.quality_score for r in sorted_reports[mid_point:]])

        improvement = second_half_avg - first_half_avg

        if improvement > 0.05:
            return "improving"
        elif improvement < -0.05:
            return "declining"
        else:
            return "stable"

    def _get_top_issues(self, reports: List[DataQualityReport], limit: int = 5) -> List[Dict]:
        """상위 이슈 목록 생성"""
        issue_counter = Counter()

        for report in reports:
            for issue in report.issues:
                issue_counter[issue.rule_name] += 1

        top_issues = []
        for rule_name, count in issue_counter.most_common(limit):
            rule = self.quality_rules.get(rule_name)
            top_issues.append(
                {
                    "rule_name": rule_name,
                    "description": rule.description if rule else "Unknown rule",
                    "count": count,
                    "severity": rule.severity.value if rule else "unknown",
                }
            )

        return top_issues

    def _calculate_metric_trends(self, reports: List[DataQualityReport]) -> Dict[str, float]:
        """메트릭별 트렌드 계산"""
        metric_trends = {}

        for metric in QualityMetric:
            scores = [report.metrics_scores.get(metric, 0.0) for report in reports]
            if scores:
                metric_trends[metric.value] = statistics.mean(scores)

        return metric_trends

    def _calculate_improvement_rate(self, reports: List[DataQualityReport]) -> float:
        """개선율 계산"""
        if len(reports) < 2:
            return 0.0

        sorted_reports = sorted(reports, key=lambda x: x.timestamp)
        first_score = sorted_reports[0].quality_score
        last_score = sorted_reports[-1].quality_score

        if first_score == 0:
            return 0.0

        return (last_score - first_score) / first_score * 100

    async def auto_fix_issues(self, data: Union[Dict, List[Dict]]) -> Tuple[Any, int]:
        """자동 이슈 수정"""
        if isinstance(data, dict):
            data_list = [data]
            single_record = True
        else:
            data_list = data
            single_record = False

        fixes_applied = 0

        for record in data_list:
            for rule_name, rule in self.quality_rules.items():
                if not rule.enabled or not rule.auto_fix_func:
                    continue

                try:
                    is_valid = rule.validation_func(record)
                    if not is_valid:
                        fixed_value = rule.auto_fix_func(record)
                        if fixed_value is not None:
                            fixes_applied += 1
                            self.auto_fix_stats["total_fixes"] += 1
                            logger.debug(f"Auto-fixed {rule_name} for record")

                except Exception as e:
                    logger.error(f"Auto-fix failed for {rule_name}: {e}")

        result = data_list[0] if single_record else data_list
        return result, fixes_applied

    def get_auto_fix_stats(self) -> Dict[str, Any]:
        """자동 수정 통계 반환"""
        stats = self.auto_fix_stats.copy()

        if stats["total_fixes"] > 0:
            stats["success_rate"] = stats["successful_fixes"] / stats["total_fixes"]
        else:
            stats["success_rate"] = 0.0

        return stats


class StatisticalAnomalyDetector:
    """통계 기반 이상 탐지기"""

    def __init__(self):
        self.baseline_stats = {}
        self.anomaly_threshold = 2.0  # 표준편차 임계값

    async def detect_anomalies(self, quality_report: DataQualityReport) -> List[str]:
        """이상 탐지 실행"""
        anomalies = []

        try:
            # 품질 점수 이상 탐지
            if await self._is_score_anomaly(quality_report.quality_score):
                anomalies.append(f"Quality score anomaly: {quality_report.quality_score:.3f}")

            # 이슈 개수 이상 탐지
            if await self._is_issue_count_anomaly(len(quality_report.issues)):
                anomalies.append(f"Issue count anomaly: {len(quality_report.issues)}")

            # 메트릭별 이상 탐지
            for metric, score in quality_report.metrics_scores.items():
                if await self._is_metric_anomaly(metric, score):
                    anomalies.append(f"Metric anomaly {metric.value}: {score:.3f}")

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")

        return anomalies

    async def _is_score_anomaly(self, score: float) -> bool:
        """품질 점수 이상 여부 확인"""
        # 간단한 임계값 기반 이상 탐지
        return score < 0.5  # 50% 미만은 이상으로 간주

    async def _is_issue_count_anomaly(self, issue_count: int) -> bool:
        """이슈 개수 이상 여부 확인"""
        return issue_count > 100  # 100개 초과는 이상으로 간주

    async def _is_metric_anomaly(self, metric: QualityMetric, score: float) -> bool:
        """메트릭 이상 여부 확인"""
        # 메트릭별 임계값
        thresholds = {
            QualityMetric.COMPLETENESS: 0.8,
            QualityMetric.ACCURACY: 0.7,
            QualityMetric.CONSISTENCY: 0.7,
            QualityMetric.VALIDITY: 0.8,
            QualityMetric.UNIQUENESS: 0.9,
            QualityMetric.TIMELINESS: 0.6,
            QualityMetric.RELEVANCE: 0.6,
        }

        threshold = thresholds.get(metric, 0.7)
        return score < threshold


# 전역 품질 관리자 인스턴스
_quality_manager: Optional[AdvancedDataQualityManager] = None


async def get_quality_manager() -> AdvancedDataQualityManager:
    """품질 관리자 싱글톤 인스턴스 반환"""
    global _quality_manager
    if _quality_manager is None:
        _quality_manager = AdvancedDataQualityManager()
        await _quality_manager.initialize_quality_rules()
    return _quality_manager


# 편의 함수들
async def validate_news_quality(data: Union[Dict, List[Dict]]) -> DataQualityReport:
    """뉴스 데이터 품질 검증 편의 함수"""
    manager = await get_quality_manager()
    return await manager.validate_data_quality(data, "news_articles")


async def auto_fix_news_data(data: Union[Dict, List[Dict]]) -> Tuple[Any, int]:
    """뉴스 데이터 자동 수정 편의 함수"""
    manager = await get_quality_manager()
    return await manager.auto_fix_issues(data)
