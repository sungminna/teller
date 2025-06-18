"""
Data Validator Operator for NewsTeam AI
Comprehensive validation with 99.5% accuracy target
"""
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.configuration import conf
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

from shared.models.news import NewsArticle, ProcessingLog, ProcessingStatus
from shared.utils.data_quality import DataQualityValidator

logger = logging.getLogger(__name__)


class DataValidatorOperator(BaseOperator):
    """Data validation operator ensuring 99.5% accuracy"""
    
    template_fields = ['quality_threshold', 'batch_size']
    
    @apply_defaults
    def __init__(
        self,
        quality_threshold: float = 0.7,
        batch_size: int = 100,
        duplicate_window_hours: int = 48,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.quality_threshold = quality_threshold
        self.batch_size = batch_size
        self.duplicate_window_hours = duplicate_window_hours
        
        # Database setup
        self.database_url = conf.get('core', 'sql_alchemy_conn')
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.quality_validator = DataQualityValidator()
        
    def execute(self, context):
        """Execute validation process"""
        logger.info("Starting data validation")
        
        # Get raw articles
        raw_articles = self._get_raw_articles()
        if not raw_articles:
            return {'processed_articles': 0}
            
        # Validate articles
        results = self._validate_articles(raw_articles)
        
        logger.info(f"Validation completed: {results}")
        return results
    
    def _get_raw_articles(self) -> List[NewsArticle]:
        """Get articles needing validation"""
        session = self.Session()
        try:
            return session.query(NewsArticle).filter(
                NewsArticle.status == ProcessingStatus.RAW
            ).limit(self.batch_size).all()
        finally:
            session.close()
    
    def _validate_articles(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Validate batch of articles"""
        results = {
            'processed': 0,
            'valid': 0,
            'invalid': 0,
            'duplicates': 0
        }
        
        session = self.Session()
        try:
            for article in articles:
                results['processed'] += 1
                
                # Convert to dict for validation
                article_data = {
                    'title': article.title or '',
                    'content': article.content or '',
                    'url': article.url or '',
                    'published_at': article.published_at
                }
                
                # Quality validation
                validation = self.quality_validator.validate_article(article_data)
                
                # Duplicate check
                is_duplicate = self._check_duplicate(article, session)
                
                # Update article status
                if validation['is_valid'] and not is_duplicate and validation['quality_score'] >= self.quality_threshold:
                    article.status = ProcessingStatus.VALIDATED
                    article.processed_at = datetime.utcnow()
                    results['valid'] += 1
                else:
                    article.status = ProcessingStatus.FAILED
                    article.last_error = self._get_error_reason(validation, is_duplicate)
                    if is_duplicate:
                        results['duplicates'] += 1
                    else:
                        results['invalid'] += 1
                
                article.processing_attempts += 1
                
            session.commit()
            
        except Exception as e:
            session.rollback()
            logger.error(f"Validation error: {str(e)}")
            raise
        finally:
            session.close()
            
        return results
    
    def _check_duplicate(self, article: NewsArticle, session) -> bool:
        """Check for duplicate articles"""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.duplicate_window_hours)
        
        existing = session.query(NewsArticle).filter(
            and_(
                NewsArticle.url == article.url,
                NewsArticle.id != article.id,
                NewsArticle.collected_at >= cutoff_time
            )
        ).first()
        
        return existing is not None
    
    def _get_error_reason(self, validation: Dict[str, Any], is_duplicate: bool) -> str:
        """Get error reason for failed validation"""
        if is_duplicate:
            return "Duplicate article"
        elif not validation['is_valid']:
            return f"Validation failed: {'; '.join(validation.get('issues', []))}"
        else:
            return f"Quality too low: {validation['quality_score']:.2f}"


class DataQualityReportOperator(BaseOperator):
    """
    Generate comprehensive data quality reports
    """
    
    @apply_defaults
    def __init__(
        self,
        report_hours: int = 24,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.report_hours = report_hours
        
        # Initialize database connection
        self.database_url = conf.get('core', 'sql_alchemy_conn')
        self.engine = create_engine(self.database_url)
        self.Session = sessionmaker(bind=self.engine)
    
    def execute(self, context):
        """Generate data quality report"""
        logger.info(f"Generating data quality report for last {self.report_hours} hours")
        
        session = self.Session()
        try:
            # Define time window
            cutoff_time = datetime.utcnow() - timedelta(hours=self.report_hours)
            
            # Get articles from the time window
            articles = session.query(NewsArticle).filter(
                NewsArticle.collected_at >= cutoff_time
            ).all()
            
            # Calculate quality metrics
            report = self._generate_quality_report(articles)
            
            logger.info(f"Quality report generated: {report}")
            return report
            
        finally:
            session.close()
    
    def _generate_quality_report(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        if not articles:
            return {'error': 'No articles found in time window'}
            
        total_articles = len(articles)
        valid_articles = len([a for a in articles if a.status == ProcessingStatus.VALIDATED])
        failed_articles = len([a for a in articles if a.status == ProcessingStatus.FAILED])
        processing_articles = len([a for a in articles if a.status == ProcessingStatus.RAW])
        
        # Calculate quality scores
        quality_scores = []
        for article in articles:
            if article.metadata and 'quality_score' in article.metadata:
                quality_scores.append(article.metadata['quality_score'])
                
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        # Processing time analysis
        processing_times = []
        for article in articles:
            if article.processed_at and article.collected_at:
                time_diff = (article.processed_at - article.collected_at).total_seconds()
                processing_times.append(time_diff)
                
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        report = {
            'report_generated_at': datetime.utcnow().isoformat(),
            'time_window_hours': self.report_hours,
            'total_articles': total_articles,
            'valid_articles': valid_articles,
            'failed_articles': failed_articles,
            'processing_articles': processing_articles,
            'validation_success_rate': (valid_articles / total_articles * 100) if total_articles > 0 else 0.0,
            'average_quality_score': round(avg_quality, 3),
            'average_processing_time_seconds': round(avg_processing_time, 2),
            'quality_distribution': {
                'high': len([s for s in quality_scores if s >= 0.8]),
                'medium': len([s for s in quality_scores if 0.6 <= s < 0.8]),
                'low': len([s for s in quality_scores if s < 0.6])
            }
        }
        
        # Check if we meet quality targets
        report['meets_quality_target'] = report['validation_success_rate'] >= 99.5
        report['meets_processing_target'] = avg_processing_time <= 300  # 5 minutes
        
        return report 