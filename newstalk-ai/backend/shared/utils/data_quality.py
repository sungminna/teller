"""
Data Quality Validation Utilities for NewsTeam AI
"""
import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
import logging

from ..models.news import NewsArticle, ProcessingStatus

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Comprehensive data quality validator for news articles"""
    
    def __init__(self):
        # Quality thresholds
        self.min_title_length = 10
        self.max_title_length = 500
        self.min_content_length = 100
        self.max_content_length = 50000
        self.min_word_count = 20
        self.max_duplicate_similarity = 0.85
        
        # Common spam/low-quality indicators
        self.spam_keywords = {
            'click here', 'buy now', 'limited time', 'act now', 
            'free money', 'guaranteed', 'miracle', 'secret',
            '광고', '홍보', '무료', '할인', '특가'
        }
        
        # Valid domains for Korean news sources
        self.trusted_domains = {
            'naver.com', 'daum.net', 'chosun.com', 'joongang.co.kr',
            'donga.com', 'hankyung.com', 'mk.co.kr', 'mt.co.kr',
            'ytn.co.kr', 'sbs.co.kr', 'kbs.co.kr', 'mbc.co.kr',
            'hani.co.kr', 'khan.co.kr', 'pressian.com', 'ohmynews.com'
        }

    def validate_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation of a news article
        
        Args:
            article_data: Dictionary containing article data
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'quality_score': 1.0,
            'issues': [],
            'warnings': [],
            'metadata': {}
        }
        
        try:
            # Basic field validation
            self._validate_required_fields(article_data, validation_result)
            
            # Content quality validation
            self._validate_content_quality(article_data, validation_result)
            
            # URL validation
            self._validate_url(article_data, validation_result)
            
            # Timestamp validation
            self._validate_timestamps(article_data, validation_result)
            
            # Spam detection
            self._detect_spam(article_data, validation_result)
            
            # Language detection
            self._validate_language(article_data, validation_result)
            
            # Calculate overall quality score
            self._calculate_quality_score(validation_result)
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            validation_result['is_valid'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
            validation_result['quality_score'] = 0.0
            
        return validation_result

    def _validate_required_fields(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate required fields are present and not empty"""
        required_fields = ['title', 'url']
        
        for field in required_fields:
            if field not in article_data or not article_data[field]:
                result['is_valid'] = False
                result['issues'].append(f"Missing required field: {field}")
            elif isinstance(article_data[field], str) and not article_data[field].strip():
                result['is_valid'] = False
                result['issues'].append(f"Empty required field: {field}")

    def _validate_content_quality(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate content quality metrics"""
        title = article_data.get('title', '').strip()
        content = article_data.get('content', '').strip()
        
        # Title validation
        if title:
            if len(title) < self.min_title_length:
                result['warnings'].append(f"Title too short ({len(title)} chars)")
                result['quality_score'] *= 0.9
            elif len(title) > self.max_title_length:
                result['warnings'].append(f"Title too long ({len(title)} chars)")
                result['quality_score'] *= 0.95
                
        # Content validation
        if content:
            word_count = len(content.split())
            result['metadata']['word_count'] = word_count
            
            if len(content) < self.min_content_length:
                result['warnings'].append(f"Content too short ({len(content)} chars)")
                result['quality_score'] *= 0.8
            elif len(content) > self.max_content_length:
                result['warnings'].append(f"Content too long ({len(content)} chars)")
                result['quality_score'] *= 0.9
                
            if word_count < self.min_word_count:
                result['warnings'].append(f"Too few words ({word_count})")
                result['quality_score'] *= 0.8
                
            # Check for repetitive content
            if self._is_repetitive_content(content):
                result['warnings'].append("Content appears repetitive")
                result['quality_score'] *= 0.7

    def _validate_url(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate URL format and domain"""
        url = article_data.get('url', '')
        
        if not url:
            return
            
        try:
            parsed_url = urlparse(url)
            
            # Check URL format
            if not parsed_url.scheme or not parsed_url.netloc:
                result['issues'].append("Invalid URL format")
                result['is_valid'] = False
                return
                
            # Check domain trust level
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
                
            result['metadata']['domain'] = domain
            
            # Check if domain is in trusted list
            if any(trusted in domain for trusted in self.trusted_domains):
                result['metadata']['trusted_domain'] = True
                result['quality_score'] *= 1.1  # Boost for trusted domains
            else:
                result['metadata']['trusted_domain'] = False
                result['warnings'].append(f"Unknown domain: {domain}")
                result['quality_score'] *= 0.95
                
        except Exception as e:
            result['issues'].append(f"URL validation error: {str(e)}")
            result['quality_score'] *= 0.8

    def _validate_timestamps(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Validate timestamp fields"""
        published_at = article_data.get('published_at')
        
        if published_at:
            try:
                if isinstance(published_at, str):
                    # Try to parse string timestamp
                    published_dt = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                elif isinstance(published_at, datetime):
                    published_dt = published_at
                else:
                    result['warnings'].append("Invalid published_at format")
                    return
                    
                now = datetime.utcnow()
                
                # Check if publication date is in the future
                if published_dt > now + timedelta(hours=1):  # 1 hour buffer
                    result['warnings'].append("Publication date in future")
                    result['quality_score'] *= 0.9
                    
                # Check if publication date is too old (older than 30 days)
                elif published_dt < now - timedelta(days=30):
                    result['warnings'].append("Publication date too old")
                    result['quality_score'] *= 0.95
                    
                result['metadata']['article_age_hours'] = (now - published_dt).total_seconds() / 3600
                
            except Exception as e:
                result['warnings'].append(f"Timestamp parsing error: {str(e)}")
                result['quality_score'] *= 0.9

    def _detect_spam(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Detect spam indicators in content"""
        title = article_data.get('title', '').lower()
        content = article_data.get('content', '').lower()
        
        spam_indicators = 0
        
        # Check for spam keywords
        for keyword in self.spam_keywords:
            if keyword in title or keyword in content:
                spam_indicators += 1
                
        # Check for excessive exclamation marks
        if title.count('!') > 3 or content.count('!') > 10:
            spam_indicators += 1
            
        # Check for excessive capitalization
        if sum(1 for c in title if c.isupper()) > len(title) * 0.3:
            spam_indicators += 1
            
        # Check for suspicious patterns
        if re.search(r'\d{10,}', content):  # Long numbers (phone numbers, etc.)
            spam_indicators += 1
            
        if spam_indicators > 0:
            result['warnings'].append(f"Potential spam indicators: {spam_indicators}")
            result['quality_score'] *= max(0.5, 1.0 - (spam_indicators * 0.1))
            result['metadata']['spam_indicators'] = spam_indicators

    def _validate_language(self, article_data: Dict[str, Any], result: Dict[str, Any]):
        """Basic language validation"""
        title = article_data.get('title', '')
        content = article_data.get('content', '')
        expected_lang = article_data.get('language', 'ko')
        
        # Simple Korean text detection
        if expected_lang == 'ko':
            korean_chars = sum(1 for c in title + content if '\uac00' <= c <= '\ud7af')
            total_chars = len(title + content)
            
            if total_chars > 0:
                korean_ratio = korean_chars / total_chars
                result['metadata']['korean_ratio'] = korean_ratio
                
                if korean_ratio < 0.3:  # Less than 30% Korean characters
                    result['warnings'].append(f"Low Korean content ratio: {korean_ratio:.2f}")
                    result['quality_score'] *= 0.9

    def _is_repetitive_content(self, content: str) -> bool:
        """Check if content is repetitive"""
        if len(content) < 100:
            return False
            
        sentences = re.split(r'[.!?]\s+', content)
        if len(sentences) < 3:
            return False
            
        # Check for repeated sentences
        sentence_counts = {}
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if len(sentence) > 10:
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
                
        # If any sentence appears more than 3 times, consider it repetitive
        return any(count > 3 for count in sentence_counts.values())

    def _calculate_quality_score(self, result: Dict[str, Any]):
        """Calculate final quality score based on issues and warnings"""
        base_score = result['quality_score']
        
        # Penalize for issues (more severe)
        issue_penalty = len(result['issues']) * 0.2
        
        # Penalize for warnings (less severe)
        warning_penalty = len(result['warnings']) * 0.05
        
        final_score = max(0.0, base_score - issue_penalty - warning_penalty)
        result['quality_score'] = round(final_score, 3)
        
        # Mark as invalid if quality is too low
        if final_score < 0.5:
            result['is_valid'] = False
            result['issues'].append(f"Quality score too low: {final_score}")

    def check_duplicate(self, article_data: Dict[str, Any], existing_articles: List[NewsArticle]) -> Dict[str, Any]:
        """
        Check for duplicate articles using content hash and similarity
        
        Args:
            article_data: New article data
            existing_articles: List of existing articles to compare against
            
        Returns:
            Dictionary with duplicate check results
        """
        duplicate_result = {
            'is_duplicate': False,
            'duplicate_score': 0.0,
            'similar_articles': [],
            'content_hash': None
        }
        
        try:
            # Generate content hash
            content_to_hash = f"{article_data.get('title', '')}{article_data.get('url', '')}"
            content_hash = hashlib.sha256(content_to_hash.encode('utf-8')).hexdigest()
            duplicate_result['content_hash'] = content_hash
            
            # Check for exact URL matches
            url = article_data.get('url', '')
            for existing in existing_articles:
                if existing.url == url:
                    duplicate_result['is_duplicate'] = True
                    duplicate_result['duplicate_score'] = 1.0
                    duplicate_result['similar_articles'].append({
                        'id': existing.id,
                        'title': existing.title,
                        'similarity': 1.0,
                        'reason': 'identical_url'
                    })
                    break
                    
            # Check for content hash matches
            if not duplicate_result['is_duplicate']:
                for existing in existing_articles:
                    if existing.content_hash == content_hash:
                        duplicate_result['is_duplicate'] = True
                        duplicate_result['duplicate_score'] = 1.0
                        duplicate_result['similar_articles'].append({
                            'id': existing.id,
                            'title': existing.title,
                            'similarity': 1.0,
                            'reason': 'identical_content_hash'
                        })
                        break
                        
            # If not exact duplicate, check for similarity
            if not duplicate_result['is_duplicate']:
                title = article_data.get('title', '').lower()
                for existing in existing_articles:
                    similarity = self._calculate_title_similarity(title, existing.title.lower())
                    if similarity > self.max_duplicate_similarity:
                        duplicate_result['similar_articles'].append({
                            'id': existing.id,
                            'title': existing.title,
                            'similarity': similarity,
                            'reason': 'title_similarity'
                        })
                        
                if duplicate_result['similar_articles']:
                    max_similarity = max(item['similarity'] for item in duplicate_result['similar_articles'])
                    duplicate_result['duplicate_score'] = max_similarity
                    if max_similarity > 0.95:
                        duplicate_result['is_duplicate'] = True
                        
        except Exception as e:
            logger.error(f"Duplicate check error: {str(e)}")
            
        return duplicate_result

    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles using simple word overlap"""
        if not title1 or not title2:
            return 0.0
            
        # Simple word-based similarity
        words1 = set(re.findall(r'\w+', title1.lower()))
        words2 = set(re.findall(r'\w+', title2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def generate_quality_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report for a batch of articles
        
        Args:
            articles: List of article data dictionaries
            
        Returns:
            Quality report dictionary
        """
        report = {
            'total_articles': len(articles),
            'valid_articles': 0,
            'invalid_articles': 0,
            'average_quality_score': 0.0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'common_issues': {},
            'common_warnings': {},
            'processing_time': datetime.utcnow()
        }
        
        if not articles:
            return report
            
        total_quality = 0.0
        all_issues = []
        all_warnings = []
        
        for article_data in articles:
            validation_result = self.validate_article(article_data)
            
            if validation_result['is_valid']:
                report['valid_articles'] += 1
            else:
                report['invalid_articles'] += 1
                
            quality_score = validation_result['quality_score']
            total_quality += quality_score
            
            # Categorize quality
            if quality_score >= 0.8:
                report['quality_distribution']['high'] += 1
            elif quality_score >= 0.6:
                report['quality_distribution']['medium'] += 1
            else:
                report['quality_distribution']['low'] += 1
                
            all_issues.extend(validation_result['issues'])
            all_warnings.extend(validation_result['warnings'])
            
        # Calculate averages and common issues
        report['average_quality_score'] = total_quality / len(articles)
        
        # Count common issues
        for issue in all_issues:
            report['common_issues'][issue] = report['common_issues'].get(issue, 0) + 1
            
        for warning in all_warnings:
            report['common_warnings'][warning] = report['common_warnings'].get(warning, 0) + 1
            
        return report 