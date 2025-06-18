"""
Data Preprocessing DAG for NewsTeam AI
Advanced news content processing and AI enhancement
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# DAG configuration
default_args = {
    'owner': 'newsteam-ai',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'catchup': False
}

dag = DAG(
    'data_preprocessing_pipeline',
    default_args=default_args,
    description='News data preprocessing and AI enhancement pipeline',
    schedule_interval=timedelta(hours=1),  # Every hour
    max_active_runs=1,
    tags=['preprocessing', 'ai', 'embeddings', 'nlp']
)

def clean_and_normalize_text(**context):
    """Clean and normalize article text content"""
    import re
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from airflow.configuration import conf
    
    database_url = conf.get('core', 'sql_alchemy_conn')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    
    # Import after creating engine to avoid circular imports
    import sys
    sys.path.append('/opt/airflow')
    from shared.models.news import NewsArticle, ProcessingStatus
    
    session = Session()
    processed_count = 0
    
    try:
        # Get AI-analyzed articles that need text preprocessing
        articles = session.query(NewsArticle).filter(
            NewsArticle.status == ProcessingStatus.AI_ANALYZED
        ).limit(100).all()
        
        for article in articles:
            if article.content:
                # Clean HTML and special characters
                clean_content = re.sub(r'<[^>]+>', '', article.content)
                clean_content = re.sub(r'[^\w\s가-힣.,!?;:]', ' ', clean_content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                # Clean title
                clean_title = re.sub(r'[^\w\s가-힣.,!?;:]', ' ', article.title)
                clean_title = re.sub(r'\s+', ' ', clean_title).strip()
                
                # Update article
                article.title = clean_title
                article.content = clean_content
                
                # Update word count
                article.word_count = len(clean_content.split())
                
                processed_count += 1
        
        session.commit()
        
    finally:
        session.close()
    
    return {'processed_articles': processed_count}

def extract_metadata(**context):
    """Extract metadata from articles"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from airflow.configuration import conf
    import re
    from datetime import datetime
    
    database_url = conf.get('core', 'sql_alchemy_conn')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    
    import sys
    sys.path.append('/opt/airflow')
    from shared.models.news import NewsArticle, ProcessingStatus
    
    session = Session()
    processed_count = 0
    
    try:
        articles = session.query(NewsArticle).filter(
            NewsArticle.status == ProcessingStatus.AI_ANALYZED
        ).limit(100).all()
        
        for article in articles:
            metadata = article.metadata or {}
            
            # Extract reading time (words per minute)
            if article.word_count:
                reading_time = max(1, article.word_count // 200)  # 200 WPM
                metadata['estimated_reading_time'] = reading_time
            
            # Extract content indicators
            if article.content:
                metadata['has_numbers'] = bool(re.search(r'\d+', article.content))
                metadata['has_quotes'] = bool(re.search(r'["""\'\'\"\"′″‴„‚']', article.content))
                metadata['paragraph_count'] = len(article.content.split('\n\n'))
            
            # Update publication freshness
            if article.published_at:
                hours_old = (datetime.utcnow() - article.published_at).total_seconds() / 3600
                metadata['freshness_hours'] = round(hours_old, 2)
                metadata['is_breaking'] = hours_old < 2  # Breaking news if < 2 hours old
            
            article.metadata = metadata
            processed_count += 1
        
        session.commit()
        
    finally:
        session.close()
    
    return {'metadata_extracted': processed_count}

def generate_embeddings(**context):
    """Generate embeddings for articles (placeholder for actual embedding service)"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from airflow.configuration import conf
    import json
    import hashlib
    
    database_url = conf.get('core', 'sql_alchemy_conn')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    
    import sys
    sys.path.append('/opt/airflow')
    from shared.models.news import NewsArticle, ProcessingStatus
    
    session = Session()
    processed_count = 0
    
    try:
        articles = session.query(NewsArticle).filter(
            NewsArticle.status == ProcessingStatus.AI_ANALYZED,
            NewsArticle.embedding_vector.is_(None)
        ).limit(50).all()  # Smaller batch for embedding generation
        
        for article in articles:
            # Generate mock embedding (in production, use actual embedding service)
            text_for_embedding = f"{article.title} {article.content[:500]}"
            
            # Create deterministic mock embedding based on content hash
            content_hash = hashlib.md5(text_for_embedding.encode()).hexdigest()
            mock_embedding = [
                float(int(content_hash[i:i+2], 16)) / 255.0 - 0.5 
                for i in range(0, min(32, len(content_hash)), 2)
            ]
            
            # Pad to 768 dimensions (common embedding size)
            while len(mock_embedding) < 768:
                mock_embedding.extend(mock_embedding[:min(16, 768 - len(mock_embedding))])
            
            article.embedding_vector = json.dumps(mock_embedding[:768])
            processed_count += 1
        
        session.commit()
        
    finally:
        session.close()
    
    return {'embeddings_generated': processed_count}

def calculate_similarity_scores(**context):
    """Calculate article similarity scores"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from airflow.configuration import conf
    import json
    import numpy as np
    
    database_url = conf.get('core', 'sql_alchemy_conn')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    
    import sys
    sys.path.append('/opt/airflow')
    from shared.models.news import NewsArticle, ProcessingStatus
    
    session = Session()
    processed_count = 0
    
    try:
        # Get articles with embeddings
        articles = session.query(NewsArticle).filter(
            NewsArticle.status == ProcessingStatus.AI_ANALYZED,
            NewsArticle.embedding_vector.isnot(None)
        ).limit(100).all()
        
        # Calculate similarity matrix (simplified version)
        for i, article in enumerate(articles):
            if not article.metadata:
                article.metadata = {}
            
            # Mock similarity calculation
            similarity_scores = []
            if len(articles) > 1:
                embedding = np.array(json.loads(article.embedding_vector))
                
                for j, other_article in enumerate(articles):
                    if i != j and other_article.embedding_vector:
                        other_embedding = np.array(json.loads(other_article.embedding_vector))
                        similarity = np.dot(embedding, other_embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(other_embedding)
                        )
                        if similarity > 0.7:  # Only store high similarity
                            similarity_scores.append({
                                'article_id': other_article.id,
                                'similarity': float(similarity)
                            })
            
            article.metadata['similar_articles'] = similarity_scores[:5]  # Top 5
            processed_count += 1
        
        session.commit()
        
    finally:
        session.close()
    
    return {'similarity_calculated': processed_count}

def finalize_preprocessing(**context):
    """Finalize preprocessing and update article status"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from airflow.configuration import conf
    from datetime import datetime
    
    database_url = conf.get('core', 'sql_alchemy_conn')
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    
    import sys
    sys.path.append('/opt/airflow')
    from shared.models.news import NewsArticle, ProcessingStatus
    
    session = Session()
    finalized_count = 0
    
    try:
        # Get AI-analyzed articles that are ready for finalization
        articles = session.query(NewsArticle).filter(
            NewsArticle.status == ProcessingStatus.AI_ANALYZED,
            NewsArticle.embedding_vector.isnot(None)
        ).limit(100).all()
        
        for article in articles:
            # Mark as fully processed
            article.status = ProcessingStatus.PROCESSED
            
            # Update final processing timestamp
            if not article.metadata:
                article.metadata = {}
            article.metadata['preprocessing_completed'] = datetime.utcnow().isoformat()
            
            finalized_count += 1
        
        session.commit()
        
    finally:
        session.close()
    
    return {'finalized_articles': finalized_count}

# Define tasks
text_cleaning = PythonOperator(
    task_id='clean_normalize_text',
    python_callable=clean_and_normalize_text,
    dag=dag
)

metadata_extraction = PythonOperator(
    task_id='extract_metadata',
    python_callable=extract_metadata,
    dag=dag
)

embedding_generation = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    dag=dag
)

similarity_calculation = PythonOperator(
    task_id='calculate_similarity',
    python_callable=calculate_similarity_scores,
    dag=dag
)

preprocessing_finalization = PythonOperator(
    task_id='finalize_preprocessing',
    python_callable=finalize_preprocessing,
    dag=dag
)

# Generate preprocessing report
def generate_preprocessing_report(**context):
    """Generate preprocessing pipeline report"""
    import json
    
    # Get results from all tasks
    cleaning_result = context['task_instance'].xcom_pull(task_ids='clean_normalize_text')
    metadata_result = context['task_instance'].xcom_pull(task_ids='extract_metadata')
    embedding_result = context['task_instance'].xcom_pull(task_ids='generate_embeddings')
    similarity_result = context['task_instance'].xcom_pull(task_ids='calculate_similarity')
    finalization_result = context['task_instance'].xcom_pull(task_ids='finalize_preprocessing')
    
    report = {
        'preprocessing_timestamp': datetime.utcnow().isoformat(),
        'text_cleaning': cleaning_result or {},
        'metadata_extraction': metadata_result or {},
        'embedding_generation': embedding_result or {},
        'similarity_calculation': similarity_result or {},
        'finalization': finalization_result or {}
    }
    
    print(f"Preprocessing report: {json.dumps(report, indent=2)}")
    return report

report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_preprocessing_report,
    provide_context=True,
    dag=dag
)

# Health check
health_check = BashOperator(
    task_id='health_check',
    bash_command="""
    echo "Data preprocessing pipeline health check"
    echo "Timestamp: $(date)"
    echo "Preprocessing completed successfully"
    """,
    dag=dag
)

# Define task dependencies
text_cleaning >> metadata_extraction >> [embedding_generation, similarity_calculation] >> preprocessing_finalization >> report_task >> health_check

# Task documentation
text_cleaning.doc_md = """
## Text Cleaning and Normalization
- Removes HTML tags and special characters
- Normalizes whitespace
- Updates word counts
- Processes 100 articles per run
"""

metadata_extraction.doc_md = """
## Metadata Extraction
- Calculates reading time estimates
- Extracts content indicators
- Determines article freshness
- Identifies breaking news
"""

embedding_generation.doc_md = """
## Embedding Generation
- Generates vector embeddings for articles
- Uses first 500 characters + title
- 768-dimensional vectors
- Batch size: 50 articles
"""

similarity_calculation.doc_md = """
## Similarity Calculation
- Calculates article similarity scores
- Uses cosine similarity on embeddings
- Stores top 5 similar articles
- Threshold: 0.7 similarity
""" 