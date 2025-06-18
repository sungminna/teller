#!/bin/bash

# NewsTalk AI - Stage 6 Deployment Script
# Real-time Streaming & System Integration Setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="newstalk-ai"
STAGE="Stage 6: Real-time Streaming & Integration"

echo -e "${BLUE}🚀 NewsTalk AI - ${STAGE} Deployment${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.11+ first."
        exit 1
    fi
    
    # Check Node.js (for mobile app)
    if ! command -v node &> /dev/null; then
        print_warning "Node.js is not installed. Mobile app setup will be skipped."
    fi
    
    print_status "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    print_info "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_status "Created .env from env.example"
        else
            print_error "env.example file not found"
            exit 1
        fi
    else
        print_warning ".env file already exists, skipping creation"
    fi
    
    print_info "Please review and update .env file with your configuration"
    print_info "Key settings to configure:"
    echo "  - OPENAI_API_KEY: Your OpenAI API key"
    echo "  - LANGFUSE_PUBLIC_KEY & LANGFUSE_SECRET_KEY: For monitoring"
    echo "  - Database credentials if different from defaults"
    
    read -p "Press Enter to continue after updating .env file..."
}

# Start infrastructure services
start_infrastructure() {
    print_info "Starting infrastructure services..."
    
    # Create Docker network if it doesn't exist
    docker network create newstalk-network 2>/dev/null || true
    
    # Start core infrastructure
    print_info "Starting PostgreSQL, Redis 8.0, and Kafka..."
    docker-compose up -d postgres redis zookeeper kafka
    
    # Wait for services to be ready
    print_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
    
    print_status "Infrastructure services started successfully"
}

# Check service health
check_service_health() {
    print_info "Checking service health..."
    
    # Check PostgreSQL
    if docker exec newstalk-postgres pg_isready -U postgres &> /dev/null; then
        print_status "PostgreSQL is ready"
    else
        print_error "PostgreSQL is not ready"
        exit 1
    fi
    
    # Check Redis
    if docker exec newstalk-redis redis-cli ping &> /dev/null; then
        print_status "Redis is ready"
    else
        print_error "Redis is not ready"
        exit 1
    fi
    
    # Check Kafka
    if docker exec newstalk-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 &> /dev/null; then
        print_status "Kafka is ready"
    else
        print_error "Kafka is not ready"
        exit 1
    fi
}

# Initialize Kafka topics
initialize_kafka_topics() {
    print_info "Initializing Kafka topics for Stage 6..."
    
    # Stage 6 topics with optimized configurations
    topics=(
        "raw-news:3:1"
        "processed-news:3:1"
        "user-feedback:2:1"
        "real-time-updates:4:1"
    )
    
    for topic_config in "${topics[@]}"; do
        IFS=':' read -r topic partitions replication <<< "$topic_config"
        
        print_info "Creating topic: $topic"
        docker exec newstalk-kafka kafka-topics \
            --create \
            --topic "$topic" \
            --bootstrap-server localhost:9092 \
            --partitions "$partitions" \
            --replication-factor "$replication" \
            --config compression.type=gzip \
            --config retention.ms=604800000 \
            --if-not-exists
    done
    
    # Verify topics
    print_info "Verifying Kafka topics..."
    docker exec newstalk-kafka kafka-topics --list --bootstrap-server localhost:9092
    
    print_status "Kafka topics initialized successfully"
}

# Setup Redis TimeSeries
setup_redis_timeseries() {
    print_info "Setting up Redis TimeSeries for Stage 6 metrics..."
    
    # Create TimeSeries for key metrics
    timeseries=(
        "news_pipeline_runs"
        "pipeline_duration"
        "articles_processed"
        "user_interactions"
        "cache_hit_rate"
    )
    
    for ts in "${timeseries[@]}"; do
        print_info "Creating TimeSeries: $ts"
        docker exec newstalk-redis redis-cli TS.CREATE "$ts" RETENTION 604800000 LABELS type metric || true
    done
    
    print_status "Redis TimeSeries setup completed"
}

# Start monitoring services
start_monitoring() {
    print_info "Starting monitoring services..."
    
    # Start Kafka UI
    docker-compose up -d kafka-ui
    
    # Start Prometheus and Grafana
    docker-compose up -d prometheus grafana
    
    # Wait for services
    sleep 15
    
    print_status "Monitoring services started"
    print_info "Access URLs:"
    echo "  - Kafka UI: http://localhost:8080"
    echo "  - Redis Insight: http://localhost:8001"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin123)"
}

# Start Airflow services
start_airflow() {
    print_info "Starting Airflow services..."
    
    # Start Airflow components
    docker-compose up -d airflow-webserver airflow-scheduler airflow-worker
    
    # Wait for Airflow to be ready
    sleep 30
    
    print_status "Airflow services started"
    print_info "Airflow UI: http://localhost:8081 (admin/admin123)"
}

# Start FastAPI backend
start_backend() {
    print_info "Starting FastAPI backend with streaming support..."
    
    docker-compose up -d fastapi-backend
    
    # Wait for backend to be ready
    sleep 15
    
    # Check backend health
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "FastAPI backend is ready"
    else
        print_warning "FastAPI backend may not be fully ready yet"
    fi
    
    print_info "Backend API: http://localhost:8000"
    print_info "API Documentation: http://localhost:8000/docs"
}

# Start stream processor
start_stream_processor() {
    print_info "Starting Kafka stream processor..."
    
    docker-compose up -d stream-processor
    
    sleep 10
    
    print_status "Stream processor started"
}

# Setup mobile app (if Node.js is available)
setup_mobile_app() {
    if command -v node &> /dev/null; then
        print_info "Setting up mobile app..."
        
        cd mobile-app
        
        # Install dependencies
        if [ ! -d "node_modules" ]; then
            print_info "Installing mobile app dependencies..."
            npm install
        fi
        
        # Create env file if it doesn't exist
        if [ ! -f ".env" ]; then
            cat > .env << EOF
EXPO_PUBLIC_API_URL=http://localhost:8000
EXPO_PUBLIC_WS_URL=ws://localhost:8000/ws
EOF
            print_status "Created mobile app .env file"
        fi
        
        cd ..
        
        print_status "Mobile app setup completed"
        print_info "To start mobile app: cd mobile-app && npx expo start"
    else
        print_warning "Node.js not found, skipping mobile app setup"
    fi
}

# Run health checks
run_health_checks() {
    print_info "Running comprehensive health checks..."
    
    # Check all services
    services=(
        "postgres:5432"
        "redis:6379"
        "kafka:9092"
        "fastapi-backend:8000"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if nc -z localhost "$port" 2>/dev/null; then
            print_status "$name is accessible on port $port"
        else
            print_warning "$name may not be ready on port $port"
        fi
    done
    
    # Test API endpoints
    print_info "Testing API endpoints..."
    
    if curl -f http://localhost:8000/health &> /dev/null; then
        print_status "Backend health check passed"
    else
        print_warning "Backend health check failed"
    fi
    
    if curl -f http://localhost:8000/api/v1/streaming/health &> /dev/null; then
        print_status "Streaming service health check passed"
    else
        print_warning "Streaming service health check failed"
    fi
}

# Display deployment summary
show_deployment_summary() {
    echo ""
    echo "=================================================="
    echo -e "${GREEN}🎉 Stage 6 Deployment Complete!${NC}"
    echo "=================================================="
    echo ""
    echo -e "${BLUE}📋 Service URLs:${NC}"
    echo "  🖥️  FastAPI Backend:     http://localhost:8000"
    echo "  📚 API Documentation:   http://localhost:8000/docs"
    echo "  🌊 Streaming Health:    http://localhost:8000/api/v1/streaming/health"
    echo "  🔄 Airflow UI:          http://localhost:8081"
    echo "  📊 Kafka UI:            http://localhost:8080"
    echo "  🗄️  Redis Insight:       http://localhost:8001"
    echo "  📈 Prometheus:          http://localhost:9090"
    echo "  📊 Grafana:             http://localhost:3000"
    echo ""
    echo -e "${BLUE}🔧 Key Features Enabled:${NC}"
    echo "  ✅ Kafka Real-time Streaming (raw-news, processed-news, user-feedback, real-time-updates)"
    echo "  ✅ Redis 8.0 Advanced Caching (24h sessions, 6h content, permanent voice files)"
    echo "  ✅ Server-Sent Events (SSE) for real-time updates"
    echo "  ✅ 5-minute Processing Pipeline Guarantee"
    echo "  ✅ Redis TimeSeries for Performance Monitoring"
    echo "  ✅ Comprehensive Health Monitoring"
    echo ""
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "  1. Update .env file with your API keys"
    echo "  2. Test real-time streaming: curl http://localhost:8000/api/v1/streaming/events"
    echo "  3. Start mobile app: cd mobile-app && npx expo start"
    echo "  4. Monitor pipeline performance in Grafana"
    echo "  5. Check Airflow DAGs: http://localhost:8081"
    echo ""
    echo -e "${BLUE}📖 Documentation:${NC}"
    echo "  📚 Full README: ./README.md"
    echo "  🔧 Configuration: ./.env"
    echo "  🐛 Troubleshooting: Check Docker logs with 'docker-compose logs [service]'"
    echo ""
    echo -e "${GREEN}Happy streaming! 🎊${NC}"
}

# Main deployment function
main() {
    echo -e "${BLUE}Starting Stage 6 deployment...${NC}"
    
    # Run deployment steps
    check_prerequisites
    setup_environment
    start_infrastructure
    initialize_kafka_topics
    setup_redis_timeseries
    start_monitoring
    start_airflow
    start_backend
    start_stream_processor
    setup_mobile_app
    run_health_checks
    show_deployment_summary
}

# Handle script interruption
trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 1' INT

# Run main function
main "$@" 