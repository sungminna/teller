#!/bin/bash

# NewsTalk AI Deployment Script - Stage 8
# Usage: ./scripts/deploy.sh [environment] [options]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT=${1:-"local"}
NAMESPACE="newstalk-ai"
REGISTRY="ghcr.io"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    cat << EOF
NewsTalk AI Deployment Script

Usage: $0 [ENVIRONMENT] [OPTIONS]

ENVIRONMENTS:
    local       Deploy using Docker Compose (default)
    staging     Deploy to staging Kubernetes cluster
    production  Deploy to production Kubernetes cluster

OPTIONS:
    --build     Force rebuild of Docker images
    --clean     Clean up existing deployment before deploying
    --skip-tests Skip running tests before deployment
    --help      Show this help message

EXAMPLES:
    $0 local --build
    $0 staging --clean
    $0 production
    
PREREQUISITES:
    - Docker and Docker Compose (for local)
    - kubectl configured for target cluster (for k8s)
    - Required environment variables set
    
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites for $ENVIRONMENT deployment..."
    
    case $ENVIRONMENT in
        "local")
            if ! command -v docker &> /dev/null; then
                log_error "Docker is not installed or not in PATH"
                exit 1
            fi
            
            if ! command -v docker-compose &> /dev/null; then
                log_error "Docker Compose is not installed or not in PATH"
                exit 1
            fi
            ;;
            
        "staging"|"production")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed or not in PATH"
                exit 1
            fi
            
            # Check kubectl connection
            if ! kubectl cluster-info &> /dev/null; then
                log_error "Cannot connect to Kubernetes cluster"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

build_images() {
    if [[ "$*" == *"--build"* ]] || [[ "$ENVIRONMENT" != "local" ]]; then
        log_info "Building Docker images..."
        
        cd "$PROJECT_ROOT"
        
        # Build backend image
        log_info "Building backend image..."
        docker build -f infrastructure/docker/Dockerfile.backend \
            -t newstalk-ai/backend:latest \
            --build-arg BUILD_ENV=production \
            backend/
        
        # Build LangGraph image
        log_info "Building LangGraph image..."
        docker build -f infrastructure/docker/Dockerfile.langgraph \
            -t newstalk-ai/langgraph:latest \
            --build-arg BUILD_ENV=production \
            backend/
        
        # Build Airflow image
        log_info "Building Airflow image..."
        docker build -f infrastructure/docker/Dockerfile.airflow \
            -t newstalk-ai/airflow:latest \
            --build-arg BUILD_ENV=production \
            backend/
        
        # Build Stream Processor image
        log_info "Building Stream Processor image..."
        docker build -f infrastructure/docker/Dockerfile.stream-processor \
            -t newstalk-ai/stream-processor:latest \
            --build-arg BUILD_ENV=production \
            backend/
        
        log_success "All images built successfully"
    fi
}

run_tests() {
    if [[ "$*" != *"--skip-tests"* ]]; then
        log_info "Running tests..."
        
        cd "$PROJECT_ROOT/backend"
        
        # Install dependencies
        if command -v poetry &> /dev/null; then
            poetry install --with dev
            poetry run pytest tests/ -v
        else
            pip install -r requirements.txt
            pip install pytest pytest-cov
            pytest tests/ -v
        fi
        
        log_success "Tests passed"
    else
        log_warning "Skipping tests as requested"
    fi
}

deploy_local() {
    log_info "Deploying to local environment with Docker Compose..."
    
    cd "$PROJECT_ROOT/infrastructure/docker"
    
    if [[ "$*" == *"--clean"* ]]; then
        log_info "Cleaning up existing deployment..."
        docker-compose down -v --remove-orphans
    fi
    
    # Start services
    log_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health checks
    log_info "Running health checks..."
    
    # Check FastAPI
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "FastAPI backend is healthy"
    else
        log_error "FastAPI backend health check failed"
        return 1
    fi
    
    # Check Airflow
    if curl -f http://localhost:8081/health &> /dev/null; then
        log_success "Airflow is healthy"
    else
        log_warning "Airflow health check failed (may still be starting)"
    fi
    
    log_success "Local deployment completed!"
    log_info "Services available at:"
    log_info "  - FastAPI Backend: http://localhost:8000"
    log_info "  - Airflow: http://localhost:8081"
    log_info "  - Grafana: http://localhost:3000"
    log_info "  - Kafka UI: http://localhost:8080"
    log_info "  - Prometheus: http://localhost:9090"
}

deploy_kubernetes() {
    log_info "Deploying to $ENVIRONMENT Kubernetes environment..."
    
    cd "$PROJECT_ROOT"
    
    # Check if namespace exists
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_info "Creating namespace $NAMESPACE..."
        kubectl apply -f infrastructure/k8s/namespace.yaml
    fi
    
    if [[ "$*" == *"--clean"* ]]; then
        log_info "Cleaning up existing deployment..."
        kubectl delete -f infrastructure/k8s/ --ignore-not-found=true
        sleep 10
    fi
    
    # Deploy in order
    log_info "Deploying ConfigMaps and Secrets..."
    kubectl apply -f infrastructure/k8s/configmap.yaml
    kubectl apply -f infrastructure/k8s/secrets.yaml
    
    log_info "Deploying infrastructure services..."
    kubectl apply -f infrastructure/k8s/postgres-deployment.yaml
    kubectl apply -f infrastructure/k8s/redis-deployment.yaml
    kubectl apply -f infrastructure/k8s/kafka-deployment.yaml
    
    # Wait for infrastructure
    log_info "Waiting for infrastructure services..."
    kubectl rollout status deployment/postgres -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/redis -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/zookeeper -n $NAMESPACE --timeout=300s
    kubectl rollout status deployment/kafka -n $NAMESPACE --timeout=300s
    
    log_info "Deploying application services..."
    kubectl apply -f infrastructure/k8s/backend-deployment.yaml
    kubectl apply -f infrastructure/k8s/airflow-deployment.yaml
    
    # Wait for application services
    log_info "Waiting for application services..."
    kubectl rollout status deployment/fastapi-backend -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/langgraph-service -n $NAMESPACE --timeout=600s
    kubectl rollout status deployment/airflow-webserver -n $NAMESPACE --timeout=600s
    
    log_info "Deploying ingress..."
    kubectl apply -f infrastructure/k8s/ingress.yaml
    
    # Health checks
    log_info "Running health checks..."
    sleep 60
    
    # Port forward for health check
    kubectl port-forward svc/fastapi-backend 8000:8000 -n $NAMESPACE &
    PF_PID=$!
    sleep 10
    
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Application health check passed"
    else
        log_error "Application health check failed"
        kill $PF_PID 2>/dev/null || true
        return 1
    fi
    
    kill $PF_PID 2>/dev/null || true
    
    log_success "$ENVIRONMENT deployment completed!"
    
    # Show service information
    log_info "Deployment status:"
    kubectl get pods -n $NAMESPACE
    kubectl get services -n $NAMESPACE
    kubectl get ingress -n $NAMESPACE
}

main() {
    # Parse arguments
    if [[ "$*" == *"--help"* ]]; then
        show_help
        exit 0
    fi
    
    log_info "Starting NewsTalk AI deployment to $ENVIRONMENT environment"
    
    # Check prerequisites
    check_prerequisites
    
    # Run tests
    run_tests "$@"
    
    # Build images
    build_images "$@"
    
    # Deploy based on environment
    case $ENVIRONMENT in
        "local")
            deploy_local "$@"
            ;;
        "staging"|"production")
            deploy_kubernetes "$@"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Supported environments: local, staging, production"
            exit 1
            ;;
    esac
    
    log_success "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@" 