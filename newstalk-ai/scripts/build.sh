#!/bin/bash

# NewsTalk AI Build Script - Stage 8
# Usage: ./scripts/build.sh [options]

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="ghcr.io"
REPOSITORY="newstalk-ai"
TAG="latest"
BUILD_ENV="production"
PUSH_IMAGES=false
PARALLEL_BUILD=false

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
NewsTalk AI Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    --tag TAG           Set image tag (default: latest)
    --registry URL      Set container registry (default: ghcr.io)
    --push              Push images to registry after building
    --parallel          Build images in parallel
    --dev               Build development images
    --no-cache          Build without using cache
    --help              Show this help message

EXAMPLES:
    $0 --tag v1.0.0 --push
    $0 --dev --parallel
    $0 --no-cache --tag staging-$(date +%Y%m%d)
    
EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --tag)
                TAG="$2"
                shift 2
                ;;
            --registry)
                REGISTRY="$2"
                shift 2
                ;;
            --push)
                PUSH_IMAGES=true
                shift
                ;;
            --parallel)
                PARALLEL_BUILD=true
                shift
                ;;
            --dev)
                BUILD_ENV="development"
                shift
                ;;
            --no-cache)
                NO_CACHE="--no-cache"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Docker check passed"
}

build_image() {
    local service=$1
    local dockerfile=$2
    local context=$3
    local image_name="${REGISTRY}/${REPOSITORY}/${service}:${TAG}"
    
    log_info "Building $service image..."
    
    local build_args=(
        "BUILD_ENV=${BUILD_ENV}"
        "POETRY_VERSION=1.7.1"
    )
    
    local docker_args=(
        "build"
        "-f" "$dockerfile"
        "-t" "$image_name"
        "-t" "${REGISTRY}/${REPOSITORY}/${service}:latest"
    )
    
    # Add build args
    for arg in "${build_args[@]}"; do
        docker_args+=("--build-arg" "$arg")
    done
    
    # Add cache options
    if [[ "${NO_CACHE:-}" == "--no-cache" ]]; then
        docker_args+=("--no-cache")
    fi
    
    # Add context
    docker_args+=("$context")
    
    if docker "${docker_args[@]}"; then
        log_success "$service image built successfully"
        
        # Show image size
        local size=$(docker images --format "table {{.Size}}" "$image_name" | tail -n 1)
        log_info "$service image size: $size"
        
        return 0
    else
        log_error "Failed to build $service image"
        return 1
    fi
}

build_all_images() {
    cd "$PROJECT_ROOT"
    
    local services=(
        "backend:infrastructure/docker/Dockerfile.backend:backend"
        "langgraph:infrastructure/docker/Dockerfile.langgraph:backend"
        "airflow:infrastructure/docker/Dockerfile.airflow:backend"
        "stream-processor:infrastructure/docker/Dockerfile.stream-processor:backend"
    )
    
    if [[ "$PARALLEL_BUILD" == true ]]; then
        log_info "Building images in parallel..."
        
        local pids=()
        for service_info in "${services[@]}"; do
            IFS=':' read -r service dockerfile context <<< "$service_info"
            build_image "$service" "$dockerfile" "$context" &
            pids+=($!)
        done
        
        # Wait for all builds to complete
        local failed=false
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                failed=true
            fi
        done
        
        if [[ "$failed" == true ]]; then
            log_error "One or more image builds failed"
            exit 1
        fi
    else
        log_info "Building images sequentially..."
        
        for service_info in "${services[@]}"; do
            IFS=':' read -r service dockerfile context <<< "$service_info"
            if ! build_image "$service" "$dockerfile" "$context"; then
                exit 1
            fi
        done
    fi
    
    log_success "All images built successfully"
}

push_images() {
    if [[ "$PUSH_IMAGES" == true ]]; then
        log_info "Pushing images to registry..."
        
        local services=("backend" "langgraph" "airflow" "stream-processor")
        
        for service in "${services[@]}"; do
            local image_name="${REGISTRY}/${REPOSITORY}/${service}:${TAG}"
            local latest_name="${REGISTRY}/${REPOSITORY}/${service}:latest"
            
            log_info "Pushing $service image..."
            
            if docker push "$image_name" && docker push "$latest_name"; then
                log_success "$service image pushed successfully"
            else
                log_error "Failed to push $service image"
                exit 1
            fi
        done
        
        log_success "All images pushed successfully"
    fi
}

show_build_summary() {
    log_info "Build Summary:"
    echo "=================="
    echo "Tag: $TAG"
    echo "Registry: $REGISTRY"
    echo "Build Environment: $BUILD_ENV"
    echo "Push Images: $PUSH_IMAGES"
    echo "Parallel Build: $PARALLEL_BUILD"
    echo "=================="
    
    log_info "Built images:"
    local services=("backend" "langgraph" "airflow" "stream-processor")
    
    for service in "${services[@]}"; do
        local image_name="${REGISTRY}/${REPOSITORY}/${service}:${TAG}"
        if docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" "$image_name" 2>/dev/null | tail -n +2; then
            continue
        else
            log_warning "$service image not found"
        fi
    done
}

cleanup_build_cache() {
    log_info "Cleaning up build cache..."
    
    # Remove dangling images
    if docker images -f "dangling=true" -q | grep -q .; then
        docker rmi $(docker images -f "dangling=true" -q) || true
    fi
    
    # Prune build cache
    docker builder prune -f || true
    
    log_success "Build cache cleaned up"
}

main() {
    log_info "Starting NewsTalk AI image build process..."
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Check prerequisites
    check_docker
    
    # Build images
    build_all_images
    
    # Push images if requested
    push_images
    
    # Show summary
    show_build_summary
    
    # Cleanup
    cleanup_build_cache
    
    log_success "Build process completed successfully!"
    
    if [[ "$PUSH_IMAGES" == true ]]; then
        log_info "Images are available at:"
        local services=("backend" "langgraph" "airflow" "stream-processor")
        for service in "${services[@]}"; do
            echo "  - ${REGISTRY}/${REPOSITORY}/${service}:${TAG}"
        done
    fi
}

# Run main function with all arguments
main "$@" 