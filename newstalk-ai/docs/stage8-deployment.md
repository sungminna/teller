# Stage 8: Docker Containerization and Deployment

## 🎯 Overview

Stage 8 implements comprehensive containerization and deployment infrastructure for NewsTalk AI, providing production-ready Docker containers, Kubernetes orchestration, and automated CI/CD pipelines.

## 🏗️ Architecture

### Containerization Strategy

```mermaid
graph TB
    subgraph "Multi-stage Docker Builds"
        Builder[Builder Stage<br/>- Poetry dependencies<br/>- Compilation<br/>- Optimization]
        Runtime[Runtime Stage<br/>- Minimal base image<br/>- Non-root user<br/>- Security hardening]
        Builder --> Runtime
    end
    
    subgraph "Service Containers"
        Backend[FastAPI Backend<br/>- 4 workers<br/>- Health checks<br/>- Resource limits]
        LangGraph[LangGraph Service<br/>- AI agents<br/>- Checkpoint storage<br/>- GPU optimization]
        Airflow[Airflow Services<br/>- Webserver<br/>- Scheduler<br/>- Workers]
        StreamProc[Stream Processor<br/>- Kafka consumer<br/>- Real-time processing]
    end
    
    subgraph "Infrastructure Containers"
        Postgres[PostgreSQL<br/>- Persistent storage<br/>- Backup ready]
        Redis[Redis Stack<br/>- TimeSeries<br/>- Memory optimization]
        Kafka[Kafka + Zookeeper<br/>- Persistent logs<br/>- JMX monitoring]
    end
    
    subgraph "Monitoring Containers"
        Prometheus[Prometheus<br/>- Metrics collection<br/>- Alert rules]
        Grafana[Grafana<br/>- Dashboards<br/>- Visualization]
        NodeExporter[Node Exporter<br/>- System metrics]
    end
```

### Kubernetes Deployment

```mermaid
graph TB
    subgraph "Kubernetes Cluster"
        subgraph "newstalk-ai Namespace"
            subgraph "Application Layer"
                FastAPI[FastAPI Deployment<br/>3 replicas + HPA]
                LangGraphSvc[LangGraph Deployment<br/>2 replicas + HPA]
                StreamProc[Stream Processor<br/>2 replicas]
            end
            
            subgraph "Data Layer"
                PostgresDB[PostgreSQL<br/>Persistent Volume]
                RedisCache[Redis<br/>Persistent Volume]
                KafkaCluster[Kafka + Zookeeper<br/>Persistent Volumes]
            end
            
            subgraph "Airflow Layer"
                AirflowWeb[Airflow Webserver]
                AirflowScheduler[Airflow Scheduler]
                AirflowWorker[Airflow Workers<br/>2 replicas]
            end
        end
        
        subgraph "Ingress Layer"
            NginxIngress[NGINX Ingress<br/>- SSL termination<br/>- Rate limiting<br/>- Security headers]
            CertManager[Cert Manager<br/>- Let's Encrypt<br/>- Auto renewal]
        end
        
        subgraph "Monitoring Layer"
            PrometheusSvc[Prometheus<br/>- Metrics scraping<br/>- Alert manager]
            GrafanaSvc[Grafana<br/>- Dashboards<br/>- Alerts]
        end
    end
```

## 🐳 Docker Configuration

### Multi-stage Builds

All services use optimized multi-stage builds:

#### Backend Service (`Dockerfile.backend`)
```dockerfile
# Builder stage - Install dependencies
FROM python:3.11-slim as builder
RUN apt-get update && apt-get install -y gcc g++ curl git build-essential libpq-dev
RUN pip install poetry==1.7.1
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-root

# Runtime stage - Minimal production image
FROM python:3.11-slim as runtime
RUN groupadd -r -g 1001 newstalk && useradd -r -g newstalk -u 1001 -m newstalk
COPY --from=builder --chown=newstalk:newstalk /app/.venv /app/.venv
USER newstalk
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Key Features:
- **Multi-stage builds** for optimized image size
- **Non-root users** for security
- **Health checks** for reliability
- **Resource limits** for performance
- **Security scanning** with Trivy

### Docker Compose Configuration

Production-ready Docker Compose with:

```yaml
services:
  fastapi-backend:
    build:
      context: ../../backend
      dockerfile: ../infrastructure/docker/Dockerfile.backend
      args:
        BUILD_ENV: production
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ☸️ Kubernetes Deployment

### Namespace and Resource Management

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: newstalk-ai
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: newstalk-ai-quota
  namespace: newstalk-ai
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
```

### Application Deployments

#### FastAPI Backend
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-backend
  namespace: newstalk-ai
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: fastapi-backend
        image: newstalk-ai/backend:latest
        resources:
          requests:
            memory: "1Gi"
            cpu: "1000m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 40
          periodSeconds: 30
```

#### Auto-scaling Configuration
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastapi-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastapi-backend
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Persistent Storage

#### PostgreSQL with Persistent Volume
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd
```

### Ingress and External Access

#### NGINX Ingress with SSL
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: newstalk-ai-ingress
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit-rpm: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.newstalk-ai.com
    secretName: newstalk-ai-tls
  rules:
  - host: api.newstalk-ai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fastapi-backend
            port:
              number: 8000
```

## 🚀 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes 6 stages:

#### Stage 1: Code Quality & Testing
```yaml
test:
  strategy:
    matrix:
      python-version: ['3.11']
      test-type: ['unit', 'integration', 'security']
  steps:
  - name: Run unit tests
    run: poetry run pytest tests/unit/ -v --cov=. --cov-report=xml
  - name: Run security tests
    run: |
      poetry run bandit -r . -f json
      poetry run safety check --json
```

#### Stage 2: Build Docker Images
```yaml
build:
  steps:
  - name: Build and push Backend image
    uses: docker/build-push-action@v5
    with:
      context: ./backend
      file: ./infrastructure/docker/Dockerfile.backend
      push: true
      cache-from: type=gha
      cache-to: type=gha,mode=max
```

#### Stage 3: Security Scanning
```yaml
security-scan:
  steps:
  - name: Run Trivy vulnerability scanner
    uses: aquasecurity/trivy-action@master
    with:
      image-ref: ${{ needs.build.outputs.backend-image }}
      format: 'sarif'
```

#### Stage 4: Deploy to Staging
```yaml
deploy-staging:
  environment: staging
  steps:
  - name: Deploy to staging
    run: |
      kubectl apply -f infrastructure/k8s/
      kubectl rollout status deployment/fastapi-backend -n newstalk-ai
```

#### Stage 5: Deploy to Production
```yaml
deploy-production:
  environment: production
  steps:
  - name: Blue-Green Deployment
    run: |
      kubectl apply -f infrastructure/k8s/
      kubectl rollout status deployment/fastapi-backend -n newstalk-ai
```

#### Stage 6: Post-deployment Monitoring
```yaml
post-deployment:
  steps:
  - name: Run performance tests
    run: echo "Running performance tests..."
  - name: Check monitoring metrics
    run: echo "Checking system metrics..."
```

## 📜 Deployment Scripts

### Main Deployment Script

```bash
./scripts/deploy.sh [environment] [options]

# Examples:
./scripts/deploy.sh local --build
./scripts/deploy.sh staging --clean
./scripts/deploy.sh production
```

**Options:**
- `--build`: Force rebuild of Docker images
- `--clean`: Clean up existing deployment
- `--skip-tests`: Skip running tests

### Build Script

```bash
./scripts/build.sh [options]

# Examples:
./scripts/build.sh --tag v1.0.0 --push
./scripts/build.sh --dev --parallel
./scripts/build.sh --no-cache
```

**Options:**
- `--tag TAG`: Set image tag
- `--push`: Push to registry
- `--parallel`: Build in parallel
- `--dev`: Development build

## 🔧 Configuration Management

### Environment Variables

#### Production Environment
```bash
# Database
DATABASE_URL=postgresql://postgres:password@postgres:5432/newstalk_ai

# Redis
REDIS_URL=redis://redis:6379/0

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:29092

# API Keys (from secrets)
OPENAI_API_KEY=${OPENAI_API_KEY}
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY}
SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL}

# Application Settings
PIPELINE_MAX_PROCESSING_TIME=300
SSE_MAX_CONNECTIONS=1000
ENABLE_STREAMING=true
```

### ConfigMaps and Secrets

#### ConfigMap for Application Settings
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: newstalk-ai-config
data:
  POSTGRES_DB: "newstalk_ai"
  REDIS_HOST: "redis"
  KAFKA_BOOTSTRAP_SERVERS: "kafka:29092"
  PIPELINE_MAX_PROCESSING_TIME: "300"
  SSE_MAX_CONNECTIONS: "1000"
```

#### Secrets for Sensitive Data
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: newstalk-ai-secrets
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
  LANGFUSE_SECRET_KEY: <base64-encoded-key>
  SLACK_WEBHOOK_URL: <base64-encoded-url>
```

## 📊 Monitoring and Health Checks

### Health Check Endpoints

All services implement comprehensive health checks:

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "services": {
            "database": await check_database(),
            "redis": await check_redis(),
            "kafka": await check_kafka()
        }
    }
```

### Prometheus Metrics

Automatic metrics collection for:
- Request duration and count
- Error rates
- Resource usage
- Business metrics

### Grafana Dashboards

Pre-configured dashboards for:
- System overview
- Application performance
- Infrastructure metrics
- Business KPIs

## 🔒 Security

### Container Security

1. **Non-root users** in all containers
2. **Minimal base images** (Alpine/slim)
3. **Security scanning** with Trivy
4. **Read-only filesystems** where possible
5. **Resource limits** to prevent DoS

### Network Security

1. **Network policies** for pod-to-pod communication
2. **Ingress rate limiting** and DDoS protection
3. **TLS termination** with automatic certificate management
4. **Security headers** via NGINX annotations

### Secrets Management

1. **Kubernetes secrets** for sensitive data
2. **Base64 encoding** for secret values
3. **RBAC** for secret access control
4. **Automatic secret rotation** support

## 🚀 Deployment Environments

### Local Development

```bash
# Start all services locally
./scripts/deploy.sh local --build

# Services available at:
# - FastAPI: http://localhost:8000
# - Airflow: http://localhost:8081
# - Grafana: http://localhost:3000
```

### Staging Environment

```bash
# Deploy to staging Kubernetes
./scripts/deploy.sh staging --clean

# Automated testing and validation
# Blue-green deployment strategy
```

### Production Environment

```bash
# Deploy to production Kubernetes
./scripts/deploy.sh production

# Zero-downtime deployment
# Comprehensive health checks
# Automatic rollback on failure
```

## 📈 Performance Optimization

### Resource Allocation

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|-------------|-----------|----------------|--------------|
| FastAPI Backend | 1000m | 2000m | 1Gi | 2Gi |
| LangGraph Service | 1000m | 2000m | 2Gi | 4Gi |
| Stream Processor | 500m | 1000m | 512Mi | 1Gi |
| PostgreSQL | 500m | 1000m | 512Mi | 1Gi |
| Redis | 250m | 500m | 256Mi | 768Mi |

### Auto-scaling Policies

- **FastAPI Backend**: 3-10 replicas based on CPU (70%) and memory (80%)
- **LangGraph Service**: 2-6 replicas based on CPU (75%) and memory (85%)
- **Stream Processor**: 2-4 replicas based on queue depth

### Storage Classes

- **fast-ssd**: High-performance SSD for databases
- **standard**: Standard storage for logs and temporary data
- **backup**: Cold storage for backups and archives

## 🛠️ Troubleshooting

### Common Issues

#### Container Build Failures
```bash
# Check build logs
docker build --no-cache -f infrastructure/docker/Dockerfile.backend backend/

# Debug with intermediate images
docker run -it <intermediate-image-id> /bin/bash
```

#### Kubernetes Deployment Issues
```bash
# Check pod status
kubectl get pods -n newstalk-ai

# View pod logs
kubectl logs -f deployment/fastapi-backend -n newstalk-ai

# Describe pod for events
kubectl describe pod <pod-name> -n newstalk-ai
```

#### Service Connectivity Issues
```bash
# Test service connectivity
kubectl port-forward svc/fastapi-backend 8000:8000 -n newstalk-ai
curl http://localhost:8000/health

# Check service endpoints
kubectl get endpoints -n newstalk-ai
```

### Performance Debugging

#### Resource Usage
```bash
# Check resource usage
kubectl top pods -n newstalk-ai
kubectl top nodes

# View resource limits
kubectl describe deployment fastapi-backend -n newstalk-ai
```

#### Application Metrics
```bash
# Access Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090 -n newstalk-ai

# View Grafana dashboards
kubectl port-forward svc/grafana 3000:3000 -n newstalk-ai
```

## 📚 Best Practices

### Docker Best Practices

1. **Use multi-stage builds** to minimize image size
2. **Run as non-root user** for security
3. **Use specific base image tags** for reproducibility
4. **Implement proper health checks** for reliability
5. **Set resource limits** to prevent resource exhaustion

### Kubernetes Best Practices

1. **Use namespaces** for environment isolation
2. **Implement resource quotas** and limits
3. **Use ConfigMaps and Secrets** for configuration
4. **Implement proper RBAC** for security
5. **Use horizontal pod autoscaling** for scalability

### CI/CD Best Practices

1. **Run tests before deployment** to catch issues early
2. **Use security scanning** to identify vulnerabilities
3. **Implement blue-green deployment** for zero downtime
4. **Use environment-specific configurations** for consistency
5. **Monitor deployment metrics** for early issue detection

## 🎯 Success Metrics

### Deployment Metrics

- **Build Time**: < 10 minutes for all images
- **Deployment Time**: < 5 minutes for full stack
- **Zero Downtime**: 99.9% uptime during deployments
- **Rollback Time**: < 2 minutes for automatic rollback

### Performance Metrics

- **Container Startup**: < 30 seconds for application containers
- **Resource Efficiency**: < 80% CPU and memory utilization
- **Storage Performance**: < 10ms average disk latency
- **Network Performance**: < 1ms pod-to-pod latency

### Security Metrics

- **Vulnerability Scans**: Zero critical vulnerabilities
- **Security Policies**: 100% compliance with security policies
- **Secret Rotation**: Automated secret rotation every 90 days
- **Access Control**: RBAC implemented for all resources

## 🔄 Continuous Improvement

### Monitoring and Alerting

1. **Set up comprehensive monitoring** for all components
2. **Implement intelligent alerting** with proper thresholds
3. **Create runbooks** for common operational tasks
4. **Regular security audits** and vulnerability assessments

### Optimization Opportunities

1. **Resource right-sizing** based on actual usage patterns
2. **Image optimization** to reduce build and pull times
3. **Network optimization** for inter-service communication
4. **Storage optimization** for cost and performance

### Future Enhancements

1. **GitOps integration** with ArgoCD or Flux
2. **Service mesh** implementation with Istio
3. **Advanced monitoring** with distributed tracing
4. **Multi-cloud deployment** for high availability

---

## 📋 Quick Reference

### Essential Commands

```bash
# Build all images
./scripts/build.sh --tag v1.0.0

# Deploy locally
./scripts/deploy.sh local --build

# Deploy to production
./scripts/deploy.sh production

# Check deployment status
kubectl get pods -n newstalk-ai
kubectl get services -n newstalk-ai
kubectl get ingress -n newstalk-ai

# View logs
kubectl logs -f deployment/fastapi-backend -n newstalk-ai

# Scale deployment
kubectl scale deployment fastapi-backend --replicas=5 -n newstalk-ai
```

### Service URLs

| Service | Local URL | Production URL |
|---------|-----------|----------------|
| FastAPI Backend | http://localhost:8000 | https://api.newstalk-ai.com |
| Airflow | http://localhost:8081 | https://airflow.newstalk-ai.com |
| Grafana | http://localhost:3000 | https://grafana.newstalk-ai.com |
| Kafka UI | http://localhost:8080 | https://kafka-ui.newstalk-ai.com |

This completes the comprehensive Stage 8 implementation with production-ready containerization, Kubernetes orchestration, and automated CI/CD pipelines! 🚀 