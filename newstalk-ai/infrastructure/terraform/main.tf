# NewsTalk AI Production Infrastructure - Stage 10
# AWS EKS, RDS PostgreSQL 17.4, ElastiCache Redis 8.0, MSK Kafka

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket = "newstalk-ai-terraform-state"
    key    = "production/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "NewsTalk AI"
      Environment = var.environment
      Stage       = "10"
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "newstalk-ai-prod"
}

variable "cluster_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.r6g.xlarge"
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.r7g.large"
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  database_subnets = ["10.0.201.0/24", "10.0.202.0/24", "10.0.203.0/24"]

  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  vpc_id                         = module.vpc.vpc_id
  subnet_ids                     = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name = "general"
      
      instance_types = ["m5.large", "m5.xlarge"]
      
      min_size     = 2
      max_size     = 10
      desired_size = 3

      disk_size = 50
      disk_type = "gp3"

      labels = {
        role = "general"
      }

      taints = []

      tags = {
        NodeGroup = "general"
      }
    }

    # AI/ML workload nodes
    ai_workload = {
      name = "ai-workload"
      
      instance_types = ["c5.2xlarge", "c5.4xlarge"]
      
      min_size     = 1
      max_size     = 5
      desired_size = 2

      disk_size = 100
      disk_type = "gp3"

      labels = {
        role = "ai-workload"
        workload = "fact-checking"
      }

      taints = [
        {
          key    = "ai-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      tags = {
        NodeGroup = "ai-workload"
      }
    }

    # Memory optimized for Redis and caching
    memory_optimized = {
      name = "memory-optimized"
      
      instance_types = ["r5.large", "r5.xlarge"]
      
      min_size     = 1
      max_size     = 3
      desired_size = 1

      disk_size = 50
      disk_type = "gp3"

      labels = {
        role = "memory-optimized"
      }

      tags = {
        NodeGroup = "memory-optimized"
      }
    }
  }

  # Cluster access entry
  # To add the current caller identity as an administrator
  enable_cluster_creator_admin_permissions = true

  tags = {
    Environment = var.environment
    Terraform   = "true"
  }
}

# RDS PostgreSQL 17.4
resource "aws_db_subnet_group" "newstalk_ai" {
  name       = "${var.cluster_name}-db-subnet-group"
  subnet_ids = module.vpc.database_subnets

  tags = {
    Name = "${var.cluster_name} DB subnet group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "${var.cluster_name}-rds-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-rds-sg"
  }
}

resource "aws_db_parameter_group" "newstalk_ai" {
  family = "postgres17"
  name   = "${var.cluster_name}-postgres17-params"

  parameter {
    name  = "shared_preload_libraries"
    value = "pg_stat_statements"
  }

  parameter {
    name  = "log_statement"
    value = "all"
  }

  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  parameter {
    name  = "max_connections"
    value = "200"
  }

  tags = {
    Name = "${var.cluster_name}-postgres17-params"
  }
}

resource "aws_db_instance" "newstalk_ai" {
  identifier = "${var.cluster_name}-postgres"

  engine         = "postgres"
  engine_version = "17.4"
  instance_class = var.db_instance_class

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true

  db_name  = "newstalk_ai_prod"
  username = "newstalk_admin"
  password = random_password.db_password.result

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.newstalk_ai.name
  parameter_group_name   = aws_db_parameter_group.newstalk_ai.name

  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"

  skip_final_snapshot = false
  final_snapshot_identifier = "${var.cluster_name}-postgres-final-snapshot"

  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn

  tags = {
    Name = "${var.cluster_name}-postgres"
  }
}

resource "random_password" "db_password" {
  length  = 32
  special = true
}

# IAM role for RDS monitoring
resource "aws_iam_role" "rds_monitoring" {
  name = "${var.cluster_name}-rds-monitoring-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "monitoring.rds.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "rds_monitoring" {
  role       = aws_iam_role.rds_monitoring.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}

# ElastiCache Redis 8.0
resource "aws_elasticache_subnet_group" "newstalk_ai" {
  name       = "${var.cluster_name}-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_security_group" "redis" {
  name_prefix = "${var.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-redis-sg"
  }
}

resource "aws_elasticache_parameter_group" "redis" {
  name   = "${var.cluster_name}-redis8-params"
  family = "redis8.x"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "timeout"
    value = "300"
  }
}

resource "aws_elasticache_replication_group" "newstalk_ai" {
  replication_group_id       = "${var.cluster_name}-redis"
  description                = "NewsTalk AI Redis cluster"

  node_type            = var.redis_node_type
  port                 = 6379
  parameter_group_name = aws_elasticache_parameter_group.redis.name

  num_cache_clusters = 2

  engine_version = "8.0"
  
  subnet_group_name  = aws_elasticache_subnet_group.newstalk_ai.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = random_password.redis_auth_token.result

  automatic_failover_enabled = true
  multi_az_enabled          = true

  snapshot_retention_limit = 5
  snapshot_window         = "03:00-05:00"

  tags = {
    Name = "${var.cluster_name}-redis"
  }
}

resource "random_password" "redis_auth_token" {
  length  = 32
  special = false
}

# MSK Kafka Cluster
resource "aws_msk_configuration" "newstalk_ai" {
  kafka_versions = ["3.6.0"]
  name           = "${var.cluster_name}-msk-config"

  server_properties = <<PROPERTIES
auto.create.topics.enable=false
default.replication.factor=3
min.insync.replicas=2
num.partitions=3
log.retention.hours=168
log.retention.bytes=1073741824
PROPERTIES
}

resource "aws_security_group" "msk" {
  name_prefix = "${var.cluster_name}-msk-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port       = 9092
    to_port         = 9092
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  ingress {
    from_port       = 9094
    to_port         = 9094
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  ingress {
    from_port       = 2181
    to_port         = 2181
    protocol        = "tcp"
    security_groups = [module.eks.cluster_security_group_id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.cluster_name}-msk-sg"
  }
}

resource "aws_msk_cluster" "newstalk_ai" {
  cluster_name           = "${var.cluster_name}-msk"
  kafka_version          = "3.6.0"
  number_of_broker_nodes = 3
  configuration_info {
    arn      = aws_msk_configuration.newstalk_ai.arn
    revision = aws_msk_configuration.newstalk_ai.latest_revision
  }

  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    client_subnets  = module.vpc.private_subnets
    security_groups = [aws_security_group.msk.id]
    storage_info {
      ebs_storage_info {
        volume_size = 100
      }
    }
  }

  encryption_info {
    encryption_at_rest_kms_key_id = aws_kms_key.msk.arn
    encryption_in_transit {
      client_broker = "TLS"
      in_cluster    = true
    }
  }

  logging_info {
    broker_logs {
      cloudwatch_logs {
        enabled   = true
        log_group = aws_cloudwatch_log_group.msk.name
      }
    }
  }

  tags = {
    Name = "${var.cluster_name}-msk"
  }
}

resource "aws_kms_key" "msk" {
  description = "KMS key for MSK encryption"
  
  tags = {
    Name = "${var.cluster_name}-msk-kms"
  }
}

resource "aws_cloudwatch_log_group" "msk" {
  name              = "/aws/msk/${var.cluster_name}"
  retention_in_days = 14

  tags = {
    Name = "${var.cluster_name}-msk-logs"
  }
}

# S3 Bucket for static assets and backups
resource "aws_s3_bucket" "newstalk_ai_assets" {
  bucket = "${var.cluster_name}-assets-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.cluster_name}-assets"
  }
}

resource "aws_s3_bucket" "newstalk_ai_backups" {
  bucket = "${var.cluster_name}-backups-${random_id.bucket_suffix.hex}"

  tags = {
    Name = "${var.cluster_name}-backups"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_versioning" "assets_versioning" {
  bucket = aws_s3_bucket.newstalk_ai_assets.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "backups_versioning" {
  bucket = aws_s3_bucket.newstalk_ai_backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "assets_encryption" {
  bucket = aws_s3_bucket.newstalk_ai_assets.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups_encryption" {
  bucket = aws_s3_bucket.newstalk_ai_backups.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.newstalk_ai.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.newstalk_ai.port
}

output "redis_endpoint" {
  description = "Redis cluster endpoint"
  value       = aws_elasticache_replication_group.newstalk_ai.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis cluster port"
  value       = aws_elasticache_replication_group.newstalk_ai.port
}

output "msk_bootstrap_brokers" {
  description = "MSK bootstrap brokers"
  value       = aws_msk_cluster.newstalk_ai.bootstrap_brokers_tls
  sensitive   = true
}

output "s3_assets_bucket" {
  description = "S3 bucket for assets"
  value       = aws_s3_bucket.newstalk_ai_assets.bucket
}

output "s3_backups_bucket" {
  description = "S3 bucket for backups"
  value       = aws_s3_bucket.newstalk_ai_backups.bucket
} 