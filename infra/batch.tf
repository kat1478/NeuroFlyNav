# Network: the default VPC. Jobs get a public IP to reach ECR and the hemibrain download
# (no NAT gateway, which would cost money every hour). Inbound traffic is not allowed.
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

resource "aws_security_group" "jobs" {
  name        = "${var.name}-jobs"
  description = "Batch jobs: outbound only"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Compute environment: Fargate Spot, scales to zero when idle.
resource "aws_batch_compute_environment" "spot" {
  name = "${var.name}-fargate-spot"
  type = "MANAGED"

  compute_resources {
    type               = "FARGATE_SPOT"
    max_vcpus          = var.max_vcpus
    subnets            = data.aws_subnets.default.ids
    security_group_ids = [aws_security_group.jobs.id]
  }
}

resource "aws_batch_job_queue" "jobs" {
  name     = var.name
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_definition" "trained" {
  name                  = "${var.name}-trained"
  type                  = "container"
  platform_capabilities = ["FARGATE"]

  timeout {
    attempt_duration_seconds = var.job_timeout_hours * 3600
  }

  # Retry only when Spot capacity is taken away; real errors fail at once.
  retry_strategy {
    attempts = 3
    evaluate_on_exit {
      on_status_reason = "Your Spot Task was interrupted*"
      action           = "RETRY"
    }
    evaluate_on_exit {
      on_reason = "*"
      action    = "EXIT"
    }
  }

  container_properties = jsonencode({
    image            = "${aws_ecr_repository.jobs.repository_url}:${var.image_tag}"
    executionRoleArn = aws_iam_role.execution.arn
    jobRoleArn       = aws_iam_role.job.arn
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.job_vcpus) },
      { type = "MEMORY", value = tostring(var.job_memory_mib) },
    ]
    environment = [
      { name = "BUCKET", value = aws_s3_bucket.results.bucket },
      { name = "OMP_NUM_THREADS", value = tostring(var.job_vcpus) },
    ]
    networkConfiguration         = { assignPublicIp = "ENABLED" }
    fargatePlatformConfiguration = { platformVersion = "LATEST" }
    runtimePlatform = {
      operatingSystemFamily = "LINUX"
      cpuArchitecture       = "X86_64"
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.jobs.name
        "awslogs-region"        = var.region
        "awslogs-stream-prefix" = "trained"
      }
    }
  })
}
