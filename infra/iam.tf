data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Execution role: used by Fargate itself to pull the image and write logs.
resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Job role: what the code inside the container may do. Only write results under runs/.
resource "aws_iam_role" "job" {
  name               = "${var.name}-job"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "job" {
  statement {
    actions   = ["s3:PutObject"]
    resources = ["${aws_s3_bucket.results.arn}/runs/*"]
  }
}

resource "aws_iam_role_policy" "job" {
  name   = "write-results"
  role   = aws_iam_role.job.id
  policy = data.aws_iam_policy_document.job.json
}
