output "bucket" {
  value = aws_s3_bucket.results.bucket
}

output "ecr_repository_url" {
  value = aws_ecr_repository.jobs.repository_url
}

output "job_queue" {
  value = aws_batch_job_queue.jobs.name
}

output "job_definition" {
  value = aws_batch_job_definition.trained.name
}
