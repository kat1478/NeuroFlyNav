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

output "site_bucket" {
  value = aws_s3_bucket.site.bucket
}

output "site_distribution_id" {
  value = aws_cloudfront_distribution.site.id
}

output "site_url" {
  value = "https://${aws_cloudfront_distribution.site.domain_name}"
}
