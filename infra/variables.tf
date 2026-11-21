variable "region" {
  description = "AWS Region of the project (fixed by the AWS project: eu-north-1)."
  type        = string
  default     = "eu-north-1"
}

variable "name" {
  description = "Prefix for resource names."
  type        = string
  default     = "neuroflynav"
}

variable "budget_email" {
  description = "Address for budget alerts. Set it in terraform.tfvars (gitignored)."
  type        = string
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget; alerts at 50, 80 and 100 percent."
  type        = number
  default     = 20
}

variable "max_vcpus" {
  description = "Upper limit of vCPUs running at once (Fargate Spot quota is 12)."
  type        = number
  default     = 12
}

variable "job_vcpus" {
  description = "vCPUs per job."
  type        = number
  default     = 4
}

variable "job_memory_mib" {
  description = "Memory per job in MiB (Fargate: 8192-30720 for 4 vCPU)."
  type        = number
  default     = 8192
}

variable "job_timeout_hours" {
  description = "A job running longer than this is stopped."
  type        = number
  default     = 3
}

variable "image_tag" {
  description = "Image tag used by the job definition (the git SHA pushed to ECR)."
  type        = string
  default     = "latest"
}

variable "image_repository" {
  description = "Image repository for the job definition; empty = the project ECR repository. Images built by GitHub Actions: ghcr.io/kat1478/neuroflynav (public)."
  type        = string
  default     = ""
}
