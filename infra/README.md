# Infrastructure (AWS, Terraform)

Everything the cloud sweeps and the web-app mirror need, in one AWS project (eu-north-1). See
[docs/cloud.md](../docs/cloud.md) for how jobs run and [docs/app.md](../docs/app.md) for the app.

| File | Resources |
| --- | --- |
| `budget.tf` | Monthly cost budget, e-mail alerts at 50/80/100% (actual) and 100% (forecast) |
| `storage.tf` | Private S3 results bucket (encrypted, `runs/` expire after 90 days), ECR repository (scan on push, last 10 images), CloudWatch log group (14 days) |
| `iam.tf` | Execution role (pull image, write logs) and job role (`s3:PutObject` on `runs/*` only) |
| `batch.tf` | Security group (outbound only), Fargate Spot compute environment (max 12 vCPU, scales to zero), job queue, job definition (4 vCPU, 8 GB, 3 h timeout, retry only on Spot interruption) |
| `site.tf` | Web-app mirror: private S3 site bucket, CloudFront distribution with Origin Access Control, HTTPS only, AWS-managed caching and security-header policies |
| `variables.tf`, `outputs.tf`, `versions.tf` | Inputs (`image_tag`, `image_repository`, `budget_email`, sizes), outputs (bucket, ECR URL, queue, job definition, site URL), provider and default tags |

All resources carry the tags `project=neuroflynav` and `managed_by=terraform`.
State is local and gitignored; nothing secret is committed.

## Use

```bash
aws login --profile neuroflynav --region eu-north-1
# Terraform reads short-lived credentials from the environment
eval "$(aws configure export-credentials --profile neuroflynav --format env)"
cp infra/terraform.tfvars.example infra/terraform.tfvars   # set budget_email
terraform -chdir=infra init
terraform -chdir=infra plan
terraform -chdir=infra apply      # only after reviewing the plan
terraform -chdir=infra destroy    # removes everything, including stored images and results
```

Idle cost is close to zero: Fargate scales to zero; only stored images (~0.1 USD/month per
image, about 1 GB compressed), results and the small site bucket cost money, and CloudFront
traffic at portfolio scale stays within the free allowance.
