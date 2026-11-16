# Infrastructure (AWS, Terraform)

Everything the M3 cloud sweep needs, in one AWS project (eu-north-1). See
[docs/cloud.md](../docs/cloud.md) for how jobs run.

| File | Resources |
| --- | --- |
| `budget.tf` | Monthly cost budget, e-mail alerts at 50/80/100% (actual) and 100% (forecast) |
| `storage.tf` | Private S3 results bucket (encrypted, `runs/` expire after 90 days), ECR repository (scan on push, last 10 images), CloudWatch log group (14 days) |
| `iam.tf` | Execution role (pull image, write logs) and job role (`s3:PutObject` on `runs/*` only) |
| `batch.tf` | Security group (outbound only), Fargate Spot compute environment (max 12 vCPU, scales to zero), job queue, job definition (4 vCPU, 8 GB, 3 h timeout, retry only on Spot interruption) |

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

Idle cost is close to zero: Fargate scales to zero; only stored images (~0.3 USD/month for
one 3.4 GB image) and results cost money.
