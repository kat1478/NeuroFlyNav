.PHONY: quality

quality:
	python -m pytest
	ruff format --check .
	ruff check .
	python -m mypy src tests

.PHONY: deploy-aws
# Upload app/dist to the AWS mirror and refresh the CDN (needs `aws login` and an applied infra/).
# --delete removes files from the bucket that are no longer in app/dist (old hashed bundles).
deploy-aws:
	cd app && npm run build
	aws s3 sync app/dist "s3://$$(terraform -chdir=infra output -raw site_bucket)" --delete
	aws cloudfront create-invalidation --paths "/*" \
		--distribution-id "$$(terraform -chdir=infra output -raw site_distribution_id)" > /dev/null
	@echo "Deployed: $$(terraform -chdir=infra output -raw site_url)"
