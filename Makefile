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
	app/scripts/fetch-models.sh models-v1
	cd app && npm run build
	# CloudFront compresses only files under 10 MB and not application/octet-stream, so the
	# runtime (.wasm, ~15 MB) and the models (.onnx) are uploaded pre-gzipped (~26 MB -> ~9 MB).
	rm -rf app/dist-aws
	cp -r app/dist app/dist-aws
	find app/dist-aws \( -name "*.wasm" -o -name "*.onnx" \) -exec gzip -9n {} \; \
		-exec sh -c 'mv "$$1.gz" "$$1"' _ {} \;
	aws s3 sync app/dist-aws "s3://$$(terraform -chdir=infra output -raw site_bucket)" --delete \
		--exclude "*.wasm" --exclude "*.onnx"
	aws s3 sync app/dist-aws "s3://$$(terraform -chdir=infra output -raw site_bucket)" --delete \
		--exclude "*" --include "*.wasm" --content-encoding gzip --content-type application/wasm
	aws s3 sync app/dist-aws "s3://$$(terraform -chdir=infra output -raw site_bucket)" --delete \
		--exclude "*" --include "*.onnx" --content-encoding gzip --content-type application/octet-stream
	aws cloudfront create-invalidation --paths "/*" \
		--distribution-id "$$(terraform -chdir=infra output -raw site_distribution_id)" > /dev/null
	@echo "Deployed: $$(terraform -chdir=infra output -raw site_url)"
