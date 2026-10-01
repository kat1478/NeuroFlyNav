.PHONY: quality

quality:
	python -m pytest
	ruff format --check .
	ruff check .
	python -m mypy src tests
