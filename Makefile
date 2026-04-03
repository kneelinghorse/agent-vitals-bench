.PHONY: evaluate test lint typecheck clean install

CORPUS ?= v1
DETECTORS ?= all
MIN_CONFIDENCE ?= 0.8

install:
	pip install -e "../agent-vitals[dev]"
	pip install -e ".[dev]"

evaluate:
	python -m evaluator.runner \
		--corpus $(CORPUS) \
		--detectors $(DETECTORS) \
		--min-confidence $(MIN_CONFIDENCE)

test:
	pytest tests/ -v --tb=short

lint:
	ruff check generators/ evaluator/ elicitation/ tests/
	ruff format --check generators/ evaluator/ elicitation/ tests/

typecheck:
	mypy generators/ evaluator/ elicitation/

clean:
	rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
