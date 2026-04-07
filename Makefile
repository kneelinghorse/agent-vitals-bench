.PHONY: evaluate test lint typecheck clean install elicit-cross-model five-paradigm-comparative

CORPUS ?= v1
DETECTORS ?= all
MIN_CONFIDENCE ?= 0.8
PROFILE ?=
# The five-paradigm comparative report needs torch + gtda + agent-vitals editable.
# The sibling tda-experiment Python 3.12 venv has the full stack; bench's main
# venv (Python 3.14) does not. Override with FIVE_PARADIGM_PYTHON=... to use a
# different interpreter.
FIVE_PARADIGM_PYTHON ?= ../tda-experiment/.venv/bin/python

install:
	pip install -e "../agent-vitals[dev]"
	pip install -e ".[dev]"

evaluate:
	python -m evaluator.runner \
		--corpus $(CORPUS) \
		--detectors $(DETECTORS) \
		--min-confidence $(MIN_CONFIDENCE) \
		$(if $(PROFILE),--profile $(PROFILE))

test:
	pytest tests/ -v --tb=short

lint:
	ruff check generators/ evaluator/ elicitation/ prototypes/ tests/
	ruff format --check generators/ evaluator/ elicitation/ prototypes/ tests/

typecheck:
	mypy generators/ evaluator/ elicitation/ prototypes/

elicit-cross-model:
	python scripts/elicit_cross_model.py \
		--tiers frontier mid-range volume \
		--detectors confabulation thrash runaway_cost \
		--traces-per-tier 20

five-paradigm-comparative:
	PYTHONPATH=$(CURDIR) $(FIVE_PARADIGM_PYTHON) scripts/run_five_paradigm_comparative.py

clean:
	rm -rf __pycache__ .mypy_cache .ruff_cache .pytest_cache *.egg-info dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
