.PHONY: install install-dev test test-unit test-integration lint format typecheck clean coverage help

PYTHON  := python3
PIP     := $(PYTHON) -m pip
PYTEST  := $(PYTHON) -m pytest
RUFF    := $(PYTHON) -m ruff
MYPY    := $(PYTHON) -m mypy
COV_MIN := 90

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in editable mode (runtime deps only)
	$(PIP) install -e .

install-dev:  ## Install the package with all dev dependencies
	$(PIP) install -e ".[dev]"

test:  ## Run the full test suite with coverage
	$(PYTEST) tests/ \
	  --cov=src/sepflows \
	  --cov-report=term-missing \
	  --cov-report=xml:coverage.xml \
	  --cov-fail-under=$(COV_MIN) \
	  -v

test-unit:  ## Run only unit tests (fast, no integration)
	$(PYTEST) tests/unit/ -v

test-integration:  ## Run only integration tests
	$(PYTEST) tests/integration/ -v -m integration

lint:  ## Run ruff linter checks (no auto-fix)
	$(RUFF) check src/ tests/

format:  ## Auto-format and fix lint issues with ruff
	$(RUFF) format src/ tests/
	$(RUFF) check --fix src/ tests/

typecheck:  ## Run mypy static type analysis
	$(MYPY) src/sepflows/

coverage:  ## Generate HTML coverage report and open it
	$(PYTEST) tests/ --cov=src/sepflows --cov-report=html:htmlcov -q
	@echo "Coverage report written to htmlcov/index.html"

clean:  ## Remove all build artifacts, caches, and coverage files
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov/ coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docs:  ## Build Sphinx HTML documentation
	cd docs && $(MAKE) html

release-check:  ## Dry-run package build (check metadata before publishing)
	$(PYTHON) -m build --no-isolation
	$(PYTHON) -m twine check dist/*
