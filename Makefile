.DEFAULT_GOAL := help
PY := .venv/bin/python
PIP := .venv/bin/pip

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create venv and install dev dependencies (no model weights)
	python3.12 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e '.[dev]'

setup-full: setup ## Install everything including Surya, FAISS, sentence-transformers
	$(PIP) install -e '.[all]'

test: ## Unit tests: no network, no models, no tokens
	$(PY) -m pytest tests -q

lint: ## Ruff lint + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

typecheck: ## mypy strict over src/
	$(PY) -m mypy

check: lint typecheck test ## Everything CI runs

ingest: ## Fetch PDFs, OCR with Surya, detect headers, chunk (slow first run)
	$(PY) -m rag.cli ingest

index: ## Embed staged chunks and build the indexes
	$(PY) -m rag.cli index

eval: ## Deterministic eval metrics against the golden set (free)
	$(PY) -m rag.cli eval

eval-judge: ## Adds LLM-judged faithfulness/correctness (spends tokens)
	$(PY) -m rag.cli eval --judge

headers: ## Score header detection against hand-labelled sections
	$(PY) -m rag.cli headers

.PHONY: help setup setup-full test lint typecheck check ingest index eval eval-judge headers
