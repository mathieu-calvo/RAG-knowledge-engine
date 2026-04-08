.PHONY: install install-dev install-all test lint format app notebooks clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[dev,app,eval]"

test:
	pytest -v -m "not requires_api_key"

test-all:
	pytest -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf chroma_db/ .pytest_cache/ .mypy_cache/ .ruff_cache/ dist/ build/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
