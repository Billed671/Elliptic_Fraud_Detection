.PHONY: install test run run-classical clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

run:
	elliptic-fraud --models all

run-classical:
	elliptic-fraud --models classical

clean:
	rm -rf outputs .pytest_cache .ruff_cache
