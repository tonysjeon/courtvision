PYTHON := uv run python
VIDEO ?= data/sample/sample.mp4
CALIBRATION ?= data/sample/calibration.json
OUTPUT ?= data/processed/court_preview.png

.PHONY: install lint format test process-video

install:
	uv sync --dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

process-video:
	$(PYTHON) -m courtvision process-video \
		--video "$(VIDEO)" \
		--calibration "$(CALIBRATION)" \
		--output "$(OUTPUT)"
