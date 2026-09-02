PYTHON := uv run python
VIDEO ?= data/sample/sample.mp4
CALIBRATION ?= data/sample/calibration.json
OUTPUT ?= data/processed/court_preview.png
TRACKING_VIDEO ?= data/processed/player_tracking.mp4
TRACKING_EVENTS ?= data/processed/player_tracking.jsonl

.PHONY: install lint format test process-video track-players

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

track-players:
	$(PYTHON) -m courtvision track-players \
		--video "$(VIDEO)" \
		--calibration "$(CALIBRATION)" \
		--output-video "$(TRACKING_VIDEO)" \
		--output-events "$(TRACKING_EVENTS)"
