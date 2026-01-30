# Makefile for Stock Market Prediction Project

.PHONY: help install test clean run-api docker-build docker-run lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-coverage - Run tests with coverage"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean temporary files"
	@echo "  make run-api       - Run Flask API"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make quickstart    - Run quickstart demo"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	pylint src/
	flake8 src/

# Format code
format:
	black src/ tests/
	black *.py

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/

# Run Flask API
run-api:
	python flask_app.py

# Build Docker image
docker-build:
	docker build -t stock-market-predictor .

# Run Docker container
docker-run:
	docker run -p 5000:5000 stock-market-predictor

# Docker compose up
docker-compose-up:
	docker-compose up -d

# Docker compose down
docker-compose-down:
	docker-compose down

# Run quickstart
quickstart:
	python quickstart.py

# Setup project
setup:
	pip install -e .
	mkdir -p logs models data/raw data/processed

# Run MLOps pipeline
mlops:
	python mlops_pipeline.py
