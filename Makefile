.PHONY: help install train test run-api run-dashboard docker-build clean

help:
	@echo "Credit Risk & Lending System Commands:"
	@echo "  make install        Install Python dependencies"
	@echo "  make train          Train multi-model suite and generate benchmark JSON"
	@echo "  make test           Run Pytest unit test suite"
	@echo "  make run-api        Start FastAPI backend service on port 8000"
	@echo "  make run-dashboard  Start Streamlit regulatory dashboard on port 8501"
	@echo "  make docker-build   Build Docker container image"

install:
	pip install -r requirements.txt

train:
	python -m src.modeling.train

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run-api:
	uvicorn src.app.main:app --reload --port 8000

run-dashboard:
	streamlit run src/frontend/app.py

docker-build:
	docker build -t credit-risk-system:latest .

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
