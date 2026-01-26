
# Credit Risk & Lending System

A comprehensive system for credit default prediction, loan assessment, and risk management.

## Setup & usage

### Prerequisites
- Python 3.11+
- [Docker](https://www.docker.com/) (optional, for containerized execution)

**macOS Users**: You must install `libomp` for LightGBM to work:
```bash
brew install libomp
```

### 1. Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Locally

#### Backend API
Start the FastAPI backend:
```bash
uvicorn src.app.main:app --reload --port 8000
```
The API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

You can also run a quick local test script:
```bash
python scripts/run_local_test.py
```

#### Frontend Dashboard
Start the Streamlit application:
```bash
streamlit run src/frontend/app.py
```
Access the dashboard at [http://localhost:8501](http://localhost:8501).

### 3. Running with Docker

Build and run the entire stack (API):

```bash
docker-compose up --build
```
The API will be available at [http://localhost:8000](http://localhost:8000).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── app                     <- FastAPI application
    │   ├── main.py             <- App entry point
    │   └── api                 <- API routes
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── frontend                <- Streamlit dashboard
    │   └── app.py
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

