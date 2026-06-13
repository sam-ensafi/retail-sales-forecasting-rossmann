# Retail Sales Forecasting (Rossmann)

Data science major research project forecasting daily sales for Rossmann
drug stores using historical sales, store, and promotion data.

## Project structure

- `notebooks/` exploratory data analysis
- `src/` reusable Python code (preprocessing, features, models, evaluation)
- `data/` raw and processed data (not tracked in git)
- `models/` saved model artifacts (not tracked in git)
- `reports/figures/` generated figures
- `results/` metrics and model comparison tables

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The Rossmann dataset is not included in this repo. Download the files and
place them in `data/raw/`.
