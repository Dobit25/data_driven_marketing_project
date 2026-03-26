# 🎮 Video Game Concurrent Players Forecasting

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Production-grade time series forecasting pipeline** for predicting concurrent player counts in video games using classical statistical models and deep learning approaches.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Pipeline Stages](#-pipeline-stages)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

This project implements an end-to-end time series forecasting pipeline designed to predict the number of concurrent players for a video game. Accurate forecasting supports critical business decisions such as **server capacity planning**, **marketing campaign timing**, and **player engagement optimization**.

### Key Features

| Feature | Description |
|---|---|
| **Multiple Models** | ARIMA, SARIMA, Prophet, and LSTM implementations |
| **Statistical Testing** | Automated stationarity (ADF, KPSS) and autocorrelation analysis |
| **Comprehensive Diagnostics** | Residual analysis, RMSE, MAE, MAPE evaluation |
| **Configuration-Driven** | YAML-based config — no hardcoded parameters |
| **MLOps-Ready** | Structured for CI/CD, logging, model versioning, and reproducibility |

---

## 📂 Project Structure

```
time_series_project/
│
├── configs/                     # Configuration files
│   └── config.yaml              # Model parameters and data paths
│
├── data/                        # Data storage (gitignored)
│   ├── raw/                     # Original, immutable data
│   ├── interim/                 # Intermediate transformed data
│   └── processed/               # Final datasets for modeling
│
├── logs/                        # Application and training logs
│
├── models/                      # Serialized / trained model artifacts
│
├── notebooks/                   # Jupyter notebooks for research & EDA
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
│
├── reports/                     # Generated analysis reports
│   └── figures/                 # Saved plots and visualizations
│
├── src/                         # Production source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── make_dataset.py      # Data ingestion and loading
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py    # Preprocessing & statistical tests
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_model.py       # Model training pipeline
│   │   └── predict_model.py     # Prediction & evaluation
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py         # Plotting utilities
│
├── tests/                       # Unit and integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/time_series_project.git
cd time_series_project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Quick Start

```bash
# 1. Ingest raw data
python -m src.data.make_dataset configs/config.yaml

# 2. Build features & run statistical tests
python -m src.features.build_features configs/config.yaml

# 3. Train model
python -m src.models.train_model configs/config.yaml

# 4. Generate predictions & diagnostics
python -m src.models.predict_model configs/config.yaml
```

---

## 🔄 Pipeline Stages

### 1. Data Ingestion (`src/data/make_dataset.py`)
Loads raw CSV/API data, validates schema, and saves to `data/raw/`.

### 2. Feature Engineering (`src/features/build_features.py`)
- Time-based feature extraction (hour, day-of-week, month)
- Stationarity tests (ADF, KPSS)
- Autocorrelation & partial autocorrelation analysis (ACF/PACF)
- Differencing and transformations

### 3. Model Training (`src/models/train_model.py`)
- Supports ARIMA, SARIMA, Prophet, and LSTM
- Configurable hyperparameters via `configs/config.yaml`
- Automatic model serialization to `models/`

### 4. Evaluation & Diagnostics (`src/models/predict_model.py`)
- Forecast generation on holdout set
- Residual diagnostics (Ljung-Box test, normality checks)
- Metrics: RMSE, MAE, MAPE
- Visualization export to `reports/figures/`

---

## ⚙️ Configuration

All parameters are managed through `configs/config.yaml`. See the file for full documentation of available options including:

- Data file paths and column mappings
- Train/test split ratio
- Model-specific hyperparameters (ARIMA order, SARIMA seasonal order, Prophet settings, LSTM architecture)
- Logging level and output paths

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
