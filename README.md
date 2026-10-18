# EpiWatch — AI-Powered Epidemiological Intelligence Platform

EpiWatch is a comprehensive epidemiological intelligence platform designed to detect disease outbreaks, forecast trajectories, assess regional health risks, and analyze healthcare preparedness.

## Implementation Status

- [x] Phase 1: Data ingestion, validation, and feature engineering
- [x] Phase 2: EDA and descriptive epidemiology
- [x] Phase 3: Early outbreak detection
- [x] Phase 4: Disease trajectory forecasting
- [x] Phase 5: Mathematical Modeling (SIR/SIRD)
- [x] Phase 6: Environmental Risk Modeling
- [x] Phase 7: Healthcare Capacity & Preparedness Assessment
- [x] Phase 8: Spatial Analytics & Regional Clustering
- [x] Phase 9: Integrated Regional Risk Engine
- [x] Phase 10: Explainable AI (SHAP)
- [x] Phase 11: Evidence-Based Recommendation Engine
- [x] Phase 12: AI Epidemiology Agent
- [x] Phase 13: Intelligent Alerts Engine
- [x] Phase 14: Stakeholder Dashboards (Streamlit)
- [x] Phase 15: API & Deployment Configuration

## Project Architecture

The platform follows a layered architecture to cleanly separate analytical workflows from presentation and API serving:

1. **Analytical Core (`src/analytics`, `src/epidemiology`, `src/risk_engine`, etc.)**: Handles all deterministic modeling, environmental checks, AI predictions, and SHAP explainability locally. Outputs are persisted securely as flat files.
2. **Service & Alert Layer (`src/alerts`, `src/recommendations`)**: Monitors the core metrics asynchronously via configuration rules and generates structured operational data.
3. **API Layer (`src/api/`)**: A fast, lightweight programmatic REST interface implemented with FastAPI to access the aggregated analytical evidence externally.
4. **Dashboard Layer (`src/dashboard`, `pages/`)**: Interactive visualizations driven by Streamlit for 4 key stakeholder personas (Citizen, Healthcare, Authority, Research).
5. **AI Agent (`src/agent/`)**: An orchestrator LLM pipeline that connects user conversational context securely to verified platform tools.

## Running the Application

### 1. Dashboards
The interactive web dashboards can be launched via Streamlit:
```bash
streamlit run app.py
```

### 2. FastAPI Endpoints
To serve the programmatic endpoints (e.g., `/api/v1/regions`, `/api/v1/alerts`), run:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Docker Deployment
The project is containerized for simple end-to-end deployments:
```bash
docker build -t epiwatch .
docker run -p 8501:8501 -p 8000:8000 epiwatch
```

*Note: Environment variables for LLMs must be provided inside `.env`.*

---

## Phase 1 — Data Engineering Foundation


```
EpiWatch/
│
├── config.yaml                     # Central YAML pipeline configuration
├── conftest.py                      # Pytest path discovery configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Phase 1 documentation
├── .gitignore                      # Git ignore rules
│
├── Dataset/                        # Raw Bhopal source CSV dataset pack
│   ├── 01_demographics_bhopal.csv
│   ├── 02_environmental_daily_bhopal.csv
│   ├── 03_healthcare_monthly_bhopal.csv
│   ├── 04_interventions_daily_bhopal.csv
│   ├── 05_epidemiological_daily_bhopal.csv
│   ├── 06_disease_transmission_parameters.csv
│   └── 07_epiwatch_derived_features_bhopal.csv
│
├── data/
│   ├── raw/                        # Target directory for raw data ingestion
│   ├── processed/                  # Cleaned, standardized CSV outputs
│   ├── features/                   # Category & master feature engineered datasets
│   └── reference/                  # Reference schemas and parameter tables
│
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py               # Reusable CSV loader with metadata logging
│   │   └── validator.py            # Schema, data type, missingness & range validator
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py              # Column standardizer, whitespace stripper & date parser
│   │   └── missing_values.py       # Domain-aware missing value imputation
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── epidemiological.py      # Rolling averages, growth rates, CFR, incidence per 100k
│   │   ├── environmental.py        # Environmental lag & rolling statistics
│   │   ├── healthcare.py           # Beds/doctors per 100k, occupancy & utilization ratios
│   │   └── temporal.py             # Temporal components and seasonal categorization
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── data_pipeline.py        # End-to-end data pipeline orchestrator
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration manager
│       └── logging_config.py       # Structured logger
│
├── tests/
│   ├── test_loader.py              # Loader & exception tests
│   ├── test_validator.py           # Validation & range check tests
│   ├── test_cleaner.py             # Cleaning & snake_case tests
│   └── test_features.py            # Feature engineering & data leakage tests
│
├── reports/
│   └── data_quality/               # JSON reports and data_quality_summary.csv
│
└── notebooks/                      # Exploratory notebooks
```

---

## 2. Dataset Categories

The platform processes five core data categories anchored to **Bhopal, Madhya Pradesh, India**:

1. **Epidemiological Data**: Daily case counts (`new_cases`, `active_cases`, `recovered`), deaths, hospitalizations, ICU admissions, tests, and test positivity rate across Dengue, Malaria, Influenza, COVID-19, and Typhoid.
2. **Demographic Data**: Population anchors (Census 2011 anchor: 2,368,145), population density, urbanization rate, sex ratio, literacy, and age distribution.
3. **Environmental Data**: Daily temperature (°C), rainfall (mm), relative humidity (%), air quality index (AQI), wind speed, solar radiation, water accumulation, and pollution indicators.
4. **Healthcare Capacity Data**: Monthly metrics for hospital beds, ICU beds, doctors, testing capacity (tests/day), oxygen availability (%), medicine availability (%), and emergency units.
5. **Historical / Derived Data**: Historical rolling averages, growth rates, incidence rates, lagged weather indicators, and seasonal features.

---

## 3. Data Flow & Processing Pipeline

```
RAW CSV Datasets (Dataset/*.csv)
       │
       ▼
   CSV Loader (src/ingestion/loader.py)
       │
       ▼
   Schema & Data Validation (src/ingestion/validator.py)
       │
       ▼
   Data Quality Reporting (reports/data_quality/*.json & summary.csv)
       │
       ▼
   Cleaning & Standardization (src/preprocessing/cleaner.py)
       │
       ▼
   Domain-Aware Imputation (src/preprocessing/missing_values.py)
       │
       ▼
   Feature Engineering (src/features/*.py)
       │ (Preventing Future Data Leakage)
       ▼
   Safe Row-Preserving Joining (src/pipeline/data_pipeline.py)
       │
       ▼
   Processed CSVs & Master Feature Store (data/processed/ & data/features/)
```

---

## 4. How to Run the Pipeline

### Prerequisites
Ensure Python 3.11+ is installed along with dependencies:
```bash
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
Execute the pipeline module from the project root:
```bash
python -m src.pipeline.data_pipeline
```

### Running Unit Tests
To verify all ingestion, validation, cleaning, feature generation, and leakage prevention tests:
```bash
python -m pytest -v
```

---

## 5. How Validation & Quality Reporting Works

Validation is managed by `DataValidator` in `src/ingestion/validator.py`:
- **Schema Validation**: Ensures all required columns exist for each dataset. Unexpected columns generate logged warnings without crashing execution.
- **Data Type & Date Checks**: Confirms dates are parseable to `datetime64[ns]` and numeric columns contain valid numbers.
- **Missingness Reporting**: Calculates missing cell counts and percentage missing per column.
- **Duplicate Records Detection**: Detects duplicates on key combinations (e.g. `['date', 'region', 'disease']`).
- **Domain Range & Negative Value Checks**: Flags negative case counts, negative deaths, negative population/beds, and humidity percentages outside `[0.0, 100.0]`. Invalid rows are logged and reported rather than silently deleted.

Outputs generated:
- `reports/data_quality/epidemiological_quality_report.json`
- `reports/data_quality/environmental_quality_report.json`
- `reports/data_quality/healthcare_quality_report.json`
- `reports/data_quality/demographics_quality_report.json`
- `reports/data_quality/data_quality_summary.csv`

---

## 6. How Features Are Generated

### A. Epidemiological Features (`src/features/epidemiological.py`)
- **Rolling Averages**: 7-day, 14-day, and 28-day rolling averages of cases (`cases_7d_avg`, `cases_14d_avg`, `cases_28d_avg`) calculated per `['region', 'disease']`.
- **Growth Rates**:
  $$\text{growth\_rate\_7d} = \frac{\text{current\_cases} - \text{cases\_7d\_ago}}{\text{cases\_7d\_ago}} \times 100$$
- **Incidence**:
  $$\text{incidence\_per\_100k} = \frac{\text{new\_cases}}{\text{population}} \times 100{,}000$$
- **Case Fatality Rate (CFR)**:
  $$\text{CFR} = \frac{\text{deaths}}{\text{confirmed\_cases}} \times 100$$
- **Hospitalization & ICU Rates**: Ratios of hospitalizations to cases, and ICU admissions to hospitalizations.

### B. Temporal Features (`src/features/temporal.py`)
- `year`, `month`, `week`, `day_of_year`, `quarter`.
- `season` categorization for India/Bhopal:
  - **Winter**: December, January, February
  - **Summer**: March, April, May
  - **Monsoon**: June, July, August, September
  - **Post-monsoon**: October, November

### C. Environmental Features (`src/features/environmental.py`)
- **Lag Features**: Lags at 7, 14, and 21 days for rainfall (`rainfall_lag_7`, `rainfall_lag_14`, `rainfall_lag_21`), 7 and 14 days for humidity (`humidity_lag_7`, `humidity_lag_14`), and 7 days for temperature (`temperature_lag_7`).
- **Rolling Statistics**: `rainfall_7d`, `rainfall_14d`, `humidity_7d_avg`, `temperature_7d_avg`.

### D. Healthcare Features (`src/features/healthcare.py`)
- **Normalized Capacity**: `beds_per_100k`, `icu_beds_per_100k`, `doctors_per_100k`.
- **Utilization Ratios**: `bed_occupancy_ratio`, `icu_occupancy_ratio`, `testing_utilization` (`tests / testing_capacity`).

---

## 7. Data Leakage Prevention

The feature engineering pipeline strictly guarantees **no future data leakage**:
1. All data is chronologically sorted prior to feature computation.
2. Rolling window computations use historical observations only (closed past windows).
3. Lag features use `.shift(N)` with positive integer offsets.
4. Forecast targets for day $t$ rely exclusively on information available up to day $t-1$ or day $t$.
5. Unit tests (`tests/test_features.py::test_no_future_data_leakage`) explicitly verify that mutating future observations ($t+N$) leaves past feature values completely unaltered.

---

## 8. Data Provenance & Real vs. Synthetic Distinction

To preserve scientific rigor, EpiWatch maintains explicit provenance tracking via the `data_status` metadata attribute:

- **`REAL_HISTORICAL`**: Official reference data (e.g. Census 2011 demographic anchors for Bhopal: population 2,368,145; area 2,772 sq km; density 855/sq km).
- **`SOURCE_CALIBRATED_SYNTHETIC`**: Calibrated synthetic time series generated for epidemiological daily cases, weather observations, and monthly healthcare capacity.
- **`SYNTHETIC`**: General synthetic test scenarios.

*Synthetic data is never converted into or represented as official public health measurements.*

---

## 9. Generated Output Files

### Cleaned Datasets (`data/processed/`)
- `epidemiological_clean.csv`
- `environmental_clean.csv`
- `healthcare_clean.csv`
- `demographics_clean.csv`

### Feature Engineered Datasets (`data/features/`)
- `epidemiological_features.csv`
- `environmental_features.csv`
- `healthcare_features.csv`
- `epiwatch_master_features.csv` (Master feature table joining epidemiological, environmental, healthcare, and demographic features at daily granularity).
