# EpiWatch Phase 4 — Disease Trajectory Forecasting Report

## 1. Executive Summary

Predicts multi-horizon disease case trajectories (7, 14, 28 days) using baseline,
classical (SARIMA/SARIMAX/Prophet), supervised ML (XGBoost), and optional deep learning (PyTorch LSTM/GRU).
Provides 95% prediction intervals and future surge warnings.

## 2. Selected Models per Region + Disease + Horizon

| Region | Disease | Horizon | Selected Model | Validation MAE | Validation RMSE |
|--------|---------|---------|----------------|----------------|-----------------|
| Bhopal | COVID-19 | 7d | MovingAverage_7 | 2.5817 | 3.4341 |
| Bhopal | COVID-19 | 14d | XGBoost | 4.0498 | 5.1756 |
| Bhopal | COVID-19 | 28d | MovingAverage_7 | 4.0128 | 5.0116 |
| Bhopal | Dengue | 7d | MovingAverage_14 | 3.4082 | 4.0471 |
| Bhopal | Dengue | 14d | Naive | 3.3929 | 4.188 |
| Bhopal | Dengue | 28d | XGBoost | 3.3317 | 3.9494 |
| Bhopal | Influenza | 7d | GRU | 6.0021 | 7.4492 |
| Bhopal | Influenza | 14d | Naive | 4.6786 | 5.8642 |
| Bhopal | Influenza | 28d | LSTM | 5.0149 | 6.2713 |
| Bhopal | Malaria | 7d | LSTM | 2.4565 | 3.0263 |
| Bhopal | Malaria | 14d | LSTM | 2.7756 | 3.2131 |
| Bhopal | Malaria | 28d | XGBoost | 2.8018 | 3.3982 |
| Bhopal | Typhoid | 7d | XGBoost | 2.3725 | 3.0518 |
| Bhopal | Typhoid | 14d | GRU | 2.1456 | 2.6509 |
| Bhopal | Typhoid | 28d | GRU | 2.0245 | 2.6379 |

## 3. Scientific Distinctions

- **Observed Data**: Actual historical cases.
- **Outbreak Detection (Phase 3)**: Identifies if current observations are abnormal.
- **Trajectory Forecast (Phase 4)**: Predicts expected future cases.
- **Forecast Surge Signal**: Flags potential future trajectory exceeding expected baseline (+20%+).
