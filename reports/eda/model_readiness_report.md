# EpiWatch — Model Readiness Report

## 1. Which diseases have sufficient history?
All 5 diseases in the dataset (**Dengue, Malaria, Influenza, COVID-19, Typhoid**) have 5 complete years of daily observations (2021–2025, 1,826 daily entries per disease, 9,130 total daily rows). This provides sufficient time-series depth for weekly and monthly sequence modelling.

## 2. Which diseases have strong seasonality?
- **Dengue**: Strong monsoon & post-monsoon seasonality (peak cases in July–October, driven by rainfall & humidity lags).
- **Malaria**: Moderate-to-high monsoon seasonality (July–September).
- **Influenza**: Winter and monsoon bimodal seasonal peaks (December–February and July).
- **COVID-19 & Typhoid**: Lower overall seasonal indexing, showing multi-wave and year-round persistence.

## 3. Which diseases have too many missing observations?
None. The cleaning pipeline verified 0 missing cell gaps across all daily epidemiological observations.

## 4. Which environmental variables are available?
- Temperature (°C), Daily Rainfall (mm), Relative Humidity (%), Air Quality Index (AQI), Wind Speed (km/h), Water Accumulation Index, Solar Radiation (kWh/m²), Surface Water Index, Pollution Index.

## 5. Which lag periods appear potentially useful?
- **Rainfall**: 7-day, 14-day, and 21-day lags show strong positive Spearman correlation ($r_s > 0.45$) with Dengue and Malaria vector-borne outbreaks.
- **Humidity**: 7-day and 14-day lags correlate significantly with vector breeding windows.
- **Temperature**: 7-day lag provides consistent baseline environmental signals.

## 6. Are there enough observations for forecasting?
Yes. With 1,826 daily observations (260 weekly periods per disease), there are sufficient historical sequences for time-series forecasting models (SARIMA, Prophet, LSTM, XGBoost).

## 7. Are there suspicious data gaps?
No date gaps were found. Complete daily continuity is maintained from 2021-01-01 to 2025-12-31.

## 8. Are there class-label problems for future outbreak detection?
Statistical anomaly labeling (rolling 28-day mean + 2 * std) yields approximately 4–7% positive anomaly days per disease, providing a well-balanced candidate outbreak class ratio for supervised classification.

## 9. Which features are highly correlated?
- `hospitalizations` and `deaths` ($r > 0.85$).
- `rainfall_mm` and `rainfall_7d` ($r > 0.80$).
- `beds_per_100k` and `icu_beds_per_100k` ($r > 0.95$).

## 10. Which features may cause data leakage?
- Same-day `recovered` or same-day `deaths` if predicting current-day `new_cases`.
- Unshifted future rolling averages.
- *Mitigation*: The feature engineering pipeline strictly uses past-shifted `.shift(N)` lags and historical-only rolling windows (`closed='left'`).
