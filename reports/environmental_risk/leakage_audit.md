# Temporal Leakage Audit

**Objective:** Ensure that the environmental risk model and feature engineering pipeline do not inadvertently use future information to predict past or present disease cases.

## Evaluated Features

| Feature Name | Source Data | Temporal Availability | Leakage Risk | Mitigation Strategy |
|--------------|-------------|-----------------------|--------------|---------------------|
| `temperature_c_lag_X` | Observations | `t - X` days | LOW | Uses pandas `.shift(X)` enforcing strict historical lookup. |
| `rainfall_mm_lag_X` | Observations | `t - X` days | LOW | Strictly shifted. |
| `temperature_c_Xd_mean` | Observations | Up to `t` | LOW | Uses `.rolling(window=X).mean()` bounded at current `t`. No centered rolling windows were used. |
| `rainfall_mm_Xd_sum` | Observations | Up to `t` | LOW | Backward-looking rolling sum. |

## Model Splitting Strategy
- **Evaluation Split:** Chronological split.
- **Date Used:** Configured as `split_date` (e.g., `2024-01-01`).
- **Mechanism:** The model is trained strictly on data where `date < split_date` and evaluated on `date >= split_date`. Random train/test splits, which violate time series causality, were intentionally rejected.

## Test Verification
- **Test:** `test_future_leakage_prevention` in `test_environmental_features.py`
- **Result:** PASSED. Altering future rows (`t + n`) was mathematically proven not to alter the computed rolling or lagged features at time `t`.

## Conclusion
The current feature engineering and machine-learning evaluation pipelines are structurally protected against look-ahead bias and temporal leakage.
