# EpiWatch Phase 4 — Mandatory Data Leakage Audit

| Feature | Source | Allowed at Forecast Time | Leakage Risk | Mitigation |
|---------|--------|--------------------------|--------------|------------|
| target_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_2 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_3 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_14 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_21 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_28 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_mean_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_mean_14 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_mean_28 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_std_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_std_14 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_rolling_std_28 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| rainfall_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| humidity_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| temperature_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| target_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| temperature_c_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| temperature_c_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| humidity_percent_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| humidity_percent_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| rainfall_mm_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| rainfall_mm_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| air_quality_index_lag_1 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| air_quality_index_lag_7 | Target shift(1+) | YES | NONE | Strict shift(1) or explicit lag offset applied. |
| day_of_week | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| week_of_year | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| month | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| quarter | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| year | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| sin_month | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| cos_month | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| sin_dow | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |
| cos_dow | Calendar | YES (Future-Known) | NONE | Deterministic calendar date computation. |

## Audit Conclusion

✅ **PASSED**: All target-derived and environmental features strictly enforce historical lag boundaries. No future target values or test-set statistics enter model training.
