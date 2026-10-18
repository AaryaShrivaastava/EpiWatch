# EpiWatch Phase 3 — ML Model Evaluation Summary

**Best Model**: Gradient Boosting  
**ROC-AUC (test)**: 0.4964  
**Average Precision (test)**: 0.0365  
**F1-Score**: 0.0  

---

## Model Comparison

| Model | ROC-AUC | Avg Precision | Precision | Recall | F1 | CV ROC-AUC |
|-------|---------|--------------|-----------|--------|-----|-----------|
| gradient_boosting | 0.4964 | 0.0365 | 0.0 | 0.0 | 0.0 | 0.9882 ± 0.02 |
| logistic_regression | 0.4811 | 0.0282 | 0.0282 | 0.4 | 0.0526 | 0.348 ± 0.1384 |
| random_forest | 0.4787 | 0.0401 | 0.2 | 0.0182 | 0.0333 | 0.9717 ± 0.0276 |

---

## Notes

- **Target**: `potential_anomaly` — a *statistical* signal (rolling 28-day mean + 2σ),
  NOT a medically confirmed outbreak label.
- **Split method**: Chronological (no data leakage).
- **Class imbalance**: Handled with SMOTE on training set only.
- **Evaluation**: Performed on raw, unseen test set (no SMOTE).
- **Data provenance**: SOURCE_CALIBRATED_SYNTHETIC — suitable for academic capstone.
