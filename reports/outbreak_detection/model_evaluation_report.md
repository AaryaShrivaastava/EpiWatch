# EpiWatch Phase 3 — Early Outbreak Detection Evaluation Report

## 1. Executive Objective

Build a multi-layered early-warning analytical pipeline to detect emerging epidemiological
anomalies, score outbreak risk (0–100), and assign actionable alert statuses without claiming
unconfirmed medical diagnoses.

## 2. Architecture & Layering

- **Level 1 (Baseline)**: Historical 28-day rolling mean + standard deviation thresholds (+2σ, +3σ).
- **Level 2 (Statistical & Isolation Forest)**: Multi-signal normalization & unsupervised Isolation Forest.
- **Level 3 (Supervised Classifier)**: Random Forest Classifier with Calibrated Probability Output.
- **Signal Aggregator**: Weighted composite scoring (Statistical 25%, Isolation 20%, Supervised 35%, Growth 10%, Incidence 10%).

## 3. Model Comparison Metrics

| Model | Precision | Recall | F1-Score | ROC-AUC | False Positive Rate |
|-------|-----------|--------|----------|---------|---------------------|
| Statistical Baseline (Level 1) | 0.75 | 1.0 | 0.8571 | 0.9996 | 0.0102 |
| Supervised Classifier (Level 3) | 0.625 | 0.1852 | 0.2857 | 0.9668 | 0.0034 |
| Isolation Forest (Level 2) | 0.0714 | 0.0556 | 0.0625 | 0.7284 | 0.022 |

## 4. Early Warning Lead Time Analysis

- **Identified Outbreak Episodes**: 394
- **Average Early Warning Lead Time**: 2.6 days before peak daily cases

## 5. Important Project Limitations

1. **Analytical Weak Labels**: `potential_outbreak` is a statistical heuristic label, NOT a medically confirmed outbreak ground truth.
2. **Synthetic / Source-Calibrated Data**: Designed for capstone demonstration; real-world deployment requires official surveillance verification.
3. **Statistical Heuristics**: System alert statuses (NORMAL, WATCH, ELEVATED, HIGH_ALERT) represent operational warning categories.
4. **No Future Data Leakage**: All rolling features enforce strict `shift(1)` temporal availability.
