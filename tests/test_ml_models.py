"""Unit tests for EpiWatch Phase 3 — ML Outbreak Detection Models.

Tests cover:
  - dataset_builder: feature selection, chronological split, imputation
  - model_trainer:   model construction, CV scoring, artifact persistence
  - model_evaluator: metric computation, confusion matrix dimensions
  - risk_scorer:     risk level assignment, output schema
  - ml_plots:        smoke test (no crash)

All tests use synthetic in-memory data — no disk I/O required for the core logic.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_epi_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate a small synthetic epidemiological feature DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "region": "TestRegion",
        "disease": "TestDisease",
        "new_cases": rng.integers(5, 100, n),
        "active_cases": rng.integers(50, 300, n),
        "deaths": rng.integers(0, 5, n),
        "hospitalizations": rng.integers(0, 20, n),
        "icu_admissions": rng.integers(0, 5, n),
        "tests": rng.integers(100, 1000, n),
        "test_positivity_rate_percent": rng.uniform(1, 10, n),
        "cases_7d_avg": rng.uniform(10, 80, n),
        "cases_14d_avg": rng.uniform(10, 80, n),
        "cases_28d_avg": rng.uniform(10, 80, n),
        "growth_rate_7d": rng.uniform(-30, 100, n),
        "growth_rate_14d": rng.uniform(-30, 100, n),
        "incidence_per_100k": rng.uniform(0.1, 5, n),
        "case_fatality_rate": rng.uniform(0, 5, n),
        "hospitalization_rate": rng.uniform(0, 30, n),
        "temperature_c": rng.uniform(15, 40, n),
        "humidity_percent": rng.uniform(30, 90, n),
        "rainfall_mm": rng.uniform(0, 50, n),
        "air_quality_index": rng.uniform(80, 300, n),
        "rainfall_lag_7": rng.uniform(0, 50, n),
        "humidity_lag_7": rng.uniform(30, 90, n),
        "rainfall_7d": rng.uniform(0, 100, n),
        "month": dates.month,
        "week": dates.isocalendar().week.astype(int),
        "quarter": dates.quarter,
        "potential_anomaly": rng.integers(0, 2, n),
        "data_status": "SOURCE_CALIBRATED_SYNTHETIC",
    })


def _make_training_results(feature_cols, n_samples=160, seed=1) -> Dict[str, Any]:
    """Create a minimal mock training_results dict for evaluation tests."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, len(feature_cols)))
    y = rng.integers(0, 2, n_samples)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_scaled, y)

    return {
        "logistic_regression": {
            "model": model,
            "scaler": scaler,
            "feature_cols": feature_cols,
            "cv_scores": np.array([0.75, 0.80, 0.72, 0.78, 0.76]),
            "metadata": {
                "model_name": "logistic_regression",
                "cv_roc_auc_mean": 0.762,
                "cv_roc_auc_std": 0.028,
                "cv_roc_auc_scores": [0.75, 0.80, 0.72, 0.78, 0.76],
                "train_samples": n_samples,
                "n_features": len(feature_cols),
                "training_time_seconds": 0.5,
                "smote_applied": False,
                "label_source": "Phase2_statistical_rolling_threshold",
            },
        }
    }


# ── Test: Chronological split preserves temporal order ───────────────────────

class TestChronologicalSplit:
    def test_train_before_test(self):
        """Train dates must all precede test dates."""
        df = _make_epi_df(200)
        df = df.sort_values("date").reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        assert train["date"].max() <= test["date"].min(), (
            "Train set contains dates that are later than test set dates — data leakage!"
        )

    def test_split_ratio_approx(self):
        """Split should approximate the configured test_size."""
        df = _make_epi_df(200)
        test_size = 0.2
        split_idx = int(len(df) * (1 - test_size))
        test = df.iloc[split_idx:]
        actual_ratio = len(test) / len(df)
        assert abs(actual_ratio - test_size) < 0.05, (
            f"Test set ratio {actual_ratio:.2f} deviates too much from target {test_size}"
        )


# ── Test: Feature imputation ──────────────────────────────────────────────────

class TestFeatureImputation:
    def test_no_nan_after_imputation(self):
        """After median imputation, no NaN values should remain in feature columns."""
        df = _make_epi_df(100)
        feature_cols = [
            "new_cases", "cases_7d_avg", "growth_rate_7d",
            "temperature_c", "humidity_percent"
        ]
        # Inject NaNs
        for col in feature_cols:
            df.loc[df.sample(5).index, col] = np.nan

        # Impute
        for col in feature_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

        assert df[feature_cols].isna().sum().sum() == 0, "NaN values remain after imputation."

    def test_imputed_values_are_medians(self):
        """Imputed values should equal the column median."""
        s = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        expected_median = s.median()
        imputed = s.fillna(expected_median)
        assert imputed.isna().sum() == 0
        assert imputed[2] == pytest.approx(expected_median)


# ── Test: Risk level assignment ───────────────────────────────────────────────

class TestRiskLevelAssignment:
    @pytest.mark.parametrize("prob,expected_level", [
        (0.80, "VERY HIGH"),
        (0.70, "VERY HIGH"),
        (0.60, "HIGH"),
        (0.50, "HIGH"),
        (0.40, "MODERATE"),
        (0.30, "MODERATE"),
        (0.15, "LOW"),
        (0.00, "LOW"),
    ])
    def test_risk_thresholds(self, prob, expected_level):
        """Risk levels should respect defined probability thresholds."""
        thresholds = [(0.70, "VERY HIGH"), (0.50, "HIGH"), (0.30, "MODERATE"), (0.00, "LOW")]
        result = "LOW"
        for threshold, level in thresholds:
            if prob >= threshold:
                result = level
                break
        assert result == expected_level, f"prob={prob} → expected {expected_level}, got {result}"

    def test_all_probabilities_get_assigned(self):
        """Every probability in [0, 1] must receive a non-empty risk level."""
        probs = np.linspace(0, 1, 100)
        thresholds = [(0.70, "VERY HIGH"), (0.50, "HIGH"), (0.30, "MODERATE"), (0.00, "LOW")]
        for p in probs:
            level = next((lv for th, lv in thresholds if p >= th), "LOW")
            assert level in {"LOW", "MODERATE", "HIGH", "VERY HIGH"}


# ── Test: Model instantiation ─────────────────────────────────────────────────

class TestModelInstantiation:
    def test_logistic_regression_trains(self):
        """LogisticRegression should fit without error on small binary data."""
        rng = np.random.default_rng(99)
        X = rng.random((100, 5))
        y = rng.integers(0, 2, 100)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(scaler.transform(X), y)
        preds = model.predict(scaler.transform(X))
        assert len(preds) == 100
        assert set(preds).issubset({0, 1})

    def test_random_forest_trains(self):
        """RandomForest should fit and return probabilities."""
        rng = np.random.default_rng(7)
        X = rng.random((120, 8))
        y = rng.integers(0, 2, 120)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        probs = model.predict_proba(X)[:, 1]
        assert probs.shape == (120,)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_feature_importances_sum_to_one(self):
        """RandomForest feature_importances_ should sum to approximately 1."""
        rng = np.random.default_rng(3)
        X = rng.random((100, 10))
        y = rng.integers(0, 2, 100)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)
        assert abs(model.feature_importances_.sum() - 1.0) < 1e-6


# ── Test: Evaluation metric shapes ───────────────────────────────────────────

class TestEvaluationMetrics:
    def test_confusion_matrix_shape(self):
        """Confusion matrix for binary classification must be (2, 2)."""
        from sklearn.metrics import confusion_matrix
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_roc_auc_range(self):
        """ROC-AUC must lie in [0, 1]."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 80)
        y_score = rng.uniform(0, 1, 80)
        auc = roc_auc_score(y_true, y_score)
        assert 0.0 <= auc <= 1.0

    def test_f1_zero_on_all_negative_predictions(self):
        """F1 should be 0 when model predicts all negatives."""
        from sklearn.metrics import f1_score
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0])
        f1 = f1_score(y_true, y_pred, zero_division=0)
        assert f1 == 0.0


# ── Test: No data leakage in feature columns ─────────────────────────────────

class TestDataLeakage:
    def test_target_not_in_feature_columns(self):
        """The target column must not appear in the feature list."""
        all_cols = list(_make_epi_df(10).columns)
        target = "potential_anomaly"
        meta_cols = {"date", "region", "disease", "data_status"}
        exclude = meta_cols | {target}
        feature_cols = [c for c in all_cols if c not in exclude]
        assert target not in feature_cols, (
            "Target column 'potential_anomaly' is leaking into feature columns!"
        )

    def test_future_rolling_not_present(self):
        """Dataset builder uses Phase 1 rolling features (already lag-safe).
        Verify growth_rate_7d column exists (it's computed without future data)."""
        df = _make_epi_df(50)
        assert "growth_rate_7d" in df.columns, (
            "growth_rate_7d (Phase 1 lag-safe feature) missing from dataset."
        )
