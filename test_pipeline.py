"""
tests/test_pipeline.py
======================
Unit tests for the TxFlood pipeline.

Run with:
    python -m pytest tests/ -v

Author: Ahad Hashmani
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from ingestion.fetchers import (
    generate_historical_baseline,
    _synthetic_gauge,
    _synthetic_precip,
    USGS_GAUGES,
)
from pipeline.features import (
    engineer_gauge_features,
    compute_rolling_stats,
    merge_precip_features,
    build_feature_matrix,
    FLOOD_STAGE_FT,
)
from models.risk_scorer import (
    AnomalyDetector,
    FloodRiskScorer,
    RiskLevel,
    BasinRisk,
    SiteRisk,
)


# ── Fixtures ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_gauge_df():
    """Six gauge readings, one per site."""
    records = []
    for i, (name, sid) in enumerate(USGS_GAUGES.items()):
        records.append({
            "site_name":      name,
            "site_id":        sid,
            "timestamp":      datetime.utcnow() - timedelta(minutes=5 * i),
            "discharge_cfs":  100.0 + i * 20,
            "gage_height_ft": 3.0 + i * 0.5,
            "source":         "test",
        })
    return pd.DataFrame(records)


@pytest.fixture
def historical_df():
    return generate_historical_baseline(n_days=30, seed=0)


@pytest.fixture
def fitted_detector(historical_df):
    return AnomalyDetector(z_threshold=2.5).fit(historical_df)


@pytest.fixture
def scorer():
    return FloodRiskScorer()


# ── Ingestion tests ──────────────────────────────────────────────────
class TestIngestion:
    def test_historical_baseline_shape(self, historical_df):
        assert len(historical_df) > 0
        assert "discharge_cfs" in historical_df.columns
        assert "gage_height_ft" in historical_df.columns

    def test_historical_covers_all_sites(self, historical_df):
        sites = set(historical_df["site_name"].unique())
        assert sites == set(USGS_GAUGES.keys())

    def test_synthetic_gauge_returns_positive_discharge(self):
        for name, sid in USGS_GAUGES.items():
            reading = _synthetic_gauge(name, sid)
            assert reading.discharge_cfs > 0
            assert reading.gage_height_ft > 0

    def test_synthetic_precip_non_negative(self):
        precip = _synthetic_precip()
        assert precip.precip_1h_in >= 0
        assert precip.precip_6h_in >= 0
        assert precip.precip_24h_in >= 0


# ── Feature engineering tests ────────────────────────────────────────
class TestFeatureEngineering:
    def test_engineer_adds_expected_columns(self, sample_gauge_df):
        out = engineer_gauge_features(sample_gauge_df)
        for col in ["log_discharge", "pct_of_flood_stage", "seasonal_factor"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_log_discharge_non_negative(self, sample_gauge_df):
        out = engineer_gauge_features(sample_gauge_df)
        assert (out["log_discharge"] >= 0).all()

    def test_pct_of_flood_stage_range(self, sample_gauge_df):
        out = engineer_gauge_features(sample_gauge_df)
        assert (out["pct_of_flood_stage"] >= 0).all()

    def test_rolling_stats_adds_zscore(self, historical_df):
        eng = engineer_gauge_features(historical_df)
        rolled = compute_rolling_stats(eng)
        assert "discharge_roll24h_zscore" in rolled.columns

    def test_merge_precip_adds_columns(self, sample_gauge_df):
        eng = engineer_gauge_features(sample_gauge_df)
        out = merge_precip_features(eng, 0.1, 0.5, 1.2)
        assert "precip_24h_in" in out.columns
        assert "rain_stream_pressure" in out.columns
        assert (out["precip_24h_in"] == 1.2).all()

    def test_build_feature_matrix_no_crash_on_missing_cols(self, sample_gauge_df):
        eng = engineer_gauge_features(sample_gauge_df)
        # Deliberately skip rolling stats — build_feature_matrix should fill with 0
        result = build_feature_matrix(eng)
        assert len(result) == len(sample_gauge_df)


# ── Anomaly detector tests ───────────────────────────────────────────
class TestAnomalyDetector:
    def test_fit_stores_all_sites(self, fitted_detector):
        assert len(fitted_detector._baselines) == len(USGS_GAUGES)

    def test_predict_adds_is_anomaly(self, fitted_detector, sample_gauge_df):
        out = fitted_detector.predict(sample_gauge_df)
        assert "is_anomaly" in out.columns
        assert out["is_anomaly"].dtype == bool

    def test_extreme_value_flagged_as_anomaly(self, fitted_detector):
        site = list(USGS_GAUGES.keys())[0]
        row = pd.DataFrame([{
            "site_name":     site,
            "discharge_cfs": 1_000_000.0,   # absurdly high
        }])
        out = fitted_detector.predict(row)
        assert out["is_anomaly"].iloc[0]

    def test_normal_value_not_anomaly(self, fitted_detector, historical_df):
        # Use a value near the median — should not be anomalous
        site = list(USGS_GAUGES.keys())[0]
        median_q = historical_df[historical_df["site_name"] == site]["discharge_cfs"].median()
        row = pd.DataFrame([{
            "site_name":     site,
            "discharge_cfs": median_q,
        }])
        out = fitted_detector.predict(row)
        assert not out["is_anomaly"].iloc[0]


# ── Risk scorer tests ────────────────────────────────────────────────
class TestRiskScorer:
    def test_normal_conditions_score_low(self, scorer):
        row = pd.Series({
            "site_name":                  list(USGS_GAUGES.keys())[0],
            "pct_of_flood_stage":         0.20,
            "pct_of_action_stage":        0.30,
            "discharge_roll24h_zscore":   0.5,
            "rain_stream_pressure":       0.01,
            "precip_24h_in":              0.1,
        })
        result = scorer.score_site(row)
        assert result.risk_level == RiskLevel.NORMAL
        assert result.composite_score < 0.35

    def test_flood_stage_exceeded_is_critical(self, scorer):
        row = pd.Series({
            "site_name":                  list(USGS_GAUGES.keys())[0],
            "pct_of_flood_stage":         1.05,
            "pct_of_action_stage":        1.30,
            "discharge_roll24h_zscore":   4.0,
            "rain_stream_pressure":       1.5,
            "precip_24h_in":              3.5,
        })
        result = scorer.score_site(row)
        assert result.risk_level == RiskLevel.CRITICAL

    def test_score_basin_returns_basin_risk(self, scorer, sample_gauge_df):
        eng = engineer_gauge_features(sample_gauge_df)
        eng = merge_precip_features(eng, 0.0, 0.1, 0.3)
        eng["discharge_roll24h_zscore"]  = 0.5
        eng["discharge_roll168h_zscore"] = 0.3
        feature_matrix = build_feature_matrix(eng)
        basin = scorer.score_basin(feature_matrix, precip_24h=0.3)
        assert isinstance(basin, BasinRisk)
        assert len(basin.sites) == len(USGS_GAUGES)

    def test_multiple_critical_sites_elevate_basin(self, scorer):
        sites = list(USGS_GAUGES.keys())
        rows = []
        for name in sites[:3]:
            rows.append(pd.Series({
                "site_name": name,
                "pct_of_flood_stage": 1.1,
                "pct_of_action_stage": 1.3,
                "discharge_roll24h_zscore": 4.5,
                "rain_stream_pressure": 2.0,
                "precip_24h_in": 4.0,
            }))
        feature_df = pd.DataFrame(rows)
        basin = scorer.score_basin(feature_df, precip_24h=4.0)
        assert basin.overall_risk == RiskLevel.CRITICAL
        assert basin.n_critical == 3


# ── RiskLevel enum tests ─────────────────────────────────────────────
class TestRiskLevel:
    def test_ordering(self):
        assert RiskLevel.NORMAL < RiskLevel.ELEVATED < RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_labels(self):
        assert RiskLevel.CRITICAL.label() == "CRITICAL"
        assert RiskLevel.NORMAL.label() == "NORMAL"

    def test_color_codes(self):
        assert RiskLevel.CRITICAL.color_code() == "red"
        assert RiskLevel.NORMAL.color_code() == "green"
