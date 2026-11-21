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
from datetime import datetime, timedelta, timezone

from ingestion.fetchers import (
    describe_run_source,
    generate_historical_baseline,
    _precip_inches,
    _synthetic_gauge,
    _synthetic_precip,
    USGS_GAUGES,
)
from pipeline.features import (
    engineer_gauge_features,
    compute_rolling_stats,
    merge_precip_features,
    build_feature_matrix,
    recent_history_per_site,
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
            "timestamp":      datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(minutes=5 * i),
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
        assert precip.source == "synthetic"

    def test_precip_inches_null_is_missing_not_zero(self):
        assert _precip_inches(
            {"precipitationLastHour": {"value": None, "unitCode": "wmoUnit:mm"}},
            "precipitationLastHour",
        ) is None
        assert _precip_inches({"precipitationLast24Hours": None}, "precipitationLast24Hours") is None
        assert _precip_inches(
            {"precipitationLastHour": {"value": 25.4, "unitCode": "wmoUnit:mm"}},
            "precipitationLastHour",
        ) == 1.0

    def test_site_ids_match_usgs_station_names(self):
        by_id = {site_id: name for name, site_id in USGS_GAUGES.items()}
        assert by_id == {
            "08155200": "Barton_Ck_at_SH71_nr_Oak_Hill",
            "08158000": "Colorado_Rv_at_Austin",
            "08156800": "Shoal_Ck_at_W_12th_St",
            "08156650": "Shoal_Ck_at_Steck_Ave",
            "08157000": "Waller_Ck_at_38th_St",
            "08159150": "Wilbarger_Ck_nr_Pflugerville",
        }

    def test_flood_stage_keys_match_gauges(self):
        assert set(FLOOD_STAGE_FT) == set(USGS_GAUGES)

    def test_describe_run_source_live_still_marks_synthetic_baseline(self):
        label = describe_run_source(["usgs_live", "usgs_live"], "nws_live")
        assert label.startswith("LIVE readings")
        assert "180-day anomaly baseline is synthetic" in label

    def test_describe_run_source_does_not_treat_usgs_live_as_demo(self):
        # Membership on the exact string "live" used to miss "usgs_live".
        label = describe_run_source(["usgs_live"], "nws_live")
        assert not label.startswith("SYNTHETIC")

    def test_describe_run_source_incomplete_precip(self):
        label = describe_run_source(["usgs_live"], "nws_live_precip_incomplete")
        assert "missing" in label
        assert "0.0 in" in label

    def test_describe_run_source_mixed_and_demo(self):
        mixed = describe_run_source(["usgs_live", "synthetic"], "nws_live")
        demo = describe_run_source(["synthetic", "synthetic"], "synthetic")
        assert mixed.startswith("MIXED")
        assert demo.startswith("SYNTHETIC")


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

    def test_recent_history_keeps_every_site(self):
        rows = []
        start = datetime(2026, 1, 1, tzinfo=timezone.utc).replace(tzinfo=None)
        for site, count in (("A", 10), ("B", 10)):
            for i in range(count):
                rows.append({
                    "site_name": site,
                    "timestamp": start + timedelta(hours=i),
                    "discharge_cfs": float(i),
                })
        out = recent_history_per_site(pd.DataFrame(rows), hours=3)
        assert set(out["site_name"]) == {"A", "B"}
        assert out.groupby("site_name").size().tolist() == [3, 3]
        assert out.loc[out["site_name"] == "A", "discharge_cfs"].tolist() == [7.0, 8.0, 9.0]

    def test_each_site_gets_a_finite_rolling_zscore(self, historical_df):
        engineered = engineer_gauge_features(historical_df)
        current = engineered.groupby("site_name", group_keys=False).tail(1).copy()
        current["source"] = "test"
        history = recent_history_per_site(engineered, hours=168)
        combined = pd.concat([history, current], ignore_index=True)
        rolled = compute_rolling_stats(combined)
        latest = rolled[rolled["source"] == "test"]
        assert len(latest) == len(USGS_GAUGES)
        assert latest["discharge_roll24h_zscore"].notna().all()


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
    def test_non_finite_zscore_does_not_make_score_nan(self, scorer):
        row = pd.Series({
            "site_name":                list(USGS_GAUGES.keys())[0],
            "pct_of_flood_stage":       0.20,
            "pct_of_action_stage":      0.30,
            "discharge_roll24h_zscore": float("nan"),
            "rain_stream_pressure":     0.01,
            "precip_24h_in":            0.1,
        })
        result = scorer.score_site(row)
        assert result.composite_score == result.composite_score
        assert result.risk_level == RiskLevel.NORMAL

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


def _live_gauge_frame() -> pd.DataFrame:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return pd.DataFrame([
        {
            "site_name": name,
            "site_id": site_id,
            "timestamp": now,
            "discharge_cfs": 50.0,
            "gage_height_ft": 2.0,
            "source": "usgs_live",
        }
        for name, site_id in USGS_GAUGES.items()
    ])


def _measured_precip():
    from ingestion.fetchers import PrecipReading
    return PrecipReading(
        station="TEST",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        precip_1h_in=0.0,
        precip_6h_in=0.0,
        precip_24h_in=0.1,
        source="nws_live",
    )


class TestEntryPoint:
    def test_snapshot_scores_every_gauge(self, capsys, monkeypatch):
        from main import run_snapshot

        monkeypatch.setattr("ingestion.fetchers.fetch_all_gauges", _live_gauge_frame)
        monkeypatch.setattr("ingestion.fetchers.fetch_nws_precip", _measured_precip)
        run_snapshot()
        out = capsys.readouterr().out
        assert "TxFlood" in out
        assert "score=nan" not in out
        assert "LIVE readings (USGS gauges, NWS precipitation)" in out
        for name in USGS_GAUGES:
            assert name.replace("_", " ") in out

    def test_live_dashboard_stops_after_one_cycle(self, capsys, monkeypatch):
        from dashboard.display import run_live_dashboard

        monkeypatch.setattr("ingestion.fetchers.fetch_all_gauges", _live_gauge_frame)
        monkeypatch.setattr("ingestion.fetchers.fetch_nws_precip", _measured_precip)
        run_live_dashboard(refresh_seconds=1, max_cycles=1)
        out = capsys.readouterr().out
        assert "TxFlood" in out
        assert "score=nan" not in out
        assert "Press Ctrl+C to exit" in out
