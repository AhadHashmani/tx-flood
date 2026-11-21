"""
models/risk_scorer.py
=====================
Flood risk scoring using a composite weighted model and a
rolling z-score anomaly detector.

Risk levels:
    0  NORMAL      — all indicators within baseline
    1  ELEVATED    — one or more indicators above action stage
    2  HIGH        — approaching flood stage or strong anomaly
    3  CRITICAL    — at or above flood stage

Author: Ahad Hashmani
"""

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

log = logging.getLogger(__name__)


def _finite(value, default: float = 0.0) -> float:
    """Float conversion that turns missing and non-finite values into `default`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


class RiskLevel(IntEnum):
    NORMAL   = 0
    ELEVATED = 1
    HIGH     = 2
    CRITICAL = 3

    def label(self) -> str:
        return self.name

    def color_code(self) -> str:
        return {0: "green", 1: "yellow", 2: "orange", 3: "red"}[self.value]


@dataclass
class SiteRisk:
    site_name: str
    risk_level: RiskLevel
    composite_score: float          # 0.0 – 1.0
    pct_of_flood_stage: float
    discharge_zscore: float
    rain_stream_pressure: float
    anomaly_flags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = ""


@dataclass
class BasinRisk:
    timestamp: str
    overall_risk: RiskLevel
    sites: List[SiteRisk] = field(default_factory=list)
    precip_24h_in: float = 0.0
    n_critical: int = 0
    n_high: int = 0
    n_elevated: int = 0

    def summary(self) -> str:
        lines = [
            f"Basin Risk Assessment — {self.timestamp}",
            f"Overall: {self.overall_risk.label()}  |  "
            f"Precip 24h: {self.precip_24h_in:.2f} in",
            f"Critical: {self.n_critical}  High: {self.n_high}  "
            f"Elevated: {self.n_elevated}",
            "",
        ]
        for s in sorted(self.sites, key=lambda x: x.composite_score, reverse=True):
            flag_str = "  [" + ", ".join(s.anomaly_flags) + "]" if s.anomaly_flags else ""
            lines.append(
                f"  {s.site_name:<45} {s.risk_level.label():<10} "
                f"score={s.composite_score:.3f}  "
                f"flood%={s.pct_of_flood_stage:.1%}{flag_str}"
            )
        return "\n".join(lines)


class FloodRiskScorer:
    """
    Composite flood risk scorer.

    Weights:
        pct_of_flood_stage       0.40  (how close to flood stage)
        discharge_zscore         0.25  (how anomalous vs. historical)
        rain_stream_pressure     0.20  (rain * flood proximity)
        pct_of_action_stage      0.15  (configured action stage, 65% of flood stage)

    All components are clipped to [0, 1] before weighting.
    """

    WEIGHTS = {
        "pct_of_flood_stage":   0.40,
        "discharge_zscore":     0.25,
        "rain_stream_pressure": 0.20,
        "pct_of_action_stage":  0.15,
    }

    ANOMALY_THRESHOLDS = {
        "zscore_high":          2.5,    # 2.5 std devs above rolling mean
        "flood_stage_pct":      0.75,   # 75% of flood stage
        "action_stage_pct":     0.90,   # 90% of action stage
        "precip_24h_heavy":     2.0,    # 2 inches in 24h
        "rain_pressure_high":   0.5,
    }

    def score_site(self, row: pd.Series) -> SiteRisk:
        """Compute risk score for a single gauge observation."""
        flags = []

        # Normalise each component to [0, 1]. Non-finite inputs count as 0
        # so a missing rolling z-score cannot turn the composite into NaN.
        flood_pct    = float(np.clip(_finite(row.get("pct_of_flood_stage", 0)), 0, 1))
        action_pct   = float(np.clip(_finite(row.get("pct_of_action_stage", 0)), 0, 1))
        z_raw        = _finite(row.get("discharge_roll24h_zscore", 0))
        z_norm       = float(np.clip((z_raw + 3) / 6, 0, 1))   # map [-3,+3] → [0,1]
        rain_press   = float(np.clip(_finite(row.get("rain_stream_pressure", 0)) / 2.0, 0, 1))

        composite = (
            self.WEIGHTS["pct_of_flood_stage"]   * flood_pct +
            self.WEIGHTS["discharge_zscore"]      * z_norm +
            self.WEIGHTS["rain_stream_pressure"]  * rain_press +
            self.WEIGHTS["pct_of_action_stage"]   * action_pct
        )

        # Anomaly flags
        th = self.ANOMALY_THRESHOLDS
        if z_raw > th["zscore_high"]:
            flags.append(f"z={z_raw:.1f}")
        if flood_pct > th["flood_stage_pct"]:
            flags.append(f"flood%={flood_pct:.0%}")
        if action_pct > th["action_stage_pct"]:
            flags.append("near_action_stage")
        if row.get("precip_24h_in", 0) > th["precip_24h_heavy"]:
            flags.append(f"heavy_rain={row['precip_24h_in']:.1f}in")
        if rain_press > th["rain_pressure_high"]:
            flags.append("rain_stream_pressure_high")

        # Determine risk level
        if flood_pct >= 1.0 or composite >= 0.85:
            level = RiskLevel.CRITICAL
        elif flood_pct >= 0.75 or composite >= 0.60:
            level = RiskLevel.HIGH
        elif flood_pct >= 0.45 or composite >= 0.35:
            level = RiskLevel.ELEVATED
        else:
            level = RiskLevel.NORMAL

        return SiteRisk(
            site_name=str(row.get("site_name", "unknown")),
            risk_level=level,
            composite_score=round(composite, 4),
            pct_of_flood_stage=round(flood_pct, 4),
            discharge_zscore=round(z_raw, 4),
            rain_stream_pressure=round(_finite(row.get("rain_stream_pressure", 0)), 4),
            anomaly_flags=flags,
            source=str(row.get("source", "") or ""),
        )

    def score_basin(
        self,
        feature_df: pd.DataFrame,
        precip_24h: float = 0.0,
    ) -> BasinRisk:
        """Score all sites and aggregate to basin-level risk."""
        site_risks = [self.score_site(row) for _, row in feature_df.iterrows()]

        n_critical = sum(1 for s in site_risks if s.risk_level == RiskLevel.CRITICAL)
        n_high     = sum(1 for s in site_risks if s.risk_level == RiskLevel.HIGH)
        n_elevated = sum(1 for s in site_risks if s.risk_level == RiskLevel.ELEVATED)

        if n_critical > 0:
            overall = RiskLevel.CRITICAL
        elif n_high >= 2:
            overall = RiskLevel.HIGH
        elif n_high == 1 or n_elevated >= 3:
            overall = RiskLevel.ELEVATED
        else:
            overall = RiskLevel.NORMAL

        from datetime import datetime, timezone
        return BasinRisk(
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            overall_risk=overall,
            sites=site_risks,
            precip_24h_in=round(precip_24h, 3),
            n_critical=n_critical,
            n_high=n_high,
            n_elevated=n_elevated,
        )


class AnomalyDetector:
    """
    Fits a per-site baseline from historical data and flags
    observations that exceed a configurable z-score threshold.
    """

    def __init__(self, z_threshold: float = 2.5):
        self.z_threshold = z_threshold
        self._baselines: Dict[str, Tuple[float, float]] = {}   # site → (mean, std)

    def fit(self, historical_df: pd.DataFrame) -> "AnomalyDetector":
        """Compute per-site mean and std from historical discharge."""
        for site, group in historical_df.groupby("site_name"):
            mu  = group["discharge_cfs"].mean()
            std = group["discharge_cfs"].std()
            self._baselines[site] = (mu, max(std, 1e-3))
            log.debug("Baseline for %s: mean=%.1f, std=%.1f", site, mu, std)
        log.info("AnomalyDetector fitted on %d sites", len(self._baselines))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds columns:
            baseline_mean, baseline_std, historical_zscore, is_anomaly
        """
        df = df.copy()
        means, stds = [], []

        for _, row in df.iterrows():
            mu, std = self._baselines.get(row["site_name"], (0.0, 1.0))
            means.append(mu)
            stds.append(std)

        df["baseline_mean"]    = means
        df["baseline_std"]     = stds
        df["historical_zscore"] = (
            (df["discharge_cfs"] - df["baseline_mean"]) / df["baseline_std"]
        ).round(4)
        df["is_anomaly"] = df["historical_zscore"].abs() > self.z_threshold
        return df

    def anomaly_report(self, df: pd.DataFrame) -> str:
        flagged = df[df["is_anomaly"]]
        if flagged.empty:
            return "No anomalies detected across monitored gauges."
        lines = [f"Anomaly Report — {len(flagged)} gauge(s) flagged:"]
        for _, row in flagged.iterrows():
            direction = "ABOVE" if row["historical_zscore"] > 0 else "BELOW"
            lines.append(
                f"  {row['site_name']}: z={row['historical_zscore']:.2f} "
                f"({direction} baseline)  discharge={row['discharge_cfs']:.1f} cfs"
            )
        return "\n".join(lines)
