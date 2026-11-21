"""
pipeline/features.py
====================
Feature engineering for flood risk modeling.

Transforms raw gauge + precipitation readings into a feature matrix
suitable for anomaly detection and risk scoring.

Author: Ahad Hashmani
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Configured stage constants, keyed like USGS_GAUGES. They are not fetched
# from NWS or LCRA. Each value stayed with its site number when station
# names were corrected. Action stage is 65% of the configured flood stage.
FLOOD_STAGE_FT: dict = {
    "Barton_Ck_at_SH71_nr_Oak_Hill": 13.0,
    "Colorado_Rv_at_Austin": 28.0,
    "Shoal_Ck_at_W_12th_St": 9.0,
    "Shoal_Ck_at_Steck_Ave": 8.5,
    "Waller_Ck_at_38th_St": 7.0,
    "Wilbarger_Ck_nr_Pflugerville": 10.0,
}

ACTION_STAGE_FT: dict = {k: v * 0.65 for k, v in FLOOD_STAGE_FT.items()}


def engineer_gauge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from a DataFrame of gauge readings.

    Input columns expected:
        site_name, timestamp, discharge_cfs, gage_height_ft

    Output adds:
        log_discharge, log_gage_height, pct_of_flood_stage,
        pct_of_action_stage, hour_of_day, month, is_nighttime
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["log_discharge"]   = np.log1p(df["discharge_cfs"])
    df["log_gage_height"] = np.log1p(df["gage_height_ft"])

    df["flood_stage_ft"]  = df["site_name"].map(FLOOD_STAGE_FT).fillna(10.0)
    df["action_stage_ft"] = df["site_name"].map(ACTION_STAGE_FT).fillna(6.5)

    df["pct_of_flood_stage"]  = (df["gage_height_ft"] / df["flood_stage_ft"]).round(4)
    df["pct_of_action_stage"] = (df["gage_height_ft"] / df["action_stage_ft"]).round(4)

    df["hour_of_day"]  = df["timestamp"].dt.hour
    df["month"]        = df["timestamp"].dt.month
    df["is_nighttime"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    # Seasonal factor: spring months have higher baseline
    df["seasonal_factor"] = (
        1.0 + 0.5 * np.cos((df["month"] - 4) * np.pi / 6)
    ).round(4)

    return df


def recent_history_per_site(df: pd.DataFrame, hours: int = 168) -> pd.DataFrame:
    """Return the last `hours` rows of each site.

    The historical frame is stacked site-by-site, so DataFrame.tail on the
    whole frame keeps only the last site.
    """
    ordered = df.sort_values(["site_name", "timestamp"])
    return ordered.groupby("site_name", group_keys=False).tail(hours)


def compute_rolling_stats(
    df: pd.DataFrame,
    windows: tuple = (6, 24, 168),   # 6h, 24h, 1 week
) -> pd.DataFrame:
    """
    Compute rolling mean, std, and z-score for discharge per site.
    Requires the DataFrame to be sorted by (site_name, timestamp).
    """
    df = df.sort_values(["site_name", "timestamp"]).copy()
    results = []

    for site, group in df.groupby("site_name"):
        g = group.copy()
        for w in windows:
            col = f"discharge_roll{w}h"
            g[f"{col}_mean"] = (
                g["discharge_cfs"].rolling(w, min_periods=max(1, w // 4)).mean()
            )
            g[f"{col}_std"] = (
                g["discharge_cfs"].rolling(w, min_periods=max(1, w // 4)).std()
            ).clip(lower=1e-3)
            g[f"{col}_zscore"] = (
                (g["discharge_cfs"] - g[f"{col}_mean"]) / g[f"{col}_std"]
            ).round(4)
        results.append(g)

    return pd.concat(results, ignore_index=True)


def merge_precip_features(
    gauge_df: pd.DataFrame,
    precip_1h: float,
    precip_6h: float,
    precip_24h: float,
) -> pd.DataFrame:
    """
    Broadcast precipitation readings onto each gauge row.
    In a production system these would be spatially interpolated
    to each watershed centroid.
    """
    df = gauge_df.copy()
    df["precip_1h_in"]  = precip_1h
    df["precip_6h_in"]  = precip_6h
    df["precip_24h_in"] = precip_24h

    # Combined pressure index: heavy rain on already-high streams is worst case
    df["rain_stream_pressure"] = (
        df["precip_6h_in"] * df["pct_of_flood_stage"]
    ).round(4)

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return the final feature columns used for risk scoring.
    """
    feature_cols = [
        "site_name",
        "timestamp",
        "discharge_cfs",
        "gage_height_ft",
        "log_discharge",
        "log_gage_height",
        "pct_of_flood_stage",
        "pct_of_action_stage",
        "seasonal_factor",
        "precip_1h_in",
        "precip_6h_in",
        "precip_24h_in",
        "rain_stream_pressure",
        "discharge_roll24h_zscore",
        "discharge_roll168h_zscore",
        "is_nighttime",
        "source",
    ]
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        log.warning("Missing feature columns (filled with defaults): %s", missing)
        for col in missing:
            df[col] = "" if col == "source" else 0.0

    return df[feature_cols].reset_index(drop=True)
