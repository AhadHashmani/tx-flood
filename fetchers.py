"""
ingestion/fetchers.py
=====================
Fetches real-time stream gauge data from the USGS National Water
Information System (NWIS) and precipitation data from the NOAA
Climate Data Online API.

Falls back to synthetic data if APIs are unreachable, so the full
pipeline always runs in demo mode without credentials.

Author: Ahad Hashmani
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── USGS gauge IDs for Austin-area watersheds ───────────────────────
USGS_GAUGES: Dict[str, str] = {
    "Barton_Creek_at_SH71":       "08155200",
    "Colorado_River_at_Austin":   "08158000",
    "Shoal_Creek_at_12th_St":     "08156800",
    "Bull_Creek_at_Loop_360":     "08156650",
    "Waller_Creek_at_23rd_St":    "08157000",
    "Williamson_Creek_at_Slaughter": "08159150",
}

USGS_BASE = "https://waterservices.usgs.gov/nwis/iv/"
NOAA_BASE = "https://api.weather.gov/points/30.2672,-97.7431"   # Austin centroid


# ── HTTP session with retry ─────────────────────────────────────────
def _make_session(retries: int = 3, backoff: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _make_session()


# ── USGS Stream Gauge ───────────────────────────────────────────────
@dataclass
class GaugeReading:
    site_name: str
    site_id: str
    timestamp: datetime
    discharge_cfs: float          # cubic feet per second
    gage_height_ft: float
    source: str = "usgs_live"


def fetch_usgs_gauge(site_name: str, site_id: str) -> Optional[GaugeReading]:
    """
    Pull the most recent instantaneous streamflow value from USGS NWIS.
    Returns None on failure (caller falls back to synthetic).
    """
    params = {
        "format": "json",
        "sites": site_id,
        "parameterCd": "00060,00065",   # discharge + gage height
        "siteStatus": "active",
    }
    try:
        resp = SESSION.get(USGS_BASE, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        ts_series = data["value"]["timeSeries"]
        discharge = gage_height = None

        for series in ts_series:
            code = series["variable"]["variableCode"][0]["value"]
            values = series["values"][0]["value"]
            if not values:
                continue
            latest = values[-1]
            val = float(latest["value"]) if latest["value"] != "-999999" else None
            if code == "00060":
                discharge = val
            elif code == "00065":
                gage_height = val

        if discharge is None:
            return None

        return GaugeReading(
            site_name=site_name,
            site_id=site_id,
            timestamp=datetime.utcnow(),
            discharge_cfs=discharge,
            gage_height_ft=gage_height or 0.0,
        )

    except Exception as exc:
        log.warning("USGS fetch failed for %s (%s): %s", site_name, site_id, exc)
        return None


def fetch_all_gauges() -> pd.DataFrame:
    """Fetch all configured Austin gauges; fall back to synthetic on failure."""
    records = []
    for name, sid in USGS_GAUGES.items():
        reading = fetch_usgs_gauge(name, sid)
        if reading is None:
            reading = _synthetic_gauge(name, sid)
        records.append({
            "site_name":       reading.site_name,
            "site_id":         reading.site_id,
            "timestamp":       reading.timestamp,
            "discharge_cfs":   reading.discharge_cfs,
            "gage_height_ft":  reading.gage_height_ft,
            "source":          reading.source,
        })
        time.sleep(0.1)   # be polite to USGS rate limits

    return pd.DataFrame(records)


# ── NOAA Precipitation ──────────────────────────────────────────────
@dataclass
class PrecipReading:
    station: str
    timestamp: datetime
    precip_1h_in: float
    precip_6h_in: float
    precip_24h_in: float
    source: str = "noaa_live"


def fetch_noaa_precip() -> PrecipReading:
    """
    Pull latest hourly precipitation observation from NWS for Austin.
    Falls back to synthetic on failure.
    """
    try:
        # Step 1: resolve observation station URL
        meta = SESSION.get(NOAA_BASE, timeout=8, headers={"User-Agent": "TxFlood/1.0"})
        meta.raise_for_status()
        obs_url = meta.json()["properties"]["observationStations"]

        stations = SESSION.get(obs_url, timeout=8, headers={"User-Agent": "TxFlood/1.0"})
        stations.raise_for_status()
        first_station = stations.json()["features"][0]["id"]

        obs = SESSION.get(
            f"{first_station}/observations/latest",
            timeout=8,
            headers={"User-Agent": "TxFlood/1.0"},
        )
        obs.raise_for_status()
        props = obs.json()["properties"]

        precip_1h = props.get("precipitationLastHour", {}).get("value") or 0.0
        precip_6h = props.get("precipitationLast6Hours", {}).get("value") or 0.0
        precip_24h = props.get("precipitationLast24Hours", {}).get("value") or 0.0

        # NOAA returns mm; convert to inches
        def mm_to_in(v): return round(v / 25.4, 4) if v else 0.0

        return PrecipReading(
            station=first_station.split("/")[-1],
            timestamp=datetime.utcnow(),
            precip_1h_in=mm_to_in(precip_1h),
            precip_6h_in=mm_to_in(precip_6h),
            precip_24h_in=mm_to_in(precip_24h),
        )

    except Exception as exc:
        log.warning("NOAA fetch failed: %s — using synthetic data", exc)
        return _synthetic_precip()


# ── Synthetic fallback generators ──────────────────────────────────
def _synthetic_gauge(name: str, sid: str) -> GaugeReading:
    """Generate plausible synthetic gauge readings for demo mode."""
    rng = np.random.default_rng(abs(hash(name + str(datetime.utcnow().hour))) % (2**32))
    base_discharge = rng.lognormal(mean=4.5, sigma=0.6)
    return GaugeReading(
        site_name=name,
        site_id=sid,
        timestamp=datetime.utcnow(),
        discharge_cfs=round(base_discharge, 2),
        gage_height_ft=round(rng.uniform(1.5, 8.0), 2),
        source="synthetic",
    )


def _synthetic_precip() -> PrecipReading:
    rng = np.random.default_rng(datetime.utcnow().hour)
    return PrecipReading(
        station="KAUS_synthetic",
        timestamp=datetime.utcnow(),
        precip_1h_in=round(float(rng.exponential(0.05)), 3),
        precip_6h_in=round(float(rng.exponential(0.20)), 3),
        precip_24h_in=round(float(rng.exponential(0.50)), 3),
        source="synthetic",
    )


# ── Historical baseline loader ──────────────────────────────────────
def generate_historical_baseline(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic historical record mimicking real Austin streamflow
    seasonality. Used to fit the anomaly detector baseline.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(
        end=datetime.utcnow(), periods=n_days * 24, freq="h"
    )

    records = []
    for site_name, site_id in USGS_GAUGES.items():
        # Seasonal discharge pattern: higher in spring (Mar-May), lower in summer
        month_factor = np.array([
            dates[i].month for i in range(len(dates))
        ])
        seasonal = 1.0 + 0.6 * np.cos((month_factor - 4) * np.pi / 6)

        discharge = (
            rng.lognormal(mean=4.2, sigma=0.5, size=len(dates)) * seasonal
        )
        gage = discharge ** 0.45 * rng.uniform(0.08, 0.12, size=len(dates))

        df = pd.DataFrame({
            "site_name":      site_name,
            "site_id":        site_id,
            "timestamp":      dates,
            "discharge_cfs":  discharge.round(2),
            "gage_height_ft": gage.round(2),
            "source":         "synthetic_historical",
        })
        records.append(df)

    return pd.concat(records, ignore_index=True)
