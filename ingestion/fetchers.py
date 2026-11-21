"""
ingestion/fetchers.py
=====================
Fetches instantaneous stream gauge data from the USGS National Water
Information System (NWIS) and precipitation from the National Weather
Service API (api.weather.gov).

Falls back to synthetic data if a request fails, so the pipeline still
runs without credentials. The 180-day anomaly baseline is always synthetic.

Author: Ahad Hashmani
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)


def _utcnow_naive() -> datetime:
    """UTC now as a naive datetime, so it sorts with the other naive timestamps."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# Site numbers are what the fetcher requests. Names are the USGS station
# names for those numbers (NWIS site service). Three older labels did not
# match the station file: 08156650 is Shoal Creek at Steck Ave (not Bull
# Creek at Loop 360), 08157000 is Waller Creek at 38th St (not 23rd St),
# and 08159150 is Wilbarger Creek near Pflugerville (not Williamson Creek
# at Slaughter).
USGS_GAUGES: Dict[str, str] = {
    "Barton_Ck_at_SH71_nr_Oak_Hill": "08155200",
    "Colorado_Rv_at_Austin": "08158000",
    "Shoal_Ck_at_W_12th_St": "08156800",
    "Shoal_Ck_at_Steck_Ave": "08156650",
    "Waller_Ck_at_38th_St": "08157000",
    "Wilbarger_Ck_nr_Pflugerville": "08159150",
}

USGS_BASE = "https://waterservices.usgs.gov/nwis/iv/"
# Austin centroid. NWS uses this point to choose a grid and a station list.
NWS_POINTS_URL = "https://api.weather.gov/points/30.2672,-97.7431"
NWS_HEADERS = {"User-Agent": "TxFlood/1.0 (https://github.com/AhadHashmani/tx-flood)"}


# ── HTTP session with retry ─────────────────────────────────────────
REQUEST_TIMEOUT_S = 20


def _make_session(retries: int = 1, backoff: float = 0.5) -> requests.Session:
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
        resp = SESSION.get(USGS_BASE, params=params, timeout=REQUEST_TIMEOUT_S)
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
            log.warning(
                "USGS returned no discharge for %s (%s); using synthetic fallback",
                site_name,
                site_id,
            )
            return None

        return GaugeReading(
            site_name=site_name,
            site_id=site_id,
            timestamp=_utcnow_naive(),
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


# ── NWS Precipitation ───────────────────────────────────────────────
@dataclass
class PrecipReading:
    station: str
    timestamp: datetime
    precip_1h_in: float
    precip_6h_in: float
    precip_24h_in: float
    source: str = "nws_live"


def _precip_inches(props: dict, key: str) -> Optional[float]:
    """Convert one NWS precipitation quantity from millimeters to inches.

    Returns None when the field is absent or its value is null. A missing
    quantity is not the same as a measured zero.
    """
    raw = props.get(key)
    if not isinstance(raw, dict) or raw.get("value") is None:
        return None
    return round(float(raw["value"]) / 25.4, 4)


def fetch_nws_precip() -> PrecipReading:
    """
    Pull the latest observation from the first NWS station listed for
    the Austin centroid. Falls back to synthetic data if the request fails.

    NWS precipitation quantities are millimeters. Missing quantities are
    stored as 0.0 inches and the source is marked incomplete.
    """
    try:
        meta = SESSION.get(NWS_POINTS_URL, timeout=REQUEST_TIMEOUT_S, headers=NWS_HEADERS)
        meta.raise_for_status()
        obs_url = meta.json()["properties"]["observationStations"]

        stations = SESSION.get(obs_url, timeout=REQUEST_TIMEOUT_S, headers=NWS_HEADERS)
        stations.raise_for_status()
        first_station = stations.json()["features"][0]["id"]

        obs = SESSION.get(
            f"{first_station}/observations/latest",
            timeout=REQUEST_TIMEOUT_S,
            headers=NWS_HEADERS,
        )
        obs.raise_for_status()
        props = obs.json()["properties"]

        precip_1h = _precip_inches(props, "precipitationLastHour")
        precip_6h = _precip_inches(props, "precipitationLast6Hours")
        precip_24h = _precip_inches(props, "precipitationLast24Hours")
        missing = any(v is None for v in (precip_1h, precip_6h, precip_24h))

        return PrecipReading(
            station=first_station.split("/")[-1],
            timestamp=_utcnow_naive(),
            precip_1h_in=0.0 if precip_1h is None else precip_1h,
            precip_6h_in=0.0 if precip_6h is None else precip_6h,
            precip_24h_in=0.0 if precip_24h is None else precip_24h,
            source="nws_live_precip_incomplete" if missing else "nws_live",
        )

    except Exception as exc:
        log.warning("NWS fetch failed: %s — using synthetic data", exc)
        return _synthetic_precip()


def describe_run_source(gauge_sources, precip_source: str) -> str:
    """Describe whether this run used live readings.

    The anomaly baseline is always the synthetic 180-day series, including
    when the gauge and precipitation requests succeed. A source value of
    ``usgs_live`` is not the string ``live``.
    """
    gauges = set(gauge_sources)
    gauges_all_live = bool(gauges) and gauges == {"usgs_live"}
    gauges_any_live = "usgs_live" in gauges
    precip_measured = precip_source == "nws_live"
    precip_incomplete = precip_source == "nws_live_precip_incomplete"

    if gauges_all_live and precip_measured:
        readings = "LIVE readings (USGS gauges, NWS precipitation)"
    elif gauges_all_live and precip_incomplete:
        readings = (
            "LIVE USGS gauge readings; NWS precipitation quantities were missing "
            "and are stored as 0.0 in"
        )
    elif gauges_any_live or precip_measured or precip_incomplete:
        readings = (
            "MIXED readings (at least one request used synthetic fallback "
            "or incomplete NWS precipitation)"
        )
        if precip_incomplete:
            readings += (
                "; NWS precipitation quantities were missing and are stored as 0.0 in"
            )
    else:
        readings = "SYNTHETIC readings (demo mode)"
    return readings + "; 180-day anomaly baseline is synthetic"


# ── Synthetic fallback generators ──────────────────────────────────
def _synthetic_gauge(name: str, sid: str) -> GaugeReading:
    """Generate plausible synthetic gauge readings for demo mode."""
    rng = np.random.default_rng(abs(hash(name + str(_utcnow_naive().hour))) % (2**32))
    base_discharge = rng.lognormal(mean=4.5, sigma=0.6)
    return GaugeReading(
        site_name=name,
        site_id=sid,
        timestamp=_utcnow_naive(),
        discharge_cfs=round(base_discharge, 2),
        gage_height_ft=round(rng.uniform(1.5, 8.0), 2),
        source="synthetic",
    )


def _synthetic_precip() -> PrecipReading:
    rng = np.random.default_rng(_utcnow_naive().hour)
    return PrecipReading(
        station="KAUS_synthetic",
        timestamp=_utcnow_naive(),
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
        end=_utcnow_naive(), periods=n_days * 24, freq="h"
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
