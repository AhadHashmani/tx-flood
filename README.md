# TxFlood — Austin Watershed Flood Risk Intelligence System

Real-time flood risk monitoring for Austin, TX watersheds. Ingests live stream gauge data from the USGS National Water Information System and precipitation data from NOAA, computes per-site composite risk scores, detects statistical anomalies against a historical baseline, and renders a colour-coded terminal dashboard.

Built as an independent project extending the environmental data analysis work from a spatiotemporal E. coli research project (advised by Prof. Timothy Keitt, UT Austin).

---

## Architecture

```
txflood/
├── ingestion/
│   └── fetchers.py        USGS NWIS + NOAA API clients with retry logic
├── pipeline/
│   └── features.py        Feature engineering: rolling stats, flood stage %, seasonal factors
├── models/
│   └── risk_scorer.py     Composite risk scorer + z-score anomaly detector
├── dashboard/
│   └── display.py         ANSI terminal dashboard with live refresh
├── tests/
│   └── test_pipeline.py   Unit tests (pytest)
└── main.py                Entry point
```

---

## Monitored Gauges

| Watershed | USGS Site ID | Flood Stage (ft) |
|---|---|---|
| Barton Creek at SH-71 | 08155200 | 13.0 |
| Colorado River at Austin | 08158000 | 28.0 |
| Shoal Creek at 12th St | 08156800 | 9.0 |
| Bull Creek at Loop 360 | 08156650 | 8.5 |
| Waller Creek at 23rd St | 08157000 | 7.0 |
| Williamson Creek at Slaughter | 08159150 | 10.0 |

Flood stages sourced from the Lower Colorado River Authority and NOAA.

---

## Risk Model

Each gauge is scored on four components:

| Component | Weight | Description |
|---|---|---|
| % of flood stage | 40% | How close gage height is to NOAA flood stage |
| Discharge z-score | 25% | How anomalous current flow is vs. 24h rolling baseline |
| Rain-stream pressure | 20% | Precipitation × flood stage proximity |
| % of action stage | 15% | NOAA action stage threshold proximity |

Risk levels:

| Level | Composite Score | Condition |
|---|---|---|
| NORMAL | < 0.35 | All indicators within baseline |
| ELEVATED | 0.35 – 0.60 | One or more indicators above action stage |
| HIGH | 0.60 – 0.85 | Approaching flood stage or strong anomaly |
| CRITICAL | ≥ 0.85 | At or above flood stage |

Basin-level risk is derived from the distribution of site-level risks — two or more HIGH sites triggers a HIGH basin status; any CRITICAL site triggers a CRITICAL basin status.

---

## Anomaly Detection

A per-site historical baseline (mean, std) is fitted from 180 days of synthetic data mimicking Austin's seasonal streamflow patterns (higher in spring, lower in late summer). Live readings are scored as anomalies when their z-score exceeds 2.5 standard deviations from baseline.

---

## Quickstart

```bash
# Clone
git clone https://github.com/ahadhashmani/txflood
cd txflood

# Install
pip install -r requirements.txt

# Single snapshot (falls back to synthetic data if APIs unavailable)
python main.py

# Live dashboard with 5-minute refresh
python main.py --live

# Demo mode: 3 cycles then exit
python main.py --live --cycles 3 --refresh 10

# Run tests
python -m pytest tests/ -v
```

---

## Demo Mode

If the USGS or NOAA APIs are unreachable, the system automatically falls back to synthetic data that mimics real Austin streamflow patterns. All pipeline logic, risk scoring, and anomaly detection run identically in demo mode. This is clearly indicated in the dashboard footer.

---

## Data Sources

- **Stream gauges**: [USGS National Water Information System](https://waterdata.usgs.gov/nwis) — public domain, no API key required
- **Precipitation**: [NOAA National Weather Service API](https://api.weather.gov) — public domain, no API key required
- **Flood stages**: [Lower Colorado River Authority](https://www.lcra.org) / NOAA

---

## Stack

Python 3.10+ · pandas · NumPy · SciPy · requests · pytest

---

## Author

Ahad Hashmani · [linkedin.com/in/ahadhashmani](https://linkedin.com/in/ahadhashmani) · UT Austin
