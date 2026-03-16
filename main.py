"""
main.py
=======
TxFlood entry point.

Usage
-----
Run a single risk assessment snapshot:
    python main.py

Run the live refreshing dashboard (default 5-min refresh):
    python main.py --live

Run in demo mode with N refresh cycles then exit:
    python main.py --live --cycles 3

Author: Ahad Hashmani
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("txflood")


def run_snapshot() -> None:
    """Single-pass risk assessment printed to stdout."""
    from ingestion.fetchers import (
        fetch_all_gauges,
        fetch_noaa_precip,
        generate_historical_baseline,
    )
    from pipeline.features import (
        engineer_gauge_features,
        compute_rolling_stats,
        merge_precip_features,
        build_feature_matrix,
    )
    from models.risk_scorer import AnomalyDetector, FloodRiskScorer
    from dashboard.display import render_full_dashboard

    log.info("Building historical baseline (180 days)...")
    historical = generate_historical_baseline(n_days=180)
    historical = engineer_gauge_features(historical)
    historical = compute_rolling_stats(historical)

    detector = AnomalyDetector(z_threshold=2.5).fit(historical)
    scorer   = FloodRiskScorer()

    log.info("Fetching current gauge readings...")
    gauge_df = fetch_all_gauges()
    precip   = fetch_noaa_precip()

    log.info("Engineering features...")
    feat = engineer_gauge_features(gauge_df)

    tail     = historical.tail(168 * len(gauge_df)).copy()
    combined = __import__("pandas").concat([tail, feat], ignore_index=True)
    combined = compute_rolling_stats(combined)
    current  = combined[combined["source"] != "synthetic_historical"].copy()

    current = merge_precip_features(
        current, precip.precip_1h_in, precip.precip_6h_in, precip.precip_24h_in
    )
    current  = detector.predict(current)
    features = build_feature_matrix(current)

    basin = scorer.score_basin(features, precip_24h=precip.precip_24h_in)

    print(render_full_dashboard(basin))
    print(detector.anomaly_report(current))
    print(f"\nData source: {'LIVE (USGS/NOAA)' if 'live' in gauge_df['source'].values else 'SYNTHETIC (demo mode)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TxFlood — Austin Flood Risk Monitor")
    parser.add_argument("--live",   action="store_true", help="Run live dashboard (auto-refresh)")
    parser.add_argument("--cycles", type=int, default=0, help="Stop after N cycles (0=infinite)")
    parser.add_argument("--refresh",type=int, default=300, help="Refresh interval in seconds")
    args = parser.parse_args()

    if args.live:
        from dashboard.display import run_live_dashboard
        run_live_dashboard(refresh_seconds=args.refresh, max_cycles=args.cycles)
    else:
        run_snapshot()


if __name__ == "__main__":
    main()
