"""
dashboard/display.py
====================
Terminal-based flood risk dashboard.
Renders a colour-coded risk table using ANSI escape codes —
no external UI dependency required.

Author: Ahad Hashmani
"""

import os
import time
from typing import Optional

import pandas as pd

from models.risk_scorer import BasinRisk, RiskLevel, SiteRisk

# ANSI colour codes
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
ORANGE = "\033[93m"
YELLOW = "\033[33m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
GRAY   = "\033[90m"
WHITE  = "\033[97m"

LEVEL_COLOR = {
    RiskLevel.NORMAL:   GREEN,
    RiskLevel.ELEVATED: YELLOW,
    RiskLevel.HIGH:     ORANGE,
    RiskLevel.CRITICAL: RED,
}

LEVEL_ICON = {
    RiskLevel.NORMAL:   "●",
    RiskLevel.ELEVATED: "▲",
    RiskLevel.HIGH:     "■",
    RiskLevel.CRITICAL: "✖",
}

WIDTH = 90


def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for flood stage percentage."""
    filled = int(min(value, 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    if value >= 1.0:
        color = RED
    elif value >= 0.75:
        color = ORANGE
    elif value >= 0.45:
        color = YELLOW
    else:
        color = GREEN
    return f"{color}{bar}{RESET}"


def _level_str(level: RiskLevel) -> str:
    color = LEVEL_COLOR[level]
    icon  = LEVEL_ICON[level]
    return f"{color}{BOLD}{icon} {level.label():<10}{RESET}"


def render_header(basin: BasinRisk) -> str:
    color = LEVEL_COLOR[basin.overall_risk]
    lines = [
        f"\n{CYAN}{BOLD}{'─' * WIDTH}{RESET}",
        f"{CYAN}{BOLD}  TxFlood — Austin Watershed Flood Risk Monitor{RESET}",
        f"{GRAY}  {basin.timestamp}{RESET}",
        f"{'─' * WIDTH}",
        f"  Basin Status : {_level_str(basin.overall_risk)}",
        f"  Precip 24h   : {basin.precip_24h_in:.2f} in",
        f"  Critical: {RED}{basin.n_critical}{RESET}  "
        f"High: {ORANGE}{basin.n_high}{RESET}  "
        f"Elevated: {YELLOW}{basin.n_elevated}{RESET}",
        f"{'─' * WIDTH}",
    ]
    return "\n".join(lines)


def render_site_row(site: SiteRisk) -> str:
    name_trunc = site.site_name.replace("_", " ")[:38]
    bar = _bar(site.pct_of_flood_stage)
    level_str = _level_str(site.risk_level)
    score_str = f"{site.composite_score:.3f}"
    flood_pct = f"{site.pct_of_flood_stage:.0%}"
    z_str = f"z={site.discharge_zscore:+.1f}"

    flags = ""
    if site.source == "synthetic":
        flags = f"  {GRAY}synthetic fallback{RESET}"
    if site.anomaly_flags:
        flags += f"  {GRAY}[{', '.join(site.anomaly_flags[:2])}]{RESET}"

    return (
        f"  {WHITE}{name_trunc:<38}{RESET} "
        f"{level_str} "
        f"{bar} "
        f"{GRAY}{flood_pct:>5} of flood stage  "
        f"score={score_str}  {z_str}{RESET}"
        f"{flags}"
    )


def render_footer(data_source: Optional[str] = None, live_hint: bool = False) -> str:
    source_line = f"  {GRAY}Data: {data_source}{RESET}\n" if data_source else ""
    hint = f"  {GRAY}Press Ctrl+C to exit{RESET}\n" if live_hint else ""
    return (
        f"{'─' * WIDTH}\n"
        f"  {GRAY}Sources: USGS NWIS instantaneous values · NWS api.weather.gov precipitation\n"
        f"  Flood-stage thresholds are configured constants, not a live flood-stage feed.{RESET}\n"
        f"{source_line}"
        f"{hint}"
        f"{'─' * WIDTH}\n"
    )


def render_full_dashboard(
    basin: BasinRisk,
    data_source: Optional[str] = None,
    live_hint: bool = False,
) -> str:
    lines = [render_header(basin)]
    sites_sorted = sorted(basin.sites, key=lambda s: s.composite_score, reverse=True)
    for site in sites_sorted:
        lines.append(render_site_row(site))
    lines.append(render_footer(data_source=data_source, live_hint=live_hint))
    return "\n".join(lines)


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def run_live_dashboard(
    refresh_seconds: int = 300,
    max_cycles: int = 0,
) -> None:
    """
    Continuously refresh the dashboard.

    Parameters
    ----------
    refresh_seconds : seconds between data fetches
    max_cycles      : stop after N cycles (0 = run forever)
    """
    # Import here to avoid a cycle with the entry point
    from ingestion.fetchers import (
        describe_run_source,
        fetch_all_gauges,
        fetch_nws_precip,
        generate_historical_baseline,
    )
    from pipeline.features import (
        engineer_gauge_features,
        compute_rolling_stats,
        merge_precip_features,
        build_feature_matrix,
        recent_history_per_site,
    )
    from models.risk_scorer import AnomalyDetector, FloodRiskScorer

    print(f"{CYAN}TxFlood initialising — building historical baseline...{RESET}")
    historical = generate_historical_baseline(n_days=180)
    historical = engineer_gauge_features(historical)
    historical = compute_rolling_stats(historical)

    detector = AnomalyDetector(z_threshold=2.5).fit(historical)
    scorer   = FloodRiskScorer()

    cycle = 0
    try:
        while True:
            cycle += 1
            gauge_df = fetch_all_gauges()
            precip   = fetch_nws_precip()

            feat_df = engineer_gauge_features(gauge_df)

            # Append to historical tail and recompute rolling stats
            tail = recent_history_per_site(historical, hours=168)
            combined = pd.concat([tail, feat_df], ignore_index=True)
            combined = compute_rolling_stats(combined)
            current = combined[combined["source"] != "synthetic_historical"].copy()

            current = merge_precip_features(
                current,
                precip.precip_1h_in,
                precip.precip_6h_in,
                precip.precip_24h_in,
            )
            current = detector.predict(current)
            feature_matrix = build_feature_matrix(current)

            basin = scorer.score_basin(feature_matrix, precip_24h=precip.precip_24h_in)
            data_source = describe_run_source(gauge_df["source"], precip.source)

            clear_screen()
            print(render_full_dashboard(basin, data_source=data_source, live_hint=True))
            print(detector.anomaly_report(current))

            if max_cycles and cycle >= max_cycles:
                break

            print(f"\n  {GRAY}Next refresh in {refresh_seconds}s...{RESET}")
            time.sleep(refresh_seconds)

    except KeyboardInterrupt:
        print(f"\n{CYAN}TxFlood exiting.{RESET}\n")
