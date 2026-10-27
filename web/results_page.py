"""Benchmark results tab: read-only CSV, methodology, and in-memory charts."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "benchmark_100_trials"
SUMMARY_CSV = OUTPUT_DIR / "benchmark_summary.csv"
TRIALS_CSV = OUTPUT_DIR / "benchmark_trials.csv"
METHODOLOGY_MD = OUTPUT_DIR / "methodology_report.md"

ALGORITHM_COLORS = {
    "Best-First": "#00bcd4",
    "Beam Search": "#ff9800",
    "Hill Climb": "#4caf50",
    "A* Belief": "#e91e63",
    "Genetic Alg": "#9c27b0",
}


def _load_summary_rows() -> List[dict]:
    if not SUMMARY_CSV.is_file():
        return []
    with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _rows_to_chart_data(rows: List[dict]) -> Dict[str, dict]:
    data: Dict[str, dict] = {}
    for r in rows:
        data[r["algorithm"]] = {
            "detection_rate": float(r["detection_rate_pct"]),
            "mean_steps": None if r["mean_steps_to_detection"] == "N/A" else float(r["mean_steps_to_detection"]),
            "mean_distance": float(r["mean_distance_all_trials"]),
            "mean_battery_pct": float(r["mean_battery_consumed_pct"]),
            "mean_coverage": float(r["mean_coverage_pct"]),
            "mean_time_ms": float(r["mean_computation_ms_all_trials"]),
        }
    return data


def _bar_figure(title: str, algorithms: List[str], values: List[float], ylabel: str, fmt: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("#0f1621")
    ax.set_facecolor("#0f1621")
    colors = [ALGORITHM_COLORS.get(a, "#888888") for a in algorithms]
    bars = ax.bar(algorithms, values, color=colors, edgecolor="#2d3748")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_ylabel(ylabel, color="#cbd5e0")
    ax.tick_params(axis="x", rotation=20, colors="#cbd5e0", labelsize=8)
    ax.tick_params(axis="y", colors="#cbd5e0")
    for spine in ax.spines.values():
        spine.set_color("#4a5568")
    max_v = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_v * 0.02,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
        )
    fig.tight_layout()
    return fig


def render() -> None:
    st.header("Benchmark results (read-only)")
    st.caption(
        "Data loaded from saved benchmark artifacts. This tab does not rerun or overwrite benchmarks."
    )

    if not OUTPUT_DIR.is_dir():
        st.error(f"Results directory not found: {OUTPUT_DIR}")
        return

    rows = _load_summary_rows()
    if not rows:
        st.warning(f"Summary CSV not found at `{SUMMARY_CSV}`.")
        return

    meta = rows[0]
    st.markdown(
        f"**Trials per algorithm:** {meta.get('num_trials', '—')}  \n"
        f"**Benchmark seed:** {meta.get('benchmark_seed', '—')}  \n"
        f"**Max steps / trial:** {meta.get('max_steps_per_trial', '—')}  \n"
        f"**Generated (UTC):** {meta.get('generated_at_utc', '—')}"
    )

    st.subheader("Summary table")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if TRIALS_CSV.is_file():
        with st.expander("Per-trial data (first 200 rows)"):
            with open(TRIALS_CSV, newline="", encoding="utf-8") as f:
                trial_rows = list(csv.DictReader(f))
            st.dataframe(trial_rows[:200], use_container_width=True, hide_index=True)
            st.caption(f"Full file: `{TRIALS_CSV}` ({len(trial_rows)} rows)")

    data = _rows_to_chart_data(rows)
    algorithms = list(data.keys())
    st.subheader("Charts (rendered from CSV, not saved to disk)")

    charts = [
        ("Detection rate (%) — all trials", [data[a]["detection_rate"] for a in algorithms], "Rate (%)", "{:.1f}"),
        (
            "Mean steps to detection — successful trials",
            [data[a]["mean_steps"] or 0 for a in algorithms],
            "Steps",
            "{:.1f}",
        ),
        ("Mean distance — all trials", [data[a]["mean_distance"] for a in algorithms], "Grid steps", "{:.1f}"),
        ("Mean battery consumed (%)", [data[a]["mean_battery_pct"] for a in algorithms], "Battery (%)", "{:.1f}"),
        ("Mean coverage (%)", [data[a]["mean_coverage"] for a in algorithms], "Coverage (%)", "{:.1f}"),
        ("Mean computation time (ms)", [data[a]["mean_time_ms"] for a in algorithms], "ms", "{:.1f}"),
    ]
    for title, values, ylabel, fmt in charts:
        fig = _bar_figure(title, algorithms, values, ylabel, fmt)
        st.pyplot(fig, clear_figure=True)

    png_names = [
        "00_dashboard.png",
        "01_detection_rate.png",
        "02_mean_steps_to_detection.png",
        "03_mean_distance.png",
        "04_battery_usage.png",
        "05_coverage.png",
        "06_runtime.png",
    ]
    existing_pngs = [OUTPUT_DIR / name for name in png_names if (OUTPUT_DIR / name).is_file()]
    if existing_pngs:
        st.subheader("Saved graph files")
        for path in existing_pngs:
            st.image(str(path), caption=path.name, use_container_width=True)

    if METHODOLOGY_MD.is_file():
        st.subheader("Methodology report")
        with open(METHODOLOGY_MD, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.info(f"No methodology report at `{METHODOLOGY_MD}`.")
