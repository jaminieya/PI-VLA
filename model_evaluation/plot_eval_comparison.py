#!/usr/bin/env python3
"""
Create visual statistics comparing multiple evaluation result folders.

Each input folder must contain a `summary.json` produced by:
- evaluate_test_dataset.py
- evaluate_ntfield_oracle.py

Outputs:
- comparison_table.json
- comparison_table.csv
- success_rates.png
- distance_stats.png
- latent_stats.png
- latent_error_stats.png

Example:
  python model_evaluation/plot_eval_comparison.py \
    --run_dirs \
      /path/to/run_a \
      /path/to/run_b \
      /path/to/run_c \
    --output_dir /path/to/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _safe_name(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def _read_summary(run_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_float(d: Dict[str, Any], key: str) -> Optional[float]:
    v = d.get(key)
    if isinstance(v, (int, float)) and not np.isnan(float(v)):
        return float(v)
    return None


def _row_from_summary(label: str, run_dir: str, s: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "label": label,
        "run_dir": run_dir,
        "mode": s.get("mode"),
        "model_type": s.get("model_type"),
        "checkpoint": s.get("checkpoint"),
        "ntfield_checkpoint": s.get("ntfield_checkpoint"),
        "n_total": s.get("n_total"),
        "n_evaluated": s.get("n_evaluated"),
        "n_errors": s.get("n_errors"),
        "n_success": s.get("n_success"),
        "success_rate": _get_float(s, "success_rate"),
        "n_ee_success": s.get("n_ee_success"),
        "ee_success_rate": _get_float(s, "ee_success_rate"),
        "n_between_fingers_success": s.get("n_between_fingers_success"),
        "between_fingers_success_rate": _get_float(s, "between_fingers_success_rate"),
        "ee_dist_mean_m": _get_float(s, "ee_dist_mean_m"),
        "ee_dist_median_m": _get_float(s, "ee_dist_median_m"),
        "ee_dist_std_m": _get_float(s, "ee_dist_std_m"),
        "finger_midpoint_to_target_xy_distance_mean_m": _get_float(
            s, "finger_midpoint_to_target_xy_distance_mean_m"
        ),
        "finger_midpoint_to_target_xy_distance_median_m": _get_float(
            s, "finger_midpoint_to_target_xy_distance_median_m"
        ),
        "finger_midpoint_to_target_xy_distance_std_m": _get_float(
            s, "finger_midpoint_to_target_xy_distance_std_m"
        ),
        "finger_midpoint_to_target_z_diff_mean_m": _get_float(
            s, "finger_midpoint_to_target_z_diff_mean_m"
        ),
        "finger_midpoint_to_target_z_diff_median_m": _get_float(
            s, "finger_midpoint_to_target_z_diff_median_m"
        ),
        "finger_midpoint_to_target_z_diff_std_m": _get_float(
            s, "finger_midpoint_to_target_z_diff_std_m"
        ),
        "final_latent_dist_mean": _get_float(s, "final_latent_dist_mean"),
        "final_latent_dist_median": _get_float(s, "final_latent_dist_median"),
        "final_latent_dist_std": _get_float(s, "final_latent_dist_std"),
        "path_len_mean": _get_float(s, "path_len_mean"),
        "latent_l2_mean": _get_float(s, "latent_l2_mean"),
        "latent_l2_std": _get_float(s, "latent_l2_std"),
        "latent_l2_median": _get_float(s, "latent_l2_median"),
        "latent_cos_dist_mean": _get_float(s, "latent_cos_dist_mean"),
    }


def _plot_success(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.25

    succ = [r["success_rate"] if r["success_rate"] is not None else np.nan for r in rows]
    ee_succ = [r["ee_success_rate"] if r["ee_success_rate"] is not None else np.nan for r in rows]
    between_succ = [
        r["between_fingers_success_rate"] if r["between_fingers_success_rate"] is not None else np.nan
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    ax.bar(x - width, succ, width=width, label="success_rate")
    ax.bar(x, ee_succ, width=width, label="ee_success_rate")
    ax.bar(x + width, between_succ, width=width, label="between_fingers_success_rate")
    ax.set_ylabel("Rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Success Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_distances(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.25

    ee_mean = [r["ee_dist_mean_m"] if r["ee_dist_mean_m"] is not None else np.nan for r in rows]
    xy_mean = [
        r["finger_midpoint_to_target_xy_distance_mean_m"]
        if r["finger_midpoint_to_target_xy_distance_mean_m"] is not None else np.nan
        for r in rows
    ]
    z_abs_mean = [
        abs(r["finger_midpoint_to_target_z_diff_mean_m"])
        if r["finger_midpoint_to_target_z_diff_mean_m"] is not None else np.nan
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    ax.bar(x - width, ee_mean, width=width, label="ee_dist_mean_m")
    ax.bar(x, xy_mean, width=width, label="finger_mid_xy_mean_m")
    ax.bar(x + width, z_abs_mean, width=width, label="abs(finger_mid_z_diff_mean_m)")
    ax.set_ylabel("Metres")
    ax.set_title("Distance Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_latent(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.4

    final_latent = [
        r["final_latent_dist_mean"] if r["final_latent_dist_mean"] is not None else np.nan
        for r in rows
    ]
    latent_l2 = [r["latent_l2_mean"] if r["latent_l2_mean"] is not None else np.nan for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 5))
    ax.bar(x - width / 2, final_latent, width=width, label="final_latent_dist_mean")
    ax.bar(x + width / 2, latent_l2, width=width, label="latent_l2_mean")
    ax.set_ylabel("Value")
    ax.set_title("Latent Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_latent_error(rows: List[Dict[str, Any]], out_path: str) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.4

    latent_l2 = [r["latent_l2_mean"] if r["latent_l2_mean"] is not None else np.nan for r in rows]
    latent_cos = [
        r["latent_cos_dist_mean"] if r["latent_cos_dist_mean"] is not None else np.nan
        for r in rows
    ]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(max(8, len(labels) * 1.3), 8),
        sharex=True,
    )

    ax1.bar(x, latent_l2, width=width, label="latent_l2_mean", color="#1f77b4")
    ax1.set_ylabel("L2 Error")
    ax1.set_title("Latent Error Comparison")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend()

    ax2.bar(x, latent_cos, width=width, label="latent_cos_dist_mean", color="#ff7f0e")
    ax2.set_ylabel("Cosine Distance")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    keys = list(rows[0].keys()) if rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot comparison across evaluation runs.")
    parser.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="Run directories that contain summary.json",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each run dir (same length as --run_dirs).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write comparison artifacts.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.run_dirs)):
        raise SystemExit("--labels must be omitted or match --run_dirs length exactly.")

    labels = args.labels if args.labels else [_safe_name(d) for d in args.run_dirs]

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for label, run_dir in zip(labels, args.run_dirs):
        run_abs = os.path.abspath(run_dir)
        s = _read_summary(run_abs)
        rows.append(_row_from_summary(label=label, run_dir=run_abs, s=s))

    table_json = os.path.join(out_dir, "comparison_table.json")
    table_csv = os.path.join(out_dir, "comparison_table.csv")
    with open(table_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    _write_csv(rows, table_csv)

    _plot_success(rows, os.path.join(out_dir, "success_rates.png"))
    _plot_distances(rows, os.path.join(out_dir, "distance_stats.png"))
    _plot_latent(rows, os.path.join(out_dir, "latent_stats.png"))
    _plot_latent_error(rows, os.path.join(out_dir, "latent_error_stats.png"))

    print(f"Wrote comparison artifacts to: {out_dir}")
    print(f"- {table_json}")
    print(f"- {table_csv}")
    print("- success_rates.png")
    print("- distance_stats.png")
    print("- latent_stats.png")
    print("- latent_error_stats.png")


if __name__ == "__main__":
    main()
