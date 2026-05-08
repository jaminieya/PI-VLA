#!/usr/bin/env python3
"""
Analyze batch benchmark results from run_student_multi_benchmark_batch.py output.

Usage:
    # From batch_summary.json (recommended)
    python analyze_batch_results.py --summary /path/to/batch_summary.json

    # From a root directory containing run_*/result.json files
    python analyze_batch_results.py --out-root /path/to/student_multi_batch_dir

    # Compare multiple batches (e.g. mdn vs regression)
    python analyze_batch_results.py \
        --summary /path/to/mdn/batch_summary.json \
        --summary /path/to/regression/batch_summary.json \
        --labels mdn regression

    # Save CSV of per-run stats
    python analyze_batch_results.py --summary /path/to/batch_summary.json --csv stats.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(d: dict, *keys, default=None):
    """Safely traverse nested dicts."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, None)
        if d is None:
            return default
    return d


def _ee_to_target_dist(result: dict) -> float | None:
    """
    Euclidean distance from EE proxy center (at selected/terminal grasp config)
    to the designated target object's world position.
    """
    planner_key = result.get("_planner_key", "student_ntfield")
    ee_center = _safe(result, planner_key, "ee_object_pair_collision_terminal_waypoint", "ee_center_world_m")
    if ee_center is None:
        ee_center = _safe(result, "ee_object_pair_collision_selected_grasp", "ee_center_world_m")
    target_pos = _safe(result, "designated_object_pose_world_m")
    if ee_center is None or target_pos is None:
        return None
    ee = list(ee_center)
    tp = list(target_pos)
    if len(ee) < 3 or len(tp) < 3:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(ee, tp)))


def _latent_converged(result: dict) -> bool | None:
    """True if NTField stopped due to latent tolerance (not max steps)."""
    planner_key = result.get("_planner_key", "student_ntfield")
    stopped = _safe(result, planner_key, "planner_stopped")
    if stopped is None:
        conv = _safe(result, planner_key, "converged_within_tol")
        return bool(conv) if conv is not None else None
    return stopped.startswith("latent_tol")


def _path_length_rad(result: dict) -> float | None:
    """Total path arc-length in joint space (L2 per segment, summed)."""
    planner_key = result.get("_planner_key", "student_ntfield")
    return _safe(result, planner_key, "motion", "path_segment_l2_sum_rad")


def _num_waypoints(result: dict) -> int | None:
    planner_key = result.get("_planner_key", "student_ntfield")
    n = _safe(result, planner_key, "num_waypoints_after_postprocess")
    if n is not None:
        return n
    n = _safe(result, planner_key, "num_waypoints")
    if n is not None:
        return n
    path = _safe(result, planner_key, "trajectory_waypoints_rad")
    if isinstance(path, list):
        return len(path)
    return None


def _planning_wall_s(result: dict) -> float | None:
    planner_key = result.get("_planner_key", "student_ntfield")
    v = _safe(result, planner_key, "planning_wall_s")
    if v is not None:
        return v
    return _safe(result, planner_key, "planning_wall_s_for_get_path2grasp_only")


def _final_latent_dist(result: dict) -> float | None:
    planner_key = result.get("_planner_key", "student_ntfield")
    return _safe(result, planner_key, "final_latent_dist")


def _planner_success(result: dict) -> bool:
    """Raw planner success flag (path exists, no EE override applied)."""
    planner_key = result.get("_planner_key", "student_ntfield")
    return bool(_safe(result, planner_key, "success", default=False))


def _ee_grasp_success(result: dict) -> bool:
    """
    True iff EE proxy touched target AND did not touch any obstacle.
    This is the 'real correct' success metric.
    """
    planner_key = result.get("_planner_key", "student_ntfield")
    planner_terminal = _safe(result, planner_key, "ee_object_pair_collision_terminal_waypoint")
    if isinstance(planner_terminal, dict):
        return bool(_safe(planner_terminal, "success", default=False))
    return bool(_safe(result, "ee_object_pair_collision_selected_grasp", "success", default=False))


def _target_collision(result: dict) -> bool:
    planner_key = result.get("_planner_key", "student_ntfield")
    planner_terminal = _safe(result, planner_key, "ee_object_pair_collision_terminal_waypoint")
    if isinstance(planner_terminal, dict):
        return bool(_safe(planner_terminal, "target", "collision", default=False))
    return bool(_safe(result, "ee_object_pair_collision_selected_grasp", "target", "collision", default=False))


def _obstacle_collision(result: dict) -> bool:
    """True if EE touched ANY obstacle object."""
    planner_key = result.get("_planner_key", "student_ntfield")
    planner_terminal = _safe(result, planner_key, "ee_object_pair_collision_terminal_waypoint")
    if isinstance(planner_terminal, dict):
        obstacles = _safe(planner_terminal, "obstacles", default=[])
    else:
        obstacles = _safe(result, "ee_object_pair_collision_selected_grasp", "obstacles", default=[])
    return any(ob.get("collision", False) for ob in obstacles)


def _net_joint_l1(result: dict) -> float | None:
    planner_key = result.get("_planner_key", "student_ntfield")
    return _safe(result, planner_key, "motion", "joint_net_abs_delta_l1_rad")


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results_from_summary(summary_path: str) -> list[dict]:
    """Load individual result dicts from a batch_summary.json."""
    with open(summary_path) as f:
        summary = json.load(f)
    out = []
    for run in summary.get("runs", []):
        result = run.get("result")
        if result is None:
            continue
        # Annotate with grid position and run index from the batch summary
        result["_run_index"] = run.get("index", -1)
        result["_object_xy_m"] = run.get("object_xy_m", result.get("designated_object_pose_world_m", [])[:2])
        result["_returncode"] = run.get("returncode", 0)
        out.append(result)
    return out


def load_results_from_dir(out_root: str) -> list[dict]:
    """Scan a directory for run_*/result.json files."""
    root = Path(out_root)
    results = []
    for json_path in sorted(root.glob("run_*/result.json")):
        try:
            with open(json_path) as f:
                result = json.load(f)
            result["_run_index"] = len(results)
            result["_object_xy_m"] = result.get("designated_object_pose_world_m", [None, None])[:2]
            result["_returncode"] = 0
            results.append(result)
        except Exception as e:
            print(f"  [warn] Could not load {json_path}: {e}", file=sys.stderr)
    return results


def discover_run_type_dirs(root_dir: str) -> list[Path]:
    """
    Find immediate child directories that look like batch outputs:
    child/run_*/result.json exists.
    """
    root = Path(root_dir)
    if not root.is_dir():
        return []
    run_type_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any(child.glob("run_*/result.json")):
            run_type_dirs.append(child)
    return run_type_dirs


# ---------------------------------------------------------------------------
# Per-run stats row
# ---------------------------------------------------------------------------

def compute_row(result: dict) -> dict:
    run_idx = result.get("_run_index", -1)
    xy = result.get("_object_xy_m", [None, None])

    ee_dist = _ee_to_target_dist(result)
    ee_ok = _ee_grasp_success(result)
    target_hit = _target_collision(result)
    obs_hit = _obstacle_collision(result)
    plan_ok = _planner_success(result)
    converged = _latent_converged(result)
    path_len = _path_length_rad(result)
    n_wp = _num_waypoints(result)
    plan_s = _planning_wall_s(result)
    fld = _final_latent_dist(result)
    net_l1 = _net_joint_l1(result)
    planner_key = result.get("_planner_key", "student_ntfield")
    stopped = _safe(result, planner_key, "planner_stopped")
    if stopped is None:
        conv = _safe(result, planner_key, "converged_within_tol")
        if conv is True:
            stopped = "converged_within_tol"
        elif conv is False:
            stopped = "not_converged_within_tol"
    num_contacts_target = _safe(
        result, "ee_object_pair_collision_selected_grasp", "target", "num_contacts", default=0
    )

    return {
        "run": run_idx,
        "planner": planner_key,
        "obj_x": xy[0] if len(xy) > 0 else None,
        "obj_y": xy[1] if len(xy) > 1 else None,
        # ---- Key success metrics ----
        "ee_grasp_success": ee_ok,           # Target hit AND no obstacle hit → "correct"
        "target_collision": target_hit,       # EE touched target (ignoring obstacles)
        "obstacle_collision": obs_hit,        # EE touched any non-target object
        "planner_has_path": plan_ok,          # Planner produced ≥2 waypoints
        "latent_converged": converged,        # Stopped via latent tolerance (not max steps)
        # ---- Distance / geometry ----
        "ee_to_target_dist_m": ee_dist,       # Euclidean EE→target center (lower=better)
        "num_target_contacts": num_contacts_target,
        # ---- Trajectory quality ----
        "path_length_rad": path_len,          # Sum of L2 joint-space segment lengths
        "net_joint_l1_rad": net_l1,           # |q_goal - q_start| L1 sum
        "num_waypoints": n_wp,
        # ---- Timing / convergence ----
        "planning_wall_s": plan_s,
        "final_latent_dist": fld,
        "stopped_reason": stopped,
    }


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _pct(bools):
    bools = [b for b in bools if b is not None]
    return 100.0 * sum(bools) / len(bools) if bools else float("nan")


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n_runs": n,
        # Success rates
        "success_rate_pct": _pct([r["ee_grasp_success"] for r in rows]),
        "target_hit_rate_pct": _pct([r["target_collision"] for r in rows]),
        "obstacle_hit_rate_pct": _pct([r["obstacle_collision"] for r in rows]),
        "planner_path_rate_pct": _pct([r["planner_has_path"] for r in rows]),
        "latent_converged_rate_pct": _pct([r["latent_converged"] for r in rows]),
        # Distance (EE → target)
        "mean_ee_to_target_dist_m": _mean([r["ee_to_target_dist_m"] for r in rows]),
        "mean_ee_to_target_dist_m_success_only": _mean(
            [r["ee_to_target_dist_m"] for r in rows if r["ee_grasp_success"]]
        ),
        "mean_ee_to_target_dist_m_fail_only": _mean(
            [r["ee_to_target_dist_m"] for r in rows if not r["ee_grasp_success"]]
        ),
        # Trajectory quality
        "mean_path_length_rad": _mean([r["path_length_rad"] for r in rows]),
        "mean_net_joint_l1_rad": _mean([r["net_joint_l1_rad"] for r in rows]),
        "mean_num_waypoints": _mean([r["num_waypoints"] for r in rows]),
        # Timing
        "mean_planning_wall_s": _mean([r["planning_wall_s"] for r in rows]),
        "mean_final_latent_dist": _mean([r["final_latent_dist"] for r in rows]),
        # Stopped-reason breakdown
        "stopped_reasons": _stopped_breakdown(rows),
        # Failure breakdown
        "failure_breakdown": _failure_breakdown(rows),
    }


def _stopped_breakdown(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        k = r["stopped_reason"] or "unknown"
        counts[k] = counts.get(k, 0) + 1
    total = len(rows)
    return {k: {"count": v, "pct": 100.0 * v / total} for k, v in sorted(counts.items())}


def _failure_breakdown(rows: list[dict]) -> dict:
    """Categorize failures for diagnosis."""
    no_path = sum(1 for r in rows if not r["planner_has_path"])
    has_path_miss_target = sum(1 for r in rows if r["planner_has_path"] and not r["target_collision"])
    has_path_hit_obstacle = sum(1 for r in rows if r["planner_has_path"] and r["target_collision"] and r["obstacle_collision"])
    success = sum(1 for r in rows if r["ee_grasp_success"])
    total = len(rows)
    return {
        "success": success,
        "fail_no_path": no_path,
        "fail_has_path_missed_target": has_path_miss_target,
        "fail_has_path_hit_obstacle_only": has_path_hit_obstacle,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt(v, fmt=".4f"):
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    try:
        return format(v, fmt)
    except Exception:
        return str(v)


def print_per_run_table(rows: list[dict], label: str = "") -> None:
    header = f"  {'run':>3}  {'obj_x':>6}  {'obj_y':>6}  {'success':>7}  {'tgt_hit':>7}  {'obs_hit':>7}  {'ee_dist_m':>9}  {'plan_s':>7}  {'latent_d':>9}  {'stopped'}"
    if label:
        print(f"\n{'='*80}")
        print(f"  {label}")
    print(f"{'='*80}")
    print(header)
    print(f"  {'-'*3}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*20}")
    for r in rows:
        print(
            f"  {r['run']:>3}  "
            f"{_fmt(r['obj_x'], '.2f'):>6}  "
            f"{_fmt(r['obj_y'], '.2f'):>6}  "
            f"{'YES' if r['ee_grasp_success'] else 'no':>7}  "
            f"{'YES' if r['target_collision'] else 'no':>7}  "
            f"{'YES' if r['obstacle_collision'] else 'no':>7}  "
            f"{_fmt(r['ee_to_target_dist_m'], '.4f'):>9}  "
            f"{_fmt(r['planning_wall_s'], '.2f'):>7}  "
            f"{_fmt(r['final_latent_dist'], '.5f'):>9}  "
            f"{r['stopped_reason'] or '':}"
        )


def print_aggregate(agg: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*80}")
        print(f"  AGGREGATE: {label}")
    print(f"{'='*80}")
    n = agg["n_runs"]
    print(f"  Runs evaluated          : {n}")
    print()
    print(f"  ── Success Rates ──────────────────────────────────────────")
    print(f"  Correct grasp success   : {agg['success_rate_pct']:.1f}%  (target hit AND no obstacle)")
    print(f"  Target hit rate         : {agg['target_hit_rate_pct']:.1f}%  (EE touched target, ignoring obstacles)")
    print(f"  Obstacle collision rate : {agg['obstacle_hit_rate_pct']:.1f}%  (EE touched any obstacle)")
    print(f"  Planner produced path   : {agg['planner_path_rate_pct']:.1f}%")
    print(f"  Latent tol. converged   : {agg['latent_converged_rate_pct']:.1f}%")
    print()
    print(f"  ── EE → Target Distance (m) ────────────────────────────────")
    print(f"  Mean (all runs)         : {_fmt(agg['mean_ee_to_target_dist_m'], '.4f')}")
    print(f"  Mean (successes only)   : {_fmt(agg['mean_ee_to_target_dist_m_success_only'], '.4f')}")
    print(f"  Mean (failures only)    : {_fmt(agg['mean_ee_to_target_dist_m_fail_only'], '.4f')}")
    print()
    print(f"  ── Trajectory Quality ──────────────────────────────────────")
    print(f"  Mean path length (rad)  : {_fmt(agg['mean_path_length_rad'], '.4f')}")
    print(f"  Mean net joint L1 (rad) : {_fmt(agg['mean_net_joint_l1_rad'], '.4f')}")
    print(f"  Mean waypoints          : {_fmt(agg['mean_num_waypoints'], '.1f')}")
    print()
    print(f"  ── Planning ────────────────────────────────────────────────")
    print(f"  Mean planning time (s)  : {_fmt(agg['mean_planning_wall_s'], '.3f')}")
    print(f"  Mean final latent dist  : {_fmt(agg['mean_final_latent_dist'], '.5f')}")
    print()
    print(f"  ── Failure Breakdown ───────────────────────────────────────")
    fb = agg["failure_breakdown"]
    print(f"  Success                 : {fb['success']:>3}  /  {fb['total']}")
    print(f"  Fail – no path          : {fb['fail_no_path']:>3}  /  {fb['total']}")
    print(f"  Fail – missed target    : {fb['fail_has_path_missed_target']:>3}  /  {fb['total']}")
    print(f"  Fail – hit obstacle     : {fb['fail_has_path_hit_obstacle_only']:>3}  /  {fb['total']}")
    print()
    print(f"  ── Planner Stop Reasons ────────────────────────────────────")
    for reason, info in agg["stopped_reasons"].items():
        print(f"  {reason:<35} {info['count']:>3}  ({info['pct']:.1f}%)")


def print_comparison_table(agg_map: dict[str, dict]) -> None:
    labels = list(agg_map.keys())
    metrics = [
        ("Correct grasp success (%)",    "success_rate_pct",              ".1f"),
        ("Target hit rate (%)",          "target_hit_rate_pct",           ".1f"),
        ("Obstacle collision rate (%)",  "obstacle_hit_rate_pct",         ".1f"),
        ("Latent converged (%)",         "latent_converged_rate_pct",     ".1f"),
        ("Mean EE→target dist (m)",      "mean_ee_to_target_dist_m",      ".4f"),
        ("Mean EE→target (success, m)",  "mean_ee_to_target_dist_m_success_only", ".4f"),
        ("Mean path length (rad)",       "mean_path_length_rad",          ".4f"),
        ("Mean planning time (s)",       "mean_planning_wall_s",          ".3f"),
        ("Mean final latent dist",       "mean_final_latent_dist",        ".5f"),
    ]
    col_w = max(30, *(len(lb) + 2 for lb in labels))
    print(f"\n{'='*80}")
    print("  COMPARISON TABLE")
    print(f"{'='*80}")
    header = f"  {'Metric':<38}" + "".join(f"  {lb:>{col_w}}" for lb in labels)
    print(header)
    print("  " + "-" * (38 + (col_w + 2) * len(labels)))
    for display, key, fmt in metrics:
        row = f"  {display:<38}"
        for lb in labels:
            v = agg_map[lb].get(key)
            row += f"  {_fmt(v, fmt):>{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _ensure_plot_dir(plot_dir: str | None) -> Path:
    if plot_dir:
        out = Path(plot_dir)
    else:
        out = Path.cwd() / f"analysis_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_success_rate_bars(agg_map: dict[str, dict], out_dir: Path) -> None:
    labels = list(agg_map.keys())
    vals = [agg_map[k]["success_rate_pct"] for k in labels]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, vals)
    plt.ylim(0, 100)
    plt.ylabel("Correct grasp success (%)")
    plt.title("Success Rate by Run Type")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "success_rate_by_run_type.png", dpi=200)
    plt.close()


def _plot_planning_time_box(rows_map: dict[str, list[dict]], out_dir: Path) -> None:
    keep_labels = {"MDN", "NTField", "RRT"}
    labels = []
    data = []
    for label, rows in rows_map.items():
        if label not in keep_labels:
            continue
        vals = [r["planning_wall_s"] for r in rows if r["planning_wall_s"] is not None]
        if vals:
            labels.append(label)
            data.append(vals)
    if not data:
        return
    plt.figure(figsize=(11, 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Planning wall time (s)")
    plt.title("Planning Time Distribution by Run Type")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "planning_time_boxplot.png", dpi=200)
    plt.close()


def _plot_distance_comparison(agg_map: dict[str, dict], out_dir: Path) -> None:
    labels = list(agg_map.keys())
    all_runs = [agg_map[k]["mean_ee_to_target_dist_m"] for k in labels]
    success_only = [agg_map[k]["mean_ee_to_target_dist_m_success_only"] for k in labels]
    fail_only = [agg_map[k]["mean_ee_to_target_dist_m_fail_only"] for k in labels]

    x = list(range(len(labels)))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar([i - width for i in x], all_runs, width=width, label="all runs")
    plt.bar(x, success_only, width=width, label="success only")
    plt.bar([i + width for i in x], fail_only, width=width, label="fail only")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Mean EE->target distance (m)")
    plt.title("Distance Comparison Between Run Types")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "distance_comparison_between_runs.png", dpi=200)
    plt.close()


def _plot_trajectory_quality_comparison(agg_map: dict[str, dict], out_dir: Path) -> None:
    keep_labels = {"MDN", "NTField", "RRT"}
    labels = [lb for lb in agg_map.keys() if lb in keep_labels]
    if not labels:
        return
    x = list(range(len(labels)))
    width = 0.25

    path_len = [agg_map[k]["mean_path_length_rad"] for k in labels]
    net_l1 = [agg_map[k]["mean_net_joint_l1_rad"] for k in labels]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    ax.bar([i - width / 2 for i in x], path_len, width=width, label="mean path length (rad)")
    ax.bar([i + width / 2 for i in x], net_l1, width=width, label="mean net joint L1 (rad)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Trajectory distance metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle("Trajectory Quality Comparison Between Run Types")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_quality_comparison_between_runs.png", dpi=200)
    plt.close(fig)


def _plot_planning_comparison(agg_map: dict[str, dict], out_dir: Path) -> None:
    keep_labels = {"MDN", "NTField", "RRT"}
    labels = [lb for lb in agg_map.keys() if lb in keep_labels]
    if not labels:
        return
    x = list(range(len(labels)))
    width = 0.6
    plan_time = [agg_map[k]["mean_planning_wall_s"] for k in labels]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.bar(x, plan_time, width=width, label="mean planning time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Planning Time Comparison (MDN / NTField / RRT)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "planning_comparison_between_runs.png", dpi=200)
    plt.close(fig)


def _plot_xy_success_map(rows_map: dict[str, list[dict]], out_dir: Path) -> None:
    labels = list(rows_map.keys())
    if not labels:
        return
    n = len(labels)
    cols = min(3, n)
    rows_n = math.ceil(n / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 4 * rows_n), squeeze=False)
    for i, label in enumerate(labels):
        ax = axes[i // cols][i % cols]
        rows = rows_map[label]
        succ = [r for r in rows if r["ee_grasp_success"] and r["obj_x"] is not None and r["obj_y"] is not None]
        fail = [r for r in rows if (not r["ee_grasp_success"]) and r["obj_x"] is not None and r["obj_y"] is not None]
        if succ:
            ax.scatter([r["obj_x"] for r in succ], [r["obj_y"] for r in succ], c="green", label="success", s=70)
        if fail:
            ax.scatter([r["obj_x"] for r in fail], [r["obj_y"] for r in fail], c="red", label="fail", s=70)
        ax.set_title(label)
        ax.set_xlabel("object x (m)")
        ax.set_ylabel("object y (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    for i in range(n, rows_n * cols):
        axes[i // cols][i % cols].axis("off")
    fig.suptitle("XY Success Map by Run Type", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "xy_success_map_by_run_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_plots(rows_map: dict[str, list[dict]], agg_map: dict[str, dict], plot_dir: str | None) -> Path:
    out_dir = _ensure_plot_dir(plot_dir)
    _plot_trajectory_quality_comparison(agg_map, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def write_csv(rows_map: dict[str, list[dict]], csv_path: str) -> None:
    import csv
    all_rows = []
    for label, rows in rows_map.items():
        for r in rows:
            row = {"model": label}
            row.update(r)
            all_rows.append(row)
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved CSV: {csv_path}")


def _split_results_by_planner(results: list[dict]) -> dict[str, list[dict]]:
    planner_order = ["student_ntfield", "rrtconnect", "ntfield"]
    out: dict[str, list[dict]] = {}
    for r in results:
        present = [k for k in planner_order if isinstance(r.get(k), dict)]
        if not present:
            present = ["student_ntfield"]
        for k in present:
            rr = dict(r)
            rr["_planner_key"] = k
            out.setdefault(k, []).append(rr)
    return out


def _pretty_run_type_label(base_label: str, planner_key: str, num_planners: int) -> str:
    lower = base_label.lower()
    if "ntfield_rrt" in lower:
        if planner_key == "rrtconnect":
            return "RRT"
        if planner_key == "ntfield":
            return "NTField"
    if "student_multi_batch_mdn" in lower:
        return "MDN"
    if "student_multi_batch_regression_hybrid_contra" in lower:
        return "Contrastive"
    if "student_multi_batch_regression_hybrid_cos" in lower:
        return "Hybrid Distillation"
    if "student_multi_batch_regression_mse" in lower:
        return "MSE"
    if num_planners > 1:
        return f"{base_label}:{planner_key}"
    return base_label


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze student multi-benchmark batch results.")
    parser.add_argument("--summary", action="append", dest="summaries", metavar="PATH",
                        help="Path to batch_summary.json (repeat for multiple batches)")
    parser.add_argument("--out-root", action="append", dest="out_roots", metavar="DIR",
                        help="Root dir with run_*/result.json (repeat for multiple batches)")
    parser.add_argument("--labels", nargs="+", metavar="NAME",
                        help="Labels for each batch (in order); default: derived from path")
    parser.add_argument("--all-run-types-root", metavar="DIR", default=None,
                        help="Analyze every run-type directory under DIR (expects DIR/<run_type>/run_*/result.json)")
    parser.add_argument("--csv", metavar="FILE", default=None,
                        help="Save per-run stats as CSV")
    parser.add_argument("--plots", action="store_true",
                        help="Generate plots from analyzed batches")
    parser.add_argument("--plot-dir", metavar="DIR", default=None,
                        help="Output directory for plots (default: ./analysis_plots_<timestamp>)")
    parser.add_argument("--no-per-run", action="store_true",
                        help="Skip per-run table (useful for large batches)")
    args = parser.parse_args()

    sources: list[tuple[str, str]] = []
    if args.all_run_types_root:
        run_type_dirs = discover_run_type_dirs(args.all_run_types_root)
        if not run_type_dirs:
            print(f"[warn] No run-type dirs found under {args.all_run_types_root}", file=sys.stderr)
        for p in run_type_dirs:
            sources.append(("dir", str(p)))
    if args.summaries:
        for p in args.summaries:
            sources.append(("summary", p))
    if args.out_roots:
        for d in args.out_roots:
            sources.append(("dir", d))

    if not sources:
        parser.print_help()
        sys.exit(0)

    # Derive labels
    if args.labels:
        labels = list(args.labels)
        if len(labels) < len(sources):
            labels += [Path(s[1]).stem for s in sources[len(labels):]]
    else:
        labels = []
        for kind, path in sources:
            p = Path(path)
            labels.append(p.parent.name if kind == "summary" else p.name)

    rows_map: dict[str, list[dict]] = {}
    agg_map: dict[str, dict] = {}

    for (kind, path), label in zip(sources, labels):
        if kind == "summary":
            results = load_results_from_summary(path)
        else:
            results = load_results_from_dir(path)

        if not results:
            print(f"[warn] No results found for {label} ({path})", file=sys.stderr)
            continue

        planner_groups = _split_results_by_planner(results)
        for planner_key, planner_results in planner_groups.items():
            out_label = _pretty_run_type_label(
                base_label=label,
                planner_key=planner_key,
                num_planners=len(planner_groups),
            )
            rows = [compute_row(r) for r in planner_results]
            rows_map[out_label] = rows
            agg = aggregate(rows)
            agg_map[out_label] = agg

            if not args.no_per_run:
                print_per_run_table(rows, label=out_label)
            print_aggregate(agg, label=out_label)

    if len(agg_map) > 1:
        print_comparison_table(agg_map)

    if args.csv and rows_map:
        write_csv(rows_map, args.csv)
    if args.plots and rows_map and agg_map:
        out_dir = write_plots(rows_map, agg_map, args.plot_dir)
        print(f"\nSaved plots to: {out_dir}")


if __name__ == "__main__":
    main()