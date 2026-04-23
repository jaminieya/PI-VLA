#!/usr/bin/env python3
"""
After merged-H5 NTField training finishes, evaluate each cumulative checkpoint on
12 fixed object poses using trajectory_evaluation/comparison/run_rrt_ntfield_benchmark.py.

Produces:
  - Per-run videos (rrt.mp4, ntfield.mp4) and result JSON under --out_dir
  - results_detail.jsonl (one row per cumulative × location)
  - results_aggregate.json + results_aggregate.csv (per cumulative: success rate,
    mean planning time, mean path energy over locations)
  - scaling_plots.png (num training files vs mean energy; vs mean planning time)

Run from PI-VLA (same env as run_rrt_ntfield_benchmark.py):
  cd /path/to/PI-VLA
  python NTField_scaling/eval_scaling_ntfield.py --device cuda:0

Replot only (after editing jsonl or fixing manifest):
  python NTField_scaling/eval_scaling_ntfield.py --replot

Options:
  --eval_only cumulative_005
  --overwrite_detail
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _pi_vla_root() -> Path:
    return _script_dir().parent


def load_num_files_map(manifest_path: Path) -> Dict[str, int]:
    data = json.loads(manifest_path.read_text())
    out = {}
    for row in data.get("levels", []):
        out[row["id"]] = int(row["num_files"])
    return out


def find_latest_checkpoint(experiments_root: Path, cumulative_name: str) -> Optional[Path]:
    cum_dir = experiments_root / cumulative_name
    if not cum_dir.is_dir():
        return None
    traj_dirs = [p for p in cum_dir.iterdir() if p.is_dir() and p.name.startswith("trajectory_")]
    if not traj_dirs:
        return None
    traj_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = traj_dirs[0]
    pts = list(latest.glob("Model_Epoch_*.pt"))
    if not pts:
        return None

    def epoch_key(p: Path) -> int:
        m = re.search(r"Model_Epoch_(\d+)_", p.name)
        return int(m.group(1)) if m else -1

    return max(pts, key=epoch_key)


def checkpoint_relative_to_pi_vla(ckpt: Path, pi_vla: Path) -> str:
    ckpt = ckpt.resolve()
    pi_vla = pi_vla.resolve()
    try:
        return str(ckpt.relative_to(pi_vla))
    except ValueError:
        return str(ckpt)


def extract_ntfield_metrics(result: Dict[str, Any]) -> Tuple[bool, bool, Optional[float], Optional[float]]:
    """success (has path), converged, planning_wall_s, energy (path_segment_l1_sum_rad)."""
    nt = result.get("ntfield") or {}
    motion = nt.get("motion") or {}
    success = bool(nt.get("success"))
    converged = bool(nt.get("converged_within_tol"))
    plan_t = nt.get("planning_wall_s")
    plan_t_f = float(plan_t) if plan_t is not None else None
    energy = motion.get("path_segment_l1_sum_rad")
    energy_f = float(energy) if energy is not None else None
    return success, converged, plan_t_f, energy_f


def run_one_benchmark(
    pi_vla: Path,
    benchmark_py: Path,
    ckpt_rel: str,
    object_x: float,
    object_y: float,
    object_z: float,
    record_dir: Path,
    output_json: Path,
    device: str,
    ntfield_max_steps: int,
) -> Dict[str, Any]:
    record_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(benchmark_py),
        "--object_x",
        str(object_x),
        "--object_y",
        str(object_y),
        "--object_z",
        str(object_z),
        "--ntfield_checkpoint",
        ckpt_rel,
        "--ntfield_device",
        device,
        "--ntfield_max_steps",
        str(ntfield_max_steps),
        "--record_dir",
        str(record_dir),
        "--output_json",
        str(output_json),
    ]

    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(pi_vla), env=env, capture_output=True, text=True)
    if output_json.is_file():
        return json.loads(output_json.read_text())
    return {
        "error": f"benchmark failed rc={proc.returncode}; no output_json",
        "stderr_tail": (proc.stderr or "")[-2000:],
        "stdout_tail": (proc.stdout or "")[-2000:],
    }


def aggregate_and_plot(
    detail_path: Path,
    num_files_map: Dict[str, int],
    plot_path: Path,
    aggregate_json: Path,
    aggregate_csv: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    with open(detail_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Last row wins if the same (cumulative_id, location_id) appears twice (re-runs).
    dedup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in rows:
        key = (str(r["cumulative_id"]), int(r["location_id"]))
        dedup[key] = r
    rows = list(dedup.values())

    by_cum: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_cum.setdefault(r["cumulative_id"], []).append(r)

    agg: List[Dict[str, Any]] = []
    for cum_id in sorted(by_cum.keys(), key=lambda s: int(s.split("_")[-1])):
        xs = by_cum[cum_id]
        n = len(xs)
        conv = sum(1 for r in xs if r.get("nt_converged"))
        energies = [r["energy_l1_path_rad"] for r in xs if r.get("energy_l1_path_rad") is not None]
        times = [r["planning_wall_s"] for r in xs if r.get("planning_wall_s") is not None]
        agg.append(
            {
                "cumulative_id": cum_id,
                "num_training_files": num_files_map.get(cum_id),
                "n_trials": n,
                "success_rate_converged": conv / n if n else 0.0,
                "mean_planning_wall_s": sum(times) / len(times) if times else None,
                "mean_energy_path_l1_rad": sum(energies) / len(energies) if energies else None,
                "n_valid_energy": len(energies),
                "n_valid_plan_time": len(times),
            }
        )

    aggregate_json.write_text(json.dumps(agg, indent=2))

    with open(aggregate_csv, "w", newline="", encoding="utf-8") as cf:
        if not agg:
            cf.write("")
        else:
            w = csv.DictWriter(cf, fieldnames=list(agg[0].keys()))
            w.writeheader()
            w.writerows(agg)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip scaling_plots.png")
        return

    nums = []
    e_means = []
    t_means = []
    for a in agg:
        n = a["num_training_files"]
        if n is None:
            continue
        nums.append(n)
        e_means.append(a["mean_energy_path_l1_rad"])
        t_means.append(a["mean_planning_wall_s"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axes

    e_plot = [(n, e) for n, e in zip(nums, e_means) if e is not None]
    t_plot = [(n, t) for n, t in zip(nums, t_means) if t is not None]
    if e_plot:
        n0, e0 = zip(*e_plot)
        ax0.plot(n0, e0, "o-", linewidth=2, markersize=6)
    if t_plot:
        n1, t1 = zip(*t_plot)
        ax1.plot(n1, t1, "s-", color="darkorange", linewidth=2, markersize=6)
    ax0.set_xlabel("Number of training trajectories (H5 files)")
    ax0.set_ylabel("Mean NTField path L1 joint travel (rad)\n(avg over 12 poses)")
    ax0.set_title("Data scaling vs motion cost proxy")
    ax0.grid(True, alpha=0.3)

    ax1.set_xlabel("Number of training trajectories (H5 files)")
    ax1.set_ylabel("Mean NTField planning wall time (s)\n(avg over 12 poses)")
    ax1.set_title("Data scaling vs planning time")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {plot_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="NTField scaling eval via run_rrt_ntfield_benchmark.py")
    ap.add_argument("--pi_vla_root", type=Path, default=_pi_vla_root())
    ap.add_argument(
        "--merged_manifest",
        type=Path,
        default=_script_dir() / "merged_h5" / "seed0" / "cumulative_manifest.json",
        help="cumulative_manifest.json (num_files per cumulative_*)",
    )
    ap.add_argument(
        "--experiments_root",
        type=Path,
        default=_pi_vla_root() / "ntrl-demo" / "Experiments" / "UR5_scaling_merged_seed0",
    )
    ap.add_argument(
        "--locations_json",
        type=Path,
        default=_script_dir() / "object_locations_12.json",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=_script_dir() / "eval_runs",
        help="Session output root (videos + JSON per run)",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ntfield_max_steps", type=int, default=200)
    ap.add_argument("--eval_only", type=str, default=None, help="Only cumulative_00N (e.g. cumulative_003)")
    ap.add_argument("--skip_existing", action="store_true", help="Skip if benchmark_result.json exists")
    ap.add_argument(
        "--overwrite_detail",
        action="store_true",
        help="Delete results_detail.jsonl at start (avoid duplicate lines on re-run)",
    )
    ap.add_argument(
        "--replot",
        action="store_true",
        help="Only rebuild results_aggregate.* and scaling_plots.png from results_detail.jsonl",
    )
    args = ap.parse_args()

    pi_vla = args.pi_vla_root.resolve()
    benchmark_py = pi_vla / "trajectory_evaluation" / "comparison" / "run_rrt_ntfield_benchmark.py"
    if not benchmark_py.is_file():
        raise SystemExit(f"Missing benchmark script: {benchmark_py}")

    num_files_map = load_num_files_map(args.merged_manifest.resolve())

    session_dir = args.out_dir.resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    detail_path = session_dir / "results_detail.jsonl"
    if args.overwrite_detail and detail_path.is_file():
        detail_path.unlink()
    aggregate_json = session_dir / "results_aggregate.json"
    aggregate_csv = session_dir / "results_aggregate.csv"
    plot_path = session_dir / "scaling_plots.png"

    if args.replot:
        if not detail_path.is_file():
            raise SystemExit(f"No {detail_path}; run eval first")
        aggregate_and_plot(detail_path, num_files_map, plot_path, aggregate_json, aggregate_csv)
        print(f"Updated {aggregate_json}, {aggregate_csv}, {plot_path}")
        return

    loc_data = json.loads(args.locations_json.read_text())
    locations: List[Dict[str, Any]] = loc_data["locations"]

    cumulative_ids = sorted(num_files_map.keys(), key=lambda s: int(s.split("_")[-1]))
    if args.eval_only:
        if args.eval_only not in num_files_map:
            raise SystemExit(f"Unknown cumulative id {args.eval_only}")
        cumulative_ids = [args.eval_only]

    detail_f = open(detail_path, "a", encoding="utf-8")

    try:
        for cum_id in cumulative_ids:
            ckpt = find_latest_checkpoint(args.experiments_root.resolve(), cum_id)
            if ckpt is None:
                print(f"Skip {cum_id}: no checkpoint under {args.experiments_root / cum_id}")
                continue
            ckpt_rel = checkpoint_relative_to_pi_vla(ckpt, pi_vla)
            n_train = num_files_map[cum_id]

            for loc in locations:
                lid = int(loc["id"])
                run_dir = session_dir / cum_id / f"loc_{lid:02d}"
                result_json = run_dir / "benchmark_result.json"

                if args.skip_existing and result_json.is_file():
                    result = json.loads(result_json.read_text())
                else:
                    result = run_one_benchmark(
                        pi_vla=pi_vla,
                        benchmark_py=benchmark_py,
                        ckpt_rel=ckpt_rel,
                        object_x=float(loc["x"]),
                        object_y=float(loc["y"]),
                        object_z=float(loc["z"]),
                        record_dir=run_dir,
                        output_json=result_json,
                        device=args.device,
                        ntfield_max_steps=args.ntfield_max_steps,
                    )

                success, converged, plan_t, energy = extract_ntfield_metrics(result)
                nt = result.get("ntfield") or {}
                row = {
                    "cumulative_id": cum_id,
                    "num_training_files": n_train,
                    "location_id": lid,
                    "object_x": loc["x"],
                    "object_y": loc["y"],
                    "object_z": loc["z"],
                    "checkpoint": ckpt_rel,
                    "nt_success": success,
                    "nt_converged": converged,
                    "planning_wall_s": plan_t,
                    "energy_l1_path_rad": energy,
                    "video_rrt": (result.get("rrtconnect") or {}).get("video_path"),
                    "video_ntfield": nt.get("video_path"),
                    "session_dir": str(run_dir),
                    "benchmark_error": result.get("error"),
                }
                detail_f.write(json.dumps(row) + "\n")
                detail_f.flush()
                print(
                    f"{cum_id} loc_{lid:02d} converged={converged} "
                    f"plan_t={plan_t} energy={energy}"
                )
    finally:
        detail_f.close()

    aggregate_and_plot(detail_path, num_files_map, plot_path, aggregate_json, aggregate_csv)
    print(f"Wrote {detail_path}, {aggregate_json}, {aggregate_csv}")


if __name__ == "__main__":
    main()
