#
# Batch runner: RRTConnect vs NTField benchmark over a fixed (x, y) grid of object poses.
#
# From PI-VLA repository root:
#   python trajectory_evaluation/comparison/run_rrt_ntfield_benchmark_batch.py
#
# --planner-playback {direct,settle} is forwarded to each benchmark (default: direct).
# Optional: pass extra Isaac Gym / benchmark flags after -- :
#   python trajectory_evaluation/comparison/run_rrt_ntfield_benchmark_batch.py --planner-playback settle -- --sim_device cpu
#
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PI_VLA_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = _PI_VLA_ROOT / "trajectory_evaluation" / "comparison" / "run_rrt_ntfield_benchmark.py"

DEFAULT_CHECKPOINT = (
    "ntrl-demo/Experiments/UR5_trajectory_no_wall_accuracy_check/"
    "trajectory_03_25_20_28/Model_Epoch_05000_ValLoss_7.820605e-01.pt"
)

# User-specified grid (x, y) in world meters; z is fixed via --object-z unless overridden.
OBJECT_XY_GRID = [
    (0.5, 0.3),
    (0.7, 0.3),
    (0.9, 0.3),
    (0.5, 0.1),
    (0.7, 0.1),
    (0.9, 0.1),
    (0.5, -0.1),
    (0.7, -0.1),
    (0.9, -0.1),
    (0.5, -0.3),
    (0.7, -0.3),
    (0.9, -0.3),
]


def _subdir_name(x: float, y: float, index: int) -> str:
    return f"run_{index:02d}_x{x:+.4f}_y{y:+.4f}".replace("+", "")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_rrt_ntfield_benchmark.py for each (x,y) in OBJECT_XY_GRID."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="NTField checkpoint path relative to PI-VLA root or absolute",
    )
    parser.add_argument(
        "--object-z",
        type=float,
        default=0.18,
        help="Mustard actor z (world m); not varied on the grid",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Parent directory for this batch (default: output/trajectory_evaluation/batch_<timestamp>/)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Passed to each benchmark run")
    parser.add_argument(
        "--planner-playback",
        dest="planner_playback",
        type=str,
        choices=("direct", "settle"),
        default="direct",
        help="Forward --planner_playback to each run (settle = multi-step dwell per waypoint)",
    )
    parser.add_argument("--no-video", action="store_true", help="Forward --no_video to each run")
    parser.add_argument("--use-viewer", action="store_true", help="Forward --use_viewer to each run")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Args after -- go to each benchmark invocation (e.g. -- --sim_device cuda:0)",
    )
    args = parser.parse_args()
    extra = list(args.passthrough)
    if extra and extra[0] == "--":
        extra = extra[1:]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_root:
        out_root = Path(args.out_root).resolve()
    else:
        out_root = _PI_VLA_ROOT / "output" / "trajectory_evaluation" / f"batch_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_arg = args.checkpoint
    if not os.path.isabs(ckpt_arg):
        ckpt_resolved = (_PI_VLA_ROOT / ckpt_arg).resolve()
    else:
        ckpt_resolved = Path(ckpt_arg).resolve()
    if not ckpt_resolved.is_file():
        print(f"Checkpoint not found: {ckpt_resolved}", file=sys.stderr)
        sys.exit(1)

    rows = []
    failed = 0

    for i, (ox, oy) in enumerate(OBJECT_XY_GRID):
        sub = _subdir_name(ox, oy, i)
        run_dir = out_root / sub
        run_dir.mkdir(parents=True, exist_ok=True)
        out_json = run_dir / "result.json"

        cmd = [
            sys.executable,
            str(_BENCHMARK),
            "--object_x",
            str(ox),
            "--object_y",
            str(oy),
            "--object_z",
            str(args.object_z),
            "--ntfield_checkpoint",
            ckpt_arg if not os.path.isabs(ckpt_arg) else str(ckpt_resolved),
            "--record_dir",
            str(run_dir),
            "--output_json",
            str(out_json),
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        cmd.extend(["--planner_playback", args.planner_playback])
        if args.no_video:
            cmd.append("--no_video")
        if args.use_viewer:
            cmd.append("--use_viewer")
        cmd.extend(extra)

        print("-" * 60)
        print(f"[{i + 1}/{len(OBJECT_XY_GRID)}] {sub}")
        print(" ", " ".join(cmd))

        if args.dry_run:
            rows.append({"object_xy": [ox, oy], "run_dir": str(run_dir), "dry_run": True})
            continue

        env = os.environ.copy()
        proc = subprocess.run(cmd, cwd=str(_PI_VLA_ROOT), env=env)
        row: dict = {
            "index": i,
            "object_xy_m": [ox, oy],
            "object_z_m": args.object_z,
            "run_dir": str(run_dir),
            "returncode": proc.returncode,
        }
        if out_json.is_file():
            try:
                with open(out_json) as jf:
                    row["result"] = json.load(jf)
            except json.JSONDecodeError as e:
                row["result_read_error"] = str(e)
        else:
            row["result_missing"] = True

        rows.append(row)
        if proc.returncode != 0:
            failed += 1

    summary_path = out_root / "batch_summary.json"
    summary = {
        "created": datetime.now().isoformat(),
        "pi_vla_root": str(_PI_VLA_ROOT),
        "checkpoint": str(ckpt_resolved),
        "planner_playback": args.planner_playback,
        "object_z_m": args.object_z,
        "num_runs": len(OBJECT_XY_GRID),
        "num_failed": failed,
        "runs": rows,
    }
    with open(summary_path, "w") as sf:
        json.dump(summary, sf, indent=2)
    print("-" * 60)
    print(f"Wrote {summary_path}")
    if failed:
        print(f"{failed} run(s) exited non-zero.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
