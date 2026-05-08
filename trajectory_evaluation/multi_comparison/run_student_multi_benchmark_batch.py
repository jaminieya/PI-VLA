'''

python trajectory_evaluation/multi_comparison/run_student_multi_benchmark_batch.py \
  --student-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep90_20260505_114200.pth \
  --checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
  --seed 123 \
  --save-final-geometric-debug \
  --out-root /home/hojinsohn/VLM-NT/PI-VLA/output/trajectory_evaluation/student_multi_batch_mdn_new

python trajectory_evaluation/multi_comparison/run_student_multi_benchmark_batch.py \
  --student-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_mse_bs256_lr3em4_ep90_20260507_155428.pth \
  --checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
  --seed 123 \
  --no-video \
  --save-final-geometric-debug \
  --out-root /home/hojinsohn/VLM-NT/PI-VLA/output/trajectory_evaluation/student_multi_batch_regression_mse

python trajectory_evaluation/multi_comparison/run_student_multi_benchmark_batch.py \
  --student-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_contra_bs256_lr3em4_ep40_20260507_201445.pth \
  --checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
  --seed 123 \
  --no-video \
  --save-final-geometric-debug \
  --out-root /home/hojinsohn/VLM-NT/PI-VLA/output/trajectory_evaluation/student_multi_batch_regression_hybrid_contra

python trajectory_evaluation/multi_comparison/run_student_multi_benchmark_batch.py \
  --student-checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_bs256_lr3em4_ep40_20260507_180418.pth \
  --checkpoint /home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt \
  --seed 123 \
  --no-video \
  --save-final-geometric-debug \
  --out-root /home/hojinsohn/VLM-NT/PI-VLA/output/trajectory_evaluation/student_multi_batch_regression_hybrid_cos
  

'''
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PI_VLA_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK = _PI_VLA_ROOT / "trajectory_evaluation" / "multi_comparison" / "run_student_multi_benchmark.py"

DEFAULT_CHECKPOINT = (
    "/home/hojinsohn/VLM-NT/PI-VLA/teacher_model.pt"
)

# Grid for designated target object pose (x, y), z set by --object-z
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
        description="Run run_student_multi_benchmark.py for each (x,y) in OBJECT_XY_GRID."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="NTField checkpoint path relative to PI-VLA root or absolute",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=str,
        required=True,
        help="Student checkpoint path relative to PI-VLA root or absolute",
    )
    parser.add_argument(
        "--object-z",
        type=float,
        default=0.18,
        help="Designated target object z (world m); not varied on the grid",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Parent directory for this batch (default: output/trajectory_evaluation/multi_batch_<timestamp>/)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Passed to each benchmark run")
    parser.add_argument(
        "--ntfield-waypoint-mode",
        dest="ntfield_waypoint_mode",
        type=str,
        choices=("full", "two_point"),
        default="full",
        help="Forward --ntfield_waypoint_mode to each run.",
    )
    parser.add_argument(
        "--ntfield-fixed-waypoints",
        dest="ntfield_fixed_waypoints",
        type=int,
        default=0,
        help="Forward --ntfield_fixed_waypoints to each run (0 disables resampling).",
    )
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
    parser.add_argument(
        "--save-final-geometric-debug",
        action="store_true",
        help="Forward --save_final_geometric_debug to each run.",
    )
    parser.add_argument(
        "--final-geometric-debug-name",
        type=str,
        default="final_geometric_debug.png",
        help="Per-run filename for --final_geometric_debug_path (saved under each run dir).",
    )
    parser.add_argument(
        "--require-ee-object-collision",
        action="store_true",
        help="Forward --require_ee_object_collision to each run.",
    )
    parser.add_argument(
        "--ee-proxy-radius-m",
        type=float,
        default=0.09,
        help="Legacy forward arg for --ee_proxy_radius_m (ignored by benchmark script).",
    )
    parser.add_argument(
        "--ee-proxy-center-mode",
        type=str,
        choices=("ee_origin", "between_fingertips"),
        default="between_fingertips",
        help="Forward --ee_proxy_center_mode to each run.",
    )
    parser.add_argument(
        "--ee-proxy-max-radius-m",
        type=float,
        default=0.04,
        help="Forward --ee_proxy_max_radius_m to each run.",
    )
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
        out_root = _PI_VLA_ROOT / "output" / "trajectory_evaluation" / f"multi_batch_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    ckpt_arg = args.checkpoint
    student_ckpt_arg = args.student_checkpoint
    if not os.path.isabs(ckpt_arg):
        ckpt_resolved = (_PI_VLA_ROOT / ckpt_arg).resolve()
    else:
        ckpt_resolved = Path(ckpt_arg).resolve()
    if not os.path.isabs(student_ckpt_arg):
        student_ckpt_resolved = (_PI_VLA_ROOT / student_ckpt_arg).resolve()
    else:
        student_ckpt_resolved = Path(student_ckpt_arg).resolve()
    if not ckpt_resolved.is_file():
        print(f"Checkpoint not found: {ckpt_resolved}", file=sys.stderr)
        sys.exit(1)
    if not student_ckpt_resolved.is_file():
        print(f"Student checkpoint not found: {student_ckpt_resolved}", file=sys.stderr)
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
            "--student_checkpoint",
            student_ckpt_arg if not os.path.isabs(student_ckpt_arg) else str(student_ckpt_resolved),
            "--record_dir",
            str(run_dir),
            "--output_json",
            str(out_json),
        ]
        if args.seed is not None:
            # Deterministic but distinct RNG stream per grid position.
            cmd.extend(["--seed", str(int(args.seed) + i)])
        cmd.extend(["--ntfield_waypoint_mode", args.ntfield_waypoint_mode])
        cmd.extend(["--ntfield_fixed_waypoints", str(args.ntfield_fixed_waypoints)])
        cmd.extend(["--planner_playback", args.planner_playback])
        if args.no_video:
            cmd.append("--no_video")
        if args.use_viewer:
            cmd.append("--use_viewer")
        if args.save_final_geometric_debug:
            cmd.append("--save_final_geometric_debug")
            cmd.extend(["--final_geometric_debug_path", str(run_dir / args.final_geometric_debug_name)])
        if args.require_ee_object_collision:
            cmd.append("--require_ee_object_collision")
        cmd.extend(["--ee_proxy_radius_m", str(args.ee_proxy_radius_m)])
        cmd.extend(["--ee_proxy_center_mode", args.ee_proxy_center_mode])
        cmd.extend(["--ee_proxy_max_radius_m", str(args.ee_proxy_max_radius_m)])
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
        "benchmark_script": str(_BENCHMARK),
        "checkpoint": str(ckpt_resolved),
        "student_checkpoint": str(student_ckpt_resolved),
        "planner_playback": args.planner_playback,
        "ntfield_waypoint_mode": args.ntfield_waypoint_mode,
        "ntfield_fixed_waypoints": int(args.ntfield_fixed_waypoints),
        "save_final_geometric_debug": bool(args.save_final_geometric_debug),
        "final_geometric_debug_name": args.final_geometric_debug_name,
        "require_ee_object_collision": bool(args.require_ee_object_collision),
        "ee_proxy_radius_m": float(args.ee_proxy_radius_m),
        "ee_proxy_center_mode": args.ee_proxy_center_mode,
        "ee_proxy_max_radius_m": float(args.ee_proxy_max_radius_m),
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
