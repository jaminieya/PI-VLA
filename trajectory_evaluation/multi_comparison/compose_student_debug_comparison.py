#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


DEFAULT_STUDENT_TYPES = [
    "student_multi_batch_mdn",
    "student_multi_batch_regression_hybrid_contra",
    "student_multi_batch_regression_hybrid_cos",
    "student_multi_batch_regression_mse",
]

DISPLAY_NAMES = {
    "student_multi_batch_mdn": "MDN",
    "student_multi_batch_regression_hybrid_contra": "Contrastive",
    "student_multi_batch_regression_hybrid_cos": "Hybrid Distillation",
    "student_multi_batch_regression_mse": "MSE",
}


def _find_run_dirs(student_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in sorted(student_dir.glob("run_*")):
        if d.is_dir():
            out[d.name] = d
    return out


def _load_result_json(run_dir: Path) -> dict | None:
    p = run_dir / "result.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _resolve_debug_image(run_dir: Path) -> Path | None:
    result = _load_result_json(run_dir)
    if isinstance(result, dict):
        p = result.get("ee_object_pair_collision_selected_grasp", {}).get("debug_image_path")
        if isinstance(p, str):
            path = Path(p)
            if path.is_file():
                return path
    fallback = run_dir / "final_geometric_debug.png"
    if fallback.is_file():
        return fallback
    return None


def _compose_one_run(run_name: str, run_paths: list[tuple[str, Path]], out_path: Path) -> None:
    n = len(run_paths)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, run_dir) in zip(axes, run_paths):
        img_path = _resolve_debug_image(run_dir)
        if img_path is None:
            ax.text(0.5, 0.5, "Missing\nfinal_geometric_debug.png", ha="center", va="center")
            ax.set_facecolor((0.95, 0.95, 0.95))
        else:
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side comparison images for final geometric debug images "
            "across multiple student run types."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("PI-VLA/output/trajectory_evaluation"),
        help="Root containing student_multi_batch_* directories.",
    )
    parser.add_argument(
        "--student-types",
        nargs="+",
        default=DEFAULT_STUDENT_TYPES,
        help="Run-type directory names under --output-root.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Specific run folder name to compose (e.g. run_00_x0.5000_y0.3000).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("PI-VLA/output/trajectory_evaluation/student_debug_comparison"),
        help="Output directory for composed side-by-side images.",
    )
    args = parser.parse_args()

    type_to_runs: dict[str, dict[str, Path]] = {}
    for st in args.student_types:
        st_dir = args.output_root / st
        if not st_dir.is_dir():
            continue
        type_to_runs[st] = _find_run_dirs(st_dir)

    if not type_to_runs:
        raise RuntimeError("No valid student type directories found.")

    all_run_names = set()
    for runs in type_to_runs.values():
        all_run_names.update(runs.keys())
    if args.run_name:
        run_names = [args.run_name]
    else:
        run_names = sorted(all_run_names)

    made = 0
    for run_name in run_names:
        run_paths: list[tuple[str, Path]] = []
        for st in args.student_types:
            runs = type_to_runs.get(st, {})
            rd = runs.get(run_name)
            if rd is None:
                continue
            run_paths.append((DISPLAY_NAMES.get(st, st), rd))
        if not run_paths:
            continue
        out_path = args.out_dir / f"{run_name}_comparison.png"
        _compose_one_run(run_name, run_paths, out_path)
        made += 1
        print(f"Wrote {out_path}")

    print(f"Done. Created {made} comparison image(s).")


if __name__ == "__main__":
    main()
