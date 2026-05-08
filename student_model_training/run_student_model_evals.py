"""
Run evaluate_student_model.py over multiple checkpoints and save plot-ready stats.

Default checkpoints match the list requested in evaluate_student_model.py docstring.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_CHECKPOINTS = [
    "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_mse_bs256_lr3em4_ep90_20260507_155428.pth",
    "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_contra_bs256_lr3em4_ep40_20260507_201445.pth",
    "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_bs256_lr3em4_ep40_20260507_180418.pth",
    "/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep90_20260505_114200.pth",
]


def _row_from_payload(payload: dict):
    metrics = payload["metrics"]
    row = {
        "checkpoint": payload["checkpoint"],
        "checkpoint_name": Path(payload["checkpoint"]).name,
        "model_type": payload["model_type"],
        "split": payload["split"],
        "batch_size": payload["batch_size"],
        "seed": payload["seed"],
        "val_fraction": payload["val_fraction"],
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "cos_distance": metrics["cos_distance"],
        "l2_mean": metrics["l2_mean"],
        "l2_median": metrics["l2_median"],
        "n_samples": metrics["n_samples"],
    }
    for thr, acc in metrics.get("accuracy_at_l2_threshold", {}).items():
        key = f"acc_l2_le_{str(thr).replace('.', 'p')}"
        row[key] = acc
    return row


def _display_name(row: dict) -> str:
    name = row["checkpoint_name"].replace(".pth", "")
    name = name.replace("best_z_goal_model_", "")
    return name


def _save_comparison_plots(rows, run_dir: Path):
    if not rows:
        return []

    labels = [_display_name(r) for r in rows]
    metric_specs = [
        ("mae", "MAE (lower is better)"),
        ("mse", "MSE (lower is better)"),
        ("cos_distance", "Cosine Distance (lower is better)"),
        ("l2_mean", "L2 Mean (lower is better)"),
        ("l2_median", "L2 Median (lower is better)"),
    ]

    saved = []

    # Plot 1: core error metrics as bar charts
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()
    for i, (metric_key, title) in enumerate(metric_specs):
        ax = axes[i]
        vals = [float(r[metric_key]) for r in rows]
        bars = ax.bar(range(len(rows)), vals, alpha=0.9)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{v:.4g}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Hide unused subplot (6th slot)
    axes[-1].axis("off")
    fig.tight_layout()
    p_core = run_dir / "compare_core_metrics.png"
    fig.savefig(p_core, dpi=160)
    plt.close(fig)
    saved.append(p_core)

    # Plot 2: threshold accuracies
    acc_keys = sorted(k for k in rows[0].keys() if k.startswith("acc_l2_le_"))
    if acc_keys:
        plt.figure(figsize=(12, 7))
        for r in rows:
            y = [float(r.get(k, 0.0)) for k in acc_keys]
            x = list(range(len(acc_keys)))
            plt.plot(x, y, marker="o", linewidth=2, label=_display_name(r))
        xlabels = [k.replace("acc_l2_le_", "").replace("p", ".") for k in acc_keys]
        plt.xticks(range(len(acc_keys)), xlabels)
        plt.ylim(0.0, 1.05)
        plt.xlabel("L2 Error Threshold")
        plt.ylabel("Accuracy (fraction <= threshold)")
        plt.title("Threshold Accuracy Comparison")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        p_acc = run_dir / "compare_threshold_accuracy.png"
        plt.savefig(p_acc, dpi=160)
        plt.close()
        saved.append(p_acc)

    return saved


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Batch-run evaluate_student_model.py and save combined stats."
    )
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=here / "evaluate_student_model.py",
        help="Path to evaluate_student_model.py",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
        help="Checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "eval_outputs",
        help="Directory to save per-checkpoint and summary outputs.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=here / "data/pt_shards_multi",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.5,1.0")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    args = parser.parse_args()

    if not args.evaluate_script.is_file():
        raise FileNotFoundError(f"Evaluate script not found: {args.evaluate_script}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    for ckpt_str in args.checkpoints:
        ckpt = Path(ckpt_str)
        if not ckpt.is_file():
            failures.append({"checkpoint": str(ckpt), "error": "checkpoint_not_found"})
            print(f"[skip] missing checkpoint: {ckpt}")
            continue

        per_json = run_dir / f"{ckpt.stem}.json"
        cmd = [
            args.python_exe,
            str(args.evaluate_script),
            "--checkpoint",
            str(ckpt),
            "--dataset-root",
            str(args.dataset_root),
            "--split",
            args.split,
            "--val-fraction",
            str(args.val_fraction),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--thresholds",
            args.thresholds,
            "--save-json",
            str(per_json),
        ]

        print(f"[run] evaluating {ckpt.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            failures.append(
                {
                    "checkpoint": str(ckpt),
                    "error": "evaluation_failed",
                    "returncode": result.returncode,
                    "stderr": result.stderr[-5000:],
                    "stdout_tail": result.stdout[-5000:],
                }
            )
            print(f"[fail] {ckpt.name} (code={result.returncode})")
            continue

        with per_json.open() as f:
            payload = json.load(f)
        rows.append(_row_from_payload(payload))
        print(f"[ok] {ckpt.name}")

    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"

    summary = {
        "run_dir": str(run_dir),
        "evaluate_script": str(args.evaluate_script),
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "thresholds": args.thresholds,
        "num_success": len(rows),
        "num_failures": len(failures),
        "results": rows,
        "failures": failures,
    }
    with summary_json.open("w") as f:
        json.dump(summary, f, indent=2)

    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        plot_paths = _save_comparison_plots(rows, run_dir)
    else:
        plot_paths = []

    print(f"\nSaved summary JSON: {summary_json}")
    if rows:
        print(f"Saved summary CSV:  {summary_csv}")
    for p in plot_paths:
        print(f"Saved plot:         {p}")
    if failures:
        print(f"Completed with failures: {len(failures)}")


if __name__ == "__main__":
    main()
