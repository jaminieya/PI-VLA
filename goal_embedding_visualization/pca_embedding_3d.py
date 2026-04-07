#!/usr/bin/env python3
"""
3D PCA visualization for 256-D (or any D) goal / latent embeddings.

Uses NumPy SVD only (no scikit-learn). Fits PCA on stacked samples so you can
overlay teacher vs student embeddings in one space when two arrays share rows.

Example:
  python pca_embedding_3d.py --embeddings z_teacher.npy --out pc1.png
  python pca_embedding_3d.py \\
    --embeddings z_teacher.npy --embeddings z_student.npy \\
    --out pca_teacher_student.png

After ``collect_data.py``, extract teacher latents then PCA (see
``extract_teacher_goal_embeddings_from_h5.py``). Bundles and PNGs default under
``PI-VLA/output/embedding_visualization/<grasp_6dof_demo_*>/``:

  python extract_teacher_goal_embeddings_from_h5.py --h5 '.../grasp_6dof_demo_T.h5' --checkpoint .../Model.pt
  python pca_embedding_3d.py --h5 '.../grasp_6dof_demo_T.h5'

  # Frame-colored trajectory (z_goal only):
  python pca_embedding_3d.py --h5 '.../grasp_6dof_demo_T.h5'
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PI_VLA_ROOT = os.path.dirname(_SCRIPT_DIR)
_OUTPUT_EMBED_ROOT = os.path.join(_PI_VLA_ROOT, "output", "embedding_visualization")


def _load_matrix(path: str, npz_key: str) -> np.ndarray:
    path = os.path.abspath(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if npz_key not in data:
            raise KeyError(f"--npz_key {npz_key!r} not in {path}; keys: {list(data.keys())}")
        x = np.asarray(data[npz_key], dtype=np.float64)
    else:
        x = np.load(path)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float64)
        else:
            x = x.astype(np.float64, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (N, D), got shape {x.shape} from {path}")
    return x


def _labels_load(path: Optional[str], n: int) -> Optional[np.ndarray]:
    if not path:
        return None
    lab = np.load(path)
    if lab.ndim != 1:
        lab = lab.reshape(-1)
    if lab.shape[0] != n:
        raise ValueError(f"Labels length {lab.shape[0]} != N={n}")
    return lab


def fit_transform_pca(
    x: np.ndarray,
    n_components: int = 3,
    standardize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      scores: (N, n_components) projected coordinates
      mean: (D,)
      components: (n_components, D) principal axes (rows)
      explained_var_ratio: (n_components,) fraction of total variance each
    """
    n, d = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples for PCA, got N={n}")
    if n_components > min(n - 1, d):
        raise ValueError(
            f"n_components={n_components} too large for N={n}, D={d} (max {min(n - 1, d)})"
        )

    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean

    if standardize:
        std = xc.std(axis=0, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)
        xc = xc / std

    # SVD: xc = U @ diag(s) @ Vt ; rows of Vt are principal directions
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    # Sample variance along each PC (matches sklearn with row-centered X)
    explained = (s**2) / max(n - 1, 1)
    total_var = explained.sum()
    ratio = explained[:n_components] / total_var if total_var > 0 else np.zeros(n_components, dtype=np.float64)
    comp = vt[:n_components, :]
    scores = xc @ comp.T
    return scores, mean.ravel(), comp, ratio


def main() -> None:
    p = argparse.ArgumentParser(description="3D PCA plot of goal / latent embeddings")
    p.add_argument(
        "--embeddings",
        type=str,
        action="append",
        default=None,
        help="Path to .npy (N,D) or .npz; pass twice for two sets (same N, same D). Omit if --npz_bundle.",
    )
    p.add_argument(
        "--h5",
        type=str,
        default="",
        help="Grasp demo .h5 path: sets --npz_bundle to output/embedding_visualization/<stem>/teacher_z_goal_bundle.npz",
    )
    p.add_argument(
        "--h5_glob",
        type=str,
        default="",
        help="Glob of grasp demo .h5 files for one combined PCA over all matching bundles",
    )
    p.add_argument(
        "--npz_bundle",
        type=str,
        default="",
        help="Single .npz from extract_teacher_goal_embeddings_from_h5.py (keys z_goal, labels_demo, ...)",
    )
    p.add_argument(
        "--embed_key",
        type=str,
        default="z_goal",
        help="Array inside --npz_bundle (default: z_goal)",
    )
    p.add_argument(
        "--label_key",
        type=str,
        default="labels_frame",
        help="Label array inside --npz_bundle for coloring (default: labels_frame=trajectory index; "
        "use labels_demo for multi-file bundles; empty = no labels)",
    )
    p.add_argument(
        "--npz_key",
        type=str,
        default="embeddings",
        help="Array key when loading generic .npz via --embeddings (default: embeddings)",
    )
    p.add_argument(
        "--labels",
        type=str,
        default="",
        help="Optional .npy (N,) labels for coloring first set (overrides --label_key from bundle)",
    )
    p.add_argument(
        "--standardize",
        action="store_true",
        help="Per-dimension z-score before PCA (after stacking all sets)",
    )
    p.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Random subsample size (0 = use all). Applied before fit; seed --seed",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="PNG path (default: next to --npz_bundle, or output/embedding_visualization/pca_embedding_3d.png)",
    )
    p.add_argument(
        "--out_variance",
        type=str,
        default="",
        help="Optional PNG path for PC1–K variance bar chart (K=--variance_k, default 20)",
    )
    p.add_argument("--variance_k", type=int, default=20)
    p.add_argument("--point_size", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=0.65)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--color_by_topk_pc",
        type=int,
        default=0,
        help="If >0, color each point by argmax(|PC score|) among first K PCs (e.g., 7).",
    )
    p.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    args = p.parse_args()

    if args.h5:
        if args.npz_bundle or args.h5_glob:
            p.error("Pass only one of --h5, --h5_glob, or --npz_bundle")
        stem = os.path.splitext(os.path.basename(os.path.abspath(args.h5)))[0]
        args.npz_bundle = os.path.join(_OUTPUT_EMBED_ROOT, stem, "teacher_z_goal_bundle.npz")

    bundle_labels = None
    data = None
    if args.h5_glob:
        if args.embeddings:
            p.error("Do not combine --h5_glob with --embeddings")
        h5_paths = sorted(glob.glob(os.path.abspath(args.h5_glob)))
        h5_paths = [p for p in h5_paths if os.path.isfile(p)]
        if not h5_paths:
            p.error(f"No files matched --h5_glob: {args.h5_glob}")

        z_list = []
        demo_labels = []
        for demo_idx, h5_path in enumerate(h5_paths):
            stem = os.path.splitext(os.path.basename(h5_path))[0]
            bpath = os.path.join(_OUTPUT_EMBED_ROOT, stem, "teacher_z_goal_bundle.npz")
            if not os.path.isfile(bpath):
                print(f"WARNING: missing bundle for {h5_path}; skipping {bpath}")
                continue
            d = np.load(bpath, allow_pickle=True)
            if args.embed_key not in d:
                raise KeyError(f"--embed_key {args.embed_key!r} not in {bpath}; keys: {list(d.keys())}")
            z = np.asarray(d[args.embed_key], dtype=np.float64)
            z_list.append(z)
            demo_labels.append(np.full(z.shape[0], demo_idx, dtype=np.int32))
        if not z_list:
            p.error("No valid bundles found for --h5_glob. Run extraction first.")
        mats = [np.vstack(z_list)]
        if not args.labels and args.label_key == "labels_demo":
            bundle_labels = np.concatenate(demo_labels, axis=0)
        paths = [f"h5_glob:{args.h5_glob}"]
    elif args.npz_bundle:
        bpath = os.path.abspath(args.npz_bundle)
        data = np.load(bpath, allow_pickle=True)
        embed_key_eff = args.embed_key
        if embed_key_eff not in data:
            raise KeyError(
                f"--embed_key {args.embed_key!r} not in {bpath}; keys: {list(data.keys())}"
            )
        mats = [np.asarray(data[embed_key_eff], dtype=np.float64)]
        if (
            embed_key_eff == "z_goal"
            and mats[0].shape[0] > 1
            and np.allclose(mats[0], mats[0][0], atol=1e-5, rtol=1e-5)
        ):
            print(
                "WARNING: z_goal has zero row-wise variance for this bundle "
                "(typical when one .h5 has a fixed final_joint_config)."
            )
        if args.label_key and args.label_key in data and not args.labels:
            bundle_labels = np.asarray(data[args.label_key]).reshape(-1)
        elif args.label_key and args.label_key not in data:
            raise KeyError(
                f"--label_key {args.label_key!r} not in {bpath}; keys: {list(data.keys())}"
            )
        paths = [bpath]
    else:
        paths = args.embeddings
        if not paths:
            p.error("Pass --embeddings (one or more) or --npz_bundle")
        mats = [_load_matrix(path, args.npz_key) for path in paths]
    d0 = mats[0].shape[1]
    for i, m in enumerate(mats):
        if m.shape[1] != d0:
            raise ValueError(f"Embedding dim mismatch: {paths[0]} has D={d0}, {paths[i]} has D={m.shape[1]}")
    n0 = mats[0].shape[0]
    for i, m in enumerate(mats):
        if m.shape[0] != n0:
            raise ValueError(
                f"Row count mismatch: {paths[0]} has N={n0}, {paths[i]} has N={m.shape[0]} "
                "(use one file or match sample counts for overlay)."
            )

    rng = np.random.default_rng(args.seed)
    idx = np.arange(n0)
    if args.max_points and args.max_points < n0:
        idx = rng.choice(idx, size=args.max_points, replace=False)

    stacks = [m[idx] for m in mats]
    x_all = np.vstack(stacks)
    n_fit = x_all.shape[0]

    scores_all, _, _, ratio3 = fit_transform_pca(x_all, n_components=3, standardize=args.standardize)
    k_report = min(7, min(x_all.shape[0] - 1, x_all.shape[1]))
    _, _, _, ratio7 = fit_transform_pca(x_all, n_components=k_report, standardize=args.standardize)

    pc_color_labels = None
    if args.color_by_topk_pc and args.color_by_topk_pc > 0:
        k_color = min(max(1, args.color_by_topk_pc), min(x_all.shape[0] - 1, x_all.shape[1]))
        scores_k, _, _, _ = fit_transform_pca(x_all, n_components=k_color, standardize=args.standardize)
        pc_color_labels = np.argmax(np.abs(scores_k), axis=1).astype(np.int32)

    # Split scores back per original set
    chunks = []
    off = 0
    for m in stacks:
        k = m.shape[0]
        chunks.append(scores_all[off : off + k, :])
        off += k

    if pc_color_labels is not None:
        labels = pc_color_labels
    elif args.labels:
        labels = _labels_load(args.labels, n0)
    elif bundle_labels is not None:
        if bundle_labels.shape[0] != n0:
            raise ValueError(f"Bundle labels length {bundle_labels.shape[0]} != N={n0}")
        labels = bundle_labels
    else:
        labels = None
    if labels is not None:
        labels = labels[idx]

    if args.out:
        out_png = os.path.abspath(args.out)
    elif args.h5_glob:
        os.makedirs(_OUTPUT_EMBED_ROOT, exist_ok=True)
        out_png = os.path.join(_OUTPUT_EMBED_ROOT, "pca_embedding_3d_all_h5.png")
    elif args.npz_bundle:
        out_png = os.path.join(
            os.path.dirname(os.path.abspath(args.npz_bundle)), "pca_embedding_3d.png"
        )
    else:
        os.makedirs(_OUTPUT_EMBED_ROOT, exist_ok=True)
        out_png = os.path.join(_OUTPUT_EMBED_ROOT, "pca_embedding_3d.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    names = ("set_A", "set_B", "set_C", "set_D")

    if len(chunks) == 1:
        sc = None
        if labels is not None:
            cmap = "tab10"
            norm = None
            if pc_color_labels is not None:
                k = int(args.color_by_topk_pc)
                base = plt.cm.get_cmap("tab10", k)
                cmap = ListedColormap(base(np.arange(k)))
                norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1.0), cmap.N)
            sc = ax.scatter(
                chunks[0][:, 0],
                chunks[0][:, 1],
                chunks[0][:, 2],
                c=labels,
                cmap=cmap,
                norm=norm,
                s=args.point_size,
                alpha=args.alpha,
            )
            if pc_color_labels is not None:
                cbar_lbl = f"dominant PC index in top-{args.color_by_topk_pc}"
            elif args.label_key == "labels_demo":
                cbar_lbl = "demo index"
            elif args.label_key == "labels_frame":
                cbar_lbl = "frame index"
            else:
                cbar_lbl = "label"
            cbar = fig.colorbar(sc, ax=ax, shrink=0.6, label=cbar_lbl)
            if pc_color_labels is not None:
                k = int(args.color_by_topk_pc)
                ticks = np.arange(0, k)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"PC{i+1}" for i in ticks])
        else:
            ax.scatter(
                chunks[0][:, 0],
                chunks[0][:, 1],
                chunks[0][:, 2],
                c=colors[0],
                s=args.point_size,
                alpha=args.alpha,
                label=names[0],
            )
    else:
        for i, ch in enumerate(chunks):
            ax.scatter(
                ch[:, 0],
                ch[:, 1],
                ch[:, 2],
                c=colors[i % len(colors)],
                s=args.point_size,
                alpha=args.alpha,
                label=names[i] if i < len(names) else f"set_{i}",
            )
        ax.legend(loc="upper left", fontsize=9)

    ratio7_txt = ", ".join([f"{v*100:.2f}%" for v in ratio7])
    var_lines = (
        f"PC1–{k_report} explained variance (of total): "
        f"{ratio7_txt} (cumulative {(ratio7.sum())*100:.2f}%)"
    )
    ax.set_title("PCA 3D — dominant linear variance")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.text(0.02, 0.02, var_lines, fontsize=9, wrap=True)

    std_note = "standardized per dim" if args.standardize else "centered only"
    fig.text(0.02, 0.06, f"N_fit={n_fit}, D={d0}, {std_note}", fontsize=9)

    fig.tight_layout()
    plt.savefig(out_png, dpi=args.dpi)
    print(out_png)
    print(var_lines)

    if args.out_variance:
        k_plot = min(max(args.variance_k, 1), min(n_fit - 1, d0))
        _, _, _, ratio_k = fit_transform_pca(x_all, n_components=k_plot, standardize=args.standardize)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        xs = np.arange(1, k_plot + 1)
        ax2.bar(xs, ratio_k * 100.0, color="steelblue")
        csum = np.cumsum(ratio_k) * 100.0
        ax2.plot(xs, csum, color="darkorange", marker="o", linewidth=1.5, label="cumulative %")
        ax2.set_xlabel("Principal component")
        ax2.set_ylabel("Explained variance (%)")
        ax2.set_title(f"Scree / cumulative variance (first {k_plot} PCs)")
        ax2.legend()
        ax2.set_xticks(xs)
        fig2.tight_layout()
        vpath = os.path.abspath(args.out_variance)
        _vd = os.path.dirname(vpath)
        if _vd:
            os.makedirs(_vd, exist_ok=True)
        plt.savefig(vpath, dpi=args.dpi)
        print(vpath)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
