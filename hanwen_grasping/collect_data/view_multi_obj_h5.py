#!/usr/bin/env python3
"""Inspect grasp_multi3_demo_*.h5 files (print contents; optional image window or PNG)."""

import argparse
import os
import sys

import h5py
import numpy as np


def _write_start_image_png(h5_path: str, out_png: str) -> None:
    with h5py.File(h5_path, "r") as f:
        img = np.array(f["start_image"])
    try:
        from PIL import Image

        Image.fromarray(img).save(out_png)
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6.75))
        ax.imshow(img)
        ax.set_title(os.path.basename(h5_path))
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote {out_png}")


def _show_interactive(h5_path: str) -> None:
    import matplotlib

    if os.environ.get("MPLBACKEND"):
        matplotlib.use(os.environ["MPLBACKEND"], force=True)
    else:
        for backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue

    import matplotlib.pyplot as plt

    backend = matplotlib.get_backend().lower()
    if backend == "agg":
        default_out = os.path.splitext(h5_path)[0] + "_start_image.png"
        print(
            "Matplotlib is using the Agg backend (no display). "
            f"Writing PNG instead: {default_out}",
            file=sys.stderr,
        )
        _write_start_image_png(h5_path, default_out)
        print(
            "Tip: set a GUI backend, e.g. `export MPLBACKEND=TkAgg` or use `--save out.png`.",
            file=sys.stderr,
        )
        return

    with h5py.File(h5_path, "r") as f:
        img = np.array(f["start_image"])
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(os.path.basename(h5_path))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="View multi-object grasp HDF5 samples")
    parser.add_argument("h5_path", help="Path to .h5 file")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open start_image in a matplotlib window (needs a GUI backend / display)",
    )
    parser.add_argument(
        "--save",
        metavar="PNG",
        help="Write start_image to a PNG file (works without a display; e.g. SCP and open locally)",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.h5_path)
    if not os.path.isfile(path):
        print(f"Not found: {path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(path, "r") as f:
        print(f"File: {path}\n")
        print("Attributes:")
        for k, v in f.attrs.items():
            print(f"  {k}: {v}")
        print("\nDatasets:")
        for name in f:
            d = f[name]
            print(f"  {name}: shape={d.shape} dtype={d.dtype}")
            if name in ("object_names", "object_id_folders"):
                for i in range(len(d)):
                    s = d[i]
                    if isinstance(s, bytes):
                        s = s.decode("utf-8")
                    print(f"    [{i}] {s}")
            elif name in (
                "start_joint_config",
                "goal_joint_config",
                "object_locations",
                "goal_joint_configs",
            ):
                arr = np.array(d)
                print(f"    {arr}")

    if args.save:
        _write_start_image_png(path, os.path.abspath(args.save))

    if args.show:
        _show_interactive(path)


if __name__ == "__main__":
    main()
