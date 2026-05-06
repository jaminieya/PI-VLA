import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# python PI-VLA/student_model_training/visualize_shard_sample.py --shard PI-VLA/student_model_training/data/pt_shards_multi/shard_51.pt --index 14 --output PI-VLA/student_model_training/output/shard_51_sample_14.png
def to_display_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) tensor to matplotlib-ready (H,W,C)."""
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got shape {tuple(image_tensor.shape)}")

    if image_tensor.shape[0] == 3:
        image_tensor = image_tensor.permute(1, 2, 0)

    image_np = image_tensor.detach().cpu().float().numpy()

    # Robustly clamp regardless of whether data is [0,1] or [0,255].
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    image_np = np.clip(image_np, 0.0, 1.0)
    return image_np


def visualize_one_sample(shard_path: Path, sample_idx: int, output_path: Path) -> None:
    samples = torch.load(shard_path, map_location="cpu")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Expected non-empty list in shard, got {type(samples)}")
    if sample_idx < 0 or sample_idx >= len(samples):
        raise IndexError(f"sample_idx={sample_idx} is out of range [0, {len(samples) - 1}]")

    sample = samples[sample_idx]
    image_np = to_display_image(sample["image"])

    object_name = sample.get("object_name", "N/A")
    object_id_folder = sample.get("object_id_folder", "N/A")
    source_file = sample.get("source_file", "N/A")

    object_location = sample.get("object_location")
    if isinstance(object_location, torch.Tensor):
        object_location = object_location.detach().cpu().numpy().tolist()

    q_goal = sample.get("q_goal")
    if isinstance(q_goal, torch.Tensor):
        q_goal = q_goal.detach().cpu().numpy().tolist()

    z_goal = sample.get("z_goal")
    if isinstance(z_goal, torch.Tensor):
        z_goal_preview = z_goal.detach().cpu().numpy()[:8].tolist()
    else:
        z_goal_preview = "N/A"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].imshow(image_np)
    axes[0].set_title(f"Sample {sample_idx}: {object_name}")
    axes[0].axis("off")

    metadata_text = (
        f"Shard: {shard_path.name}\n"
        f"Sample index: {sample_idx}\n\n"
        f"Object: {object_name}\n"
        f"Object folder: {object_id_folder}\n"
        f"Source file: {source_file}\n\n"
        f"Object location (xyz):\n{object_location}\n\n"
        f"q_goal:\n{q_goal}\n\n"
        f"z_goal[:8]:\n{z_goal_preview}"
    )

    axes[1].axis("off")
    axes[1].text(
        0.0,
        1.0,
        metadata_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved visualization to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one sample from a .pt shard with metadata.")
    parser.add_argument(
        "--shard",
        type=Path,
        default=Path("PI-VLA/student_model_training/data/pt_shards_multi/shard_51.pt"),
        help="Path to shard .pt file",
    )
    parser.add_argument("--index", type=int, default=0, help="Sample index inside shard")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("PI-VLA/student_model_training/output/shard_51_sample_0.png"),
        help="Where to save the rendered image + metadata panel",
    )
    args = parser.parse_args()

    visualize_one_sample(args.shard, args.index, args.output)


if __name__ == "__main__":
    main()
