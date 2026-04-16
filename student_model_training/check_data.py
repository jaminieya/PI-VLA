import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Use the path from your training script
DATASET_ROOT = Path("/home/hojinsohn/VLM-NT/grasp_zgoal_wonorm_dataset_shards")

def inspect_data():
    # 1. Find shards
    shard_files = sorted(DATASET_ROOT.glob("grasp_dataset_shard_*5.pt"))
    if not shard_files:
        print(f"No shards found in {DATASET_ROOT}")
        return

    print(f"Found {len(shard_files)} shards.")
    first_shard_path = shard_files[0]
    print(f"Inspecting: {first_shard_path.name}")

    # 2. Load the shard
    # Shards usually contain a dict with 'images' and 'z_goals' (or configs/obj_locs)
    shard = torch.load(first_shard_path, map_location="cpu")

    print("\n--- Keys in Shard ---")
    for key in shard.keys():
        if isinstance(shard[key], torch.Tensor):
            shape_str = str(list(shard[key].shape))
            print(f"Key: '{key:10}' | Shape: {shape_str:15} | Dtype: {shard[key].dtype}")
        else:
            print(f"Key: '{key:10}' | Type: {type(shard[key])}")

    # 3. Validate Image Format
    images = shard["images"]
    # Check if images are 0-255 (uint8) or 0-1 (float)
    print(f"\nImage Range: min={images.min().item()}, max={images.max().item()}")

    # 4. Visualize a few samples
    n_samples = min(4, images.shape[0])
    fig, axes = plt.subplots(1, n_samples, figsize=(15, 5))
    
    # Identify which label key is being used
    label_key = None
    for k in ["z_goals", "configs", "obj_locs"]:
        if k in shard:
            label_key = k
            break

    for i in range(n_samples):
        img = images[3*i]
        
        # Handle (C, H, W) vs (H, W, C)
        if img.shape[0] == 3:
            img = img.permute(1, 2, 0)
        
        # Convert to numpy for matplotlib (ensure it's 0-1 range)
        img_np = img.float().numpy()
        if img_np.max() > 1.0:
            img_np /= 255.0

        axes[i].imshow(img_np)
        if label_key:
            label_val = shard[label_key][i]
            # If it's a long vector, just show the first few dimensions
            label_str = f"{label_key}[:6]:\n{label_val[:6].numpy()}"
            axes[i].set_title(label_str, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    save_path = "dataset_inspection.png"
    plt.savefig(save_path)
    print(f"\nInspection plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    inspect_data()