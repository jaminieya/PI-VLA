#!/usr/bin/env python3
"""
extract_features.py

Pre-computes and caches ResNet-50 image embeddings and CLIP text embeddings
from the raw image shards, producing new "feature shards" that contain only:

    "img_feat"         (N, 2048)  float32  — ResNet-50 avgpool output (before proj)
    "txt_feat"         (N, 512)   float32  — CLIP text embedding (L2-normalised)
    "object_location"  (N, 3)     float32  — XYZ target (unchanged)
    "prompt"           list[str]  length N
    "object_name"      list[str]  length N
    "object_idx"       list[int]  length N
    "source_file"      list[str]  length N
    "num_samples"      int

Why: frozen encoder forward passes dominate training time (4s/batch on RTX 3090
at 0% GPU util). Pre-extracting once means training only runs the tiny
FiLM + MLP head, reducing per-epoch time from ~hours to ~minutes.

Text embeddings are de-duplicated: there are only ~N_unique prompts
("grasp banana", "grasp sugar box", ...) so we compute each once and reuse.

Usage:
    python extract_features.py \
        --input-dir  /path/to/multi_obj_layout_shards \
        --output-dir /path/to/multi_obj_layout_feat_shards \
        --batch-size 256 \
        --device cuda

    # Then point train_fast.py at the new output-dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR  = Path("/scratch/scholar/sohn31/grasp_dataset_shards_multi")
DEFAULT_OUTPUT_DIR = Path("/scratch/scholar/sohn31/grasp_feat_shards_multi")
DEFAULT_BATCH      = 256
DEFAULT_NUM_WORKERS = 4


# ---------------------------------------------------------------------------
# Minimal dataset that reads raw image shards
# ---------------------------------------------------------------------------

class RawShardDataset(Dataset):
    def __init__(self, shard_path: Path) -> None:
        shard = torch.load(shard_path)
        self.images           = shard["image"]            # (N, 3, H, W)
        self.object_locations = shard["object_location"]  # (N, 3)
        self.prompts          = shard["prompt"]
        self.object_names     = shard["object_name"]
        self.object_idxs      = shard["object_idx"]
        self.source_files     = shard["source_file"]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {
            "image":           self.images[idx],
            "object_location": self.object_locations[idx],
            "prompt":          self.prompts[idx],
            "object_name":     self.object_names[idx],
            "object_idx":      self.object_idxs[idx],
            "source_file":     self.source_files[idx],
        }


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

def build_image_encoder(device: torch.device) -> nn.Module:
    resnet = tvm.resnet50(pretrained=True)
    encoder = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def build_text_encoder(device: torch.device):
    try:
        import open_clip
    except ImportError:
        raise ImportError("pip install open_clip_torch")

    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer  = open_clip.get_tokenizer("ViT-B-32")

    transformer       = clip.transformer.to(device).eval()
    token_embedding   = clip.token_embedding.to(device).eval()
    positional_emb    = clip.positional_embedding.to(device)
    ln_final          = clip.ln_final.to(device).eval()
    text_projection   = clip.text_projection.to(device)
    attn_mask         = clip.attn_mask.to(device) if clip.attn_mask is not None else None

    for p in transformer.parameters():    p.requires_grad_(False)
    for p in token_embedding.parameters(): p.requires_grad_(False)
    for p in ln_final.parameters():       p.requires_grad_(False)
    # text_projection and positional_embedding are plain tensors, not nn.Parameters
    # so we just detach and freeze them via no_grad context at call time

    @torch.no_grad()
    def encode(prompts: list[str]) -> torch.Tensor:
        tokens = tokenizer(prompts).to(device)
        x = token_embedding(tokens) + positional_emb
        x = x.permute(1, 0, 2)
        x = transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)
        x = ln_final(x)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        x = x @ text_projection
        return x / x.norm(dim=-1, keepdim=True)   # (B, 512) L2-normalised

    return encode


# ---------------------------------------------------------------------------
# Process one shard
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_shard(
    shard_path: Path,
    output_path: Path,
    image_encoder: nn.Module,
    text_encode_fn,
    text_cache: dict,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> int:
    dataset = RawShardDataset(shard_path)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_img_feats  = []
    all_txt_feats  = []
    all_locations  = []
    all_prompts    = []
    all_names      = []
    all_idxs       = []
    all_sources    = []

    for batch in loader:
        images  = batch["image"].to(device)
        prompts = batch["prompt"]

        # ── Image features ────────────────────────────────────────────
        img_feats = image_encoder(images).squeeze(-1).squeeze(-1)  # (B, 2048)
        all_img_feats.append(img_feats.cpu())

        # ── Text features (cached per unique prompt) ──────────────────
        unique_prompts = list(dict.fromkeys(prompts))   # deduplicate, preserve order
        for p in unique_prompts:
            if p not in text_cache:
                text_cache[p] = text_encode_fn([p])[0].cpu()  # (512,)

        txt_feats = torch.stack([text_cache[p] for p in prompts])  # (B, 512)
        all_txt_feats.append(txt_feats)

        all_locations.append(batch["object_location"])
        all_prompts.extend(prompts)
        all_names.extend(batch["object_name"])
        all_idxs.extend(
            batch["object_idx"].tolist()
            if isinstance(batch["object_idx"], torch.Tensor)
            else batch["object_idx"]
        )
        all_sources.extend(batch["source_file"])

    feat_shard = {
        "img_feat":        torch.cat(all_img_feats, dim=0),   # (N, 2048)
        "txt_feat":        torch.cat(all_txt_feats, dim=0),   # (N, 512)
        "object_location": torch.cat(all_locations, dim=0),   # (N, 3)
        "prompt":          all_prompts,
        "object_name":     all_names,
        "object_idx":      all_idxs,
        "source_file":     all_sources,
        "num_samples":     len(all_prompts),
    }
    torch.save(feat_shard, output_path)
    return len(all_prompts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    input_dir:  Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Discover shards
    manifest_path = input_dir / "manifest.pt"
    if manifest_path.exists():
        manifest    = torch.load(manifest_path)
        shard_paths = [Path(p) for p in manifest["shards"]]
    else:
        shard_paths = sorted(input_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {input_dir}")

    if args.max_shards:
        shard_paths = shard_paths[:args.max_shards]

    print(f"Found {len(shard_paths)} shard(s) to process")

    # Build encoders
    print("Loading ResNet-50...")
    image_encoder = build_image_encoder(device)
    print("Loading CLIP text encoder...")
    text_encode_fn = build_text_encoder(device)

    text_cache: dict = {}   # prompt str → (512,) tensor
    total_samples = 0
    output_shard_paths = []

    for i, shard_path in enumerate(tqdm(shard_paths, desc="Shards")):
        output_path = output_dir / shard_path.name
        output_shard_paths.append(str(output_path))

        n = process_shard(
            shard_path    = shard_path,
            output_path   = output_path,
            image_encoder = image_encoder,
            text_encode_fn= text_encode_fn,
            text_cache    = text_cache,
            batch_size    = args.batch_size,
            num_workers   = args.num_workers,
            device        = device,
        )
        total_samples += n

    # Write manifest
    feat_manifest = {
        "num_samples":  total_samples,
        "num_shards":   len(output_shard_paths),
        "shards":       output_shard_paths,
        "shard_size":   manifest["shard_size"] if manifest_path.exists() else args.batch_size,
        "tensor_keys":  ["img_feat", "txt_feat", "object_location"],
        "meta_keys":    ["prompt", "object_name", "object_idx", "source_file"],
        "img_feat_dim": 2048,
        "txt_feat_dim": 512,
    }
    torch.save(feat_manifest, output_dir / "manifest.pt")

    print(f"\nDone.")
    print(f"  Shards    : {len(output_shard_paths)}")
    print(f"  Samples   : {total_samples}")
    print(f"  Unique prompts cached: {len(text_cache)} → {list(text_cache.keys())}")
    print(f"  Output    : {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",    type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size",   type=int,  default=DEFAULT_BATCH,
                   help="Batch size for encoder inference (default: 256)")
    p.add_argument("--num-workers",  type=int,  default=DEFAULT_NUM_WORKERS)
    p.add_argument("--device",       type=str,  default="cuda")
    p.add_argument("--max-shards",   type=int,  default=None,
                   help="Process only first N shards (for testing)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())