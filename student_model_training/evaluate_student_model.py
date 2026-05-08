"""
Evaluate trained student latent-goal models (regression or MDN).

This script mirrors the shard loading and validation split logic from:
  - full_train_multi_regression.py
  - full_train_multi_mdn.py

Reported metrics:
  - MSE (mean squared error)
  - MAE (mean absolute error)
  - cosine distance (1 - cosine similarity)
  - L2 mean / median
  - threshold accuracies (fraction of samples with L2 error <= threshold)

Example:
  python evaluate_student_model.py \
    --checkpoint /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_mse_bs256_lr3em4_ep90_20260507_155428.pth \
    --dataset-root /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi \
    --split val --batch-size 256 --seed 42

    /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_mse_bs256_lr3em4_ep90_20260507_155428.pth
    /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_contra_bs256_lr3em4_ep40_20260507_201445.pth
    /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_regression_hybrid_bs256_lr3em4_ep40_20260507_180418.pth
    /home/hojinsohn/VLM-NT/PI-VLA/student_model_training/best_z_goal_model_mdn_mdn_K8_bs256_lr3em4_ep90_20260505_114200.pth
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from student_model_mdn import MDNStudent
from student_model_regression import RegressionStudent


def label_tensor_from_shard(shard):
    if "z_goals" in shard:
        return shard["z_goals"]
    if "configs" in shard:
        c = shard["configs"]
        return c[:, -1, :] if c.dim() == 3 else c
    if "obj_locs" in shard:
        return shard["obj_locs"]
    raise ValueError("Shard must contain 'z_goals', 'configs', or 'obj_locs'.")


def _is_list_shard(s):
    return isinstance(s, list)


def discover_shards(root: Path):
    shard_files = sorted(root.glob("grasp_dataset_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(root.glob("shard_*.pt"))
    if not shard_files:
        raise ValueError(f"No shard files found under {root}")

    cumulative = []
    total = 0
    z_dim = None
    object_names = set()
    for sp in shard_files:
        shard = torch.load(sp, map_location="cpu")
        if _is_list_shard(shard):
            if not shard:
                continue
            n = len(shard)
            s0 = shard[0]
            if "z_goal" not in s0 or "image" not in s0:
                raise ValueError(f"List shard missing 'image'/'z_goal' in {sp}")
            if z_dim is None:
                z_dim = int(s0["z_goal"].numel())
            for dp in shard:
                if "object_name" in dp:
                    object_names.add(str(dp["object_name"]))
        else:
            if "images" not in shard:
                raise ValueError(f"Missing 'images' in {sp}")
            labels = label_tensor_from_shard(shard)
            n = shard["images"].shape[0]
            if z_dim is None:
                z_dim = int(labels.shape[-1])
        total += n
        cumulative.append(total)

    if not cumulative:
        raise ValueError(f"All discovered shards are empty under {root}")
    return shard_files, cumulative, int(z_dim), sorted(object_names)


def build_train_val_split(n_total: int, val_fraction: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    val_size = int(val_fraction * n_total)
    train_size = n_total - val_size
    train_idx = set(perm[:train_size].tolist())
    val_idx = set(perm[train_size:].tolist())
    return train_idx, val_idx


def train_val_indices_for_shard(shard_idx, cumulative, train_idx, val_idx):
    start = 0 if shard_idx == 0 else cumulative[shard_idx - 1]
    shard_len = cumulative[shard_idx] - start
    train_locals, val_locals = [], []
    for j in range(shard_len):
        g = start + j
        if g in train_idx:
            train_locals.append(j)
        elif g in val_idx:
            val_locals.append(j)
    return train_locals, val_locals


def prompt_from_object_name(name: str):
    return f"grasp {str(name).strip().lower()}"


def _tokenize_prompt(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_text_vocab(object_names: Sequence[str]):
    token_set = set()
    for name in object_names:
        token_set.update(_tokenize_prompt(prompt_from_object_name(name)))
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for tok in sorted(token_set):
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    return token_to_id


def encode_prompts(prompts: Sequence[str], token_to_id: Dict[str, int], max_len: int):
    pad_id = token_to_id["<pad>"]
    unk_id = token_to_id["<unk>"]
    out = torch.full((len(prompts), max_len), pad_id, dtype=torch.long)
    for i, p in enumerate(prompts):
        toks = _tokenize_prompt(p)[:max_len]
        if not toks:
            toks = ["<unk>"]
        ids = [token_to_id.get(t, unk_id) for t in toks]
        out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return out


def parse_thresholds(text: str) -> List[float]:
    vals = []
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("At least one threshold must be provided.")
    return vals


def infer_model_type(ckpt: dict) -> str:
    model_type = str(ckpt.get("model_type", "")).strip().lower()
    if model_type in {"regression", "mdn"}:
        return model_type
    if "n_components" in ckpt:
        return "mdn"
    return "regression"


def build_model(
    ckpt: dict,
    model_type: str,
    z_dim: int,
    object_names: Sequence[str],
    mdn_max_prompt_len: int,
    dropout_p: float,
    device: torch.device,
):
    if model_type == "mdn":
        token_to_id = ckpt.get("token_to_id")
        if not isinstance(token_to_id, dict):
            token_to_id = build_text_vocab(object_names)
            token_to_id.setdefault("<pad>", 0)
            token_to_id.setdefault("<unk>", 1)

        max_prompt_len = int(ckpt.get("max_prompt_len", mdn_max_prompt_len))
        n_components = int(ckpt.get("n_components", 8))
        vocab_size = int(ckpt.get("vocab_size", len(token_to_id)))
        model = MDNStudent(
            output_dim=z_dim,
            vocab_size=vocab_size,
            n_components=n_components,
            dropout_p=dropout_p,
        ).to(device)
        return model, token_to_id, max_prompt_len

    model = RegressionStudent(
        output_dim=z_dim,
        dropout_p=dropout_p,
    ).to(device)
    return model, None, None


@torch.no_grad()
def evaluate(
    model,
    model_type: str,
    device: torch.device,
    shard_files: Sequence[Path],
    cumulative: Sequence[int],
    split_name: str,
    split_indices: set,
    normalize,
    batch_size: int,
    thresholds: Sequence[float],
    token_to_id=None,
    max_prompt_len=None,
):
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    cos_dist_sum = 0.0
    l2_sum = 0.0
    n_samples = 0

    sq_err = nn.MSELoss(reduction="sum")
    abs_err = nn.L1Loss(reduction="sum")
    threshold_hits = np.zeros(len(thresholds), dtype=np.int64)
    l2_errors_all = []

    for si, sp in enumerate(shard_files):
        if split_name == "all":
            start = 0 if si == 0 else cumulative[si - 1]
            shard_len = cumulative[si] - start
            local_indices = list(range(shard_len))
        else:
            train_locals, val_locals = train_val_indices_for_shard(
                si, cumulative, split_indices if split_name == "train" else set(),
                split_indices if split_name == "val" else set()
            )
            local_indices = train_locals if split_name == "train" else val_locals

        if not local_indices:
            continue

        shard = torch.load(sp, map_location="cpu")
        list_shard = _is_list_shard(shard)
        if not list_shard:
            images = shard["images"]
            labels = label_tensor_from_shard(shard)

        for start_idx in range(0, len(local_indices), batch_size):
            chunk = local_indices[start_idx : start_idx + batch_size]
            if list_shard:
                dps = [shard[i] for i in chunk]
                x = torch.stack([dp["image"] for dp in dps]).to(device, non_blocking=True)
                y = torch.stack([dp["z_goal"] for dp in dps]).to(device, non_blocking=True)
                prompts = [prompt_from_object_name(dp.get("object_name", "object")) for dp in dps]
            else:
                x = images[chunk].to(device, non_blocking=True)
                y = labels[chunk].to(device, non_blocking=True)
                prompts = ["grasp object"] * len(chunk)

            x = normalize(x)
            if model_type == "mdn":
                text_tokens = encode_prompts(prompts, token_to_id, max_prompt_len).to(
                    device, non_blocking=True
                )
                pred = model.predict_best(x, text_tokens)
            else:
                pred = model(x)

            b = x.size(0)
            n_samples += b
            mse_sum += sq_err(pred, y).item()
            mae_sum += abs_err(pred, y).item()

            cos_sim = nn.functional.cosine_similarity(pred, y, dim=1)
            cos_dist_sum += (1 - cos_sim).sum().item()

            l2_batch = torch.norm(pred - y, dim=1)
            l2_sum += l2_batch.sum().item()
            l2_np = l2_batch.detach().cpu().numpy()
            l2_errors_all.append(l2_np)
            for i, thr in enumerate(thresholds):
                threshold_hits[i] += int((l2_batch <= thr).sum().item())

    if n_samples == 0:
        raise ValueError("No samples evaluated. Check dataset and split settings.")

    l2_errors = np.concatenate(l2_errors_all, axis=0)
    metrics = {
        "mse": mse_sum / n_samples,
        "mae": mae_sum / n_samples,
        "cos_distance": cos_dist_sum / n_samples,
        "l2_mean": l2_sum / n_samples,
        "l2_median": float(np.median(l2_errors)),
        "n_samples": int(n_samples),
        "accuracy_at_l2_threshold": {
            str(thr): float(hit / n_samples) for thr, hit in zip(thresholds, threshold_hits.tolist())
        },
    }
    return metrics


def select_indices(
    split: str,
    n_total: int,
    val_fraction: float,
    seed: int,
):
    split = split.lower()
    if split == "all":
        return None

    train_idx, val_idx = build_train_val_split(n_total, val_fraction=val_fraction, seed=seed)
    if split == "val":
        return val_idx
    if split == "train":
        return train_idx
    raise ValueError("split must be one of: train, val, all")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained student latent models.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/hojinsohn/VLM-NT/PI-VLA/student_model_training/data/pt_shards_multi"),
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mdn-max-prompt-len", type=int, default=8)
    parser.add_argument("--thresholds", type=str, default="0.1,0.2,0.5,1.0")
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root not found: {args.dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    thresholds = parse_thresholds(args.thresholds)

    shard_files, cumulative, z_dim_data, object_names = discover_shards(args.dataset_root)
    n_total = cumulative[-1]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = infer_model_type(ckpt)
    z_dim_ckpt = int(ckpt.get("z_dim", z_dim_data))
    if z_dim_ckpt != z_dim_data:
        print(
            f"[warn] checkpoint z_dim={z_dim_ckpt} differs from data z_dim={z_dim_data}. "
            f"Using checkpoint z_dim.",
            flush=True,
        )

    model, token_to_id, max_prompt_len = build_model(
        ckpt=ckpt,
        model_type=model_type,
        z_dim=z_dim_ckpt,
        object_names=object_names,
        mdn_max_prompt_len=args.mdn_max_prompt_len,
        dropout_p=args.dropout,
        device=device,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    split_indices = select_indices(
        split=args.split,
        n_total=n_total,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    metrics = evaluate(
        model=model,
        model_type=model_type,
        device=device,
        shard_files=shard_files,
        cumulative=cumulative,
        split_name=args.split,
        split_indices=split_indices,
        normalize=normalize,
        batch_size=args.batch_size,
        thresholds=thresholds,
        token_to_id=token_to_id,
        max_prompt_len=max_prompt_len,
    )

    payload = {
        "checkpoint": str(args.checkpoint),
        "model_type": model_type,
        "split": args.split,
        "dataset_root": str(args.dataset_root),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "thresholds": thresholds,
        "metrics": metrics,
    }

    print(json.dumps(payload, indent=2))
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
