#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

try:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except Exception as exc:
    raise SystemExit(
        "Missing dependency for SAM. Install with:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        "Original import error:\n"
        f"  {exc}"
    )


def load_first_image_from_h5(h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with h5py.File(h5_path, "r") as f:
        image = f["images"][0]
        object_location = f["object_location"][:] if "object_location" in f else None
    return image, object_location


def ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    return image


def select_table_mask_auto(masks: List[Dict], image_rgb: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # White tabletop prior: low saturation + high brightness.
    white_prior = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 145)

    best_score = -1.0
    best_mask = None

    for m in masks:
        seg = m["segmentation"]
        area = float(m.get("area", float(seg.sum())))
        x, y, bw, bh = m.get("bbox", [0, 0, w, h])
        x_center = x + bw * 0.5
        y_center = y + bh * 0.5
        aspect = bw / (bh + 1e-6)
        stability = float(m.get("stability_score", 0.0))
        pred_iou = float(m.get("predicted_iou", 0.0))

        # Heuristic for "table-like" region:
        # - relatively large area
        # - lower half of image
        # - horizontally extended, rectangular, mostly white
        # - avoid "whole environment" masks that touch too many borders
        area_ratio = area / float(h * w)
        lower_bias = max(0.0, min(1.0, (y_center / h - 0.35) / 0.65))
        horizontal_bias = max(0.0, min(1.0, (aspect - 1.0) / 3.0))
        center_bias = 1.0 - min(1.0, abs(x_center / w - 0.5) * 1.7)
        bbox_area = float(max(1.0, bw * bh))
        fill_ratio = min(1.0, area / bbox_area)
        white_ratio = float(white_prior[seg].mean()) if seg.any() else 0.0
        top_touch = float(seg[0, :].mean())
        left_touch = float(seg[:, 0].mean())
        right_touch = float(seg[:, -1].mean())
        border_penalty = top_touch + 0.7 * left_touch + 0.7 * right_touch

        score = (
            0.8 * area_ratio
            + 1.2 * lower_bias
            + 1.0 * horizontal_bias
            + 0.6 * center_bias
            + 1.6 * fill_ratio
            + 3.0 * white_ratio
            + 0.8 * stability
            + 0.6 * pred_iou
            - 1.6 * border_penalty
        )
        # Reject tiny masks and very large "scene-like" masks.
        if area_ratio < 0.05 or area_ratio > 0.60:
            continue
        if white_ratio < 0.12:
            continue
        if score > best_score:
            best_score = score
            best_mask = seg

    if best_mask is None:
        raise RuntimeError("Failed to find table mask. Try changing view or SAM settings.")
    best_mask = best_mask.astype(np.uint8)
    # Keep the dominant connected component and smooth boundaries.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(best_mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        best_mask = (labels == largest).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return best_mask.astype(bool)


def save_outputs(
    image_rgb: np.ndarray,
    table_mask: np.ndarray,
    out_prefix: str,
    crop_margin_ratio: float,
) -> None:
    mask_u8 = (table_mask.astype(np.uint8) * 255)
    table_only = image_rgb.copy()
    table_only[~table_mask] = 0

    overlay = image_rgb.copy()
    overlay[table_mask] = (0.35 * overlay[table_mask] + 0.65 * np.array([0, 255, 0])).astype(
        np.uint8
    )

    cv2.imwrite(out_prefix + "_mask.png", mask_u8)
    cv2.imwrite(out_prefix + "_table_only.png", cv2.cvtColor(table_only, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_prefix + "_overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    ys, xs = np.where(table_mask)
    if len(xs) == 0 or len(ys) == 0:
        return

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    h, w = image_rgb.shape[:2]
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    mx = int(bw * crop_margin_ratio)
    my = int(bh * crop_margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w - 1, x2 + mx)
    y2 = min(h - 1, y2 + my)

    # Keeps all content inside table region bounds (banana stays if on table),
    # while discarding most surrounding background by spatial cropping.
    crop = image_rgb[y1 : y2 + 1, x1 : x2 + 1]
    crop_masked = table_only[y1 : y2 + 1, x1 : x2 + 1]
    cv2.imwrite(out_prefix + "_table_crop.png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        out_prefix + "_table_crop_masked.png", cv2.cvtColor(crop_masked, cv2.COLOR_RGB2BGR)
    )


def build_mask_generator(args: argparse.Namespace) -> SamAutomaticMaskGenerator:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )


def parse_timestamp_from_h5_name(h5_path: str) -> str:
    base = os.path.splitext(os.path.basename(h5_path))[0]
    parts = base.split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    return base


def process_one_h5(
    h5_path: str, output_dir: str, args: argparse.Namespace, mask_generator: SamAutomaticMaskGenerator
) -> Dict:
    image_rgb, object_location = load_first_image_from_h5(h5_path)
    image_rgb = ensure_uint8_rgb(image_rgb)
    masks = mask_generator.generate(image_rgb)
    table_mask = select_table_mask_auto(masks, image_rgb)
    out_prefix = os.path.join(output_dir, "first_image")
    save_outputs(
        image_rgb=image_rgb,
        table_mask=table_mask,
        out_prefix=out_prefix,
        crop_margin_ratio=args.crop_margin_ratio,
    )
    meta = {"source_h5": h5_path}
    if object_location is not None:
        fixed = np.array([object_location[0], object_location[1], 0.12], dtype=np.float32)
        meta["object_location_original"] = [float(v) for v in object_location.tolist()]
        meta["object_location_z_fixed"] = [float(v) for v in fixed.tolist()]
    with open(os.path.join(output_dir, "object_location.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def run(args: argparse.Namespace) -> None:
    if args.h5_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        h5_files = sorted(
            [
                os.path.join(args.h5_dir, fn)
                for fn in os.listdir(args.h5_dir)
                if fn.endswith(".h5")
            ]
        )
        if not h5_files:
            raise FileNotFoundError(f"No .h5 files found in directory: {args.h5_dir}")
        mask_generator = build_mask_generator(args)
        print(f"Found {len(h5_files)} h5 files.")
        success = 0
        for i, h5_path in enumerate(h5_files, start=1):
            timestamp = parse_timestamp_from_h5_name(h5_path)
            sample_dir = os.path.join(args.output_dir, timestamp)
            os.makedirs(sample_dir, exist_ok=True)
            try:
                process_one_h5(
                    h5_path=h5_path,
                    output_dir=sample_dir,
                    args=args,
                    mask_generator=mask_generator,
                )
                success += 1
                print(f"[{i}/{len(h5_files)}] OK  {os.path.basename(h5_path)} -> {sample_dir}")
            except Exception as exc:
                print(f"[{i}/{len(h5_files)}] ERR {os.path.basename(h5_path)}: {exc}")
        print(f"Completed. Success: {success}/{len(h5_files)}")
        return

    if args.h5_path is None and args.image_path is None:
        raise ValueError("Provide either --h5-path/--image-path, or --h5-dir for batch mode")
    if args.h5_path is not None and args.image_path is not None:
        raise ValueError("Provide only one of --h5-path or --image-path")
    if args.image_path is not None:
        raise ValueError("Single-image mode supports --h5-path only for object_location export")

    os.makedirs(args.output_dir, exist_ok=True)
    mask_generator = build_mask_generator(args)
    base_name = os.path.splitext(os.path.basename(args.h5_path))[0]
    sample_dir = os.path.join(args.output_dir, base_name)
    os.makedirs(sample_dir, exist_ok=True)
    meta = process_one_h5(
        h5_path=args.h5_path,
        output_dir=sample_dir,
        args=args,
        mask_generator=mask_generator,
    )
    print(f"Saved directory: {sample_dir}")
    print("object_location_original:", meta.get("object_location_original"))
    print("object_location_z_fixed:", meta.get("object_location_z_fixed"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract table region from first H5 image using Segment Anything."
    )
    parser.add_argument("--h5-dir", type=str, default=None, help="Directory of .h5 files (batch)")
    parser.add_argument("--h5-path", type=str, default=None, help="Path to one .h5 data file")
    parser.add_argument("--image-path", type=str, default=None, help="Path to one RGB image")
    parser.add_argument("--sam-checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model variant",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--output-dir", type=str, default="img2objloc_model/outputs")

    # Automatic mask tuning knobs.
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--min-mask-region-area", type=int, default=1000)
    parser.add_argument(
        "--crop-margin-ratio",
        type=float,
        default=0.06,
        help="Extra margin around table bbox for crop image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
