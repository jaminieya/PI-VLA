#!/usr/bin/env python3
"""
Run inference with GoalLatentPredictorWithFiLM (prompt + image + start conditioning -> z_goal_hat).

This mirrors train_goal_rep_alignment.py:
  - image: Resize((224,224)) + ResNet normalization (ImageNet mean/std)
  - start_cond teacher_z (default in new checkpoints): E(q_start) from frozen NTField + learned proj
  - start_cond joints (legacy): 6D q_start into a small MLP

Optional evaluation:
  If you also provide --q_goal, this will compute the teacher z_goal latent and report
  mse/cosine between z_goal_hat and z_goal.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
_HANWEN = os.path.abspath(os.path.join(_ROOT, "..", "hanwen_grasping"))
if _HANWEN not in sys.path:
    sys.path.insert(0, _HANWEN)

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from train_goal_rep_alignment import (
    GoalLatentPredictorWithFiLM,
    build_coords_batch,
    encode_teacher_joint_latent,
    load_teacher,
    normalize_coords_tensor,
)


def parse_q_ntfield(s: str) -> list[float]:
    """
    Parse q_start/q_goal string.

    Same parsing behavior as new_setup.py:
      - comma-separated 6 values
      - allows "pi" (mapped to math.pi) in expressions via eval
    """
    s = s.strip().replace("pi", "math.pi")
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(parts)}")
    return [float(eval(p, {"math": math})) for p in parts]


def main() -> None:
    p = argparse.ArgumentParser(description="Infer goal latent using trained student")
    p.add_argument(
        "--student",
        type=str,
        required=True,
        help="Path to saved student .pt checkpoint (e.g. goal_rep_student_film_2.pt or goal_rep_student_film_epoch005.pt)",
    )
    p.add_argument("--image", type=str, required=True, help="Path to an RGB image file")
    p.add_argument("--prompt", type=str, required=True, help="Text prompt used by the student")
    p.add_argument(
        "--q_start",
        type=str,
        required=True,
        help='6 comma-separated joint values in radians, e.g. "0.1,-0.2,0.0,1.57,1.57,0"',
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_z", type=str, default="", help="Optional path to save z_goal_hat as .npy")

    # Optional evaluation against the NTField teacher
    p.add_argument(
        "--q_goal",
        type=str,
        default="",
        help="Optional 6D q_goal string; if provided, compute cosine/mse against teacher z_goal.",
    )
    args = p.parse_args()

    device = torch.device(args.device)

    student_path = os.path.abspath(args.student)
    payload = torch.load(student_path, map_location=device)

    ntfield_h = int(payload["ntfield_h"])
    normalize_coords = bool(payload.get("normalize_coords", False))
    loss_name = str(payload.get("loss", "mse"))
    image_key = str(payload.get("image_key", "images"))
    teacher_ckpt = str(payload["checkpoint_teacher"])
    start_cond = str(payload.get("start_cond", "joints"))

    student = GoalLatentPredictorWithFiLM(ntfield_h=ntfield_h, start_cond=start_cond).to(device)
    student.load_state_dict(payload["student_state_dict"], strict=True)
    student.eval()

    # ---- image preprocessing (match training dataset transform) ----
    # train_goal_rep_alignment.py uses:
    #   Resize((224,224)) + Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    img = Image.open(args.image).convert("RGB")
    tfm = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_t = tfm(img).unsqueeze(0).to(device)  # (1,3,224,224)

    # ---- joints ----
    qs = torch.tensor(parse_q_ntfield(args.q_start), dtype=torch.float32, device=device).unsqueeze(0)  # (1,6)

    teacher = None
    with torch.no_grad():
        if start_cond == "teacher_z":
            teacher, _ = load_teacher(teacher_ckpt, device)
            teacher.eval()
            qs_n = normalize_coords_tensor(qs, normalize_coords)
            z_in = encode_teacher_joint_latent(teacher, qs_n)
        else:
            z_in = qs
        z_goal_hat = student(img_t, [args.prompt], z_in)  # (1, ntfield_h)

    z_goal_hat_cpu = z_goal_hat.detach().cpu().numpy()
    print(f"z_goal_hat shape: {z_goal_hat_cpu.shape}")
    print(f"z_goal_hat L2 norm: {float(np.linalg.norm(z_goal_hat_cpu)):.6f}")

    if args.out_z:
        out_path = os.path.abspath(args.out_z)
        np.save(out_path, z_goal_hat_cpu)
        print(f"Saved z_goal_hat to {out_path}")

    # ---- optional evaluation vs teacher ----
    if args.q_goal:
        if teacher is None:
            teacher, _ = load_teacher(teacher_ckpt, device)
            teacher.eval()
        qg = torch.tensor(parse_q_ntfield(args.q_goal), dtype=torch.float32, device=device).unsqueeze(0)  # (1,6)

        coords = build_coords_batch(qs, qg, use_scale=normalize_coords).to(device)
        with torch.no_grad():
            _, z_tgt = teacher.encode_pair_latents(coords)

        if loss_name == "mse":
            metric = torch.nn.functional.mse_loss(z_goal_hat, z_tgt).item()
            print(f"Teacher mse loss: {metric:.6f}")
        else:
            z_hat_n = torch.nn.functional.normalize(z_goal_hat, dim=1)
            z_tgt_n = torch.nn.functional.normalize(z_tgt, dim=1)
            cosine_loss = (1.0 - (z_hat_n * z_tgt_n).sum(dim=1)).mean().item()
            cosine_sim = (z_hat_n * z_tgt_n).sum(dim=1).mean().item()
            print(f"Teacher cosine loss (1-cos): {cosine_loss:.6f}")
            print(f"Teacher cosine similarity: {cosine_sim:.6f}")

    # Avoid unused-variable warnings (image_key is only informative here)
    _ = image_key


if __name__ == "__main__":
    main()

