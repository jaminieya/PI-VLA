#!/usr/bin/env python3
"""
Compare NTField latent extraction:

  (1) Stacked: ``encode_pair_latents(concat(q_s, q_g))`` — production path (one forward, batch 2B before norm).

  (2) Split: ``_embed_configs(q_s)`` and ``_embed_configs(q_g)`` — two forwards (batch B each before norm).

Because ``apply_encoder_norm`` normalizes over the batch dimension, (1) and (2) need not match in
``training`` mode or when running stats are absent (batch statistics differ). In ``eval`` mode with
consistent ``encoder_norm`` buffers, they may still differ if you process different batch sizes.

Run from ``ntrl-demo`` root::

    cd ntrl-demo && python models/metric_arm/test_stacked_vs_split_embed.py
    cd ntrl-demo && python models/metric_arm/test_stacked_vs_split_embed.py --checkpoint /path/to/Model_Epoch_....pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Repo root: .../ntrl-demo
_NTRL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _NTRL_ROOT not in sys.path:
    sys.path.insert(0, _NTRL_ROOT)

from models.metric_arm import model_network_metric as model_network # noqa: E402


def _load_network(device: torch.device, dim: int, checkpoint: str | None):
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        B = ckpt["B_state_dict"]
        if not torch.is_tensor(B):
            B = torch.as_tensor(B)
        net = model_network.NN(str(device), dim, B)
        net.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        B = 0.2 * torch.normal(0, 1, size=(128, dim))
        net = model_network.NN(str(device), dim, B)
        net.apply(net.init_weights)
    net.to(device)
    return net


def main():
    parser = argparse.ArgumentParser(description="Stacked vs split NTField embed comparison")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional Model_Epoch_*.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--train_mode", action="store_true", help="Use net.train() instead of eval()")
    args = parser.parse_args()

    dim = 6
    device = torch.device(args.device)
    net = _load_network(device, dim, args.checkpoint)

    if args.train_mode:
        net.train()
    else:
        net.eval()

    torch.manual_seed(0)
    q_s = (torch.rand(args.batch, dim, device=device) - 0.5)
    q_g = (torch.rand(args.batch, dim, device=device) - 0.5)
    coords = torch.cat([q_s, q_g], dim=1)

    with torch.no_grad():
        z_s_stacked, z_g_stacked = net.encode_pair_latents(coords)
        z_s_split, _ = net._embed_configs(q_s)
        z_g_split, _ = net._embed_configs(q_g)

    d_s = (z_s_stacked - z_s_split).abs().max().item()
    d_g = (z_g_stacked - z_g_split).abs().max().item()
    print(f"mode: {'train' if args.train_mode else 'eval'}")
    print(f"max |z_s stacked - z_s split|: {d_s:.6e}")
    print(f"max |z_g stacked - z_g split|: {d_g:.6e}")
    if d_s < 1e-6 and d_g < 1e-6:
        print("Note: exact match (unusual unless norm sees identical batch stats).")
    else:
        print(
            "Note: mismatch expected when InstanceNorm stats are computed on B vs 2B rows; "
            "production path remains stacked via _embed_start_goal."
        )


if __name__ == "__main__":
    main()
