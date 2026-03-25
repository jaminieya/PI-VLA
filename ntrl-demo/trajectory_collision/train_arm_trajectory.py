#!/usr/bin/env python3
"""
Train NTField from trajectory data WITH collision avoidance.

Uses trajectory supervision + Eikonal (F=1) + collision loss.
Requires --obstacle_mesh for collision avoidance.

Usage:
    cd ntrl-demo && python trajectory_collision/train_arm_trajectory.py \\
        --data_path ./datasets/arm/UR5_trajectory \\
        --obstacle_mesh datasets/arm/UR5/realpc_scaled.off
"""

import sys
import os

_ntrl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ntrl_root)
os.chdir(_ntrl_root)

if os.environ.get('MPLBACKEND') == 'Agg' or os.environ.get('MATPLOTLIB_HEADLESS'):
    import matplotlib
    matplotlib.use('Agg')

import argparse
import numpy as np
import torch
from datetime import datetime, timedelta

from models import data_mlp as db
from models.metric_arm import model_network_metric as model_network
from trajectory_collision.model_function_metric import Function as CollisionFunction

torch.backends.cudnn.benchmark = True


class FastTensorDataLoader:
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        self.n_batches = n_batches + (1 if remainder > 0 else 0)

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def main():
    parser = argparse.ArgumentParser(description="Train NTField with collision avoidance")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./Experiments/UR5_trajectory_collision")
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batches_per_epoch", type=int, default=None)
    parser.add_argument("--obstacle_mesh", type=str, required=True,
                        help="Path to .off obstacle mesh (required for collision)")
    parser.add_argument("--weight_collision", type=float, default=1e-3)
    parser.add_argument("--collision_margin", type=float, default=0.05)
    parser.add_argument("--collision_freq", type=int, default=1)
    parser.add_argument("--collision_sub_batch", type=int, default=32)
    args = parser.parse_args()

    dim = 6
    device = args.device
    data_path = os.path.abspath(args.data_path)
    model_path = os.path.abspath(args.model_path)

    mesh_path = os.path.abspath(args.obstacle_mesh)
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Obstacle mesh not found: {mesh_path}")

    from trajectory_collision.collision_utils import (
        build_obstacle_kdtree,
        build_arm_chain,
        arm_obstacle_distance_batch,
    )
    kdtree, v_obs = build_obstacle_kdtree(mesh_path, device=device)
    chain, mesh_list = build_arm_chain()
    scale = np.pi / 0.5
    margin = args.collision_margin
    offset = margin / 10.0
    sub_batch = args.collision_sub_batch

    def collision_callback_fn(points):
        th = points * scale
        q_s = th[:, :dim]
        q_g = th[:, dim:]
        n = points.shape[0]
        dist_s_list, norm_s_list, dist_g_list, norm_g_list = [], [], [], []
        for i in range(0, n, sub_batch):
            end = min(i + sub_batch, n)
            ds, ns = arm_obstacle_distance_batch(q_s[i:end], chain, mesh_list, kdtree, v_obs, device)
            dg, ng = arm_obstacle_distance_batch(q_g[i:end], chain, mesh_list, kdtree, v_obs, device)
            dist_s_list.append(ds)
            norm_s_list.append(ns)
            dist_g_list.append(dg)
            norm_g_list.append(ng)
        dist_s = torch.cat(dist_s_list, dim=0)
        norm_s = torch.cat(norm_s_list, dim=0)
        dist_g = torch.cat(dist_g_list, dim=0)
        norm_g = torch.cat(norm_g_list, dim=0)
        dist_s = dist_s - 0.01
        dist_g = dist_g - 0.01
        y0 = torch.clamp(dist_s, offset, margin) / margin
        y1 = torch.clamp(dist_g, offset, margin) / margin
        Yobs = torch.stack([y0, y1], dim=1)
        norm_s = torch.nn.functional.normalize(norm_s, dim=1)
        norm_g = torch.nn.functional.normalize(norm_g, dim=1)
        norm_s = torch.nan_to_num(norm_s, nan=0.0, posinf=0.0, neginf=0.0)
        norm_g = torch.nan_to_num(norm_g, nan=0.0, posinf=0.0, neginf=0.0)
        normal = torch.cat([norm_s, norm_g], dim=1)
        return Yobs, normal

    current_time = datetime.utcnow() - timedelta(hours=5)
    folder = os.path.join(model_path, "trajectory_collision_" + current_time.strftime("%m_%d_%H_%M"))
    os.makedirs(folder, exist_ok=True)

    import shutil
    source_folder = "./models/metric_arm"
    dest_folder = os.path.join(folder, "models")
    os.makedirs(dest_folder, exist_ok=True)
    for fname in os.listdir(source_folder):
        src = os.path.join(source_folder, fname)
        dst = os.path.join(dest_folder, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    B = 0.2 * torch.normal(0, 1, size=(128, dim))
    network = model_network.NN(device, dim, B)
    network.apply(network.init_weights)
    network.to(device)

    function = CollisionFunction(folder, device, network, dim)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=0.5)

    dataset = db.DatabaseTrajectory(data_path)
    points_tensor = dataset.data[:, :2 * dim]
    tau_obs_tensor = dataset.data[:, -1]
    dataloader = FastTensorDataLoader(
        points_tensor, tau_obs_tensor,
        batch_size=args.batch_size,
        shuffle=True
    )

    total_train_loss = []
    beta = 1.0
    batches_per_epoch = args.batches_per_epoch or len(dataloader)

    print(f"Collision avoidance enabled: mesh={mesh_path}, weight={args.weight_collision}, margin={margin}m")
    print(f"Training on {len(dataset)} samples, {batches_per_epoch} batches/epoch")
    print(f"Output: {folder}")

    for epoch in range(1, args.epochs + 1):
        total_diff = 0.0
        total_traj_loss = 0.0
        batch_count = 0

        for batch_idx, (points_batch, tau_batch) in enumerate(dataloader):
            if batch_idx >= batches_per_epoch:
                break

            points = points_batch.to(device)
            tau_obs = tau_batch.to(device)

            use_collision = batch_idx % args.collision_freq == 0
            cb = collision_callback_fn if use_collision else None
            loss_value, loss_n, traj_loss = function.LossTrajectory(
                points, tau_obs, beta, epoch,
                collision_callback=cb,
                weight_collision=args.weight_collision,
            )

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            total_diff += loss_n.item()
            total_traj_loss += traj_loss.item()
            batch_count += 1

            del points, tau_obs, loss_value, loss_n, traj_loss

        if batch_count > 0:
            total_diff /= batch_count
            total_traj_loss /= batch_count

        total_train_loss.append(total_diff)
        beta = 1.0 / max(total_diff, 1e-6)

        if epoch % args.print_every == 0:
            print(f"Epoch {epoch} -- Loss = {total_diff:.4e} -- TrajLoss = {total_traj_loss:.4e}")

        if (epoch % args.save_every == 0) or (epoch == args.epochs) or (epoch == 1):
            torch.save({
                "epoch": epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "B_state_dict": B,
                "train_loss": total_train_loss,
            }, os.path.join(folder, f"Model_Epoch_{str(epoch).zfill(5)}_ValLoss_{total_diff:.6e}.pt"))

    print("Training complete.")


if __name__ == "__main__":
    main()
