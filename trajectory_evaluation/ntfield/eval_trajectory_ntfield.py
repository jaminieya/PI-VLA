"""
Load trajectory-trained NTField checkpoints (train/train_arm_trajectory.py).

Used by trajectory_evaluation/comparison/run_rrt_ntfield_benchmark.py and
final_integrate/run_integrated_pipeline.py.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch


class _ModelShim:
    """gradient_planner_trajectory.plan expects ``model.function.Gradient``."""

    def __init__(self, function: Any) -> None:
        self.function = function


def load_network_and_function(
    checkpoint_path: str,
    experiment_dir: Optional[str],
    device: torch.device,
    dim: int = 6,
) -> Tuple[Any, Any]:
    """
    Restore NN + Function from a Model_Epoch_*.pt saved by train_arm_trajectory.py.

    Args:
        checkpoint_path: Absolute path to .pt file.
        experiment_dir: Folder that contains copied ``models/metric_arm`` (usually
            the run directory next to the checkpoint). If None, uses dirname(checkpoint).
        device: torch.device
        dim: joint dimension (6 for UR5e)
    """
    checkpoint_path = os.path.abspath(checkpoint_path)
    if experiment_dir is None:
        experiment_dir = os.path.dirname(checkpoint_path)
    else:
        experiment_dir = os.path.abspath(experiment_dir)

    ckpt = torch.load(checkpoint_path, map_location=device)
    B = ckpt["B_state_dict"]
    if not torch.is_tensor(B):
        B = torch.as_tensor(B)

    from models.metric_arm import model_function_metric as model_function
    from models.metric_arm import model_network_metric as model_network

    network = model_network.NN(str(device), dim, B)
    network.load_state_dict(ckpt["model_state_dict"], strict=True)
    network.to(device)
    network.eval()

    fn = model_function.Function(experiment_dir, device, network, dim)
    return network, fn
