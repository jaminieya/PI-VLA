"""
Load a trajectory-trained NTField checkpoint for gradient planning.

Used by trajectory_evaluation/comparison/run_rrt_ntfield_benchmark.py.
Mirrors student_model_evaluation/plan_ntfield_with_student.py teacher setup.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional, Tuple

import torch

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")


def _ensure_ntrl_on_path() -> None:
    if os.path.isdir(_NTRL_DEMO) and _NTRL_DEMO not in sys.path:
        sys.path.insert(0, _NTRL_DEMO)


class _ModelShim:
    """Expose metric Function as ``model.function`` for planning.gradient_planner_trajectory.plan."""

    __slots__ = ("function",)

    def __init__(self, function_obj: Any) -> None:
        self.function = function_obj


def load_network_and_function(
    checkpoint_path: str,
    experiment_dir: Optional[str],
    device: torch.device,
    *,
    dim: int = 6,
) -> Tuple[Any, Any]:
    """
    Returns ``(network, function)`` where ``function`` has ``.Gradient(XP)``.

    Args:
        checkpoint_path: Absolute path to ``Model_Epoch_*.pt``.
        experiment_dir: If set, directory with NTField config (ModelPath); else ``dirname(checkpoint)``.
        device: Torch device for the loaded network.
        dim: Joint dimension (UR5e = 6).
    """
    _ensure_ntrl_on_path()
    from models.metric_arm import model_test_metric as md

    ckpt = os.path.abspath(checkpoint_path)
    model_path = os.path.abspath(experiment_dir) if experiment_dir else os.path.dirname(ckpt)
    data_path = os.path.join(_NTRL_DEMO, "datasets", "arm", "UR5_trajectory")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"NTField dataset path missing: {data_path}")

    dev_str = str(device)

    teacher = md.Model(
        model_path,
        data_path,
        dim=dim,
        source=[0.0] * dim,
        device=dev_str,
    )
    teacher.load(ckpt)
    teacher.network.eval()
    return teacher.network, teacher.function


if __name__ == "__main__":
    print("This module is imported by run_rrt_ntfield_benchmark.py; no CLI here.")
