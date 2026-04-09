#!/usr/bin/env python3
"""Re-export NTField loaders; implementation lives in trajectory_evaluation/ntfield/."""
from __future__ import annotations

import sys

from trajectory_evaluation.ntfield.eval_trajectory_ntfield import _ModelShim, load_network_and_function

__all__ = ["_ModelShim", "load_network_and_function"]

if __name__ == "__main__":
    print(
        "This file only exports load_network_and_function and _ModelShim. "
        "For τ / planner metrics see trajectory_evaluation/rrtconnect/README.md.",
        file=sys.stderr,
    )
    sys.exit(2)
