#!/usr/bin/env python3
"""Backward-compatible entry point; implementation lives in trajectory_evaluation/ntfield/."""
from __future__ import annotations

import pathlib
import runpy
import sys

_target = pathlib.Path(__file__).resolve().parent.parent / "ntfield" / "eval_trajectory_ntfield.py"
if not _target.is_file():
    print(f"Missing {_target}", file=sys.stderr)
    sys.exit(1)
runpy.run_path(str(_target), run_name="__main__")
