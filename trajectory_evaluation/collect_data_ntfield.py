#!/usr/bin/env python3
"""Shim: use trajectory_evaluation/ntfield/collect_data.py (same CLI)."""
from __future__ import annotations

import pathlib
import runpy
import sys

_here = pathlib.Path(__file__).resolve()
_target = _here.parent / "ntfield" / "collect_data.py"
if not _target.is_file():
    print(f"Missing {_target}", file=sys.stderr)
    sys.exit(1)
runpy.run_path(str(_target), run_name="__main__")
