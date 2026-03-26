#!/usr/bin/env python3
"""
Automatically run collect_data.py multiple times to collect grasp trajectories.
Usage:
    cd starter_code && python run_collect_auto.py --num_episodes 10
    python run_collect_auto.py --num_episodes 5 --target_idx 0
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Auto-run collect_data.py for data collection")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to collect")
    parser.add_argument("--target_idx", type=int, default=0, help="Object index to grasp (0 with single object)")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default: True)")
    parser.add_argument("--no-headless", action="store_true", help="Run with viewer")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    collect_script = os.path.join(script_dir, "collect_data.py")
    if not os.path.exists(collect_script):
        print(f"Error: {collect_script} not found")
        sys.exit(1)

    headless = args.headless and not args.no_headless
    cmd = [sys.executable, collect_script, "--target_idx", str(args.target_idx)]
    if headless:
        cmd.append("--headless")

    print(f"Collecting {args.num_episodes} episodes (target_idx={args.target_idx}, headless={headless})")
    print(f"Output: ../collected_data/grasp_6dof_demo_*.h5\n")

    success_count = 0
    for i in range(args.num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{args.num_episodes} - {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        try:
            result = subprocess.run(cmd, cwd=script_dir)
            if result.returncode in (0, 1):
                success_count += 1
                print(f"✓ Episode {i+1} completed")
            else:
                print(f"✗ Episode {i+1} failed (exit code {result.returncode})")
        except Exception as e:
            print(f"✗ Episode {i+1} failed: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break

    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{args.num_episodes} episodes collected")
    print("="*60)


if __name__ == "__main__":
    main()
