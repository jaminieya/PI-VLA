import argparse
import time
from typing import Optional

import cv2
import h5py
import numpy as np


def _get_dataset(f, name_primary: str, name_fallback: str):
    if name_primary in f:
        return f[name_primary]
    if name_fallback in f:
        return f[name_fallback]
    raise KeyError(f"Neither dataset '{name_primary}' nor '{name_fallback}' exists.")


def _format_joints(joint_row):
    # joint_row: (joint_dim,)
    return ", ".join([f"{float(j):.3f}" for j in joint_row])


def view_or_export_top_view(
    file_path: str,
    start: int = 0,
    end: Optional[int] = None,
    fps: float = 10.0,
    out_path: Optional[str] = None,
    no_gui: bool = False,
):
    with h5py.File(file_path, "r") as f:
        prompt = f.attrs.get("prompt", "")
        num_samples = int(f.attrs.get("num_samples", 0) or 0)

        images_ds = _get_dataset(f, "images_top", "images")
        joint_configs_ds = f["joint_configs"] if "joint_configs" in f else None

        if num_samples == 0:
            num_samples = int(images_ds.shape[0])

        if end is None:
            end = num_samples
        start = max(0, min(int(start), num_samples))
        end = max(start, min(int(end), num_samples))

        final_joint = f["final_joint_config"][:] if "final_joint_config" in f else None
        object_location = f["object_location"][:] if "object_location" in f else None

        print(f"--- Top-view viewer ---")
        print(f"File: {file_path}")
        print(f"Frames: {num_samples} (showing {start}:{end})")
        if prompt:
            print(f"Prompt: {prompt}")
        if final_joint is not None:
            print(f"Final joint config: {_format_joints(final_joint)}")
        if object_location is not None:
            obj_str = ", ".join([f"{float(x):.4f}" for x in np.array(object_location).reshape(-1)])
            print(f"Object location (xyz): {obj_str}")

        # Determine frame size from first frame in range.
        if end <= start:
            print("No frames to show/export.")
            return

        first_rgb = images_ds[start]
        if first_rgb.ndim != 3 or first_rgb.shape[-1] < 3:
            raise ValueError(f"Expected RGB frames with shape (H,W,3); got {first_rgb.shape}")

        frame_height, frame_width = first_rgb.shape[:2]

        video_writer = None
        if out_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(out_path, fourcc, float(fps), (frame_width, frame_height))
            if not video_writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")
            print(f"Exporting MP4 to: {out_path}")

        if not no_gui:
            cv2.namedWindow("Top View", cv2.WINDOW_NORMAL)

        paused = False
        idx = start
        last_tick = time.time()

        while idx < end:
            # Read frame lazily from HDF5 (avoids loading full dataset into RAM).
            rgb = images_ds[idx]
            bgr = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)

            if joint_configs_ds is not None:
                joint_row = joint_configs_ds[idx]
                joint_str = _format_joints(joint_row)
                overlay = f"Frame {idx+1}/{num_samples} | Joints: [{joint_str}]"
            else:
                overlay = f"Frame {idx+1}/{num_samples}"

            cv2.putText(
                bgr,
                "Top-down | " + overlay,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(bgr)

            if not no_gui:
                cv2.imshow("Top View", bgr)

                # FPS pacing (only when not paused).
                if paused:
                    key = cv2.waitKey(0) & 0xFF
                else:
                    # Slightly adaptive pacing: if UI work took time, reduce wait.
                    now = time.time()
                    elapsed_ms = (now - last_tick) * 1000.0
                    target_ms = 1000.0 / max(float(fps), 1e-6)
                    wait_ms = int(max(1.0, target_ms - elapsed_ms))
                    key = cv2.waitKey(wait_ms) & 0xFF
                    last_tick = now
            else:
                # No GUI: just run as fast as possible; keep keys disabled.
                key = 0

            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord(" "):
                paused = not paused
            if key in (81, 2424832):  # Left arrow (keycode varies)
                idx = max(start, idx - 1)
                paused = True
            if key in (83, 2555904):  # Right arrow (keycode varies)
                idx = min(end - 1, idx + 1)
                paused = True
            if key in (ord("r"), ord("R")):
                idx = start
                paused = False

            if not paused:
                idx += 1

        if video_writer is not None:
            video_writer.release()
            print("Success: MP4 saved.")

        if not no_gui:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="View (and optionally export) top-down frames from a collected .h5 dataset.")
    parser.add_argument("file", help="Path to the .h5 dataset file")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (0-based, inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End frame index (0-based, exclusive)")
    parser.add_argument("--fps", type=float, default=10.0, help="Playback/export FPS")
    parser.add_argument("--out", default=None, help="If set, export top view to this MP4 path")
    parser.add_argument("--no_gui", action="store_true", help="Run without opening an OpenCV window")
    args = parser.parse_args()

    view_or_export_top_view(
        file_path=args.file,
        start=args.start,
        end=args.end,
        fps=args.fps,
        out_path=args.out,
        no_gui=args.no_gui,
    )


if __name__ == "__main__":
    main()

