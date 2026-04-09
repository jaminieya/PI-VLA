import argparse
import os

import cv2
import h5py
import numpy as np


def view_student_h5(file_path, output_image_path="student_sample_preview.png"):
    if not os.path.exists(file_path):
        print(f"Error: Could not find file {file_path}")
        return

    print(f"--- Reading student HDF5 data from {file_path} ---")
    with h5py.File(file_path, "r") as f:
        print("\n--- Metadata ---")
        for key, val in f.attrs.items():
            print(f"{key}: {val}")

        required_keys = [
            "start_image",
            "start_joint_config",
            "object_location",
            "goal_joint_config",
        ]
        missing = [k for k in required_keys if k not in f]
        if missing:
            print(f"Error: missing required dataset keys: {missing}")
            return

        start_image = f["start_image"][:]
        start_joint_config = np.array(f["start_joint_config"][:], dtype=np.float32)
        object_location = np.array(f["object_location"][:], dtype=np.float32)
        goal_joint_config = np.array(f["goal_joint_config"][:], dtype=np.float32)

    print("\n--- Stored Values ---")
    print(f"start_image shape: {start_image.shape}, dtype: {start_image.dtype}")
    print(f"start_joint_config: {start_joint_config}")
    print(f"object_location (xyz): {object_location}")
    print(f"goal_joint_config: {goal_joint_config}")

    # Convert RGB to BGR for OpenCV preview.
    bgr_start = cv2.cvtColor(start_image, cv2.COLOR_RGB2BGR)

    start_joint_str = ", ".join([f"{x:.2f}" for x in start_joint_config.tolist()])
    goal_joint_str = ", ".join([f"{x:.2f}" for x in goal_joint_config.tolist()])
    obj_str = ", ".join([f"{x:.3f}" for x in object_location.tolist()])

    cv2.putText(
        bgr_start,
        "Start image",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        bgr_start,
        f"Object xyz: [{obj_str}]",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        bgr_start,
        f"Start q: [{start_joint_str}]",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        bgr_start,
        f"Goal q: [{goal_joint_str}]",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(output_image_path, bgr_start)
    print(f"\nSuccess! Preview image saved to {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="View one-sample student grasp dataset (.h5)."
    )
    parser.add_argument("file", help="Path to the .h5 dataset file")
    parser.add_argument(
        "--out",
        default="student_sample_preview.png",
        help="Output preview image path",
    )
    args = parser.parse_args()
    view_student_h5(args.file, args.out)

# python view_saved_data_for_student.py \
#   ../collected_data/grasp_6dof_demo_20260408_140000.h5 \
#   --out student_sample_preview.png