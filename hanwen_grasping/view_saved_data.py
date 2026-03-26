import h5py
import cv2
import numpy as np
import argparse
import os

def view_and_export_h5(file_path, output_video_path="trajectory_review.mp4"):
    if not os.path.exists(file_path):
        print(f"Error: Could not find file {file_path}")
        return

    print(f"--- Reading HDF5 Data from {file_path} ---")
    
    with h5py.File(file_path, 'r') as f:
        # 1. Read Metadata
        print("\n--- Metadata ---")
        for key, val in f.attrs.items():
            print(f"{key}: {val}")
        
        num_samples = f.attrs.get('num_samples', 0)
        
        if num_samples == 0:
            print("No samples found in this file.")
            return

        # 2. Extract Datasets (images = top-down; images_top = legacy; images_side = side view)
        images = f['images_top'][:] if 'images_top' in f else f['images'][:]
        images_side = f['images_side'][:] if 'images_side' in f else None
        joint_configs = f['joint_configs'][:]
        final_joint = f['final_joint_config'][:]
        object_location = f['object_location'][:] if 'object_location' in f else None
        
        print(f"\nTotal Frames: {num_samples}")
        print(f"Target Final Joint Config: {final_joint}")
        if object_location is not None:
            print(f"Object Location (xyz): {object_location}")

    # 3. Setup Video Writer (side-by-side if side view exists)
    frame_height, frame_width = images[0].shape[:2]
    if images_side is not None:
        out_width = frame_width * 2
        out_height = frame_height
    else:
        out_width = frame_width
        out_height = frame_height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10.0  # Smooth playback for trajectory review
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    print(f"\n--- Processing Frames ---")
    for i in range(num_samples):
        # Convert RGB (saved format) to BGR (OpenCV format)
        bgr = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)

        # Overlay text
        joint_str = ", ".join([f"{j:.2f}" for j in joint_configs[i]])
        text = f"Frame {i+1}/{num_samples} | Joints: [{joint_str}]"
        
        cv2.putText(bgr, "Top-down | " + text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if images_side is not None:
            bgr_side = cv2.cvtColor(images_side[i], cv2.COLOR_RGB2BGR)
            cv2.putText(bgr_side, "Side view", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            frame = np.hstack([bgr, bgr_side])
        else:
            frame = bgr
        
        video_writer.write(frame)
        print(f"Processed frame {i+1}/{num_samples} | Joints: {joint_str}")

    video_writer.release()
    print(f"\nSuccess! Video saved to {output_video_path}")
    print("You can now download this mp4 to your local machine to view the grasp trajectory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View and export saved 6DOF grasp dataset.")
    parser.add_argument("file", help="Path to the .h5 dataset file")
    parser.add_argument("--out", default="trajectory_review.mp4", help="Output video file name")
    args = parser.parse_args()
    
    view_and_export_h5(args.file, args.out)