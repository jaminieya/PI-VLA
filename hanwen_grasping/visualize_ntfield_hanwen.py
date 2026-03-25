#!/usr/bin/env python3
"""
Visualize NTField-planned 6-DoF trajectory in Isaac Gym with hanwen table setup.

Standalone script for NTField visualization only. Uses the same table environment
as new_setup.py but without objects or grasp planning.

Usage:
    cd hanwen_grasping
    python visualize_ntfield_hanwen.py --checkpoint ../ntrl-demo/Experiments/UR5_trajectory/trajectory_XX/Model_Epoch_XXXX.pt \\
        --h5_path ../collected_data/grasp_6dof_demo_XXX.h5

    python visualize_ntfield_hanwen.py --checkpoint ... --q_start "0.2,-0.5,-1.0,1.57,1.57,0" --q_goal "-0.2,-0.5,-0.35,0.63,1.57,0"

Options:
    --no_walls     Remove side walls and upper cover (table only)
    --headless     Run without viewer (use with --record for batch; may need xvfb-run on SSH)
    --record       Record video to --record_output
    --record_output  Output path (saved to PI-VLA/visualization/; auto: ntfield_YYYYMMDD_HHMMSS.mp4 from h5, or ntfield_q_<goal>.mp4)
"""

import math
import os
import re
import sys

import h5py
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil

# Add ntrl-demo to path for model import
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pi_vla_root = os.path.dirname(_script_dir)
_ntrl_demo_path = os.path.join(_pi_vla_root, "ntrl-demo")
_visualization_dir = os.path.join(_pi_vla_root, "visualization")
if _ntrl_demo_path not in sys.path:
    sys.path.insert(0, _ntrl_demo_path)


def interpolate_path(path, steps_between=4):
    """Interpolate between consecutive waypoints to get denser trajectory."""
    if not path or len(path) < 2:
        return path
    interpolated = []
    for i in range(len(path) - 1):
        start = np.array(path[i], dtype=np.float64)
        end = np.array(path[i + 1], dtype=np.float64)
        for k in range(steps_between + 1):
            t = k / (steps_between + 1)
            pt = start + t * (end - start)
            interpolated.append(pt.tolist())
    interpolated.append(np.array(path[-1], dtype=np.float64).tolist())
    return interpolated


def parse_q(s):
    """Parse q_start or q_goal string. Supports '0.5*pi' via eval with math in scope."""
    s = s.strip().replace("pi", "math.pi")
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(parts)}")
    return [float(eval(p, {"math": math})) for p in parts]


def main():
    args = gymutil.parse_arguments(
        description="Visualize NTField with hanwen table setup",
        custom_parameters=[
            {"name": "--checkpoint", "type": str, "default": None, "help": "Path to NTField checkpoint (.pt)"},
            {"name": "--q_start", "type": str, "default": None, "help": "Start joint config: 6 comma-separated values in radians"},
            {"name": "--q_goal", "type": str, "default": None, "help": "Goal joint config: 6 comma-separated values in radians"},
            {"name": "--h5_path", "type": str, "default": None, "help": "HDF5 with joint_configs; use first/last as q_start/q_goal"},
            {"name": "--no_walls", "action": "store_true", "help": "Remove side walls and upper cover (table only)"},
            {"name": "--record", "action": "store_true", "help": "Record video to --record_output"},
            {"name": "--record_output", "type": str, "default": "ntfield_record.mp4", "help": "Output video path when --record (saved to PI-VLA/visualization/ unless absolute)"},
            {"name": "--headless", "action": "store_true", "help": "Run without viewer (for recording or batch)"},
            {"name": "--steps_between", "type": int, "default": 4, "help": "Interpolation steps between waypoints"},
        ],
    )
    if not args.checkpoint:
        print("Error: --checkpoint is required")
        sys.exit(1)

    # Resolve q_start, q_goal
    if args.h5_path:
        if not os.path.isfile(args.h5_path):
            print(f"Error: HDF5 file not found: {args.h5_path}")
            sys.exit(1)
        with h5py.File(args.h5_path, "r") as f:
            joint_configs = np.array(f["joint_configs"][:], dtype=np.float64)
        if joint_configs.ndim != 2 or joint_configs.shape[1] != 6:
            print(f"Error: joint_configs shape {joint_configs.shape}, expected (N, 6)")
            sys.exit(1)
        q_start = joint_configs[0].tolist()
        q_goal = joint_configs[-1].tolist()
        print(f"Using q_start/q_goal from HDF5: {args.h5_path}")
        # Auto-name video from H5 date/time
        if getattr(args, "record", False) and getattr(args, "record_output", None) == "ntfield_record.mp4":
            basename = os.path.basename(args.h5_path)
            m = re.search(r"(\d{8}_\d{6})", basename)
            if m:
                args.record_output = f"ntfield_{m.group(1)}.mp4"
    else:
        if not args.q_start or not args.q_goal:
            print("Error: provide --q_start and --q_goal, or --h5_path")
            sys.exit(1)
        q_start = parse_q(args.q_start)
        q_goal = parse_q(args.q_goal)
        # Auto-name video from end config
        if getattr(args, "record", False) and getattr(args, "record_output", None) == "ntfield_record.mp4":
            q_goal_arr = np.array(q_goal, dtype=np.float64)
            q_str = "_".join(f"{x:.2f}" for x in q_goal_arr)
            args.record_output = f"ntfield_q_{q_str}.mp4"

    q_start = np.array(q_start, dtype=np.float64)
    q_goal = np.array(q_goal, dtype=np.float64)

    # Load model
    if not os.path.isfile(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    from models.metric_arm import model_test_metric as md

    model_path = os.path.dirname(os.path.abspath(args.checkpoint))
    data_path = os.path.join(_ntrl_demo_path, "datasets", "arm", "UR5_trajectory")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=device)
    model.load(args.checkpoint)
    model.network.eval()

    # Plan
    from planning import plan as gradient_plan

    path = gradient_plan(
        model, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device=device
    )
    print(f"Planned path: {len(path)} waypoints")

    path = interpolate_path(path, steps_between=args.steps_between)
    print(f"Interpolated path: {len(path)} waypoints")

    # Isaac Gym setup (hanwen table environment)
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.8)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
    )
    if sim is None:
        print("*** Failed to create sim")
        sys.exit(1)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = os.path.join(_script_dir, "assets")
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)

    # Hanwen table setup (same as new_setup.py)
    table_dims = gymapi.Vec3(0.6, 0.9, 0.10)
    table_asset = gym.create_box(
        sim, table_dims.x, table_dims.y, table_dims.z, asset_options
    )

    add_walls = not getattr(args, "no_walls", False)
    piece_width = 0.03
    drawer_height = 0.40
    if add_walls:
        side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
        left_cover_asset = gym.create_box(
            sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options
        )
        right_cover_asset = gym.create_box(
            sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options
        )
        upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
        upper_cover_asset = gym.create_box(
            sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options
        )

    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)
    row_num = 1

    env = gym.create_env(sim, env_lower, env_upper, row_num)
    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

    ur5e_handle = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)

    spj = gym.find_actor_dof_handle(env, ur5e_handle, "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(env, ur5e_handle, "shoulder_lift_joint")
    ej = gym.find_actor_dof_handle(env, ur5e_handle, "elbow_joint")
    wj1 = gym.find_actor_dof_handle(env, ur5e_handle, "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(env, ur5e_handle, "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(env, ur5e_handle, "wrist_3_joint")

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    if add_walls:
        left_cover_pose = gymapi.Transform()
        left_cover_pose.p = gymapi.Vec3(
            table_pose.p.x, table_dims.y * 0.5 - 0.015, table_dims.z + side_cover_dims.z / 2.0
        )
        right_cover_pose = gymapi.Transform()
        right_cover_pose.p = gymapi.Vec3(
            table_pose.p.x, -table_dims.y * 0.5 + 0.015, table_dims.z + side_cover_dims.z / 2.0
        )
        upper_cover_pose = gymapi.Transform()
        upper_cover_pose.p = gymapi.Vec3(
            table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z + 0.015
        )
        gym.create_actor(env, left_cover_asset, left_cover_pose, "left_cover", 0, 1)
        gym.create_actor(env, right_cover_asset, right_cover_pose, "right_cover", 0, 1)
        gym.create_actor(env, upper_cover_asset, upper_cover_pose, "upper_cover", 0, 1)

    # Camera for recording (fixed viewpoint, same as new_setup)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720
    record_cam = None
    if getattr(args, "record", False):
        record_cam = gym.create_camera_sensor(env, camera_props)
        gym.set_camera_location(record_cam, env, gymapi.Vec3(3, 0, 0.3), gymapi.Vec3(0, 0, 0.5))

    headless = getattr(args, "headless", False)
    viewer = None
    if not headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer")
            gym.destroy_sim(sim)
            sys.exit(1)
        cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Settle at home pose (same as new_setup.py) before path animation
    home_pose = [0, -math.pi / 2, 0, -math.pi / 2, 0, 0]
    for _ in range(60):
        gym.set_dof_target_position(env, spj, home_pose[0])
        gym.set_dof_target_position(env, slj, home_pose[1])
        gym.set_dof_target_position(env, ej, home_pose[2])
        gym.set_dof_target_position(env, wj1, home_pose[3])
        gym.set_dof_target_position(env, wj2, home_pose[4])
        gym.set_dof_target_position(env, wj3, home_pose[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    # Main loop: one waypoint per physics step; path advances every frame (same as new_setup.py)
    path_id = 0
    record_frames = [] if getattr(args, "record", False) else None
    total_steps_headless = len(path) + 60  # path + short hold at end
    step_count = 0

    while True:
        if path_id >= len(path):
            path_id = len(path) - 1
        config = path[path_id]
        gym.set_dof_target_position(env, spj, config[0])
        gym.set_dof_target_position(env, slj, config[1])
        gym.set_dof_target_position(env, ej, config[2])
        gym.set_dof_target_position(env, wj1, config[3])
        gym.set_dof_target_position(env, wj2, config[4])
        gym.set_dof_target_position(env, wj3, config[5])
        path_id += 1

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        if record_frames is not None and record_cam is not None:
            gym.render_all_camera_sensors(sim)
            raw = gym.get_camera_image(sim, env, record_cam, gymapi.IMAGE_COLOR)
            rgba = raw.reshape(camera_props.height, camera_props.width, 4)
            record_frames.append(rgba[..., :3].copy())

        step_count += 1
        if headless and step_count >= total_steps_headless:
            break
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

    # Save video
    if record_frames and len(record_frames) > 0:
        out_path = getattr(args, "record_output", "ntfield_record.mp4")
        if not os.path.isabs(out_path):
            out_path = os.path.join(_visualization_dir, os.path.basename(out_path))
        os.makedirs(_visualization_dir, exist_ok=True)
        out_path = os.path.abspath(out_path)
        try:
            import imageio
            imageio.mimsave(out_path, record_frames, fps=60)
            print(f"Saved video to {out_path}")
        except (ImportError, ValueError):
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = record_frames[0].shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 60.0, (w, h))
                for f in record_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"Saved video to {out_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done.")


if __name__ == "__main__":
    main()
