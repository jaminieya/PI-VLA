#
# File:          run_collected_trajectory.py
# Brief:         Builds the same environment as new_setup.py and replays trajectories
#                from PI-VLA/collected_data HDF5 files. Robot follows trajectory from
#                start to goal.
#
# Run: cd hanwen_grasping && python run_collected_trajectory.py --h5_path ../collected_data/grasp_6dof_demo_YYYYMMDD_HHMMSS.h5
#      python run_collected_trajectory.py --h5_path ../collected_data/grasp_6dof_demo_20260308_174106.h5 --record
#      Default --record writes ../output/trajectory_evaluation/YYYYMMDD_HHMMSS/original.mp4 and episode_meta.txt
#      python run_collected_trajectory.py --h5_path ... --no_h5_object   # replay without YCB object
#      python run_collected_trajectory.py --h5_path ... --object_index 5 --object_pos 0.5,0.0,0.18  # manual object
#

from scipy.spatial.transform import Rotation as R
import math
import os
import re
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import fcl
import copy

file_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(file_dir, './util')
grasp_util_dir = os.path.join(file_dir, './grasp_util')
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)

from stl_reader import stl_reader
from obj_reader import obj_reader
import h5_episode_viz as h5viz

# Parameters (same structure as new_setup.py)
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))
piece_width = 0.03
max_scaling_factor = 0
ADD_COVER = False
TARGET_OBJ_INDEX = [1, 3, 5]
NUM_OF_OBJECTS = 1
JOINT_DIM = 6


def interpolate_path(path, steps_between=4):
    """Interpolate between consecutive waypoints for smoother animation."""
    if path is None or len(path) < 2:
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


def load_trajectory_from_h5(h5_path):
    """Load joint_configs from HDF5 file (collected_data format)."""
    import h5py
    h5_path = os.path.abspath(h5_path)
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "joint_configs" not in f:
            raise ValueError(f"No 'joint_configs' dataset in {h5_path}")
        joint_configs = np.array(f["joint_configs"][:], dtype=np.float64)
    if joint_configs.ndim != 2 or joint_configs.shape[1] < JOINT_DIM:
        raise ValueError(f"joint_configs shape {joint_configs.shape} invalid; need (N, {JOINT_DIM})")
    # Use first 6 columns (arm joints)
    return joint_configs[:, :JOINT_DIM]


def _assert_ur5_asset_bundle(asset_root, ur5e_asset_file):
    """
    Fail fast if URDF or expected meshes are missing. Isaac Gym often reports only
    'Failed to parse URDF' while still opening a viewer with just the table.
    """
    urdf_path = os.path.join(asset_root, ur5e_asset_file)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(
            f"UR5 URDF not found:\n  {urdf_path}\n\n"
            "Install the full hanwen_grasping/assets tree (UR5 URDF + meshes). "
            "Partial clones or machines without binary assets commonly miss this path."
        )
    coll = os.path.join(asset_root, "urdf/ur5e/meshes/collision")
    if not os.path.isdir(coll):
        raise FileNotFoundError(
            f"UR5 collision meshes directory missing:\n  {coll}\n"
            f"  urdf: {urdf_path}\n\n"
            "Without meshes, Isaac reports 'Failed to parse URDF' and the viewer may show only the table."
        )
    n_stl = sum(1 for name in os.listdir(coll) if name.lower().endswith(".stl"))
    if n_stl < 3:
        raise FileNotFoundError(
            f"UR5 collision meshes look incomplete under:\n  {coll}\n"
            f"(found {n_stl} .stl files; expected several for UR5). "
            "Copy the full urdf/ur5e/meshes tree from your data-collection setup."
        )


if __name__ == '__main__':
    gym = gymapi.acquire_gym()

    args = gymutil.parse_arguments(
        description="Replay collected trajectories in Isaac Gym",
        custom_parameters=[
            {'name': '--h5_path', 'type': str, 'default': None,
             'help': 'Path to HDF5 file in collected_data (e.g. ../collected_data/grasp_6dof_demo_20260308_174106.h5)'},
            {'name': '--seed', 'type': int, 'default': 42,
             'help': 'Random seed for environment (reproducibility)'},
            {'name': '--interpolate', 'type': int, 'default': 4,
             'help': 'Steps between waypoints for smoother playback (0 = no interpolation)'},
            {'name': '--record', 'action': 'store_true', 'help': 'Record video to --record_output'},
            {'name': '--record_output', 'type': str, 'default': 'collected_replay.mp4',
             'help': 'Output video path when --record (default: PI-VLA/output/trajectory_evaluation/YYYYMMDD_HHMMSS/original.mp4 from h5 name)'},
            {'name': '--object_index', 'type': int, 'default': None,
             'help': 'Object index (0-5) to show. Requires --object_pos. If omitted, no objects are shown.'},
            {'name': '--object_pos', 'type': str, 'default': None,
             'help': 'Object position "x,y,z" (e.g. 0.5,0.0,0.18). Requires --object_index. If omitted, uses HDF5 object_location + prompt when available.'},
            {'name': '--no_h5_object', 'action': 'store_true',
             'help': 'Do not spawn object from HDF5 object_location / prompt (overrides automatic placement).'},
            {'name': '--no_walls', 'action': 'store_true',
             'help': 'Match new_setup.py --no_walls: table only, no side rails / upper cover'},
            {'name': '--headless', 'action': 'store_true',
             'help': 'No interactive viewer; run fixed-length replay (for recording on servers without DISPLAY). Still uses step_graphics + camera sensors.'},
        ],
    )

    if not args.h5_path:
        print("Error: --h5_path is required. Example: --h5_path ../collected_data/grasp_6dof_demo_20260308_174106.h5")
        sys.exit(1)

    if args.object_index is not None and args.object_pos is None:
        print("Error: --object_index requires --object_pos (e.g. --object_pos 0.5,0.0,0.18)")
        sys.exit(1)
    if args.object_pos is not None and args.object_index is None:
        print("Error: --object_pos requires --object_index (e.g. --object_index 5)")
        sys.exit(1)

    # Resolve path: allow relative to PI-VLA root
    h5_path = args.h5_path
    if not os.path.isabs(h5_path):
        # Try relative to hanwen_grasping, then PI-VLA root
        cand = os.path.join(file_dir, h5_path)
        if os.path.isfile(cand):
            h5_path = cand
        else:
            pi_vla_root = os.path.dirname(file_dir)
            cand2 = os.path.join(pi_vla_root, "collected_data", os.path.basename(h5_path))
            if os.path.isfile(cand2):
                h5_path = cand2
            elif os.path.isfile(os.path.join(pi_vla_root, h5_path)):
                h5_path = os.path.join(pi_vla_root, h5_path)
            else:
                _c3 = os.path.normpath(os.path.join(file_dir, h5_path))
                if os.path.isfile(_c3):
                    h5_path = _c3

    if not os.path.isfile(h5_path):
        raise FileNotFoundError(
            f"HDF5 not found after path resolution: {h5_path}\n"
            f"Try: --h5_path ../output/data_collection/test/grasp_6dof_demo_*.h5 (from hanwen_grasping), "
            f"or an absolute path."
        )

    pi_vla_root = os.path.dirname(file_dir)
    _, session_eval_dir = h5viz.trajectory_evaluation_session_dir(pi_vla_root, h5_path)

    h5_ol, h5_prompt = h5viz.read_h5_object_and_prompt(h5_path)
    user_object = args.object_index is not None and args.object_pos is not None
    if user_object:
        show_object = True
    elif getattr(args, 'no_h5_object', False):
        show_object = False
    else:
        show_object = False
        if h5_ol is not None and h5_ol.size >= 3:
            asset_root_infer = os.path.join(file_dir, "assets")
            ycb_list_path = os.path.join(asset_root_infer, "urdf/ycb/object_urdf_grasp.txt")
            object_asset_files_infer = []
            if os.path.isfile(ycb_list_path):
                with open(ycb_list_path) as f:
                    for line in f:
                        object_asset_files_infer.append("urdf/ycb/" + line.strip())
            inferred_idx = h5viz.infer_ycb_index_from_prompt(h5_prompt, object_asset_files_infer)
            if inferred_idx is not None:
                args.object_index = inferred_idx
                args.object_pos = ",".join(str(float(x)) for x in h5_ol[:3])
                show_object = True
                print(
                    f"HDF5 object: prompt={h5_prompt!r} -> YCB index {inferred_idx} at ({args.object_pos})"
                )
            elif h5_prompt:
                print(
                    f"Warning: could not map prompt {h5_prompt!r} to a YCB asset; "
                    "replay without object (use --object_index and --object_pos, or check object_urdf_grasp.txt)."
                )

    robot_path = load_trajectory_from_h5(h5_path)
    print(f"Loaded trajectory: {len(robot_path)} waypoints from {h5_path}")

    if getattr(args, 'record', False) and getattr(args, 'record_output', None) == 'collected_replay.mp4':
        if session_eval_dir:
            os.makedirs(session_eval_dir, exist_ok=True)
            args.record_output = os.path.join(session_eval_dir, "original.mp4")
            meta_lines = [
                f"h5_path: {os.path.abspath(h5_path)}",
                f"prompt: {h5_prompt}",
                f"object_location (h5): {h5_ol}",
                f"object_index: {args.object_index}",
                f"object_pos: {args.object_pos}",
                f"video_original: original.mp4",
            ]
            meta_path = os.path.join(session_eval_dir, "episode_meta.txt")
            # If run_isaac_ntfield_demo.sh ran NTField first, episode_meta.txt already exists — append, do not overwrite.
            if os.path.isfile(meta_path):
                h5viz.append_session_meta(
                    session_eval_dir,
                    [
                        f"video_original: original.mp4",
                        f"object_index: {args.object_index}",
                        f"object_pos: {args.object_pos}",
                    ],
                )
            else:
                h5viz.write_session_meta(session_eval_dir, meta_lines)
        else:
            viz_dir = os.path.join(pi_vla_root, "visualization")
            os.makedirs(viz_dir, exist_ok=True)
            h5_basename = os.path.basename(h5_path)
            m = re.search(r"(\d{8}_\d{6})", h5_basename)
            if m:
                args.record_output = os.path.join(viz_dir, f"trained_trajectory_{m.group(1)}.mp4")
            else:
                args.record_output = os.path.join(viz_dir, f"trained_trajectory_{os.path.splitext(h5_basename)[0]}.mp4")

    if args.interpolate > 0:
        robot_path = interpolate_path(robot_path, steps_between=args.interpolate)
        print(f"Interpolated to {len(robot_path)} waypoints")

    # Deterministic environment
    np.random.seed(args.seed)
    choose = 0  # Use fixed small table config
    max_drawer_height = 0.40
    min_drawer_height = 0.40
    table_dims = gymapi.Vec3(0.56, 0.86, 0.10)
    drawer_height = min_drawer_height
    side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)

    # Sim
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

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        sys.exit(1)

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    # Assets
    asset_root = os.path.join(file_dir, "assets")
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    object_asset_files = []
    object_collision_files = []
    object_offset = []
    object_common_prefix = "urdf/ycb/"
    # YCB lists are only needed when placing an object (--object_index + --object_pos)
    if show_object:
        _urdf_list = os.path.join(asset_root, "urdf/ycb/object_urdf_grasp.txt")
        if not os.path.isfile(_urdf_list):
            raise FileNotFoundError(
                f"Missing YCB asset list (needed for --object_index): {_urdf_list}\n"
                "Replay without objects does not require these files, or install/copy hanwen_grasping/assets."
            )
        with open(_urdf_list) as f:
            for line in f:
                object_asset_files.append(object_common_prefix + line.strip())
        with open(os.path.join(asset_root, "urdf/ycb/object_collision_grasp.txt")) as f:
            for line in f:
                object_collision_files.append(object_common_prefix + line.strip())
        with open(os.path.join(asset_root, "urdf/ycb/object_offset_grasp.txt")) as f:
            for line in f:
                div = line.strip().split(" ")
                object_offset.append([float(x) for x in div])

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    try:
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    except (TypeError, ValueError):
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    _assert_ur5_asset_bundle(asset_root, ur5e_asset_file)
    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    if ur5e_asset is None or ur5e_asset == 0:
        raise RuntimeError(
            f"gym.load_asset failed for {ur5e_asset_file} under {asset_root}. "
            "Check Isaac log above for URDF parse errors; meshes paths must resolve."
        )
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    left_cover_asset = gym.create_box(sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options)
    right_cover_asset = gym.create_box(sim, side_cover_dims.x, side_cover_dims.y, side_cover_dims.z, asset_options)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)

    asset_options.fix_base_link = False
    object_assets = []
    if show_object:
        for ob in object_asset_files:
            object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
    asset_options.fix_base_link = True

    # Poses
    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    left_cover_pose = gymapi.Transform()
    left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y * 0.5 - 0.015, table_dims.z + side_cover_dims.z / 2.0)

    right_cover_pose = gymapi.Transform()
    right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y * 0.5 + 0.015, table_dims.z + side_cover_dims.z / 2.0)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z + 0.015)

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    # Environment
    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)

    envs = []
    ur5e_handles = []
    body_cam_handles = []
    object_handles = []

    for i in range(num_of_envs):
        envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")

        cam_link = gym.find_actor_rigid_body_handle(envs[-1], ur5e_handles[-1], "wrist_3_link")
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        body_cam_transform = gymapi.Transform()
        body_cam_transform.p = gymapi.Vec3(0.11, 0, 0.08)
        gym.attach_camera_to_body(body_cam_handles[-1], envs[-1], cam_link, body_cam_transform,
                                 gymapi.CameraFollowMode.FOLLOW_TRANSFORM)

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        add_walls = not getattr(args, 'no_walls', False)
        if add_walls:
            gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
            gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
        if add_walls and ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)

        # Objects: only shown when --object_index and --object_pos are both provided
        if show_object:
            idx = args.object_index
            if idx < 0 or idx >= len(object_assets):
                print(f"Error: --object_index {idx} out of range [0, {len(object_assets) - 1}]")
                sys.exit(1)
            parts = [p.strip() for p in args.object_pos.split(",")]
            if len(parts) != 3:
                print("Error: --object_pos must be 'x,y,z' (e.g. 0.5,0.0,0.18)")
                sys.exit(1)
            tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(tx, ty, tz)
            obj_asset = object_assets[idx]
            object_handles.append(
                gym.create_actor(envs[-1], obj_asset, object_pose, "object0" + str(i), 0, 2, 1)
            )
            gym.set_actor_scale(envs[-1], object_handles[-1], 1.0)

        # Global camera for recording
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        viewpoint_candidate = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(body_cam_handles[-1], envs[-1], viewpoint_candidate, camera_focus)

    # Store joint handles for the last env (we use envs[-1])
    spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
    slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
    ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
    wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
    wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
    wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")

    viewer = None
    if not getattr(args, 'headless', False):
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise ValueError("*** Failed to create viewer")
        cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    else:
        print("Headless mode: no Isaac viewer window (use --record to save MP4).")
        if not getattr(args, 'record', False):
            print("Warning: --headless without --record exits after replay; nothing is saved.")
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(1.0, 0.0, 0.0))

    # Brief settle at start config
    SETTLE_STEPS = 60
    for _ in range(SETTLE_STEPS):
        dof_result = robot_path[0]
        gym.set_dof_target_position(envs[-1], spj, dof_result[0])
        gym.set_dof_target_position(envs[-1], slj, dof_result[1])
        gym.set_dof_target_position(envs[-1], ej, dof_result[2])
        gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
        gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
        gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

    # Replay trajectory (mutable so nested helper can update without nonlocal-at-module-level)
    path_id_box = [0]
    record_frames = [] if getattr(args, 'record', False) else None
    headless_hold_frames = 90

    def _replay_step():
        path_id = path_id_box[0]
        if path_id >= len(robot_path):
            path_id = len(robot_path) - 1
        dof_result = robot_path[path_id]
        gym.set_dof_target_position(envs[-1], spj, dof_result[0])
        gym.set_dof_target_position(envs[-1], slj, dof_result[1])
        gym.set_dof_target_position(envs[-1], ej, dof_result[2])
        gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
        gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
        gym.set_dof_target_position(envs[-1], wj3, dof_result[5])
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
        if record_frames is not None:
            gym.render_all_camera_sensors(sim)
            raw = gym.get_camera_image(sim, envs[-1], body_cam_handles[-1], gymapi.IMAGE_COLOR)
            rgba = raw.reshape(camera_props.height, camera_props.width, 4)
            rgb = rgba[..., :3].copy()
            record_frames.append(rgb)
        path_id_box[0] = path_id + 1

    if getattr(args, 'headless', False):
        for _ in range(len(robot_path) + headless_hold_frames):
            _replay_step()
    else:
        while not gym.query_viewer_has_closed(viewer):
            _replay_step()

    # Save recorded video
    if record_frames and len(record_frames) > 0:
        out_path = getattr(args, 'record_output', 'collected_replay.mp4')
        out_path = os.path.abspath(out_path)
        saved = False
        try:
            import imageio
            imageio.mimsave(out_path, record_frames, fps=60)
            print(f"Saved video to {out_path}")
            saved = True
        except (ImportError, ValueError):
            pass
        if not saved:
            try:
                import cv2
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = record_frames[0].shape[:2]
                writer = cv2.VideoWriter(out_path, fourcc, 60.0, (w, h))
                for f in record_frames:
                    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                writer.release()
                print(f"Saved video to {out_path}")
                saved = True
            except Exception as e:
                print(f"Could not save video: {e}")
        if not saved:
            print("Install imageio[ffmpeg] or opencv-python for video recording")

    print("Replay completed.")
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sys.exit(0)
