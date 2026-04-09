#!/usr/bin/env python3
"""
Collect RRTConnect (OMPL) trajectories for NTField training.

Environment matches trajectory_evaluation/comparison/run_rrt_ntfield_benchmark.py
(fixed table 0.8x1.0x0.1, UR5e, single YCB object from TARGET_OBJ_INDEX).

Per successful episode:
  - Random object pose (world frame)
  - TracIK + grasp_dict + collision checks + feasibility RRT (same as benchmark)
  - Shortest-path RRTConnect from **actual settled** q_start to grasp q_goal via get_patha2b
  - Save HDF5: joint_configs (T,6), final_joint_config, q_start, object pose, attrs

After collection, run (optional --export_ntfield_numpy):
  ntrl-demo/dataprocessing/prepare_trajectory_dataset.py --data_dir <out> ...

Usage (from PI-VLA root):
  python new_clean_data_collect/collect_ntfield_rrt_episodes.py \\
    --num_episodes 1000 \\
    --output_dir output/data_collection/20260408

Resume after interruption (same ``--output_dir`` and ``--num_episodes`` total target):
  python new_clean_data_collect/collect_ntfield_rrt_episodes.py \\
    --num_episodes 1000 --output_dir output/data_collection --resume
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np

_PI_VLA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANWEN_GRASPING_ROOT = os.path.join(_PI_VLA_ROOT, "hanwen_grasping")
_COLLECT_DATA_DIR = os.path.join(HANWEN_GRASPING_ROOT, "collect_data")
_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "util")
_GRASP_UTIL_DIR = os.path.join(_COLLECT_DATA_DIR, "grasp_util")
_NTRL_DEMO = os.path.join(_PI_VLA_ROOT, "ntrl-demo")

for _p in (HANWEN_GRASPING_ROOT, _UTIL_DIR, _GRASP_UTIL_DIR, _PI_VLA_ROOT, _NTRL_DEMO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_EPISODE_GLOB = "ntfield_rrt_ep_*.h5"
_RNG_STATE_NAME = ".ntfield_collect_rng_state.pkl"


def _count_saved_episodes(out_dir: str) -> int:
    return len(glob.glob(os.path.join(out_dir, _EPISODE_GLOB)))


def _count_jsonl_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _save_rng_state(out_dir: str, rng: np.random.Generator) -> None:
    path = os.path.join(out_dir, _RNG_STATE_NAME)
    with open(path, "wb") as f:
        pickle.dump(rng.bit_generator.state, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_rng_state(out_dir: str) -> Optional[dict]:
    path = os.path.join(out_dir, _RNG_STATE_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def run_one_episode(
    ox: float,
    oy: float,
    oz: float,
    episode_index: int,
    out_h5_path: str,
    rng: np.random.Generator,
    use_viewer: bool,
    rrt_time_limit: float,
    argv_gym_tail: List[str],
) -> bool:
    from scipy.spatial.transform import Rotation as R
    from isaacgym import gymapi
    from isaacgym import gymutil
    import fcl
    import h5py
    import robot_arm_configuration as RC
    from stl_reader import stl_reader
    from obj_reader import obj_reader

    from trajectory_evaluation.comparison.run_rrt_ntfield_benchmark import (
        TABLE_DIMS_X,
        TABLE_DIMS_Y,
        TABLE_DIMS_Z,
        DRAWER_HEIGHT,
        NUM_OF_OBJECTS,
        TARGET_OBJ_INDEX,
        get_swept_volume_size,
        sim_dt,
    )
    from final_integrate.run_integrated_pipeline import find_grasp_q_goal

    _cwd = os.getcwd()
    os.chdir(HANWEN_GRASPING_ROOT)

    sys.argv = [sys.argv[0]] + argv_gym_tail
    gym = gymapi.acquire_gym()
    gym_args = gymutil.parse_arguments(description="ntfield_collect", headless=True, custom_parameters=[])
    gym_args.headless = not use_viewer

    table_dims = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)
    drawer_height = DRAWER_HEIGHT

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = sim_dt
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = gym_args.num_threads
    sim_params.physx.use_gpu = gym_args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = None
    viewer = None
    try:
        sim = gym.create_sim(
            gym_args.compute_device_id, gym_args.graphics_device_id, gym_args.physics_engine, sim_params
        )
        if sim is None:
            return False

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane_params)

        asset_root = "./assets/"
        ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
        ur5e_collision_parts = [
            "urdf/ur5e/meshes/collision/base.stl",
            "urdf/ur5e/meshes/collision/shoulder.stl",
            "urdf/ur5e/meshes/collision/upperarm.stl",
            "urdf/ur5e/meshes/collision/forearm.stl",
            "urdf/ur5e/meshes/collision/wrist1.stl",
            "urdf/ur5e/meshes/collision/wrist2.stl",
            "urdf/ur5e/meshes/collision/wrist3.stl",
        ]
        object_asset_files: List[str] = []
        object_collision_files: List[str] = []
        object_offset: List[List[float]] = []
        object_common_prefix = "urdf/ycb/"
        with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
            for line in f:
                object_asset_files.append(object_common_prefix + line[:-1])
        with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
            for line in f:
                object_collision_files.append(object_common_prefix + line[:-1])
        with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
            for line in f:
                object_offset.append([float(x) for x in line[:-1].split(" ")])

        ur5e_collision_models = []
        ur5e_rotations = [
            R.from_euler("x", [90], degrees=True),
            R.from_euler("xy", [90, 180], degrees=True),
            R.from_euler("xy", [180, 180], degrees=True),
            R.from_euler("z", [-180], degrees=True),
            R.from_euler("x", [-180], degrees=True),
            R.from_euler("x", [90], degrees=True),
            R.from_euler("z", [-90], degrees=True),
        ]
        ur5e_translations = [
            [0, 0, 0],
            [0, 0, 0],
            [0, -0.138, 0],
            [0, -0.007, 0],
            [0, 0.127, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for idx, parts_path in enumerate(ur5e_collision_parts):
            collision_mesh = stl_reader(asset_root + parts_path)
            m = fcl.BVHModel()
            collision_mesh.transform(ur5e_rotations[idx], ur5e_translations[idx])
            verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            ur5e_collision_models.append(m)

        if not gym_args.headless:
            viewer = gym.create_viewer(sim, gymapi.CameraProperties())
            if viewer is None:
                gym_args.headless = True

        spacing = 2
        env_lower = gymapi.Vec3(-spacing, -spacing, 0)
        env_upper = gymapi.Vec3(spacing, spacing, 0)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.use_mesh_materials = True
        ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
        table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

        asset_options.fix_base_link = False
        object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

        ur5e_pose = gymapi.Transform()
        ur5e_pose.p = gymapi.Vec3(0, 0, 0)
        ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

        plane_normal = np.array([0.0, 0.0, 1.0])
        col_plane = fcl.Plane(plane_normal, 0)
        plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())
        col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
        trans_table = fcl.Transform(np.array([table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5]))
        table_obj = fcl.CollisionObject(col_table, trans_table)
        object_collision_models = [table_obj]

        envs: List[Any] = []
        ur5e_handles: List[Any] = []
        object_handles: List[Any] = []
        object_status_list: List[Any] = []
        object_reader_tracker: List[Any] = []
        object_mesh: List[Any] = []
        object_collision_lib: List[Any] = []
        spj = slj = ej = wj1 = wj2 = wj3 = None
        target_file_idx = np.array(TARGET_OBJ_INDEX)

        envs.append(gym.create_env(sim, env_lower, env_upper, 1))
        i = 0
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs: List[Any] = []

        object_scaling_factor = np.ones(NUM_OF_OBJECTS, dtype=np.float64)

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            object_pose.p = gymapi.Vec3(ox, oy, oz)
            file_path = object_collision_files[target_file_idx[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[target_file_idx[k]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array([ox, oy, oz]))
            object_handles.append(
                gym.create_actor(
                    envs[-1],
                    object_assets[target_file_idx[k]],
                    object_pose,
                    "object" + str(k) + str(i),
                    0,
                    2 ** (k + 1),
                    k + 1,
                )
            )
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_objs.append(fcl.CollisionObject(m, t))
            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

        if viewer is not None:
            cam_pos = gymapi.Vec3(2.2, 0, 0.5)
            cam_target = gymapi.Vec3(0, 0, 0.5)
            gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

        env = envs[-1]
        ur = ur5e_handles[-1]
        real_position = False
        for t in range(2000):
            if not real_position:
                gym.set_dof_target_position(env, spj, 0)
                gym.set_dof_target_position(env, slj, -math.pi / 2)
                gym.set_dof_target_position(env, ej, 0)
                gym.set_dof_target_position(env, wj1, -math.pi / 2)
                gym.set_dof_target_position(env, wj2, 0)
                gym.set_dof_target_position(env, wj3, 0)
                real_position = True
            if t == 999:
                for ii, element in enumerate(object_handles):
                    states = gym.get_actor_rigid_body_states(env, element, 1)
                    rotation = np.array(states[0][0][1])
                    translation = np.array(states[0][0][0])
                    rotation = np.array(rotation.item())
                    translation = np.array(translation.item())
                    object_status_list[ii][0] += translation
                    r1 = R.from_quat(rotation)
                    tf = fcl.Transform(r1.as_matrix(), translation)
                    temp_obj = object_reader_tracker[ii]
                    temp_obj.set_offset(translation)
                    vertices, faces = temp_obj.get_bounding_box_mesh()
                    object_mesh.append([vertices, faces])
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            if viewer is not None:
                gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        if len(object_mesh) < 1:
            return False

        dof_snapshot = gym.get_actor_dof_states(env, ur, gymapi.STATE_POS)
        q_start_live = np.array(dof_snapshot["pos"][:6], dtype=np.float64)

        scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        file_path_rac = "./assets/urdf/ur5e/meshes/collision/"
        rac = RC.robot_arm_configuration(
            file_path_rac, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info
        )

        grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[0]].split("/")[:-1]) + "/grasp_dict.npy"
        grasp_data = np.load(grasp_file, allow_pickle=True)
        target_idx = 0
        grasp_list = np.arange(len(grasp_data))
        rng.shuffle(grasp_list)

        obj_xy = np.array([ox, oy], dtype=np.float64)
        grasp_target_q, _, _ = find_grasp_q_goal(
            rac,
            RC,
            scene_info,
            grasp_data,
            grasp_list,
            obj_xy,
            target_idx,
            object_mesh,
            object_collision_models,
            plane_obj,
            get_swept_volume_size,
        )

        if grasp_target_q is None:
            return False

        with open(os.devnull, "w", encoding="utf-8") as _devnull:
            with contextlib.redirect_stdout(_devnull):
                path_list = RC.get_patha2b(
                    rac,
                    q_start_live.tolist(),
                    grasp_target_q.tolist(),
                    scene_info,
                    target_mesh=object_mesh[target_idx],
                    time_limit=rrt_time_limit,
                    given_static_model=object_collision_models,
                )

        if path_list is None or len(path_list) < 2:
            return False

        joint_configs = np.asarray(path_list, dtype=np.float64).reshape(-1, 6)
        os.makedirs(os.path.dirname(os.path.abspath(out_h5_path)) or ".", exist_ok=True)
        with h5py.File(out_h5_path, "w") as hf:
            hf.create_dataset("joint_configs", data=joint_configs, compression="gzip", compression_opts=4)
            hf.create_dataset("final_joint_config", data=grasp_target_q.astype(np.float64))
            hf.create_dataset("q_start", data=q_start_live.astype(np.float64))
            hf.attrs["object_x"] = ox
            hf.attrs["object_y"] = oy
            hf.attrs["object_z"] = oz
            hf.attrs["episode_index"] = episode_index
            hf.attrs["created_iso"] = datetime.now().isoformat()
            hf.attrs["rrt_time_limit_s"] = rrt_time_limit
            hf.attrs["source"] = "new_clean_data_collect/collect_ntfield_rrt_episodes.py"
            hf.attrs["planner_start_end"] = "get_patha2b settled_q_start -> grasp_q_goal RRTConnect shortest"

        return True
    finally:
        if sim is not None:
            gym.destroy_sim(sim)
        if viewer is not None:
            gym.destroy_viewer(viewer)
        os.chdir(_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect RRT trajectories for NTField (benchmark scene).")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for *.h5 (default: output/data_collection/20260408 under PI-VLA)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--object_z", type=float, default=0.18)
    parser.add_argument("--ox_min", type=float, default=0.42)
    parser.add_argument("--ox_max", type=float, default=0.98)
    parser.add_argument("--oy_min", type=float, default=-0.38)
    parser.add_argument("--oy_max", type=float, default=0.38)
    parser.add_argument("--rrt_time_limit", type=float, default=30.0)
    parser.add_argument("--max_tries_total", type=int, default=50000, help="Stop after this many attempts (success+fail)")
    parser.add_argument(
        "--export_ntfield_numpy",
        action="store_true",
        help="After collection, write points.npy + tau_obs.npy into output_dir via trajectory_sampler",
    )
    parser.add_argument("--num_pairs", type=int, default=100000, help="With --export_ntfield_numpy")
    parser.add_argument("--tau_min", type=float, default=0.01)
    parser.add_argument("--tau_max", type=float, default=2.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue toward --num_episodes: count existing ntfield_rrt_ep_*.h5 in --output_dir, "
        "restore RNG state from .ntfield_collect_rng_state.pkl if present, append to collection_log.jsonl.",
    )
    args, argv_remainder = parser.parse_known_args()

    argv_gym = list(argv_remainder)
    if not args.use_viewer and "--headless" not in argv_gym:
        argv_gym.append("--headless")

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.join(_PI_VLA_ROOT, "output", "data_collection", "20260408")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "collection_log.jsonl")
    rng_state_path = os.path.join(out_dir, _RNG_STATE_NAME)

    if args.resume:
        successes = _count_saved_episodes(out_dir)
        attempts = _count_jsonl_lines(log_path)
        st = _load_rng_state(out_dir)
        if st is not None:
            rng = np.random.default_rng()
            rng.bit_generator.state = st
            print(
                f"Resume: {successes} episodes on disk, {attempts} logged attempts, RNG state restored."
            )
        else:
            rng = np.random.default_rng(args.seed)
            print(
                f"Resume: {successes} episodes on disk, {attempts} logged attempts; "
                f"no {rng_state_path!r} — starting new RNG from --seed (poses may repeat early draws)."
            )
        if successes >= args.num_episodes:
            print(f"Already have {successes} >= {args.num_episodes} episodes. Nothing to do.")
            return
    else:
        successes = 0
        attempts = 0
        rng = np.random.default_rng(args.seed)
        if os.path.isfile(rng_state_path):
            os.remove(rng_state_path)

    while successes < args.num_episodes and attempts < args.max_tries_total:
        ox = float(rng.uniform(args.ox_min, args.ox_max))
        oy = float(rng.uniform(args.oy_min, args.oy_max))
        oz = float(args.object_z)
        h5_name = f"ntfield_rrt_ep_{successes:05d}_{datetime.now().strftime('%H%M%S')}.h5"
        out_h5 = os.path.join(out_dir, h5_name)
        t0 = time.perf_counter()
        ok = run_one_episode(
            ox,
            oy,
            oz,
            successes,
            out_h5,
            rng,
            args.use_viewer,
            args.rrt_time_limit,
            argv_gym,
        )
        dt = time.perf_counter() - t0
        attempts += 1
        rec = {
            "attempt": attempts,
            "success": ok,
            "wall_s": round(dt, 3),
            "object_xyz": [ox, oy, oz],
            "path": out_h5 if ok else None,
        }
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec) + "\n")
        _save_rng_state(out_dir, rng)
        if ok:
            successes += 1
            print(f"[{successes}/{args.num_episodes}] OK  {out_h5}  ({dt:.1f}s)")
        else:
            print(f"[try {attempts}] FAIL  obj=({ox:.3f},{oy:.3f},{oz:.3f})  ({dt:.1f}s)")

    print(f"Done. Successes: {successes}/{args.num_episodes}, attempts: {attempts}, out: {out_dir}")

    if args.export_ntfield_numpy and successes > 0:
        sys.path.insert(0, _NTRL_DEMO)
        if _NTRL_DEMO + "/dataprocessing" not in sys.path:
            sys.path.insert(0, os.path.join(_NTRL_DEMO, "dataprocessing"))
        from trajectory_sampler import load_trajectories_from_h5, sample_pairs_from_trajectories

        trajs = load_trajectories_from_h5(out_dir)
        pts, tau = sample_pairs_from_trajectories(
            trajs,
            num_pairs=args.num_pairs,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            rng=np.random.default_rng(args.seed + 999),
        )
        np.save(os.path.join(out_dir, "points.npy"), pts)
        np.save(os.path.join(out_dir, "tau_obs.npy"), tau)
        print(f"Wrote points.npy {pts.shape}, tau_obs.npy {tau.shape} -> {out_dir}")


if __name__ == "__main__":
    main()
