#
# collect_multi_obj_ntfield_data.py
# Build the same multi-object tabletop environment as collect_multi_obj_data_for_student.py,
# but save only NTField-oriented data:
#   - realpc.obj / dim / ur5e.urdf scene bundle
#   - sampled_points.npy / speed.npy / normal.npy
#
# Key sampling policy:
#   - q_start: near obstacles (distance band in workspace-derived C-space metric)
#   - q_goal: uniform in normalized joint space [-0.5, 0.5]^6
#
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import math
import os
import shutil
import sys

import fcl
import h5py
import igl
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import torch
import torch_kdtree


OBJECT_NAMES = {
    "002_master_chef_can": "master chef can",
    "004_sugar_box": "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle": "mustard bottle",
    "036_wood_block": "wood block",
    "011_banana": "banana",
}

_script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.abspath(os.path.join(_script_dir, ".."))
PI_VLA_ROOT = os.path.dirname(package_root)
ASSETS_DIR = os.path.join(package_root, "assets")
if not os.path.isfile(os.path.join(ASSETS_DIR, "urdf", "ycb", "object_urdf_grasp.txt")):
    alt_assets = os.path.join(PI_VLA_ROOT, "assets")
    if os.path.isfile(os.path.join(alt_assets, "urdf", "ycb", "object_urdf_grasp.txt")):
        ASSETS_DIR = alt_assets

util_dir = os.path.join(package_root, "util")
sys.path.insert(0, package_root)
sys.path.append(util_dir)

from obj_reader import obj_reader
from stl_reader import stl_reader

# Reuse arm distance + FK utilities from ntrl-demo preprocessing.
NTRL_DEMO_ROOT = os.path.join(PI_VLA_ROOT, "ntrl-demo")
sys.path.append(os.path.join(NTRL_DEMO_ROOT, "dataprocessing"))
from speed_sampling_arm_normal import build_chain, arm_obstacle_distance


TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
table_dims = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)

# Banana (5), sugar box (1), mustard bottle (3) in object_urdf_grasp.txt indexing.
TARGET_OBJ_INDEX = [1, 3, 5]
NUM_OF_OBJECTS = 3
MAX_SCALING_FACTOR = 0


def _table_box_mesh_vertices_faces(center_xyz, dims_xyz):
    cx, cy, cz = center_xyz
    dx, dy, dz = dims_xyz
    v = np.array(
        [
            [cx - dx / 2.0, cy - dy / 2.0, cz - dz / 2.0],
            [cx - dx / 2.0, cy + dy / 2.0, cz - dz / 2.0],
            [cx + dx / 2.0, cy + dy / 2.0, cz - dz / 2.0],
            [cx + dx / 2.0, cy - dy / 2.0, cz - dz / 2.0],
            [cx - dx / 2.0, cy - dy / 2.0, cz + dz / 2.0],
            [cx - dx / 2.0, cy + dy / 2.0, cz + dz / 2.0],
            [cx + dx / 2.0, cy + dy / 2.0, cz + dz / 2.0],
            [cx + dx / 2.0, cy - dy / 2.0, cz + dz / 2.0],
        ],
        dtype=np.float64,
    )
    f = np.array(
        [
            [0, 2, 1], [0, 2, 3],
            [4, 6, 5], [4, 6, 7],
            [5, 2, 1], [5, 2, 6],
            [7, 2, 3], [7, 2, 6],
            [4, 3, 0], [4, 3, 7],
            [4, 1, 0], [4, 1, 5],
        ],
        dtype=np.int64,
    )
    return v, f


def _write_obj(path, vertices, faces):
    with open(path, "w") as f:
        for vx, vy, vz in vertices:
            f.write(f"v {vx:.8f} {vy:.8f} {vz:.8f}\n")
        for i, j, k in faces:
            f.write(f"f {i + 1} {j + 1} {k + 1}\n")


def _export_scene_bundle(output_dir, table_pose, table_dims, object_reader_tracker):
    os.makedirs(output_dir, exist_ok=True)
    vertices_all, faces_all = [], []
    v_offset = 0

    table_center = np.array([table_pose.p.x, table_pose.p.y, table_pose.p.z], dtype=np.float64)
    table_size = np.array([table_dims.x, table_dims.y, table_dims.z], dtype=np.float64)
    tv, tf = _table_box_mesh_vertices_faces(table_center, table_size)
    vertices_all.append(tv)
    faces_all.append(tf + v_offset)
    v_offset += tv.shape[0]

    for mesh_reader in object_reader_tracker:
        ov = np.asarray(mesh_reader.get_vertices(), dtype=np.float64)
        of = np.asarray(mesh_reader.get_faces(), dtype=np.int64)
        if ov.size == 0 or of.size == 0:
            continue
        vertices_all.append(ov)
        faces_all.append(of + v_offset)
        v_offset += ov.shape[0]

    vertices = np.concatenate(vertices_all, axis=0)
    faces = np.concatenate(faces_all, axis=0)
    obj_path = os.path.join(output_dir, "realpc.obj")
    _write_obj(obj_path, vertices, faces)

    dim_path = os.path.join(output_dir, "dim")
    with open(dim_path, "w") as f:
        f.write("6\n")
        f.write("wrist_3_link\n")

    urdf_src = os.path.join(ASSETS_DIR, "urdf", "ur5e", "ur5e.urdf")
    urdf_dst = os.path.join(output_dir, "ur5e.urdf")
    if os.path.isfile(urdf_src):
        shutil.copy2(urdf_src, urdf_dst)

    return obj_path, dim_path, urdf_dst


def _collect_near_obstacle_starts(
    target_n,
    chain,
    mesh_list,
    kdtree,
    v_obs,
    q_min=-0.5,
    q_max=0.5,
    near_min=0.005,
    near_max=0.05,
    batch_size=60000,
):
    scale = math.pi / 0.5
    starts_list = []
    d_list = []
    n_list = []

    while sum(x.shape[0] for x in starts_list) < target_n:
        q = (q_max - q_min) * torch.rand((batch_size, 6), device="cuda") + q_min
        d, n = arm_obstacle_distance(scale * q, chain, mesh_list, kdtree, v_obs)
        d_adj = d - 0.01
        keep = (d_adj > near_min) & (d_adj < near_max)
        if torch.any(keep):
            starts_list.append(q[keep].detach())
            d_list.append(d_adj[keep].detach())
            n_list.append(n[keep].detach())
        del q, d, n, d_adj, keep
        torch.cuda.empty_cache()

    q_start = torch.cat(starts_list, dim=0)[:target_n]
    d_start = torch.cat(d_list, dim=0)[:target_n]
    n_start = torch.cat(n_list, dim=0)[:target_n]
    n_start = n_start / torch.clamp(torch.norm(n_start, dim=1, keepdim=True), min=1e-8)
    return q_start, d_start, n_start


def _collect_uniform_goals(
    target_n,
    chain,
    mesh_list,
    kdtree,
    v_obs,
    q_min=-0.5,
    q_max=0.5,
):
    scale = math.pi / 0.5
    q_goal = (q_max - q_min) * torch.rand((target_n, 6), device="cuda") + q_min
    d_goal, n_goal = arm_obstacle_distance(scale * q_goal, chain, mesh_list, kdtree, v_obs)
    d_goal = d_goal - 0.01
    n_goal = n_goal / torch.clamp(torch.norm(n_goal, dim=1, keepdim=True), min=1e-8)
    return q_goal, d_goal, n_goal


def _build_ntfield_arrays_from_scene(
    output_dir,
    num_samples=500000,
    near_min=0.005,
    near_max=0.05,
    margin=0.05,
    offset=0.005,
    mesh_surface_samples=120000,
):
    obj_path = os.path.join(output_dir, "realpc.obj")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"Missing scene mesh: {obj_path}")

    v, f = igl.read_triangle_mesh(obj_path)
    if v.shape[0] == 0 or f.shape[0] == 0:
        raise RuntimeError("Scene mesh is empty; cannot sample NTField dataset.")
    bary, FI, _ = igl.random_points_on_mesh(mesh_surface_samples, v, f)
    face_verts = v[f[FI], :]
    v_obs_np = np.sum(bary[..., np.newaxis] * face_verts, axis=1)

    chain, mesh_list = build_chain()
    v_obs = torch.tensor(v_obs_np, dtype=torch.float32, device="cuda")
    kdtree = torch_kdtree.build_kd_tree(v_obs)

    q_start, d_start, n_start = _collect_near_obstacle_starts(
        target_n=num_samples,
        chain=chain,
        mesh_list=mesh_list,
        kdtree=kdtree,
        v_obs=v_obs,
        near_min=near_min,
        near_max=near_max,
    )
    q_goal, d_goal, n_goal = _collect_uniform_goals(
        target_n=num_samples,
        chain=chain,
        mesh_list=mesh_list,
        kdtree=kdtree,
        v_obs=v_obs,
    )

    sampled_points = torch.cat((q_start, q_goal), dim=1).detach().cpu().numpy()
    normal = torch.cat((n_start, n_goal), dim=1).detach().cpu().numpy()

    speed = np.zeros((num_samples, 2), dtype=np.float32)
    speed[:, 0] = np.clip(d_start.detach().cpu().numpy(), a_min=offset, a_max=margin) / margin
    speed[:, 1] = np.clip(d_goal.detach().cpu().numpy(), a_min=offset, a_max=margin) / margin

    np.save(os.path.join(output_dir, "sampled_points.npy"), sampled_points.astype(np.float32))
    np.save(os.path.join(output_dir, "speed.npy"), speed.astype(np.float32))
    np.save(os.path.join(output_dir, "normal.npy"), normal.astype(np.float32))
    np.save(os.path.join(output_dir, "B.npy"), np.random.normal(0, 1, size=(3, 128)).astype(np.float32))


if __name__ == "__main__":
    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(
        description="Collect multi-object environment data for NTField only",
        custom_parameters=[
            {"name": "--num_episodes", "type": int, "default": 1, "help": "Number of environment samples"},
            {"name": "--headless", "action": "store_true", "help": "Run without viewer"},
            {"name": "--output_dir", "type": str, "default": None, "help": "Output root"},
            {"name": "--num_samples", "type": int, "default": 500000, "help": "NTField samples per episode"},
            {"name": "--near_min", "type": float, "default": 0.005, "help": "Start near-obstacle lower bound"},
            {"name": "--near_max", "type": float, "default": 0.05, "help": "Start near-obstacle upper bound"},
        ],
    )

    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim.")
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())

    asset_root = ASSETS_DIR + os.sep
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    object_asset_files, object_collision_files, object_offset = [], [], []
    with open(os.path.join(ASSETS_DIR, "urdf", "ycb", "object_urdf_grasp.txt")) as f:
        for line in f:
            object_asset_files.append("urdf/ycb/" + line.strip())
    with open(os.path.join(ASSETS_DIR, "urdf", "ycb", "object_collision_grasp.txt")) as f:
        for line in f:
            object_collision_files.append("urdf/ycb/" + line.strip())
    with open(os.path.join(ASSETS_DIR, "urdf", "ycb", "object_offset_grasp.txt")) as f:
        for line in f:
            object_offset.append([float(x) for x in line.strip().split(" ")])

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True
    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
    asset_options.fix_base_link = False
    object_assets = [gym.load_asset(sim, asset_root, ob, asset_options) for ob in object_asset_files]

    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 0), 1)
    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)
    ur5e_handle = gym.create_actor(env, ur5e_asset, ur5e_pose, "ur5e", 0, 32767)
    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)
    gym.create_actor(env, table_asset, table_pose, "table", 0, 1)

    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20

    default_output_dir = os.path.join(PI_VLA_ROOT, "output", "multi_obj_ntfield")
    output_root = (
        os.path.abspath(args.output_dir)
        if args.output_dir and os.path.isabs(args.output_dir)
        else os.path.abspath(os.path.join(PI_VLA_ROOT, args.output_dir))
        if args.output_dir
        else default_output_dir
    )
    os.makedirs(output_root, exist_ok=True)

    # Build one collision manager for object placement validity checks.
    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_pose.p.x, table_pose.p.y, table_pose.p.z]))
    table_obj = fcl.CollisionObject(col_table, trans_table)

    successful = 0
    for ep in range(max(1, int(args.num_episodes))):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = os.path.join(output_root, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        target_file_idx = np.random.choice(TARGET_OBJ_INDEX, NUM_OF_OBJECTS, replace=False)
        object_handles = []
        object_reader_tracker = []
        object_collision_lib = []
        object_scaling_factor = (
            np.random.randint(0, MAX_SCALING_FACTOR + 1, size=NUM_OF_OBJECTS) / 10.0 + 1.0
        )

        obstacle_objs = [table_obj]
        gt_obj_xy = []
        for k in range(NUM_OF_OBJECTS):
            is_collision = True
            while is_collision:
                tx = np.random.uniform(table_x_min, table_x_max)
                ty = np.random.uniform(table_y_min, table_y_max)
                tz = table_dims.z + 0.08

                collision_mesh = obj_reader(asset_root + object_collision_files[target_file_idx[k]])
                collision_mesh.set_scale(object_scaling_factor[k])
                collision_mesh.add_offset(object_offset[target_file_idx[k]])
                verts, tris = collision_mesh.get_bounding_box_mesh()
                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                t = fcl.Transform(np.array([tx, ty, tz]))
                temp_obj = fcl.CollisionObject(m, t)

                is_collision = False
                req = fcl.CollisionRequest()
                result = fcl.CollisionResult()
                for prev in obstacle_objs:
                    if fcl.collide(temp_obj, prev, req, result):
                        is_collision = True
                        break
                if not is_collision:
                    for obj in gt_obj_xy:
                        if np.linalg.norm(np.array([tx, ty]) - np.array(obj)) <= 0.16:
                            is_collision = True
                            break

            gt_obj_xy.append([tx, ty])
            obstacle_objs.append(temp_obj)
            object_collision_lib.append(m)
            object_reader_tracker.append(collision_mesh)

            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(tx, ty, tz)
            h = gym.create_actor(
                env,
                object_assets[target_file_idx[k]],
                pose,
                f"object{k}",
                0,
                2 ** (k + 1),
                k + 1,
            )
            gym.set_actor_scale(env, h, object_scaling_factor[k])
            object_handles.append(h)

        # Settle + refresh mesh offsets from simulator states.
        for step in range(120):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            if viewer is not None:
                gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        for i_obj, handle in enumerate(object_handles):
            states = gym.get_actor_rigid_body_states(env, handle, gymapi.STATE_POS)
            translation = np.array(
                [
                    states["pose"]["p"]["x"].item(),
                    states["pose"]["p"]["y"].item(),
                    states["pose"]["p"]["z"].item(),
                ]
            )
            object_reader_tracker[i_obj].set_offset(translation)

        # Optional tiny metadata h5 for object identity and positions.
        display_names = [
            OBJECT_NAMES.get(object_asset_files[idx].split("/")[-2], object_asset_files[idx].split("/")[-2])
            for idx in target_file_idx
        ]
        meta_h5 = os.path.join(run_dir, f"ntfield_meta_{timestamp}.h5")
        with h5py.File(meta_h5, "w") as f:
            str_dt = h5py.special_dtype(vlen=str)
            d = f.create_dataset("object_names", (NUM_OF_OBJECTS,), dtype=str_dt)
            d[:] = np.array(display_names, dtype=object)
            f.create_dataset("object_xy", data=np.asarray(gt_obj_xy, dtype=np.float32))

        obj_path, dim_path, urdf_path = _export_scene_bundle(
            output_dir=run_dir,
            table_pose=table_pose,
            table_dims=table_dims,
            object_reader_tracker=object_reader_tracker,
        )
        _build_ntfield_arrays_from_scene(
            output_dir=run_dir,
            num_samples=int(args.num_samples),
            near_min=float(args.near_min),
            near_max=float(args.near_max),
        )

        print(f"[Episode {ep + 1}] Saved NTField dataset at: {run_dir}")
        print(f"  scene: {obj_path}, {dim_path}, {urdf_path}")
        successful += 1

    print(f"Completed {successful}/{max(1, int(args.num_episodes))} episodes.")
    os._exit(0 if successful == max(1, int(args.num_episodes)) else 1)
