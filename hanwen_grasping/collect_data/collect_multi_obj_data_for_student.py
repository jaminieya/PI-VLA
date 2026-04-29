#
# collect_multi_obj_data_for_student.py
# Same pipeline as collect_data_for_student.py but places 3 YCB objects on the table.
# Saves start_image, object_names, object_locations, object_id_folders, goal_joint_configs (3x6).
# Rows align across object_names, object_locations, and goal_joint_configs.
#
# Run (from hanwen_grasping): python collect_data/collect_multi_obj_data_for_student.py
# Output: PI-VLA/output/multi_obj/grasp_multi3_demo_*.h5
# Retries: --max_plan_attempts (default 60) re-randomizes poses until all 3 grasps plan.
# Exits with os._exit() after HDF5 flush to avoid Isaac Gym destroy_sim segfaults on Linux.
#
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import math
import time
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from PIL import Image
import numpy as np
from trac_ik_python.trac_ik import IK
import sys
import os
import open3d as o3d
import fcl
import cv2
import copy
import shutil

import h5py

OBJECT_NAMES = {
    "002_master_chef_can": "master chef can",
    "004_sugar_box": "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle": "mustard bottle",
    "036_wood_block": "wood block",
    "011_banana": "banana",
}


def get_object_display_name(urdf_path):
    folder = urdf_path.split("/")[-2] if "/" in urdf_path else urdf_path
    return OBJECT_NAMES.get(folder, folder.replace("_", " "))

_script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.abspath(os.path.join(_script_dir, ".."))


def _resolve_assets_dir():
    """URDFs/meshes: prefer hanwen_grasping/assets, then PI-VLA/assets, then VLM-NT/starter_code/assets."""
    marker = os.path.join("urdf", "ycb", "object_urdf_grasp.txt")
    candidates = [
        os.path.join(package_root, "assets"),
        os.path.join(os.path.dirname(package_root), "assets"),
        os.path.join(
            os.path.dirname(os.path.dirname(package_root)), "starter_code", "assets"
        ),
    ]
    for c in candidates:
        root = os.path.abspath(c)
        if os.path.isfile(os.path.join(root, marker)):
            return root
    return os.path.abspath(os.path.join(package_root, "assets"))


ASSETS_DIR = _resolve_assets_dir()
SAVED_RESULT_DIR = os.path.join(_script_dir, "saved_as_result")
PI_VLA_ROOT = os.path.dirname(package_root)

util_dir = os.path.join(package_root, "util")
grasp_util_dir = os.path.join(package_root, "grasp_util")
sys.path.insert(0, package_root)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)

import robot_arm_configuration as RC

try:
    import ompl.base as ob
    import ompl.util as ou
    import ompl.geometric as og
except ImportError:
    from os.path import abspath, dirname, join
    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

from stl_reader import stl_reader
from obj_reader import obj_reader

#define parameters
#*************************************************************************************************#
#global settings
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

#env settings
TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
DRAWER_HEIGHT = 0.40
max_drawer_height = DRAWER_HEIGHT
min_drawer_height = DRAWER_HEIGHT
table_dims = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)

piece_width = 0.03
max_scaling_factor = 0
fall_height = table_dims.z
ADD_COVER = False

# Banana (5), sugar box (1), mustard bottle (3) - object_urdf_grasp indices
TARGET_OBJ_INDEX = [1, 3, 5]
MIN_RADIUS = 0.03471716871486391

# Three distinct objects from TARGET_OBJ_INDEX (sugar box, mustard, banana).
NUM_OF_OBJECTS = 3

#*************************************************************************************************#

#helper functions
#*************************************************************************************************#
def write_to_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    img = Image.fromarray(new_image, 'RGB')
    img.save(image_name)
    return new_image

def convert_rgb_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw//4
    new_image = np.zeros((x_dim, y_dim, 3), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            offset = j*4
            for k in range(3):
                new_image[i][j][k] = raw_image[i][offset+k]
    return new_image

def write_to_seg_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    img = Image.fromarray(new_image)
    img.save(image_name)

def convert_seg_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    x_dim = x_dim_raw
    y_dim = y_dim_raw
    new_image = np.zeros((x_dim, y_dim), dtype = np.uint8)
    for i in range(x_dim):
        for j in range(y_dim):
            new_image[i][j] = raw_image[i][j]
    return new_image

def write_to_depth_image(raw_image, image_name):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1))
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] != mini:
                new_image[i][j][0] = - int(raw_image[i][j]*1000)
            else:
                new_image[i][j][0] = 65535

    cv2.imwrite(image_name, new_image.astype(np.uint16))
    return new_image[:,:,0]

def convert_depth_image(raw_image):
    x_dim_raw, y_dim_raw = raw_image.shape
    maxi = -sys.maxsize
    mini = sys.maxsize
    for i in range(x_dim_raw):
        for j in range(y_dim_raw):
            maxi = max(maxi, raw_image[i][j])
            mini = min(mini, raw_image[i][j])
    x_dim, y_dim = x_dim_raw, y_dim_raw
    new_image = np.zeros((x_dim, y_dim, 1), dtype = np.uint16)
    for i in range(x_dim):
        for j in range(y_dim):
            if raw_image[i][j] != mini:
                new_image[i][j][0] = -(raw_image[i][j]*1000).astype(int)
            else:
                new_image[i][j][0] = 65535
    return new_image

def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)

def get_swept_volume_size(main_swept):
    min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize
    max_x, max_y, max_z = -sys.maxsize, -sys.maxsize, -sys.maxsize
    for tx, ty, tz in main_swept:
        min_x = min(min_x, tx)
        min_y = min(min_y, ty)
        min_z = min(min_z, tz)
        max_x = max(max_x, tx)
        max_y = max(max_y, ty)
        max_z = max(max_z, tz)
    return max_y - min_y

def interpolate_path(path, steps_between=4):
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

def resample_path(path, num_waypoints=20):
    if not path or len(path) < 2:
        return path
    if len(path) == num_waypoints:
        return path
    path_arr = np.array(path, dtype=np.float64)
    indices = np.linspace(0, len(path_arr) - 1, num_waypoints, dtype=np.float64)
    resampled = []
    for i in indices:
        idx_lo, idx_hi = int(np.floor(i)), min(int(np.ceil(i)), len(path_arr) - 1)
        t = i - idx_lo
        if idx_lo == idx_hi:
            pt = path_arr[idx_lo]
        else:
            pt = (1 - t) * path_arr[idx_lo] + t * path_arr[idx_hi]
        resampled.append(pt.tolist())
    return resampled


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


def _export_train_arm_scene_dataset(output_dir, table_pose, table_dims, object_reader_tracker):
    os.makedirs(output_dir, exist_ok=True)

    vertices_all = []
    faces_all = []
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

    if not vertices_all:
        raise RuntimeError("No geometry available for scene export.")

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


def plan_init2grasp_path_for_slot(
    rac,
    plane_obj,
    scene_info,
    object_mesh,
    static_for_planning,
    grasp_data,
    GT_OBJ_POS_LIST,
    slot_idx,
    max_consecutive_failures=15,
):
    """Plan approach path for object at ``slot_idx``; return metadata dict or None."""
    num_grasp = 0
    swept_size = sys.maxsize
    grasp_list = np.arange(len(grasp_data))
    np.random.shuffle(grasp_list)
    init2grasp_path = None
    selected_grasp_idx = None
    selected_target_grasp_pos = None
    selected_target_grasp_quat = None
    consecutive_path_failures = 0
    for grasp_idx in grasp_list:
        if consecutive_path_failures >= max_consecutive_failures:
            print(
                f"Slot {slot_idx}: stuck after {max_consecutive_failures} consecutive path failures."
            )
            break
        target_grasp_pos = grasp_data[grasp_idx]["target_pos"].copy()
        target_grasp_quat = grasp_data[grasp_idx]["target_quat"]
        target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[slot_idx][:2]
        init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
        grasp2init_angels_temp = rac.grasp_verify(
            target_grasp_pos + [0, 0, 0.01], target_grasp_quat
        )
        if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
            continue

        init2grasp_collision = rac.arm_collision_free(
            init2grasp_angels_temp, plane_obj, static_for_planning, []
        )
        grasp2init_collision = rac.arm_collision_free(
            grasp2init_angels_temp, plane_obj, static_for_planning, []
        )
        if not init2grasp_collision or not grasp2init_collision:
            continue

        init2grasp_path_temp = RC.get_path2grasp(
            rac,
            init2grasp_angels_temp,
            scene_info,
            target_mesh=object_mesh[slot_idx],
            time_limit=30,
            given_static_model=static_for_planning,
        )
        if init2grasp_path_temp is None:
            consecutive_path_failures += 1
            continue

        temp_mod_bbox = rac.modify_grasp_bbox(
            init2grasp_angels_temp,
            target_mesh=object_mesh[slot_idx],
            visualize=False,
        )
        grasp2init_path_temp = RC.get_path2start(
            rac,
            grasp2init_angels_temp,
            temp_mod_bbox,
            scene_info,
            time_limit=30,
            given_static_model=static_for_planning,
        )
        if grasp2init_path_temp is None:
            consecutive_path_failures += 1
            continue

        consecutive_path_failures = 0
        swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(
            init2grasp_path_temp,
            frame_rate=60,
            scene_info=scene_info,
            animation=False,
            static_vi=False,
        )
        swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(
            grasp2init_path_temp,
            w_target=temp_mod_bbox,
            frame_rate=60,
            scene_info=scene_info,
            animation=False,
            static_vi=False,
        )
        num_grasp += 1
        swept_center_temp, swept_verts_temp = rac.get_swept_center(
            swept_verts1_temp + swept_verts2_temp, scene_info, 0.6
        )
        temp_swept_size = get_swept_volume_size(swept_verts_temp)
        if temp_swept_size < swept_size:
            swept_size = temp_swept_size
            init2grasp_path = init2grasp_path_temp
            selected_grasp_idx = int(grasp_idx)
            selected_target_grasp_pos = np.array(target_grasp_pos, dtype=np.float32)
            selected_target_grasp_quat = np.array(target_grasp_quat, dtype=np.float32)
        if num_grasp == 1:
            break

    if init2grasp_path is None:
        return None
    if len(init2grasp_path) > 1:
        init2grasp_path = interpolate_path(init2grasp_path, steps_between=2)
        init2grasp_path = resample_path(init2grasp_path, num_waypoints=10)
    return {
        "path": init2grasp_path,
        "grasp_idx": selected_grasp_idx,
        "target_grasp_pos": selected_target_grasp_pos,
        "target_grasp_quat": selected_target_grasp_quat,
    }


#*************************************************************************************************#

if __name__ == '__main__':
    gym = gymapi.acquire_gym()

    args = gymutil.parse_arguments(
        description="Multi-object student grasp data (3 YCB objects)",
        custom_parameters=[
            {"name": "--env_id", "type": int, "help": "env_id", "default": 0},
            {"name": "--num_episodes", "type": int, "default": 1, "help": "Number of episodes to collect in one invocation"},
            {
                "name": "--max_plan_attempts",
                "type": int,
                "default": 10,
                "help": "Max random layouts per episode until all 3 grasps plan (default: 60)",
            },
            {"name": "--headless", "action": "store_true", "help": "Run without creating a viewer"},
            {
                "name": "--output_dir",
                "type": str,
                "default": None,
                "help": "Output root for timestamped runs (default: PI-VLA/output/multi_obj).",
            },
            {
                "name": "--export_scene_bundle",
                "action": "store_true",
                "help": "Also export realpc.obj, dim, and ur5e.urdf per sample.",
            },
        ],
    )
    env_id = int(args.env_id)

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
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)

    asset_root = ASSETS_DIR + os.sep
    ur5e_asset_file = "urdf/ur5e/ur5e_mimic_real_gripper_test.urdf"
    ur5e_collision_parts = ["urdf/ur5e/meshes/collision/base.stl",
                            "urdf/ur5e/meshes/collision/shoulder.stl",
                            "urdf/ur5e/meshes/collision/upperarm.stl",
                            "urdf/ur5e/meshes/collision/forearm.stl",
                            "urdf/ur5e/meshes/collision/wrist1.stl",
                            "urdf/ur5e/meshes/collision/wrist2.stl",
                            "urdf/ur5e/meshes/collision/wrist3.stl"]

    object_asset_files = []
    object_collision_files = []
    object_offset = []
    object_common_prefix = "urdf/ycb/"
    with open(asset_root + "urdf/ycb/object_urdf_grasp.txt") as f:
        for line in f:
            object_asset_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_collision_grasp.txt") as f:
        for line in f:
            object_collision_files.append(object_common_prefix + line[:-1])
    with open(asset_root + "urdf/ycb/object_offset_grasp.txt") as f:
        for line in f:
            div = line[:-1].split(" ")
            object_offset.append([float(x) for x in div])

    ur5e_collision_models = []
    ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                    R.from_euler('xy', [90, 180], degrees = True),
                    R.from_euler('xy', [180, 180], degrees = True),
                    R.from_euler('z',  [-180], degrees = True),
                    R.from_euler('x',  [-180], degrees = True),
                    R.from_euler('x',  [90], degrees = True),
                    R.from_euler('z',  [-90], degrees = True)]
    ur5e_translations = [[0, 0, 0], [0, 0, 0], [0, -0.138, 0], [0, -0.007, 0], [0, 0.127, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(ur5e_collision_parts)):
        parts_path = ur5e_collision_parts[i]
        collision_mesh = stl_reader(asset_root + parts_path)
        m = fcl.BVHModel()
        collision_mesh.transform(ur5e_rotations[i], ur5e_translations[i])
        verts, tris = collision_mesh.get_vertices(), collision_mesh.get_faces()
        m.beginModel(len(verts), len(tris))
        m.addSubModel(verts, tris)
        m.endModel()
        ur5e_collision_models.append(m)

    object_collision_lib = []

    urdf_str = ''
    with open(os.path.join(ASSETS_DIR, "urdf", "ur5e", "ur5e_mimic_real_gripper_test.urdf")) as f:
        urdf_str = f.read()

    viewer = None
    if not getattr(args, "headless", False):
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            print("*** Failed to create viewer; switching to headless mode.")

    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

    drawer_height = np.random.random()*(max_drawer_height - min_drawer_height) + min_drawer_height

    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options)

    os.makedirs(SAVED_RESULT_DIR, exist_ok=True)
    saved_env_name = os.path.join(SAVED_RESULT_DIR, "env_" + str(env_id) + "_scene_info.npy")
    np.save(saved_env_name, np.array([table_dims.x, table_dims.y, table_dims.z, drawer_height]))

    asset_options.fix_base_link = False
    object_assets = []
    for ob in object_asset_files:
        object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
    asset_options.fix_base_link = True

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + drawer_height + 0.015)

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    # Valid tabletop sampling region in world frame (keep margins from edges).
    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20

    plane_normal = np.array([0.0, 0.0, 1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)

    object_collision_models = [table_obj]

    if ADD_COVER:
        col_upper_cover = fcl.Box(upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z)
        trans_upper_cover = fcl.Transform(np.array([table_pose.p.x, 0.0, table_dims.z + drawer_height + 0.015]))
        upper_cover_obj = fcl.CollisionObject(col_upper_cover, trans_upper_cover)
        object_collision_models.append(upper_cover_obj)

    envs = []
    ur5e_handles = []
    body_cam_handles = []
    object_status_list = []
    object_reader_tracker = []
    
    for i in range(num_of_envs):
        envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        if ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)

        if len(TARGET_OBJ_INDEX) < NUM_OF_OBJECTS:
            raise RuntimeError(
                f"Need at least {NUM_OF_OBJECTS} entries in TARGET_OBJ_INDEX, got {len(TARGET_OBJ_INDEX)}"
            )
        target_file_idx = np.random.choice(
            TARGET_OBJ_INDEX, NUM_OF_OBJECTS, replace=False
        )
            
        object_handles = []
        object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = NUM_OF_OBJECTS)/10.0 + 1.0

        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        GT_TARGET_POS = [np.random.uniform(max(table_x_min, 0.20 + table_dims.x/2), table_x_max),
                         np.random.uniform(table_y_min, table_y_max),
                         table_dims.z + 0.08]

        for k in range(NUM_OF_OBJECTS):
            object_pose = gymapi.Transform()
            is_collision = True

            while is_collision:
                tx = np.random.uniform(table_x_min, table_x_max)
                ty = np.random.uniform(table_y_min, table_y_max)
                tz = table_dims.z + 0.08

                object_pose.p = gymapi.Vec3(tx, ty, tz)

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
                t = fcl.Transform(np.array([tx,ty,tz]))
                
                req = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request = req)
                objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)

                is_collision = rdata.result.is_collision

                if not is_collision:
                    dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                    if dist <= 0.2:
                        is_collision = True
                        continue

                    for obj in GT_OBJ_POS_LIST:
                        dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                        if dist <= 0.16:
                            is_collision = True
                            continue

            GT_OBJ_POS_LIST.append([object_pose.p.x, object_pose.p.y])

            object_handles.append(gym.create_actor(envs[-1], 
                                                object_assets[target_file_idx[k]], 
                                                object_pose, 
                                                "object" + str(k) + str(i), 0, 2**(k+1), k+1))
            gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
            object_reader_tracker.append(collision_mesh)
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_objs.append(fcl.CollisionObject(m, t))
            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        viewpoint_candidate = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(body_cam_handles[-1], envs[-1], viewpoint_candidate, camera_focus)

        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        table_center_x = table_pose.p.x
        table_center_y = table_pose.p.y
        top_cam_pos = gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2)
        top_cam_target = gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z)
        gym.set_camera_location(top_cam_handle, envs[-1], top_cam_pos, top_cam_target)

        side_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        side_cam_pos = gymapi.Vec3(2.2, 0, 0.5)
        side_cam_target = gymapi.Vec3(0, 0, 0.5)
        gym.set_camera_location(side_cam_handle, envs[-1], side_cam_pos, side_cam_target)

    cam_pos = gymapi.Vec3(2.2, 0, 0.5)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(1.0, 0.0, 0.0))

    real_position = False
    real_position = False
    for t in range(100):
        if not real_position:
            gym.set_dof_target_position(envs[-1], spj, 0)
            gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
            gym.set_dof_target_position(envs[-1], ej,  0)
            gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
            gym.set_dof_target_position(envs[-1], wj2, 0)
            gym.set_dof_target_position(envs[-1], wj3, 0)
            real_position = True
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

    # Save original relative centers for re-randomization updates
    original_centers = [status[0].copy() for status in object_status_list]

    ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
    target_quat = gymapi.Quat(0.446, 0.560, -0.433, 0.549)
    converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
    file_path = os.path.join(ASSETS_DIR, "urdf", "ur5e", "meshes", "collision") + os.sep

    scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
    rac = RC.robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)
    seed_state = [0.0]*ik_solver2.number_of_joints

    num_episodes = max(1, int(getattr(args, "num_episodes", 1)))
    print(f"Multi-object collect: {NUM_OF_OBJECTS} YCB objects; saving all goal_joint_configs.")

    grasp_data_by_slot = []
    for k in range(NUM_OF_OBJECTS):
        grasp_file_k = os.path.join(
            ASSETS_DIR,
            *object_asset_files[target_file_idx[k]].split("/")[:-1],
            "grasp_dict.npy",
        )
        grasp_data_by_slot.append(np.load(grasp_file_k, allow_pickle=True))

    successful_episodes = 0
    max_plan_attempts = max(1, int(getattr(args, "max_plan_attempts", 10)))
    for episode_idx in range(num_episodes):
        print(f"\n================ Episode {episode_idx + 1}/{num_episodes} ================\n")
        episode_saved = False
        attempt = 0
        while attempt < max_plan_attempts and not episode_saved:
            attempt += 1
            if attempt > 1:
                print(
                    f"  ... planning retry {attempt}/{max_plan_attempts} (new object poses)\n"
                )

            # --- 1. Reset the robot arm ---
            gym.set_dof_target_position(envs[-1], spj, 0)
            gym.set_dof_target_position(envs[-1], slj, -math.pi/2)
            gym.set_dof_target_position(envs[-1], ej,  0)
            gym.set_dof_target_position(envs[-1], wj1, -math.pi/2)
            gym.set_dof_target_position(envs[-1], wj2, 0)
            gym.set_dof_target_position(envs[-1], wj3, 0)

            # --- 2. Randomize object locations and update FCL ---
            objs_manager.clear()
            obstacle_objs = []
            GT_OBJ_POS_LIST = []
            GT_TARGET_POS = [np.random.uniform(max(table_x_min, 0.20 + table_dims.x/2), table_x_max),
                             np.random.uniform(table_y_min, table_y_max),
                             table_dims.z + 0.08]

            for k in range(NUM_OF_OBJECTS):
                is_collision = True
                while is_collision:
                    tx = np.random.uniform(table_x_min, table_x_max)
                    ty = np.random.uniform(table_y_min, table_y_max)
                    tz = table_dims.z + 0.08

                    m = object_collision_lib[k]
                    t = fcl.Transform(np.array([tx, ty, tz]))
                    temp_obj = fcl.CollisionObject(m, t)

                    is_collision = False
                    req = fcl.CollisionRequest()
                    result = fcl.CollisionResult()
                    for placed_obj in obstacle_objs:
                        if fcl.collide(temp_obj, placed_obj, req, result):
                            is_collision = True
                            break

                    if not is_collision:
                        dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                        if dist <= 0.2:
                            is_collision = True
                            continue

                        for obj in GT_OBJ_POS_LIST:
                            dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                            if dist <= 0.16:
                                is_collision = True
                                break

                GT_OBJ_POS_LIST.append([tx, ty])
                obstacle_objs.append(temp_obj)

                handle = object_handles[k]
                states = gym.get_actor_rigid_body_states(envs[-1], handle, gymapi.STATE_POS)
                states['pose']['p']['x'] = tx
                states['pose']['p']['y'] = ty
                states['pose']['p']['z'] = tz
                states['pose']['r']['x'] = 0.0
                states['pose']['r']['y'] = 0.0
                states['pose']['r']['z'] = 0.0
                states['pose']['r']['w'] = 1.0

                gym.set_actor_rigid_body_states(envs[-1], handle, states, gymapi.STATE_POS)

            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

            # --- 3. Settle simulation and lock in meshes ---
            flex_collision_models = []
            object_mesh = []
            for settle_step in range(100):
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                if viewer is not None:
                    gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

                if settle_step == 99:
                    for i_obj, element in enumerate(object_handles):
                        states = gym.get_actor_rigid_body_states(envs[-1], element, gymapi.STATE_POS)
                        px = states['pose']['p']['x'].item()
                        py = states['pose']['p']['y'].item()
                        pz = states['pose']['p']['z'].item()
                        rx = states['pose']['r']['x'].item()
                        ry = states['pose']['r']['y'].item()
                        rz = states['pose']['r']['z'].item()
                        rw = states['pose']['r']['w'].item()

                        translation = np.array([px, py, pz])
                        rotation = np.array([rx, ry, rz, rw])

                        object_status_list[i_obj][0] = original_centers[i_obj] + translation

                        r1 = R.from_quat(rotation)
                        tf = fcl.Transform(r1.as_matrix(), translation)
                        flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i_obj], tf), 0])

                        temp_obj = object_reader_tracker[i_obj]
                        temp_obj.set_offset(translation)
                        vertices, faces = temp_obj.get_bounding_box_mesh()
                        object_mesh.append([vertices, faces])

            plan_results = []
            for k in range(NUM_OF_OBJECTS):
                distractor_cols = [
                    flex_collision_models[j][0]
                    for j in range(NUM_OF_OBJECTS)
                    if j != k
                ]
                static_k = object_collision_models + distractor_cols
                result_k = plan_init2grasp_path_for_slot(
                    rac,
                    plane_obj,
                    scene_info,
                    object_mesh,
                    static_k,
                    grasp_data_by_slot[k],
                    GT_OBJ_POS_LIST,
                    k,
                )
                plan_results.append(result_k)
                if result_k is not None:
                    print(f"Slot {k}: path with {len(result_k['path'])} waypoints")

            if any(p is None for p in plan_results):
                failed = [k for k, p in enumerate(plan_results) if p is None]
                print(f"No valid grasp for slot(s) {failed}; retrying with new poses.")
                continue

            init2grasp_paths = [plan_results[k]["path"] for k in range(NUM_OF_OBJECTS)]

            JOINT_DIM = 6
            START_SETTLE_STEPS = 30
            HOME_DOF = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
            TRAJ_STEPS_PER_WAYPOINT = 1
            display_names = [
                get_object_display_name(object_asset_files[target_file_idx[k]])
                for k in range(NUM_OF_OBJECTS)
            ]
            id_folders = [
                object_asset_files[target_file_idx[k]].split("/")[-2]
                for k in range(NUM_OF_OBJECTS)
            ]
            object_locations = np.stack(
                [
                    np.array(object_status_list[k][0], dtype=np.float32)
                    for k in range(NUM_OF_OBJECTS)
                ]
            )
            goal_joint_configs = np.stack(
                [
                    np.array(init2grasp_paths[k][-1][:JOINT_DIM], dtype=np.float32)
                    for k in range(NUM_OF_OBJECTS)
                ]
            )
            grasp_indices = np.array(
                [int(plan_results[k]["grasp_idx"]) for k in range(NUM_OF_OBJECTS)],
                dtype=np.int32,
            )
            target_grasp_positions = np.stack(
                [
                    np.array(plan_results[k]["target_grasp_pos"], dtype=np.float32)
                    for k in range(NUM_OF_OBJECTS)
                ]
            )
            target_grasp_quaternions = np.stack(
                [
                    np.array(plan_results[k]["target_grasp_quat"], dtype=np.float32)
                    for k in range(NUM_OF_OBJECTS)
                ]
            )

            start_dof = HOME_DOF
            for _ in range(START_SETTLE_STEPS):
                gym.set_dof_target_position(envs[-1], spj, start_dof[0])
                gym.set_dof_target_position(envs[-1], slj, start_dof[1])
                gym.set_dof_target_position(envs[-1], ej,  start_dof[2])
                gym.set_dof_target_position(envs[-1], wj1, start_dof[3])
                gym.set_dof_target_position(envs[-1], wj2, start_dof[4])
                gym.set_dof_target_position(envs[-1], wj3, start_dof[5])

                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                if viewer is not None:
                    gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            gym.render_all_camera_sensors(sim)
            raw_top = gym.get_camera_image(sim, envs[-1], top_cam_handle, gymapi.IMAGE_COLOR)
            rgba_top = raw_top.reshape(camera_props.height, camera_props.width, 4)
            start_image = rgba_top[..., :3].copy()
            raw_side = gym.get_camera_image(sim, envs[-1], side_cam_handle, gymapi.IMAGE_COLOR)
            rgba_side = raw_side.reshape(camera_props.height, camera_props.width, 4)
            start_image_side = rgba_side[..., :3].copy()

            # Replay each planned path and capture trajectory joints + top/side RGB.
            trajectory_joint_configs = []
            trajectory_images_top = []
            trajectory_images_side = []
            for k in range(NUM_OF_OBJECTS):
                traj_q = []
                traj_top = []
                traj_side = []
                for waypoint in init2grasp_paths[k]:
                    for _ in range(TRAJ_STEPS_PER_WAYPOINT):
                        gym.set_dof_target_position(envs[-1], spj, waypoint[0])
                        gym.set_dof_target_position(envs[-1], slj, waypoint[1])
                        gym.set_dof_target_position(envs[-1], ej, waypoint[2])
                        gym.set_dof_target_position(envs[-1], wj1, waypoint[3])
                        gym.set_dof_target_position(envs[-1], wj2, waypoint[4])
                        gym.set_dof_target_position(envs[-1], wj3, waypoint[5])

                        gym.simulate(sim)
                        gym.fetch_results(sim, True)
                        gym.step_graphics(sim)
                        if viewer is not None:
                            gym.draw_viewer(viewer, sim, True)
                        gym.sync_frame_time(sim)

                        gym.render_all_camera_sensors(sim)
                        raw_top_step = gym.get_camera_image(sim, envs[-1], top_cam_handle, gymapi.IMAGE_COLOR)
                        raw_side_step = gym.get_camera_image(sim, envs[-1], side_cam_handle, gymapi.IMAGE_COLOR)
                        top_step = raw_top_step.reshape(camera_props.height, camera_props.width, 4)[..., :3].copy()
                        side_step = raw_side_step.reshape(camera_props.height, camera_props.width, 4)[..., :3].copy()
                        dof_states = gym.get_actor_dof_states(envs[-1], ur5e_handles[-1], gymapi.STATE_POS)
                        q_step = np.array(dof_states["pos"][:JOINT_DIM], dtype=np.float32)

                        traj_q.append(q_step)
                        traj_top.append(top_step)
                        traj_side.append(side_step)

                trajectory_joint_configs.append(np.stack(traj_q, axis=0))
                trajectory_images_top.append(np.stack(traj_top, axis=0))
                trajectory_images_side.append(np.stack(traj_side, axis=0))

            trajectory_joint_configs = np.stack(trajectory_joint_configs, axis=0)  # (O, T, 6)
            trajectory_images_top = np.stack(trajectory_images_top, axis=0)        # (O, T, H, W, 3)
            trajectory_images_side = np.stack(trajectory_images_side, axis=0)      # (O, T, H, W, 3)

            default_output_dir = os.path.join(PI_VLA_ROOT, "output", "multi_obj")
            if args.output_dir:
                output_root = (
                    args.output_dir
                    if os.path.isabs(args.output_dir)
                    else os.path.join(PI_VLA_ROOT, args.output_dir)
                )
                output_root = os.path.abspath(output_root)
            else:
                output_root = default_output_dir
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            run_dir = os.path.join(output_root, timestamp)
            os.makedirs(run_dir, exist_ok=True)
            out_path = os.path.join(run_dir, f"grasp_multi3_demo_{timestamp}.h5")
            str_dt = h5py.special_dtype(vlen=str)
            with h5py.File(out_path, "w") as f:
                f.create_dataset("start_image", data=start_image.astype(np.uint8), compression="gzip")
                f.create_dataset("start_image_side", data=start_image_side.astype(np.uint8), compression="gzip")
                f.create_dataset("object_locations", data=object_locations)
                name_ds = f.create_dataset("object_names", (NUM_OF_OBJECTS,), dtype=str_dt)
                name_ds[:] = np.array(display_names, dtype=object)
                folder_ds = f.create_dataset("object_id_folders", (NUM_OF_OBJECTS,), dtype=str_dt)
                folder_ds[:] = np.array(id_folders, dtype=object)
                f.create_dataset("goal_joint_configs", data=goal_joint_configs)
                f.create_dataset("trajectory_joint_configs", data=trajectory_joint_configs)
                f.create_dataset("trajectory_images_top", data=trajectory_images_top.astype(np.uint8), compression="gzip")
                f.create_dataset("trajectory_images_side", data=trajectory_images_side.astype(np.uint8), compression="gzip")
                f.create_dataset("grasp_indices", data=grasp_indices)
                f.create_dataset("grasp_target_positions", data=target_grasp_positions)
                f.create_dataset("grasp_target_quaternions", data=target_grasp_quaternions)
                f.attrs["joint_dim"] = int(JOINT_DIM)
                f.attrs["num_objects"] = int(NUM_OF_OBJECTS)
                f.attrs["trajectory_steps_per_waypoint"] = int(TRAJ_STEPS_PER_WAYPOINT)
                f.flush()

            if args.export_scene_bundle:
                obj_path, dim_path, urdf_path = _export_train_arm_scene_dataset(
                    output_dir=run_dir,
                    table_pose=table_pose,
                    table_dims=table_dims,
                    object_reader_tracker=object_reader_tracker,
                )
            successful_episodes += 1
            episode_saved = True
            print(f"Saved 1 sample to {out_path}")
            if args.export_scene_bundle:
                print(f"Exported scene bundle: {obj_path}, {dim_path}, {urdf_path}")

        if not episode_saved:
            print(
                f"Episode {episode_idx + 1}: no sample saved after {max_plan_attempts} layout/plan attempts."
            )

    print(f"Collected {successful_episodes}/{num_episodes} episodes in one simulator setup.")
    print("Test Completed Successfully!!")

    # Avoid Isaac Gym native teardown segfault on many Linux/GPU setups: exit the
    # process immediately; the OS reclaims GPU resources for this PID.
    exit_code = 0 if successful_episodes == num_episodes else 1
    os._exit(exit_code)
