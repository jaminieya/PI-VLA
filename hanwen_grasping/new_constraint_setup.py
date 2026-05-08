#
# File:          run_retrieval.py
# Brief:         Comparison test on active sensing methods and rearrangement methods
# Author:        Junyoung Kim -- kim3722@purdue.edu, Hanwen Ren -- ren221@purdue.edu
# Date:          2024-05-04
# Last Modified: 2025-05-17
#

from scipy.spatial.transform import Rotation as R
import math
import re
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
from datetime import datetime

import robot_arm_configuration as RC

#import MCTS_algo_ICRA as mct
#from rearrangement_planning_util_ICRA import write_result


file_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(file_dir, './util')
grasp_util_dir = os.path.join(file_dir, './grasp_util')
pi_vla_root = os.path.dirname(file_dir)
ntrl_demo_path = os.path.abspath(os.path.join(pi_vla_root, 'ntrl-demo'))
if os.path.isdir(ntrl_demo_path) and ntrl_demo_path not in sys.path:
    sys.path.insert(0, ntrl_demo_path)
sys.path.append(util_dir)
#sys.path.append(scorenet_dir)
sys.path.append(grasp_util_dir)

try:
    import ompl.base as ob
    import ompl.util as ou
    import ompl.geometric as og
except ImportError:
    # if the ompl module is not in the PYTHONPATH assume it is installed in a
    # subdirectory of the parent directory called "py-bindings."
    from os.path import abspath, dirname, join
    sys.path.insert(
        0, join(dirname(dirname(dirname(abspath(__file__)))), 'py-bindings'))
    from ompl import util as ou
    from ompl import base as ob
    from ompl import geometric as og

from stl_reader import stl_reader
from obj_reader import obj_reader
#from global_scene import global_scene
#from grasp_util.global_scene import global_scene
#from grasp_util.pc_extractor_grasp import pc_extractor_grasp

#from test_module.camera_view import camera
#from runner import feed_forward

#define parameters
#*************************************************************************************************#
#global settings
num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

#env settings (fixed to match constraint_env/collect_constraint_data.py)
max_drawer_height = 0.40
min_drawer_height = 0.40
MIN_NUM_OBSTACLES = 5
MAX_NUM_OBSTACLES = 8
table_dims = gymapi.Vec3(0.8, 1.0, 0.10)

piece_width = 0.03
max_scaling_factor = 0
fall_height = table_dims.z
ADD_COVER = True

TARGET_OBJ_INDEX = [1, 3, 5]
MIN_RADIUS = 0.03471716871486391

NUM_OF_OBJECTS = np.random.randint(MIN_NUM_OBSTACLES + 1, MAX_NUM_OBSTACLES + 1)
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


def capture_camera_rgb_from_sensor(gym, sim, env, cam_handle, height, width):
    """
    Reliable RGB from Isaac Gym IMAGE_COLOR: step graphics, render sensors, then reshape (H,W,4)->RGB.
    """
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    raw = gym.get_camera_image(sim, env, cam_handle, gymapi.IMAGE_COLOR)
    arr = np.asarray(raw, dtype=np.uint8)
    expected = int(height) * int(width) * 4
    if arr.size == expected:
        rgba = arr.reshape((int(height), int(width), 4))
        return rgba[..., :3].copy()
    return convert_rgb_image(raw)


def save_rig_camera_outputs(
    gym,
    sim,
    env,
    output_root,
    env_id,
    camera_props,
    diag_left_handle,
    diag_right_handle,
    point_cloud_center_handle,
    diag_left_cam_pos,
    diag_left_cam_target,
    diag_right_cam_pos,
    diag_right_cam_target,
    point_cloud_center_cam_pos,
    point_cloud_center_cam_target,
    tag=None,
):
    """Save three rig cameras to env_*_scene_{diag_left,diag_right,center}_views.npz (+ matching .png)."""
    h, w = int(camera_props.height), int(camera_props.width)
    gym.set_camera_location(diag_left_handle, env, diag_left_cam_pos, diag_left_cam_target)
    gym.set_camera_location(diag_right_handle, env, diag_right_cam_pos, diag_right_cam_target)
    gym.set_camera_location(point_cloud_center_handle, env, point_cloud_center_cam_pos, point_cloud_center_cam_target)

    dl = capture_camera_rgb_from_sensor(gym, sim, env, diag_left_handle, h, w)
    dr = capture_camera_rgb_from_sensor(gym, sim, env, diag_right_handle, h, w)
    pc = capture_camera_rgb_from_sensor(gym, sim, env, point_cloud_center_handle, h, w)

    dlp = np.array([diag_left_cam_pos.x, diag_left_cam_pos.y, diag_left_cam_pos.z], dtype=np.float64)
    dlt = np.array([diag_left_cam_target.x, diag_left_cam_target.y, diag_left_cam_target.z], dtype=np.float64)
    drp = np.array([diag_right_cam_pos.x, diag_right_cam_pos.y, diag_right_cam_pos.z], dtype=np.float64)
    drt = np.array([diag_right_cam_target.x, diag_right_cam_target.y, diag_right_cam_target.z], dtype=np.float64)
    pcp = np.array(
        [point_cloud_center_cam_pos.x, point_cloud_center_cam_pos.y, point_cloud_center_cam_pos.z],
        dtype=np.float64,
    )
    pct = np.array(
        [point_cloud_center_cam_target.x, point_cloud_center_cam_target.y, point_cloud_center_cam_target.z],
        dtype=np.float64,
    )

    base = os.path.join(output_root, f"env_{env_id}_{tag}" if tag else f"env_{env_id}")
    # One bundle per rig camera: env_{id}_scene_{diag_left|diag_right|center}_views.{npz,png}
    np.savez(
        f"{base}_scene_diag_left_views.npz",
        rgb=dl,
        cam_pos=dlp,
        cam_target=dlt,
    )
    np.savez(
        f"{base}_scene_diag_right_views.npz",
        rgb=dr,
        cam_pos=drp,
        cam_target=drt,
    )
    np.savez(
        f"{base}_scene_center_views.npz",
        rgb=pc,
        cam_pos=pcp,
        cam_target=pct,
    )
    Image.fromarray(dl, "RGB").save(f"{base}_scene_diag_left_views.png")
    Image.fromarray(dr, "RGB").save(f"{base}_scene_diag_right_views.png")
    Image.fromarray(pc, "RGB").save(f"{base}_scene_center_views.png")
    print(
        f"Saved rig cameras to {base}_scene_diag_left_views.*, {base}_scene_diag_right_views.*, "
        f"{base}_scene_center_views.*",
        flush=True,
    )
    return dl, dr, pc, dlp, dlt, drp, drt, pcp, pct

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

def write_for_contact_grasp(color_image, seg_image, depth_image, k, cam_rot, cam_tran, name):
    arr = np.array({'rgb':color_image, 'depth':depth_image, 'K':k, 'seg': seg_image, 'cam_rot': cam_rot, 'cam_tran': cam_tran})
    np.save(name, arr)        

def global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0.w, quaternion0.x, quaternion0.y, quaternion0.z
    w1, x1, y1, z1 = quaternion1.w, quaternion1.x, quaternion1.y, quaternion1.z
    return gymapi.Quat(x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                       -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                       x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, 
                       -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)


def _viewer_forward_from_quat(quat):
    """Compute a world-space forward unit vector from Isaac Gym quaternion."""
    rot = R.from_quat([quat.x, quat.y, quat.z, quat.w])
    # Isaac viewer/camera look axis follows OpenGL camera convention: local -Z.
    return rot.apply(np.array([0.0, 0.0, -1.0], dtype=np.float64))


def viewer_eye_target_env_local(gym, viewer, env):
    """
    Same convention as set_camera_location / viewer logs: eye and look-at target in env-local coords.
    Target lies one unit along the viewer forward axis (sufficient for look_at direction).
    """
    cam_tf = gym.get_viewer_camera_transform(viewer, env)
    pos = cam_tf.p
    cam_dir = _viewer_forward_from_quat(cam_tf.r)
    tgt = gymapi.Vec3(
        float(pos.x + cam_dir[0]),
        float(pos.y + cam_dir[1]),
        float(pos.z + cam_dir[2]),
    )
    return pos, tgt


def _mutate_vec3(dst, src):
    dst.x = float(src.x)
    dst.y = float(src.y)
    dst.z = float(src.z)


def compute_rig_cameras_table_anchored(table_pose, table_dims):
    """
    Three fixed *relative* bearings from the tabletop center: each eye looks at a common
    target slightly above the table surface. Recomputes every call from table_pose so
    layout tracks the table actor (same idea as the global camera at (3,0,0.3)->(0,0,0),
    but tighter multi-view around the workspace).
    """
    ax = float(table_pose.p.x)
    ay = float(table_pose.p.y)
    az_top = float(table_pose.p.z + table_dims.z * 0.5)
    # Look-at point on / just above the table.
    tgt = gymapi.Vec3(ax, ay, az_top + 0.05)

    # Eyes sit in the -X half-space (robot / room side) with +/-Y for diagonals.
    diag_left_cam_pos = gymapi.Vec3(ax - 0.55, ay + 0.75, az_top + 0.40)
    diag_left_cam_target = gymapi.Vec3(tgt.x, tgt.y, tgt.z)

    diag_right_cam_pos = gymapi.Vec3(ax - 0.55, ay - 0.75, az_top + 0.40)
    diag_right_cam_target = gymapi.Vec3(tgt.x, tgt.y, tgt.z)

    point_cloud_center_cam_pos = gymapi.Vec3(ax - 1.05, ay, az_top + 0.35)
    point_cloud_center_cam_target = gymapi.Vec3(tgt.x, tgt.y, tgt.z)

    return (
        diag_left_cam_pos,
        diag_left_cam_target,
        diag_right_cam_pos,
        diag_right_cam_target,
        point_cloud_center_cam_pos,
        point_cloud_center_cam_target,
    )


def assign_viewer_pose_to_rig_slot(gym, viewer, env, cam_handle, cam_pos, cam_target, log_label, paste_var_prefix):
    """Copy current viewer eye/target into rig camera Vec3s (mutate in place) and refresh the sensor."""
    eye, tgt = viewer_eye_target_env_local(gym, viewer, env)
    _mutate_vec3(cam_pos, eye)
    _mutate_vec3(cam_target, tgt)
    gym.set_camera_location(cam_handle, env, cam_pos, cam_target)
    print(
        f"[RigAssign {log_label}] env-local eye=({cam_pos.x:.4f}, {cam_pos.y:.4f}, {cam_pos.z:.4f}) "
        f"target=({cam_target.x:.4f}, {cam_target.y:.4f}, {cam_target.z:.4f})",
        flush=True,
    )
    print(
        f"    Paste into new_constraint_setup.py:\n"
        f"        {paste_var_prefix}_pos = gymapi.Vec3({cam_pos.x:.4f}, {cam_pos.y:.4f}, {cam_pos.z:.4f})\n"
        f"        {paste_var_prefix}_target = gymapi.Vec3({cam_target.x:.4f}, {cam_target.y:.4f}, {cam_target.z:.4f})",
        flush=True,
    )


def maybe_log_viewer_camera_on_move(gym, viewer, env, cache, camera_props, pos_eps=1e-4, dir_eps=1e-3):
    """
    Print viewer camera FOV + pose whenever position or look direction changes.
    cache format: {"pos": np.array([x,y,z]), "dir": np.array([dx,dy,dz])}
    """
    if viewer is None:
        return

    cam_tf = gym.get_viewer_camera_transform(viewer, env)
    cam_pos = np.array([cam_tf.p.x, cam_tf.p.y, cam_tf.p.z], dtype=np.float64)
    cam_dir = _viewer_forward_from_quat(cam_tf.r)
    cam_target = cam_pos + cam_dir
    hfov_deg = float(camera_props.horizontal_fov)
    cw = int(camera_props.width)
    ch = int(camera_props.height)

    def _emit():
        print(
            f"[ViewerCamera env-local] hfov_deg={hfov_deg:.4f} width={cw} height={ch} "
            f"pos=({cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f}) "
            f"dir=({cam_dir[0]:.4f}, {cam_dir[1]:.4f}, {cam_dir[2]:.4f}) "
            f"target=({cam_target[0]:.4f}, {cam_target[1]:.4f}, {cam_target[2]:.4f})",
            flush=True,
        )

    if cache["pos"] is None:
        cache["pos"] = cam_pos.copy()
        cache["dir"] = cam_dir.copy()
        _emit()
        return

    moved = np.linalg.norm(cam_pos - cache["pos"]) > pos_eps
    rotated = np.linalg.norm(cam_dir - cache["dir"]) > dir_eps
    if moved or rotated:
        cache["pos"] = cam_pos.copy()
        cache["dir"] = cam_dir.copy()
        _emit()


def _normalize_vec3(v, fallback):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return fallback.copy()
    return v / n


def setup_viewer_camera_controls(
    gym, viewer, subscribe_rig_snapshot=False, subscribe_rig_assign_keys=False
):
    """Bind keyboard controls for manual real-time viewer camera control."""
    if viewer is None:
        return
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_I, "cam_fwd")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_K, "cam_back")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_J, "cam_left")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_L, "cam_right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_U, "cam_up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_O, "cam_down")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_F, "cam_yaw_left")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_H, "cam_yaw_right")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_T, "cam_pitch_up")
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_G, "cam_pitch_down")
    if subscribe_rig_snapshot:
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "rig_snapshot")
    if subscribe_rig_assign_keys:
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_1, "rig_assign_diag_left")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_2, "rig_assign_diag_right")
        gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_3, "rig_assign_point_cloud_center")


def handle_viewer_camera_input(
    gym,
    viewer,
    env,
    move_step=0.01,
    rot_step_deg=2.0,
    rig_snapshot_callback=None,
    rig_from_viewer_slots=None,
):
    """Apply keyboard camera control (viewer logs pose on every move separately).

    rig_from_viewer_slots: optional dict with keys
      'diag_left', 'diag_right', 'point_cloud_center' -> (cam_handle, cam_pos_vec3, cam_target_vec3)
    """
    if viewer is None:
        return

    changed = False
    cam_tf = gym.get_viewer_camera_transform(viewer, env)
    cam_pos = np.array([cam_tf.p.x, cam_tf.p.y, cam_tf.p.z], dtype=np.float64)
    forward = _viewer_forward_from_quat(cam_tf.r)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = _normalize_vec3(np.cross(forward, world_up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    up = _normalize_vec3(np.cross(right, forward), world_up)

    for evt in gym.query_viewer_action_events(viewer):
        if evt.value <= 0:
            continue
        if evt.action == "rig_snapshot" and rig_snapshot_callback is not None:
            rig_snapshot_callback()
            continue
        if rig_from_viewer_slots and evt.action == "rig_assign_diag_left":
            h, p, t = rig_from_viewer_slots["diag_left"]
            assign_viewer_pose_to_rig_slot(gym, viewer, env, h, p, t, "diag_left", "diag_left_cam")
            continue
        if rig_from_viewer_slots and evt.action == "rig_assign_diag_right":
            h, p, t = rig_from_viewer_slots["diag_right"]
            assign_viewer_pose_to_rig_slot(gym, viewer, env, h, p, t, "diag_right", "diag_right_cam")
            continue
        if rig_from_viewer_slots and evt.action == "rig_assign_point_cloud_center":
            h, p, t = rig_from_viewer_slots["point_cloud_center"]
            assign_viewer_pose_to_rig_slot(
                gym, viewer, env, h, p, t, "point_cloud_center", "point_cloud_center_cam"
            )
            continue
        if evt.action == "cam_fwd":
            cam_pos += forward * move_step
            changed = True
        elif evt.action == "cam_back":
            cam_pos -= forward * move_step
            changed = True
        elif evt.action == "cam_left":
            cam_pos -= right * move_step
            changed = True
        elif evt.action == "cam_right":
            cam_pos += right * move_step
            changed = True
        elif evt.action == "cam_up":
            cam_pos += up * move_step
            changed = True
        elif evt.action == "cam_down":
            cam_pos -= up * move_step
            changed = True
        elif evt.action == "cam_yaw_left":
            yaw_rot = R.from_rotvec(world_up * math.radians(rot_step_deg))
            forward = _normalize_vec3(yaw_rot.apply(forward), np.array([1.0, 0.0, 0.0], dtype=np.float64))
            right = _normalize_vec3(np.cross(forward, world_up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
            up = _normalize_vec3(np.cross(right, forward), world_up)
            changed = True
        elif evt.action == "cam_yaw_right":
            yaw_rot = R.from_rotvec(world_up * -math.radians(rot_step_deg))
            forward = _normalize_vec3(yaw_rot.apply(forward), np.array([1.0, 0.0, 0.0], dtype=np.float64))
            right = _normalize_vec3(np.cross(forward, world_up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
            up = _normalize_vec3(np.cross(right, forward), world_up)
            changed = True
        elif evt.action == "cam_pitch_up":
            pitch_rot = R.from_rotvec(right * math.radians(rot_step_deg))
            forward = _normalize_vec3(pitch_rot.apply(forward), np.array([1.0, 0.0, 0.0], dtype=np.float64))
            right = _normalize_vec3(np.cross(forward, world_up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
            up = _normalize_vec3(np.cross(right, forward), world_up)
            changed = True
        elif evt.action == "cam_pitch_down":
            pitch_rot = R.from_rotvec(right * -math.radians(rot_step_deg))
            forward = _normalize_vec3(pitch_rot.apply(forward), np.array([1.0, 0.0, 0.0], dtype=np.float64))
            right = _normalize_vec3(np.cross(forward, world_up), np.array([0.0, 1.0, 0.0], dtype=np.float64))
            up = _normalize_vec3(np.cross(right, forward), world_up)
            changed = True

    if changed:
        cam_target = cam_pos + forward
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
            gymapi.Vec3(float(cam_target[0]), float(cam_target[1]), float(cam_target[2])),
        )


def estimate_endpoint_joint_speeds(path, dt):
    """Estimate per-joint velocity at start/end from a joint path."""
    if path is None or len(path) < 2 or dt <= 0:
        nan6 = np.full((6,), np.nan, dtype=np.float64)
        return nan6, nan6
    p0 = np.array(path[0], dtype=np.float64)
    p1 = np.array(path[1], dtype=np.float64)
    pn1 = np.array(path[-2], dtype=np.float64)
    pn = np.array(path[-1], dtype=np.float64)
    v_start = (p1 - p0) / dt
    v_goal = (pn - pn1) / dt
    return v_start, v_goal

def interpolate_path_ntfield(path, steps_between=4):
    """Interpolate between consecutive waypoints for NTField path animation."""
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

def parse_q_ntfield(s):
    """Parse q_start or q_goal string. Supports '0.5*pi' via eval."""
    s = s.strip().replace("pi", "math.pi")
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 comma-separated values, got {len(parts)}")
    return [float(eval(p, {"math": math})) for p in parts]


def interpolate_joint_path(q_start, q_goal, num_steps=120):
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    num_steps = max(int(num_steps), 2)
    path = []
    for i in range(num_steps + 1):
        alpha = float(i) / float(num_steps)
        q = (1.0 - alpha) * q_start + alpha * q_goal
        path.append(q.copy())
    return path


def straight_line_collision_free_path(rac, q_start, q_goal, plane_obj, static_env_models, flex_collision_models, num_steps=120):
    path = interpolate_joint_path(q_start, q_goal, num_steps=num_steps)
    for q in path:
        if not rac.arm_collision_free(q.tolist(), plane_obj, static_env_models, flex_collision_models):
            return None
    return path


def get_random_loc(x_min, x_max, y_min, y_max, z_min, z_max):
    x_can = np.random.random()*(x_max - x_min) + x_min
    y_can = np.random.random()*(y_max - y_min) + y_min
    z_can = np.random.random()*(z_max - z_min) + z_min
    return gymapi.Vec3(x_can, y_can, z_can)

def get_best_cam_pose(scene, camera_pose_list):
    best_score = - sys.maxsize
    best_index = None
    for t in range(len(camera_pose_list)):
        pose_candidate = camera_pose_list[t]
        candidate_score = feed_forward(scene.scene_, pose_candidate)
        if candidate_score > best_score:
            best_score = candidate_score
            best_index = t
    print (f'best score is : {best_score}')
    return best_index

def get_best_cam_pose_point_check(camera_pose_list, points):
    best_score = - sys.maxsize
    best_index = None
    for t in range(len(camera_pose_list)):
        pose_candidate = camera_pose_list[t]
        cam = camera(pose_candidate[4:], pose_candidate[:4])
        cam.build()
        
        in_cam = 0
        for point in points:
            if len(point) == 3:
                target_location = point
            else:
                target_location = [point[0]/100, point[1]/100, 0.05]

            flag, depth, location = cam.inside_frame(target_location)
            if flag:
                in_cam += 1

        if best_score < in_cam:
            best_score = in_cam
            best_index = t
    
    return best_index

def cam_loc_selection_for_clusters(sim, env, test_cam, center, end_points, scene_info, points):
    camera_pose_list = []
    camera_setting_list = []
    cam_height = scene_info[3] * 0.90
    floor_height = scene_info[2] + 0.01

    dist = 0
    while len(camera_pose_list) < 10:
        for end_point in end_points:
            loc_vec = (end_point - center) / np.linalg.norm(end_point - center) / 100
            cam_loc = end_point + loc_vec * dist

            camera_loc = gymapi.Vec3(cam_loc[0], cam_loc[1], cam_height)
            camera_focus = gymapi.Vec3(center[0], center[1], floor_height)
            gym.set_camera_location(test_cam, env, 
                                    camera_loc, 
                                    camera_focus)
            target_pos = gym.get_camera_transform(sim, env, test_cam).p
            target_quat = gym.get_camera_transform(sim, env, test_cam).r
            camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                    target_pos.x, target_pos.y, target_pos.z])

            r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
            cam_offset_vector = np.array([0.11, 0, 0.08])
            rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
            converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                     target_pos.y - rot_cam_offset_vector[1],
                                                     target_pos.z - rot_cam_offset_vector[2], 
                                                     ur5e_pose.p.x, 
                                                     ur5e_pose.p.y,
                                                     ur5e_pose.p.z)
            converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

            seed_state = [0.0]*ik_solver2.number_of_joints
            dof_result = ik_solver2.get_ik(seed_state, 
                                           converted_coord[0],
                                           converted_coord[1],
                                           converted_coord[2],
                                           converted_quat.x, 
                                           converted_quat.y,
                                           converted_quat.z,
                                           converted_quat.w)

            print('dist:', dist, 'loc', cam_loc, cam_height, "dof result", True if dof_result else False, 'len', len(camera_pose_list))

            if dof_result:
                end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)

                if end_state_collision_free:
                    camera_pose_list.append(camera_pose)
                    camera_setting_list.append([camera_loc, camera_focus, dof_result])

        if cam_loc[0] < 0.05:
            cam_height -= 0.01
            dist = 0

        if cam_height <= 0.20:
            break

        dist += 1

    print("cluster")
    best_cam_pose_index = get_best_cam_pose_point_check(camera_pose_list, points)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]


def random_sample_swept_volume_selection(sim, env, test_cam, swept_center, swept_points):
    camera_pose_list = []
    camera_setting_list = []
    while len(camera_pose_list) < 50:
        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                    -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                    table_dims.z, table_dims.z + drawer_height - 0.02)
        camera_focus = gymapi.Vec3(swept_center[0], swept_center[1], swept_center[2])
        gym.set_camera_location(test_cam, env, 
                                camera_loc, 
                                camera_focus)
        target_pos = gym.get_camera_transform(sim, env, test_cam).p
        target_quat = gym.get_camera_transform(sim, env, test_cam).r
        camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                target_pos.x, target_pos.y, target_pos.z])

        r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        cam_offset_vector = np.array([0.11, 0, 0.08])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                 target_pos.y - rot_cam_offset_vector[1],
                                                 target_pos.z - rot_cam_offset_vector[2], 
                                                 ur5e_pose.p.x, 
                                                 ur5e_pose.p.y,
                                                 ur5e_pose.p.z)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

        seed_state = [0.0]*ik_solver2.number_of_joints
        dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                       converted_quat.w)
        if dof_result:
            end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)


            if end_state_collision_free:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus, dof_result])
    print("swept")
    best_cam_pose_index = get_best_cam_pose_point_check(camera_pose_list, swept_points)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]


def random_sample_guided_selection(sim, env, test_cam, scene):
    camera_pose_list = []
    camera_setting_list = []
    while len(camera_pose_list) < 50:
        camera_loc = get_random_loc(0 + 0.2, table_dims.x + 0.3,
                                    -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                    table_dims.z, table_dims.z + drawer_height - 0.02)
        camera_focus = get_random_loc(0 + 0.3, table_dims.x + 0.3,
                                      -table_dims.y*0.5 + 0.02, table_dims.y*0.5 - 0.02,
                                      table_dims.z, camera_loc.z)
        gym.set_camera_location(test_cam, env, 
                                camera_loc, 
                                camera_focus)
        target_pos = gym.get_camera_transform(sim, env, test_cam).p
        target_quat = gym.get_camera_transform(sim, env, test_cam).r
        camera_pose = np.array([target_quat.x, target_quat.y, target_quat.z, target_quat.w,
                                target_pos.x, target_pos.y, target_pos.z])

        r_rot = R.from_quat([target_quat.x, target_quat.y, target_quat.z, target_quat.w])
        cam_offset_vector = np.array([0.11, 0, 0.08])
        rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
        converted_coord = global_coord_converter(target_pos.x - rot_cam_offset_vector[0],
                                                 target_pos.y - rot_cam_offset_vector[1],
                                                 target_pos.z - rot_cam_offset_vector[2], 
                                                 ur5e_pose.p.x, 
                                                 ur5e_pose.p.y,
                                                 ur5e_pose.p.z)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)

        seed_state = [0.0]*ik_solver2.number_of_joints
        dof_result = ik_solver2.get_ik(seed_state, 
                                       converted_coord[0],
                                       converted_coord[1],
                                       converted_coord[2],
                                       converted_quat.x, 
                                       converted_quat.y,
                                       converted_quat.z,
                                       converted_quat.w)
        if dof_result:
            end_state_collision_free = rac.arm_collision_free(dof_result, plane_obj, object_collision_models, flexible_collision_models)


            if end_state_collision_free:
                camera_pose_list.append(camera_pose)
                camera_setting_list.append([camera_loc, camera_focus, dof_result])
         
    best_cam_pose_index = get_best_cam_pose(scene, camera_pose_list)
    return camera_setting_list[best_cam_pose_index][0], camera_setting_list[best_cam_pose_index][1], camera_setting_list[best_cam_pose_index][2]

def swept_coverage_check(scene, swept_verts, rac, scene_info, max_height):
    covered = 0
    new_verts = []
    for i, verts in enumerate(swept_verts):
        idx = verts * 100
        idx[0] -= 30
        idx[1] += 60
        idx = np.rint(idx).astype(int)
        checked = scene.scene_[idx[0], idx[1], idx[2]]
        if checked < 0:
            swept_verts.pop(i)
        if checked > 0:
            covered += 1
        else:
            new_verts.append(verts)

    next_center, _ = rac.get_swept_center([new_verts], scene_info, max_height)
    return covered / len(swept_verts), next_center

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


def get_unobserved_area(scene):
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_]
    unknown_area = np.argwhere(floor == 0)[:,:2]
    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]
    return unknown_area

def get_max_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    max_height = -sys.maxsize
    for i in range(len(OBJ_FILE_IDX_LIST) - 1):
        obj_name = object_asset_files[OBJ_FILE_IDX_LIST[i]]
        obj_name = asset_root + "/".join(obj_name.split('/')[:3]) + "/textured_vhacd.obj"
    
        mesh = o3d.io.read_triangle_mesh(obj_name)
        verts = np.asarray(mesh.vertices)
        min_z = sys.maxsize
        max_z = -sys.maxsize
        for tx, ty, tz in verts:
            min_z = min(min_z, tz)
            max_z = max(max_z, tz)
        height = int((max_z - min_z) * 100)

        if max_height < height:
            max_height = height
    
    return max_height + 1

def get_min_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    min_height = sys.maxsize
    for i in range(len(OBJ_FILE_IDX_LIST) - 1):
        obj_name = object_asset_files[OBJ_FILE_IDX_LIST[i]]
        obj_name = asset_root + "/".join(obj_name.split('/')[:3]) + "/textured_vhacd.obj"
    
        mesh = o3d.io.read_triangle_mesh(obj_name)
        verts = np.asarray(mesh.vertices)
        min_z = sys.maxsize
        max_z = -sys.maxsize
        for tx, ty, tz in verts:
            min_z = min(min_z, tz)
            max_z = max(max_z, tz)
        height = int((max_z - min_z) * 100)

        if min_height > height:
            min_height = height
    
    return min_height

def get_unobserved_area_w_height(scene, asset_root, object_asset_files, OBJ_FILE_IDX_LIST):
    min_height = get_min_height(asset_root, object_asset_files, OBJ_FILE_IDX_LIST)
    floor = scene.scene_[:scene.x_limit_, scene.y_left_+1 :(scene.y_left_ + scene.y_limit_-1), scene.g_height_: scene.g_height_ + min_height]

    unknown_layer = set(map(tuple, np.argwhere(floor[:,:,0] == 0)[:,:2]))
    for i in range(1, min_height):
        temp_unknown = set(map(tuple, np.argwhere(floor[:,:,i] == 0)[:,:2]))
        unknown_layer = unknown_layer.intersection(temp_unknown)
    unknown_area = np.array(list(unknown_layer))

    unknown_area[:, 0] += 29
    unknown_area[:, 1] -= int((floor.shape[1]) / 2)

    temp = copy.deepcopy(unknown_area)
    unknown_area[:, 0] = -temp[:, 1]
    unknown_area[:, 1] = temp[:, 0]

    return unknown_area

def scale_config(config):
    for pos in config:
        for i in range(3):
            pos[i] = pos[i] * 100
        temp = pos[0]
        pos[0] = -pos[1]
        pos[1] = temp

    return config


def object_prompt_from_asset(asset_path):
    """Create a simple text prompt from YCB asset path."""
    obj_folder = asset_path.split("/")[-2]
    return "grasp the " + obj_folder.replace("_", " ")

def save_scene(init2grasp_path, grasp2init_path, obj_pos_list, gt_obj_pos_list, NUM_OF_OBJECTS, scene_info,
               target_mesh, obj_mesh, target_pos, gt_target_pos, obstacles_num, w_target,
               unknown_area, valid_area, potential_centers, method, MCTS_result=None):
    gt_save_info = {"idx" : 0,
                    "init2grasp_path" : init2grasp_path,
                    "grasp2init_path" : grasp2init_path,
                    "obj_pos_list" : gt_obj_pos_list[:NUM_OF_OBJECTS-1],
                    "obj_mesh" : obj_mesh,
                    "scene_info" : scene_info,
                    "w_target" : w_target,
                    "test_name" : None,
                    "target_mesh" : target_mesh,
                    "obstacles_num" : obstacles_num,
                    "target_pos" : gt_target_pos,
                    "unknown_area" : unknown_area,
                    "valid_area" : valid_area,
                    "potential_centers" : potential_centers}
    
    save_info = {"idx" : 0,
                 "init2grasp_path" : init2grasp_path,
                 "grasp2init_path" : grasp2init_path,
                 "obj_pos_list" : obj_pos_list,
                 "obj_mesh" : obj_mesh,
                 "scene_info" : scene_info,
                 "w_target" : w_target,
                 "test_name" : None,
                 "target_mesh" : target_mesh,
                 "obstacles_num" : obstacles_num,
                 "target_pos" : target_pos,
                 "unknown_area" : unknown_area,
                 "valid_area" : valid_area,
                 "potential_centers" : potential_centers}
    
    comp = np.array([save_info])

    end = "_success" if MCTS_result else "_failed"
    if MCTS_result is None:
        end = "None"

    name = new_folder + method + "temp_scene" + str(sequence_count) + end
    np.save(name, comp)

    comp = np.array([gt_save_info])
    name = new_folder + method + "groud_truth_scene" + str(sequence_count) + end
    np.save(name, comp)

def update_MCTS_val(ML_MCTS_ins, curr_config, target_pos_MCT, obj_mesh, unknown_area, valid_area, potential_centers):
    # update MCTS values
    ML_MCTS_ins.curr_config_ = copy.deepcopy(curr_config)
    ML_MCTS_ins.goal_config_ = copy.deepcopy(curr_config)
    ML_MCTS_ins.target_pos = copy.deepcopy(target_pos_MCT)
    ML_MCTS_ins.obj_mesh = copy.deepcopy(obj_mesh)

    ML_MCTS_ins.unknown_area = copy.deepcopy(unknown_area)
    ML_MCTS_ins.valid_area = copy.deepcopy(valid_area)
    ML_MCTS_ins.potential_centers = copy.deepcopy(potential_centers)

def update_rac_val(rac, target_obj_mesh, obj_mesh_MCTS, obj_pos_MCTS):
    rac.target_mesh = copy.deepcopy(target_obj_mesh)
    rac.obj_mesh = copy.deepcopy(list(obj_mesh_MCTS.values()))
    rac.obj_pos_list = copy.deepcopy(list(obj_pos_MCTS.values()))
    rac.obstacles_num = len(rac.obj_pos_list)

def move_objs(gym, object_collision_files, gymapi, new_obj_pos_list):
    objs_manager = fcl.DynamicAABBTreeCollisionManager()
    objs_manager.setup()
    obstacle_objs = []
    GT_TARGET_POS = [np.random.uniform(0.20 + table_dims.x/2, table_dims.x),
                     np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2),
                     table_dims.z + 0.08]
    
    for k in range(NUM_OF_OBJECTS):
        object_pose = gymapi.Transform()
        is_collision = True
        # add target obj
        if k == NUM_OF_OBJECTS - 1:
            object_pose.p = gymapi.Vec3(GT_TARGET_POS[0], GT_TARGET_POS[1], GT_TARGET_POS[2])
            file_path = object_collision_files[OBJ_FILE_IDX_LIST[-1]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[-1])
            collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[-1]])
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array(GT_TARGET_POS))
            is_collision = False
        # random selec obj location
        while is_collision:
            tx = np.random.uniform(0.35, table_dims.x + 0.2)
            ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
            tz = table_dims.z + 0.08
            object_pose.p = gymapi.Vec3(tx, ty, tz)
            file_path = object_collision_files[OBJ_FILE_IDX_LIST[k]]
            collision_mesh = obj_reader(asset_root + file_path)
            collision_mesh.set_scale(object_scaling_factor[k])
            collision_mesh.add_offset(object_offset[OBJ_FILE_IDX_LIST[k]])
            
            verts, tris = collision_mesh.get_bounding_box_mesh()
            temp_center = collision_mesh.get_center()
            temp_bounding_box = collision_mesh.get_bounding_box()
            # new obj
            m = fcl.BVHModel()
            m.beginModel(len(verts), len(tris))
            m.addSubModel(verts, tris)
            m.endModel()
            t = fcl.Transform(np.array([tx,ty,tz]))
            
            # check collision
            req = fcl.CollisionRequest()
            rdata = fcl.CollisionData(request = req)
            objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)
            is_collision = rdata.result.is_collision # update collision status
            if not is_collision:
                    dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                    if dist <= 0.2:
                        is_collision = True
                        continue
                    for obj in new_obj_pos_list:
                        dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                        if dist <= 0.16:
                            is_collision = True
                            continue
        new_obj_pos_list.append([object_pose.p.x, object_pose.p.y])
        object_handles.append(gym.create_actor(envs[-1], 
                                            object_assets[OBJ_FILE_IDX_LIST[k]], 
                                            object_pose, 
                                            "object" + str(k) + str(i), 0, 2**(k+1), k+1))
        gym.set_actor_scale(envs[-1], object_handles[-1], object_scaling_factor[k])
        object_reader_tracker.append(collision_mesh)
        object_status_list.append([temp_center, temp_bounding_box])
        object_collision_lib.append(m)
        obstacle_objs.append(fcl.CollisionObject(m, t))
        objs_manager.registerObjects(obstacle_objs)
        objs_manager.setup()
    
    #set up global camera to record configuration
    body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
    viewpoint_candidate = gymapi.Vec3(3, 0, 0.3)
    gym.set_camera_location(body_cam_handles[-1], envs[-1], 
                            viewpoint_candidate, 
                            camera_focus)

    return

#*************************************************************************************************#


def resolve_collected_h5_path(h5_path, script_dir, repo_root):
    """Resolve grasp_6dof_demo_*.h5 relative to hanwen_grasping or PI-VLA/collected_data."""
    if not h5_path:
        return None
    if os.path.isfile(h5_path):
        return os.path.abspath(h5_path)
    cand = os.path.join(script_dir, h5_path)
    if os.path.isfile(cand):
        return os.path.abspath(cand)
    cand2 = os.path.join(repo_root, h5_path)
    if os.path.isfile(cand2):
        return os.path.abspath(cand2)
    cand3 = os.path.join(repo_root, "collected_data", os.path.basename(h5_path))
    if os.path.isfile(cand3):
        return os.path.abspath(cand3)
    return os.path.abspath(h5_path)


def read_h5_object_world_xyz(h5_path):
    """
    Return (x,y,z) actor root on the table for replay, or None.

    Older collect_data.h5 files stored mesh-local bbox centers in object_location
    (small values near origin) — those are ignored so we do not spawn under the robot.
    """
    import h5py
    if not h5_path or not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path, "r") as f:
        if "object_actor_world" in f:
            v = np.array(f["object_actor_world"][:], dtype=np.float64).reshape(-1)
            if v.size >= 3:
                return v[:3].copy()
            return None
        if "object_location" in f:
            v = np.array(f["object_location"][:], dtype=np.float64).reshape(-1)
            if v.size < 3:
                return None
            # Table objects in this scene sit at x ~ 0.35+ (see placement loop).
            if float(v[0]) < 0.28:
                print(
                    "Warning: HDF5 object_location looks like legacy mesh-local coords; "
                    "skipping fixed object pose (object will be placed randomly)."
                )
                return None
            return v[:3].copy()
    return None


#*************************************************************************************************#

if __name__ == '__main__':
    #initialize gym
    #*************************************************************************************************#
    gym = gymapi.acquire_gym()
    #*************************************************************************************************#

    # parse arguments
    #*************************************************************************************************#
    args = gymutil.parse_arguments(
        description="ur5e example",
        custom_parameters=[
            {'name': '--env_id', 'type': int, 'help': 'env_id', 'default': 0},
            {'name': '--ntfield', 'action': 'store_true', 'help': 'Use NTField to plan and animate robot path (requires --checkpoint)'},
            {'name': '--checkpoint', 'type': str, 'default': None, 'help': 'NTField checkpoint path (required with --ntfield)'},
            {'name': '--q_start', 'type': str, 'default': None, 'help': 'Start joint config: 6 values in radians. Use = for negative values, e.g. --q_start="0.2,-0.5,-1.0,1.57,1.57,0"'},
            {'name': '--q_goal', 'type': str, 'default': None, 'help': 'Goal joint config: 6 values in radians. Use = for negative values, e.g. --q_goal="-0.2,-0.5,-0.35,0.63,1.57,0"'},
            {'name': '--h5_path', 'type': str, 'default': None, 'help': 'collected_data grasp_6dof_demo_*.h5: joint_configs[0] -> q_start, final_joint_config (or last row) -> q_goal; object_location places object when NUM_OF_OBJECTS==1'},
            {'name': '--record', 'action': 'store_true', 'help': 'Record video of the simulation to --record_output'},
            {'name': '--record_output', 'type': str, 'default': 'ntfield_record.mp4', 'help': 'Output video path when --record. With --h5_path: default folder PI-VLA/output/trajectory_evaluation/YYYYMMDD_HHMMSS/ntfield.mp4; else ntfield_q_<goal>.mp4'},
            {'name': '--no_walls', 'action': 'store_true', 'help': 'Remove side walls and upper cover on table (keep table only)'},
            {'name': '--no_h5_spawn_object', 'action': 'store_true', 'help': 'With --ntfield --h5_path: do not place object at HDF5 object_location / prompt (use random placement).'},
            {'name': '--headless', 'action': 'store_true', 'help': 'No interactive viewer; fixed-length animation for --ntfield --record (servers without DISPLAY).'},
            {'name': '--skip_path_planning', 'action': 'store_true', 'help': 'Skip q_start->q_goal path planning in grasp mode and only save per-object goal joint configs.'},
            {'name': '--camera_capture_only', 'action': 'store_true', 'help': 'Skip motion planning; fly viewer (logs pose). Keys 1/2/3 copy viewer to diag_left/diag_right/point_cloud_center rig; S saves PNG/npz; close viewer to save again.'},
            {'name': '--rig_cams_anchored_table', 'action': 'store_true', 'help': 'Place the three rig cameras relative to tabletop center each run (re-aims with table_pose); default is fixed env-local Vec3 in script.'},
            {'name': '--save', 'action': 'store_true', 'help': 'Legacy no-op: viewer camera FOV/pos/dir/target log on every camera move when a viewer exists.'},
            {'name': '--collect_ntfield_metric', 'action': 'store_true',
             'help': 'After scene settle: sample (q_start near obstacles, q_goal uniform) in FCL and save sampled_points.npy, speed.npy, normal.npy for ntrl-demo/train/train_arm.py (no OMPL/RRT).'},
            {'name': '--metric_num_samples', 'type': int, 'default': 20000, 'help': 'Target number of accepted pairs for --collect_ntfield_metric.'},
            {'name': '--metric_output_dir', 'type': str, 'default': None,
             'help': 'Output directory for metric .npy files (default: output_root/env_<id>_ntfield_metric/).'},
            {'name': '--metric_seed', 'type': int, 'default': 0, 'help': 'RNG seed for --collect_ntfield_metric.'},
            {'name': '--metric_max_tries_factor', 'type': int, 'default': 2000,
             'help': 'Stop after this many proposals per requested sample if undersampled.'},
            {'name': '--metric_sampler_mode', 'type': str, 'default': 'fcl_uniform',
             'help': 'q_start sampler for --collect_ntfield_metric: fcl_uniform or ik_mesh.'},
            {'name': '--metric_ik_pose_trials', 'type': int, 'default': 80,
             'help': 'For --metric_sampler_mode=ik_mesh: random mesh target poses attempted per sample proposal.'},
            {'name': '--metric_ik_seed_trials', 'type': int, 'default': 6,
             'help': 'For --metric_sampler_mode=ik_mesh: IK restarts per pose target.'},
            {'name': '--metric_ik_surface_offset_min', 'type': float, 'default': 0.002,
             'help': 'For --metric_sampler_mode=ik_mesh: minimum offset (m) from sampled obstacle surface.'},
            {'name': '--metric_ik_surface_offset_max', 'type': float, 'default': 0.03,
             'help': 'For --metric_sampler_mode=ik_mesh: maximum offset (m) from sampled obstacle surface.'},
            {'name': '--metric_ik_tool_offset_x', 'type': float, 'default': 0.11,
             'help': 'For --metric_sampler_mode=ik_mesh: IK target->wrist offset x (m), aligned with new_setup.py by default.'},
            {'name': '--metric_ik_tool_offset_y', 'type': float, 'default': 0.0,
             'help': 'For --metric_sampler_mode=ik_mesh: IK target->wrist offset y (m).'},
            {'name': '--metric_ik_tool_offset_z', 'type': float, 'default': 0.08,
             'help': 'For --metric_sampler_mode=ik_mesh: IK target->wrist offset z (m), aligned with new_setup.py by default.'},
            {'name': '--metric_ik_urdf_file', 'type': str, 'default': 'ur5e_mimic_real_gripper_test.urdf',
             'help': 'URDF filename under assets/urdf/ur5e for TRAC-IK in metric collection.'},
            {'name': '--metric_log_every_tries', 'type': int, 'default': 2000,
             'help': 'For --collect_ntfield_metric: print progress every N proposals (<=0 disables periodic logs).'},
            {'name': '--metric_qstart_only', 'action': 'store_true',
             'help': 'Collect only q_start IK collision-free configurations (skip q_goal sampling and speed/normal outputs).'},
            {'name': '--metric_save_speed_normal', 'action': 'store_true',
             'help': 'When q_goal is sampled, also save speed.npy and normal.npy. Disabled by default for faster debug runs.'},
            {'name': '--metric_visualize_sampling', 'action': 'store_true',
             'help': 'For --collect_ntfield_metric: visualize sampled q_start/q_goal in viewer during collection (requires not --headless).'},
            {'name': '--metric_visualize_every_accepted', 'type': int, 'default': 100,
             'help': 'For --metric_visualize_sampling: render every N accepted samples.'},
            {'name': '--metric_visualize_hold_steps', 'type': int, 'default': 20,
             'help': 'For --metric_visualize_sampling: physics/viewer steps to hold q_start then q_goal.'},
            {'name': '--simple_grasp_collect', 'action': 'store_true',
             'help': 'Simple mode: one object, sample TRAC-IK grasp candidates (no RRT), choose straight-line collision-reachable q_g from current q_s, and save q_s->q_g path.'},
            {'name': '--simple_num_candidates', 'type': int, 'default': 100,
             'help': 'Number of TRAC-IK grasp candidates to test in --simple_grasp_collect mode.'},
            {'name': '--simple_interp_steps', 'type': int, 'default': 120,
             'help': 'Interpolation steps for straight-line collision check and saved path in --simple_grasp_collect mode.'},
        ],
    )
    args._session_eval_dir = None
    output_root = os.path.abspath(os.path.join(file_dir, "..", "output", "constrained_multi_obj"))
    os.makedirs(output_root, exist_ok=True)
    env_id = int(args.env_id)
    ntfield_h5_object_xyz = None
    if getattr(args, 'ntfield', False) and args.h5_path:
        args.h5_path = resolve_collected_h5_path(args.h5_path, file_dir, pi_vla_root)
        ntfield_h5_object_xyz = read_h5_object_world_xyz(args.h5_path)
        if ntfield_h5_object_xyz is not None:
            print(f"HDF5 object world pose (actor root): {ntfield_h5_object_xyz}")
    #*************************************************************************************************#

    #create a simulator
    #*************************************************************************************************#
    sim_params = gymapi.SimParams()
    sim_params.substeps = 2
    sim_params.dt = 1.0 / 60.0
    #*************************************************************************************************#

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0, 0, -9.8)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, 
                        args.physics_engine, sim_params)

    if sim is None:
        print("*** Failed to create sim")
        quit()
    #*************************************************************************************************#

    #configure a ground plane
    #*************************************************************************************************#
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    gym.add_ground(sim, plane_params)
    #*************************************************************************************************#

    #all assets
    #*************************************************************************************************#
    asset_root = "./assets/"
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
    object_centroid_m = []
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

    import h5_episode_viz as h5viz
    _nt_h5_resolved = None
    _nt_h5_ol = None
    _nt_h5_prompt = ""
    _nt_h5_obj_idx = None
    if getattr(args, 'ntfield', False) and args.h5_path and not getattr(args, 'no_h5_spawn_object', False):
        _nt_h5_resolved = h5viz.resolve_h5_path(args.h5_path, pi_vla_root, file_dir)
        if _nt_h5_resolved:
            _nt_h5_ol, _nt_h5_prompt = h5viz.read_h5_object_and_prompt(_nt_h5_resolved)
            _nt_h5_obj_idx = h5viz.infer_ycb_index_from_prompt(_nt_h5_prompt, object_asset_files)
            if _nt_h5_obj_idx is None or _nt_h5_ol is None or _nt_h5_ol.size < 3:
                if _nt_h5_prompt:
                    print(
                        "Warning (NTField+H5): could not place object from HDF5 (prompt/location/index); "
                        "using random object placement."
                    )
                _nt_h5_obj_idx = None
                _nt_h5_ol = None

    #setup all collision meshes
    #setup is done outside the env loop since all robots are the same
    ur5e_collision_models = []
    ur5e_rotations = [R.from_euler('x',  [90], degrees = True),
                    R.from_euler('xy', [90, 180], degrees = True),
                    R.from_euler('xy', [180, 180], degrees = True),
                    R.from_euler('z',  [-180], degrees = True),
                    R.from_euler('x',  [-180], degrees = True),
                    R.from_euler('x',  [90], degrees = True),
                    R.from_euler('z',  [-90], degrees = True)]
    ur5e_translations = [[0, 0, 0], 
                        [0, 0, 0],
                        [0, -0.138, 0],
                        [0, -0.007, 0],
                        [0, 0.127, 0],
                        [0, 0, 0],
                        [0, 0, 0]]
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
    #*************************************************************************************************#


    #calculate Inverse Kinematics
    #*************************************************************************************************#
    urdf_str = ''
    with open("./assets/urdf/ur5e/ur5e_mimic_real_gripper_test.urdf") as f:
        urdf_str = f.read()

    #*************************************************************************************************#

    # create viewer (optional: --headless for batch recording)
    # Use the same FOV as sensor cameras so tracked views match saved captures.
    #*************************************************************************************************#
    viewer = None
    if not getattr(args, 'headless', False):
        viewer_props = gymapi.CameraProperties()
        viewer_props.horizontal_fov = 70.25
        viewer = gym.create_viewer(sim, viewer_props)
        if viewer is None:
            raise ValueError('*** Failed to create viewer')
    else:
        print('Headless mode: no Isaac viewer window (use --ntfield with --record to save MP4).')
    #*************************************************************************************************#

    #set up the environment grid
    #*************************************************************************************************#
    spacing = 2
    env_lower = gymapi.Vec3(-spacing, -spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, 0)
    #*************************************************************************************************#

    #load asset
    #*************************************************************************************************#
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.use_mesh_materials = True

    ur5e_asset = gym.load_asset(sim, asset_root, ur5e_asset_file, asset_options)
    table_asset = gym.create_box(sim, table_dims.x,
                                    table_dims.y,
                                    table_dims.z,
                                    asset_options)

    #size of left/right cover will be decided by table size
    drawer_height = max_drawer_height
    side_cover_dims = gymapi.Vec3(table_dims.x, piece_width, drawer_height)
    left_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                        side_cover_dims.y,
                                        side_cover_dims.z,
                                        asset_options)
    right_cover_asset = gym.create_box(sim, side_cover_dims.x,
                                            side_cover_dims.y,
                                            side_cover_dims.z,
                                            asset_options)

    #upper cover
    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(sim, upper_cover_dims.x,
                                            upper_cover_dims.y,
                                            upper_cover_dims.z,
                                            asset_options)
    print(
        f"Table size fixed: x={table_dims.x:.3f}m, y={table_dims.y:.3f}m, z={table_dims.z:.3f}m; "
        f"drawer_height={drawer_height:.3f}m"
    )

    saved_env_name = os.path.join(output_root, f'env_{env_id}_scene_info.npy')
    np.save(saved_env_name, np.array([table_dims.x, table_dims.y, table_dims.z, drawer_height]))

    asset_options.fix_base_link = False
    object_assets = []
    test_assets = []
    for ob in object_asset_files:
        object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
    asset_options.fix_base_link = True
    test_assets.append(gym.load_asset(sim, asset_root, object_asset_files[0], asset_options))
    asset_options.fix_base_link = False

    #*************************************************************************************************#

    #initial pose
    #*************************************************************************************************#
    ur5e_pose = gymapi.Transform()
    # ur5e_pose.p = gymapi.Vec3(np.random.rand()*0.3 - 0.2, np.random.rand()*0.4 - 0.2, 0.0)
    ur5e_pose.p = gymapi.Vec3(0.2, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5*math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5)

    left_cover_pose = gymapi.Transform()
    left_cover_pose.p = gymapi.Vec3(table_pose.p.x, table_dims.y*0.5 - 0.015, 
                                    table_dims.z + side_cover_dims.z/2.0)

    right_cover_pose = gymapi.Transform()
    right_cover_pose.p = gymapi.Vec3(table_pose.p.x, -table_dims.y*0.5 + 0.015, 
                                    table_dims.z + side_cover_dims.z/2.0)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + side_cover_dims.z + 0.015)

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    #set all environment collision models

    plane_normal = np.array([0.0, 0.0, 1.0])
    col_plane = fcl.Plane(plane_normal, 0)
    plane_obj = fcl.CollisionObject(col_plane, fcl.Transform())

    col_table = fcl.Box(table_dims.x, table_dims.y, table_dims.z)
    trans_table = fcl.Transform(np.array([table_dims.x*0.5 + 0.3, 0.0, table_dims.z*0.5]))
    table_obj = fcl.CollisionObject(col_table, trans_table)

    col_left_cover = fcl.Box(side_cover_dims.x,
                            side_cover_dims.y,
                            side_cover_dims.z)
    trans_left_cover = fcl.Transform(np.array([table_pose.p.x, table_dims.y*0.5 - 0.015, 
                                            table_dims.z + side_cover_dims.z/2.0]))
    left_cover_obj = fcl.CollisionObject(col_left_cover, trans_left_cover)

    col_right_cover = fcl.Box(side_cover_dims.x,
                            side_cover_dims.y,
                            side_cover_dims.z)
    trans_right_cover = fcl.Transform(np.array([table_pose.p.x, -table_dims.y*0.5 + 0.015, 
                                                table_dims.z + side_cover_dims.z/2.0]))
    right_cover_obj = fcl.CollisionObject(col_right_cover, trans_right_cover)

    object_collision_models = [table_obj]
    if not getattr(args, 'no_walls', False):
        object_collision_models.extend([left_cover_obj, right_cover_obj])

    if not getattr(args, 'no_walls', False) and ADD_COVER:
        col_upper_cover = fcl.Box(upper_cover_dims.x,
                                upper_cover_dims.y,
                                upper_cover_dims.z)
        trans_upper_cover = fcl.Transform(np.array([table_pose.p.x, 0.0, 
                                        table_dims.z + side_cover_dims.z + 0.015]))
        upper_cover_obj = fcl.CollisionObject(col_upper_cover, trans_upper_cover)
        object_collision_models.append(upper_cover_obj)

    #*************************************************************************************************#

    #create environment
    #*************************************************************************************************#
    #create location candidates
    location_candidates = []
    start_i = 0.4
    while start_i <= table_dims.x - 0.1 + 0.3:
        start_j = - table_dims.y * 0.5 + 0.1
        temp_candidates = []
        while start_j <= table_dims.y*0.5 - 0.1:
            temp_candidates.append([start_i, start_j, table_dims.z + 0.1])
            start_j += 0.2
        location_candidates.append(temp_candidates)
        start_i += 0.3

    region_candidates = []
    start_i = 0.4
    while start_i <= table_dims.x - 0.1 + 0.3:
        start_j = - table_dims.y * 0.5 + 0.1
        temp_candidates = []
        while start_j <= table_dims.y*0.5 - 0.1:
            temp_candidates.append([start_i, start_j, table_dims.z + 0.1])
            start_j += 0.1
        region_candidates.append(temp_candidates)
        start_i += 0.1

    print(len(region_candidates), len(region_candidates[0]))

    envs = []
    ur5e_handles = []
    body_cam_handles = []
    diag_left_cam_handles = []
    diag_right_cam_handles = []
    point_cloud_center_cam_handles = []
    camera_candidates = []
    # chosen_object = []
    chosen_scale = []
    object_normalize = []

    observed_objects = []
    gripper_location = None
    object_status_list = []
    object_reader_tracker = []
    num_objects_for_run = 1 if getattr(args, "simple_grasp_collect", False) else NUM_OF_OBJECTS
    for i in range(num_of_envs):
        envs.append(gym.create_env(sim, env_lower, env_upper, row_num_of_envs))
        ur5e_handles.append(gym.create_actor(envs[-1], ur5e_asset, ur5e_pose, "ur5e" + str(i), 0, 32767))

        #get joint handler
        spj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_pan_joint")
        slj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "shoulder_lift_joint")
        ej = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "elbow_joint")
        wj1 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_1_joint")
        wj2 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_2_joint")
        wj3 = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "wrist_3_joint")
        likj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_knuckle_joint")
        lifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_inner_finger_joint")
        lokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "left_outer_knuckle_joint")
        rikj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_knuckle_joint")
        rifj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_inner_finger_joint")
        rokj = gym.find_actor_dof_handle(envs[-1], ur5e_handles[-1], "right_outer_knuckle_joint")

        #attach body camera sensor
        cam_link = gym.find_actor_rigid_body_handle(envs[-1], ur5e_handles[-1], "wrist_3_link")

        #right in front of D435i model
        cam_offset_x = 0.11
        cam_offset_z = 0.08
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        body_cam_transform = gymapi.Transform()
        body_cam_transform.p = gymapi.Vec3(cam_offset_x, 0, cam_offset_z)
        gym.attach_camera_to_body(body_cam_handles[-1], envs[-1], cam_link, body_cam_transform, 
                                gymapi.CameraFollowMode.FOLLOW_TRANSFORM)

        gym.create_actor(envs[-1], table_asset, table_pose, "table" + str(i), 0, 1)
        add_walls = not getattr(args, 'no_walls', False)
        if add_walls:
            gym.create_actor(envs[-1], left_cover_asset, left_cover_pose, "left_cover" + str(i), 0, 1)
            gym.create_actor(envs[-1], right_cover_asset, right_cover_pose, "right_cover" + str(i), 0, 1)
        if add_walls and ADD_COVER:
            gym.create_actor(envs[-1], upper_cover_asset, upper_cover_pose, "upper_cover" + str(i), 0, 1)

        # Choose target & obstacle objects-------------------------------------------------------------------------------
        h5_use_fixed = (
            _nt_h5_obj_idx is not None
            and _nt_h5_ol is not None
            and _nt_h5_ol.size >= 3
        )
        target_file_idx = np.random.choice(TARGET_OBJ_INDEX, num_objects_for_run, replace=False)
        if h5_use_fixed:
            # Keep one object consistent with the HDF5 prompt-derived target.
            target_file_idx[-1] = _nt_h5_obj_idx
        object_handles = []

        with open("object_name.txt", 'a') as f:
            for k in range(num_objects_for_run):
                f.write(object_asset_files[target_file_idx[k]])

        object_scaling_factor = np.random.randint(0, max_scaling_factor+1, size = num_objects_for_run)/10.0 + 1.0
        if h5_use_fixed:
            object_scaling_factor = np.ones(num_objects_for_run, dtype=np.float64)

        # set up objects--------------------------------------------------------------------------------------------------------------------
        # creating manager
        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        GT_TARGET_POS = [np.random.uniform(0.20 + table_dims.x/2, table_dims.x),
                         np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2),
                         table_dims.z + 0.08]

        for k in range(num_objects_for_run):
            object_pose = gymapi.Transform()
            if h5_use_fixed and k == num_objects_for_run - 1:
                tx = float(_nt_h5_ol[0])
                ty = float(_nt_h5_ol[1])
                tz = float(_nt_h5_ol[2])
                object_pose.p = gymapi.Vec3(tx, ty, tz)
                fk = target_file_idx[k]
                file_path = object_collision_files[fk]
                collision_mesh = obj_reader(asset_root + file_path)
                collision_mesh.set_scale(object_scaling_factor[k])
                collision_mesh.add_offset(object_offset[fk])
                verts, tris = collision_mesh.get_bounding_box_mesh()
                temp_center = collision_mesh.get_center()
                temp_bounding_box = collision_mesh.get_bounding_box()
                m = fcl.BVHModel()
                m.beginModel(len(verts), len(tris))
                m.addSubModel(verts, tris)
                m.endModel()
                t = fcl.Transform(np.array([tx, ty, tz]))
            else:
                is_collision = True
                # random selec obj location
                while is_collision:
                    tx = np.random.uniform(0.35, table_dims.x + 0.2)
                    ty = np.random.uniform(-table_dims.y/2 + 0.1, table_dims.y/2 - 0.2)
                    tz = table_dims.z + 0.08

                    object_pose.p = gymapi.Vec3(tx, ty, tz)

                    file_path = object_collision_files[target_file_idx[k]]
                    collision_mesh = obj_reader(asset_root + file_path)
                    collision_mesh.set_scale(object_scaling_factor[k])
                    collision_mesh.add_offset(object_offset[target_file_idx[k]])

                    verts, tris = collision_mesh.get_bounding_box_mesh()
                    temp_center = collision_mesh.get_center()
                    temp_bounding_box = collision_mesh.get_bounding_box()

                    # new obj
                    m = fcl.BVHModel()
                    m.beginModel(len(verts), len(tris))
                    m.addSubModel(verts, tris)
                    m.endModel()
                    t = fcl.Transform(np.array([tx, ty, tz]))

                    # check collision
                    req = fcl.CollisionRequest()
                    rdata = fcl.CollisionData(request = req)
                    objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)

                    is_collision = rdata.result.is_collision  # update collision status

                    if not is_collision:
                        dist = np.sqrt((tx - GT_TARGET_POS[0])**2 + (ty - GT_TARGET_POS[1])**2)
                        if dist <= 0.2:
                            is_collision = True
                            print("target contact recalc")
                            continue

                        for obj in GT_OBJ_POS_LIST:
                            dist = np.sqrt((tx - obj[0])**2 + (ty - obj[1])**2)
                            if dist <= 0.16:
                                is_collision = True
                                print("recalc")
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

        #set up global camera to record configuration
        body_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        viewpoint_candidate = gymapi.Vec3(3, 0, 0.3)
        gym.set_camera_location(body_cam_handles[-1], envs[-1], 
                                viewpoint_candidate, 
                                camera_focus)

        # Diagonal cameras (fixed env-local Vec3, or table-anchored when --rig_cams_anchored_table)
        diag_left_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        if getattr(args, "rig_cams_anchored_table", False):
            (
                diag_left_cam_pos,
                diag_left_cam_target,
                diag_right_cam_pos,
                diag_right_cam_target,
                point_cloud_center_cam_pos,
                point_cloud_center_cam_target,
            ) = compute_rig_cameras_table_anchored(table_pose, table_dims)
        else:
            diag_left_cam_pos = gymapi.Vec3(-0.1300, 0.6069, 0.5000)
            diag_left_cam_target = gymapi.Vec3(-0.8500, 1.2000, 0.8000)
            diag_right_cam_pos = gymapi.Vec3(-0.3616, -0.6252, 0.4547)
            diag_right_cam_target = gymapi.Vec3(-1.1427, -1.2303, 0.6088)
            point_cloud_center_cam_pos = gymapi.Vec3(-0.4036, -0.1921, 0.4464)
            point_cloud_center_cam_target = gymapi.Vec3(-1.3627, -0.4306, 0.5991)
        gym.set_camera_location(diag_left_cam_handles[-1], envs[-1], diag_left_cam_pos, diag_left_cam_target)

        diag_right_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        gym.set_camera_location(diag_right_cam_handles[-1], envs[-1], diag_right_cam_pos, diag_right_cam_target)

        point_cloud_center_cam_handles.append(gym.create_camera_sensor(envs[-1], camera_props))
        gym.set_camera_location(
            point_cloud_center_cam_handles[-1],
            envs[-1],
            point_cloud_center_cam_pos,
            point_cloud_center_cam_target,
        )

    #*************************************************************************************************#

    #*************************************************************************************************#
    if viewer is not None:
        # viewer_camera_look_at(..., env, eye, target) expects eye/target in **env-local** coords (see Isaac Gym docs).
        o = gym.get_env_origin(envs[-1])
        eye_w = gymapi.Vec3(2.2, 0, 0.5)
        tgt_w = gymapi.Vec3(0, 0, 0.5)
        gym.viewer_camera_look_at(
            viewer,
            envs[-1],
            gymapi.Vec3(eye_w.x - o.x, eye_w.y - o.y, eye_w.z - o.z),
            gymapi.Vec3(tgt_w.x - o.x, tgt_w.y - o.y, tgt_w.z - o.z),
        )
        setup_viewer_camera_controls(
            gym,
            viewer,
            subscribe_rig_snapshot=getattr(args, "camera_capture_only", False),
            subscribe_rig_assign_keys=getattr(args, "camera_capture_only", False),
        )
        _cam_help = (
            "Manual camera keys: I/K fwd/back, J/L left/right, U/O up/down, F/H yaw, T/G pitch. "
            "Terminal logs FOV/pos/dir/target on every camera move."
        )
        if getattr(args, "camera_capture_only", False):
            _cam_help += (
                " Capture-only: S saves rig PNG+npz; 1/2/3 copy viewer pose to diag_left / diag_right / "
                "point_cloud_center rig sensors (then S or close viewer to save)."
            )
        print(_cam_help)
    viewer_cam_cache = {"pos": None, "dir": None}
    gym.set_light_parameters(sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                    gymapi.Vec3(-1.0, 0.0, 0.0))
    gym.set_light_parameters(sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0),
                                    gymapi.Vec3(1.0, 0.0, 0.0))
    #*************************************************************************************************#

    # empty loop here to test functionalities
    #*************************************************************************************************#
    real_position = False
    flex_collision_models = []
    object_mesh = []
    trajectory_joint_configs = []
    t = 0
    #while not gym.query_viewer_has_closed(viewer):
    for t in range(200):
        if not real_position:
            # Match reference.py initialization pose.
            gym.set_dof_target_position(envs[-1], spj, 0.7)
            gym.set_dof_target_position(envs[-1], slj, -2.0)
            gym.set_dof_target_position(envs[-1], ej,  2.5)
            gym.set_dof_target_position(envs[-1], wj1, -0.3)
            gym.set_dof_target_position(envs[-1], wj2, 0.7)
            gym.set_dof_target_position(envs[-1], wj3, 0)
            real_position = True

        if t == 199:
            for i in range(len(object_handles)):
                element = object_handles[i]
                states = gym.get_actor_rigid_body_states(envs[-1], element, 1)
                rotation = np.array(states[0][0][1])
                translation = np.array(states[0][0][0])
                rotation = np.array(rotation.item())
                translation = np.array(translation.item())
                object_status_list[i][0] += translation
                r1 = R.from_quat(rotation)
                tf = fcl.Transform(r1.as_matrix(), translation)
                flex_collision_models.append([fcl.CollisionObject(object_collision_lib[i], tf), 0])

                temp_obj = object_reader_tracker[i]
                temp_obj.set_offset(translation)
                all_lines = [] 
                vertices, faces = temp_obj.get_bounding_box_mesh()
                object_mesh.append([vertices, faces])
                for v1, v2, v3 in faces:
                    all_lines += list(vertices[v1])
                    all_lines += list(vertices[v2])
                    all_lines += list(vertices[v1])
                    all_lines += list(vertices[v3])
                    all_lines += list(vertices[v2])
                    all_lines += list(vertices[v3])

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        dof_states = gym.get_actor_dof_states(envs[-1], ur5e_handles[-1], gymapi.STATE_POS)
        q_step = np.array(dof_states["pos"][:6], dtype=np.float64)
        trajectory_joint_configs.append(q_step)

        # update the viewer
        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            handle_viewer_camera_input(gym, viewer, envs[-1])
            maybe_log_viewer_camera_on_move(gym, viewer, envs[-1], viewer_cam_cache, camera_props)
            gym.sync_frame_time(sim)

    #*************************************************************************************************#
    # NTField metric data (train_arm.py): FCL-only pairs, no RRT — same clutter as this Isaac run.
    #*************************************************************************************************#
    if getattr(args, "simple_grasp_collect", False):
        scene_info_simple = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        rac_simple = RC.robot_arm_configuration(
            "./assets/urdf/ur5e/meshes/collision/",
            np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z], dtype=np.float64),
            scene_info_simple,
            ik_urdf_file="ur5e_mimic_real_gripper_test.urdf",
        )

        dof_states = gym.get_actor_dof_states(envs[-1], ur5e_handles[-1], gymapi.STATE_POS)
        q_start = np.array(dof_states["pos"][:6], dtype=np.float64)
        target_idx = 0
        grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[target_idx]].split("/")[:-1]) + "/grasp_dict.npy"
        grasp_data = np.load(grasp_file, allow_pickle=True)

        candidate_indices = np.arange(len(grasp_data), dtype=np.int64)
        np.random.shuffle(candidate_indices)
        max_trials = min(int(args.simple_num_candidates), int(len(candidate_indices)))

        flex_collision_objects = [
            entry[0] if isinstance(entry, (list, tuple)) and len(entry) > 0 else entry
            for entry in flex_collision_models
        ]

        qg_candidates = []
        for grasp_idx in candidate_indices[:max_trials]:
            target_grasp_pos = np.array(grasp_data[grasp_idx]["target_pos"], dtype=np.float64)
            target_grasp_quat = np.array(grasp_data[grasp_idx]["target_quat"], dtype=np.float64)
            target_grasp_pos[:2] = target_grasp_pos[:2] + np.array(GT_OBJ_POS_LIST[target_idx][:2], dtype=np.float64)
            qg = rac_simple.grasp_verify(target_grasp_pos, target_grasp_quat)
            if qg is None:
                continue
            qg = np.array(qg[:6], dtype=np.float64)
            if not rac_simple.arm_collision_free(
                qg.tolist(), plane_obj, object_collision_models, flex_collision_objects
            ):
                continue
            qg_candidates.append((int(grasp_idx), qg))

        reachable_paths = []
        for grasp_idx, qg in qg_candidates:
            path = straight_line_collision_free_path(
                rac_simple,
                q_start,
                qg,
                plane_obj,
                object_collision_models,
                flex_collision_objects,
                num_steps=int(args.simple_interp_steps),
            )
            if path is None:
                continue
            dist = float(np.linalg.norm(qg - q_start))
            reachable_paths.append((dist, grasp_idx, qg, path))

        if not reachable_paths:
            print(
                f"[simple_grasp_collect] No reachable q_g found. "
                f"candidates_tested={max_trials}, ik_collisionfree_candidates={len(qg_candidates)}"
            )
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            sys.exit(1)

        reachable_paths.sort(key=lambda x: x[0])
        best_dist, best_grasp_idx, best_qg, best_path = reachable_paths[0]
        best_path_arr = np.array(best_path, dtype=np.float32)

        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        simple_out_dir = os.path.join(
            output_root,
            "simple_grasp_collect",
            run_ts,
        )
        os.makedirs(simple_out_dir, exist_ok=True)
        path_file = os.path.join(simple_out_dir, "qs_to_reachable_qg_path.npy")
        meta_file = os.path.join(simple_out_dir, "meta.npz")
        np.save(path_file, best_path_arr)
        np.savez(
            meta_file,
            q_start=q_start.astype(np.float32),
            q_goal=best_qg.astype(np.float32),
            chosen_grasp_idx=np.array([best_grasp_idx], dtype=np.int32),
            qg_l2_distance=np.array([best_dist], dtype=np.float32),
            num_candidates_requested=np.array([int(args.simple_num_candidates)], dtype=np.int32),
            num_candidates_tested=np.array([max_trials], dtype=np.int32),
            num_ik_collisionfree_candidates=np.array([len(qg_candidates)], dtype=np.int32),
            num_reachable_candidates=np.array([len(reachable_paths)], dtype=np.int32),
        )
        print(
            f"[simple_grasp_collect] saved path to {path_file}, meta to {meta_file}. "
            f"tested={max_trials}, ik_collisionfree={len(qg_candidates)}, reachable={len(reachable_paths)}, "
            f"chosen_grasp_idx={best_grasp_idx}, path_len={best_path_arr.shape[0]}"
        )

        if viewer is not None:
            for q in best_path:
                gym.set_dof_target_position(envs[-1], spj, float(q[0]))
                gym.set_dof_target_position(envs[-1], slj, float(q[1]))
                gym.set_dof_target_position(envs[-1], ej, float(q[2]))
                gym.set_dof_target_position(envs[-1], wj1, float(q[3]))
                gym.set_dof_target_position(envs[-1], wj2, float(q[4]))
                gym.set_dof_target_position(envs[-1], wj3, float(q[5]))
                gym.simulate(sim)
                gym.fetch_results(sim, True)
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        sys.exit(0)

    if getattr(args, "collect_ntfield_metric", False):
        if getattr(args, "ntfield", False):
            print("Error: use --collect_ntfield_metric without --ntfield (metric collection skips learned planning).")
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            sys.exit(1)
        scene_info_metric = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        file_path_metric = "./assets/urdf/ur5e/meshes/collision/"
        rac_metric = RC.robot_arm_configuration(
            file_path_metric,
            np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z], dtype=np.float64),
            scene_info_metric,
            ik_urdf_file=str(args.metric_ik_urdf_file),
        )
        from ntfield_metric_collect_fcl import collect_metric_dataset

        metric_out = getattr(args, "metric_output_dir", None)
        if metric_out:
            metric_out = os.path.abspath(metric_out)
        else:
            metric_out = os.path.join(output_root, f"env_{env_id}_ntfield_metric")
        os.makedirs(metric_out, exist_ok=True)

        metric_visualize_cb = None
        if getattr(args, "metric_visualize_sampling", False):
            if viewer is None:
                print("Warning: --metric_visualize_sampling requested but viewer is unavailable (headless).")
            else:
                hold_steps = max(int(getattr(args, "metric_visualize_hold_steps", 20)), 1)

                def metric_visualize_cb(q_start, q_goal, accepted, tries):
                    def _set_q(q):
                        gym.set_dof_target_position(envs[-1], spj, float(q[0]))
                        gym.set_dof_target_position(envs[-1], slj, float(q[1]))
                        gym.set_dof_target_position(envs[-1], ej,  float(q[2]))
                        gym.set_dof_target_position(envs[-1], wj1, float(q[3]))
                        gym.set_dof_target_position(envs[-1], wj2, float(q[4]))
                        gym.set_dof_target_position(envs[-1], wj3, float(q[5]))

                    states_to_show = [q_start]
                    if q_goal is not None:
                        states_to_show.append(q_goal)

                    for q in states_to_show:
                        _set_q(q)
                        for _ in range(hold_steps):
                            gym.simulate(sim)
                            gym.fetch_results(sim, True)
                            gym.step_graphics(sim)
                            gym.draw_viewer(viewer, sim, True)
                            handle_viewer_camera_input(gym, viewer, envs[-1])
                            maybe_log_viewer_camera_on_move(gym, viewer, envs[-1], viewer_cam_cache, camera_props)
                            gym.sync_frame_time(sim)
                    print(
                        f"[metric_visualize] accepted={accepted}, tries={tries}, "
                        f"showing {'q_start -> q_goal' if q_goal is not None else 'q_start'}",
                        flush=True,
                    )

        collect_metric_dataset(
            rac_metric,
            plane_obj,
            object_collision_models,
            flex_collision_models,
            num_samples=int(args.metric_num_samples),
            output_dir=metric_out,
            seed=int(args.metric_seed),
            max_tries_factor=int(args.metric_max_tries_factor),
            sampler_mode=str(args.metric_sampler_mode),
            obstacle_meshes=object_mesh,
            ik_pose_trials=int(args.metric_ik_pose_trials),
            ik_seed_trials=int(args.metric_ik_seed_trials),
            ik_surface_offset_min=float(args.metric_ik_surface_offset_min),
            ik_surface_offset_max=float(args.metric_ik_surface_offset_max),
            ik_tool_offset_xyz=(
                float(args.metric_ik_tool_offset_x),
                float(args.metric_ik_tool_offset_y),
                float(args.metric_ik_tool_offset_z),
            ),
            log_every_tries=int(args.metric_log_every_tries),
            visualize_callback=metric_visualize_cb,
            visualize_every_accepted=int(args.metric_visualize_every_accepted),
            qstart_only=bool(args.metric_qstart_only),
            save_speed_normal=bool(args.metric_save_speed_normal),
        )
        print(f"NTField metric dataset saved under: {metric_out}")
        print("Train with: cd ntrl-demo && python train/train_arm.py  (set model_train_metric DataPath to this folder or copy the three .npy files into datasets/arm/UR5/).")
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        sys.exit(0)

    #*************************************************************************************************#

    if getattr(args, 'camera_capture_only', False):
        print(
            "Camera capture-only mode: logs [ViewerCamera env-local] on each camera move; "
            "keys 1/2/3 assign viewer eye/target to rig cameras; "
            "on viewer close saves rig RGB to env_*_scene_{diag_left,diag_right,center}_views.npz and matching .png."
        )
        if viewer is None:
            print("Camera capture-only mode requires viewer (do not use --headless). Exiting.")
            gym.destroy_sim(sim)
            sys.exit(0)

        _rig_from_viewer_slots = {
            "diag_left": (diag_left_cam_handles[-1], diag_left_cam_pos, diag_left_cam_target),
            "diag_right": (diag_right_cam_handles[-1], diag_right_cam_pos, diag_right_cam_target),
            "point_cloud_center": (
                point_cloud_center_cam_handles[-1],
                point_cloud_center_cam_pos,
                point_cloud_center_cam_target,
            ),
        }

        def _save_rig_snapshots():
            save_rig_camera_outputs(
                gym,
                sim,
                envs[-1],
                output_root,
                env_id,
                camera_props,
                diag_left_cam_handles[-1],
                diag_right_cam_handles[-1],
                point_cloud_center_cam_handles[-1],
                diag_left_cam_pos,
                diag_left_cam_target,
                diag_right_cam_pos,
                diag_right_cam_target,
                point_cloud_center_cam_pos,
                point_cloud_center_cam_target,
                tag=None,
            )

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            handle_viewer_camera_input(
                gym,
                viewer,
                envs[-1],
                rig_snapshot_callback=_save_rig_snapshots,
                rig_from_viewer_slots=_rig_from_viewer_slots,
            )
            maybe_log_viewer_camera_on_move(gym, viewer, envs[-1], viewer_cam_cache, camera_props)
            gym.sync_frame_time(sim)
        # Refresh sim/graphics for sensor capture (avoid draw_viewer here: window may already be invalid).
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        print("Camera capture completed; saving rig camera snapshots…", flush=True)
        try:
            _save_rig_snapshots()
        except Exception as e:
            print(f"Error saving rig camera snapshots: {e}", flush=True)
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        sys.exit(0)

    robot_path = None
    saved_q_s = np.full((6,), np.nan, dtype=np.float64)
    saved_q_g = np.full((6,), np.nan, dtype=np.float64)
    saved_grasp = np.full((7,), np.nan, dtype=np.float64)  # [x,y,z,qx,qy,qz,qw]
    saved_language = ""
    saved_speed_qs = np.full((6,), np.nan, dtype=np.float64)
    saved_speed_qg = np.full((6,), np.nan, dtype=np.float64)
    trajectory_joint_configs = []
    diag_left_rgb = None
    diag_right_rgb = None
    point_cloud_center_rgb = None
    diag_left_cam_pose = None
    diag_left_cam_focus = None
    diag_right_cam_pose = None
    diag_right_cam_focus = None
    point_cloud_center_cam_pose = None
    point_cloud_center_cam_focus = None
    if getattr(args, 'ntfield', False):
        # NTField mode: plan path with trained model and animate in this environment
        import h5py
        import torch

        # Ensure ntrl-demo is on path
        _ntfield_ntrl = ntrl_demo_path if os.path.isdir(ntrl_demo_path) else os.path.normpath(os.path.join(pi_vla_root, '..', 'ntrl-demo'))
        if not os.path.isdir(_ntfield_ntrl):
            print(f"Error: ntrl-demo not found. Tried: {ntrl_demo_path}")
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            sys.exit(1)
        if _ntfield_ntrl not in sys.path:
            sys.path.insert(0, _ntfield_ntrl)
        from models.metric_arm import model_test_metric as md
        from planning import plan as gradient_plan

        checkpoint_path = os.path.abspath(args.checkpoint)
        if not args.checkpoint or not os.path.isfile(checkpoint_path):
            print("Error: --ntfield requires --checkpoint with a valid .pt file")
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            sys.exit(1)
        if args.h5_path:
            _h5 = args.h5_path
            if not os.path.isfile(_h5):
                _c0 = os.path.join(pi_vla_root, _h5)
                if os.path.isfile(_c0):
                    _h5 = _c0
                else:
                    _c1 = os.path.join(file_dir, _h5)
                    if os.path.isfile(_c1):
                        _h5 = _c1
            if not os.path.isfile(_h5):
                print(f"Error: HDF5 not found: {args.h5_path} (tried PI-VLA root and hanwen_grasping)")
                if viewer is not None:
                    gym.destroy_viewer(viewer)
                gym.destroy_sim(sim)
                sys.exit(1)
            args.h5_path = os.path.abspath(_h5)
            with h5py.File(args.h5_path, "r") as f:
                joint_configs = np.array(f["joint_configs"][:], dtype=np.float64)
                q_start = np.array(joint_configs[0], dtype=np.float64)
                if "final_joint_config" in f:
                    q_goal = np.array(f["final_joint_config"][:], dtype=np.float64)
                else:
                    q_goal = np.array(joint_configs[-1], dtype=np.float64)
            print(f"Using q_start/q_goal from HDF5: {args.h5_path}")
            # Auto path: output/trajectory_evaluation/YYYYMMDD_HHMMSS/ntfield.mp4 when --record and default output
            if getattr(args, 'record', False) and getattr(args, 'record_output', None) == 'ntfield_record.mp4':
                _, session_dir = h5viz.trajectory_evaluation_session_dir(pi_vla_root, args.h5_path)
                if session_dir:
                    os.makedirs(session_dir, exist_ok=True)
                    args.record_output = os.path.join(session_dir, "ntfield.mp4")
                    args._session_eval_dir = session_dir
                else:
                    basename = os.path.basename(args.h5_path)
                    m = re.search(r'(\d{8}_\d{6})', basename)
                    if m:
                        args.record_output = f"ntfield_{m.group(1)}.mp4"
        else:
            if not args.q_start or not args.q_goal:
                print("Error: --ntfield requires --q_start and --q_goal, or --h5_path")
                if viewer is not None:
                    gym.destroy_viewer(viewer)
                gym.destroy_sim(sim)
                sys.exit(1)
            q_start = np.array(parse_q_ntfield(args.q_start), dtype=np.float64)
            q_goal = np.array(parse_q_ntfield(args.q_goal), dtype=np.float64)
            # Auto-name video from end config when --record and default output
            if getattr(args, 'record', False) and getattr(args, 'record_output', None) == 'ntfield_record.mp4':
                q_str = "_".join(f"{x:.2f}" for x in q_goal)
                args.record_output = f"ntfield_q_{q_str}.mp4"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.dirname(checkpoint_path)
        data_path = os.path.join(_ntfield_ntrl, "datasets", "arm", "UR5_trajectory")
        if not os.path.isdir(data_path):
            _alt_dp = os.path.join(_ntfield_ntrl, "datasets", "arm", "UR5_trajectory_vertical_train")
            if os.path.isdir(_alt_dp):
                data_path = _alt_dp
        model = md.Model(model_path, data_path, dim=6, source=[0.0] * 6, device=device)
        model.load(checkpoint_path)
        model.network.eval()

        path = gradient_plan(model, q_start, q_goal, step_size=0.02, max_steps=200, tol=0.01, device=device)
        robot_path = interpolate_path_ntfield(path, steps_between=4)
        saved_q_s = np.array(q_start, dtype=np.float64)
        saved_q_g = np.array(q_goal, dtype=np.float64)
        saved_language = str(_nt_h5_prompt) if _nt_h5_prompt else ""
        _h, _w = int(camera_props.height), int(camera_props.width)
        diag_left_rgb = capture_camera_rgb_from_sensor(
            gym, sim, envs[-1], diag_left_cam_handles[-1], _h, _w
        )
        diag_right_rgb = capture_camera_rgb_from_sensor(
            gym, sim, envs[-1], diag_right_cam_handles[-1], _h, _w
        )
        point_cloud_center_rgb = capture_camera_rgb_from_sensor(
            gym, sim, envs[-1], point_cloud_center_cam_handles[-1], _h, _w
        )
        diag_left_cam_pose = np.array([diag_left_cam_pos.x, diag_left_cam_pos.y, diag_left_cam_pos.z], dtype=np.float64)
        diag_left_cam_focus = np.array([diag_left_cam_target.x, diag_left_cam_target.y, diag_left_cam_target.z], dtype=np.float64)
        diag_right_cam_pose = np.array([diag_right_cam_pos.x, diag_right_cam_pos.y, diag_right_cam_pos.z], dtype=np.float64)
        diag_right_cam_focus = np.array([diag_right_cam_target.x, diag_right_cam_target.y, diag_right_cam_target.z], dtype=np.float64)
        point_cloud_center_cam_pose = np.array(
            [point_cloud_center_cam_pos.x, point_cloud_center_cam_pos.y, point_cloud_center_cam_pos.z], dtype=np.float64
        )
        point_cloud_center_cam_focus = np.array(
            [point_cloud_center_cam_target.x, point_cloud_center_cam_target.y, point_cloud_center_cam_target.z], dtype=np.float64
        )
        print(f"NTField planned {len(path)} waypoints, interpolated to {len(robot_path)}")

    if robot_path is None:
        # Grasp planning mode (original flow)
        ik_solver2 = IK("base_link", "wrist_3_link", urdf_string = urdf_str)
        target_quat = gymapi.Quat(0.446, 0.560, -0.433, 0.549)
        converted_quat = quaternion_multiply(gymapi.Quat(-math.sqrt(2)/2, 0, 0, math.sqrt(2)/2), target_quat)
        file_path = './assets/urdf/ur5e/meshes/collision/'

        scene_info = [table_dims.x, table_dims.y, table_dims.z, drawer_height]
        rac = RC.robot_arm_configuration(file_path, np.array([ur5e_pose.p.x, ur5e_pose.p.y, ur5e_pose.p.z]), scene_info)
        seed_state = [0.0]*ik_solver2.number_of_joints

        # Rebuild object meshes from current actor poses to keep indices aligned with user-selected target_idx.
        object_mesh = []
        for i_obj, handle in enumerate(object_handles):
            states = gym.get_actor_rigid_body_states(envs[-1], handle, 1)
            translation = np.array(states[0][0][0]).item()
            temp_obj = object_reader_tracker[i_obj]
            temp_obj.set_offset(np.array(translation))
            vertices, faces = temp_obj.get_bounding_box_mesh()
            object_mesh.append([vertices, faces])

        # Save one feasible goal joint configuration per object (if found).
        goal_joint_configs = np.full((NUM_OF_OBJECTS, 6), np.nan, dtype=np.float64)
        goal_found_mask = np.zeros((NUM_OF_OBJECTS,), dtype=np.int32)
        object_prompts = []
        for obj_idx in range(NUM_OF_OBJECTS):
            object_prompts.append(object_prompt_from_asset(object_asset_files[target_file_idx[obj_idx]]))
            grasp_file_obj = "./assets/" + "/".join(object_asset_files[target_file_idx[obj_idx]].split("/")[:-1]) + "/grasp_dict.npy"
            grasp_data_obj = np.load(grasp_file_obj, allow_pickle=True)
            for grasp_idx in range(len(grasp_data_obj)):
                target_grasp_pos = np.array(grasp_data_obj[grasp_idx]['target_pos'], dtype=np.float64)
                target_grasp_quat = grasp_data_obj[grasp_idx]['target_quat']
                target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[obj_idx][:2]
                q_goal_candidate = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
                q_lift_candidate = rac.grasp_verify(target_grasp_pos + [0, 0, 0.01], target_grasp_quat)
                if q_goal_candidate is None or q_lift_candidate is None:
                    continue
                goal_collision_free = rac.arm_collision_free(q_goal_candidate, plane_obj, object_collision_models, [])
                lift_collision_free = rac.arm_collision_free(q_lift_candidate, plane_obj, object_collision_models, [])
                if goal_collision_free and lift_collision_free:
                    goal_joint_configs[obj_idx] = np.array(q_goal_candidate, dtype=np.float64)
                    goal_found_mask[obj_idx] = 1
                    break

        # Capture multi-view scene images at save time (env-local camera poses; step_graphics + sensor render).
        (
            diag_left_rgb,
            diag_right_rgb,
            point_cloud_center_rgb,
            diag_left_cam_pose,
            diag_left_cam_focus,
            diag_right_cam_pose,
            diag_right_cam_focus,
            point_cloud_center_cam_pose,
            point_cloud_center_cam_focus,
        ) = save_rig_camera_outputs(
            gym,
            sim,
            envs[-1],
            output_root,
            env_id,
            camera_props,
            diag_left_cam_handles[-1],
            diag_right_cam_handles[-1],
            point_cloud_center_cam_handles[-1],
            diag_left_cam_pos,
            diag_left_cam_target,
            diag_right_cam_pos,
            diag_right_cam_target,
            point_cloud_center_cam_pos,
            point_cloud_center_cam_target,
            tag=None,
        )

        goal_cfg_path = os.path.join(output_root, f'env_{env_id}_all_goal_configs.npz')
        np.savez(
            goal_cfg_path,
            goal_joint_configs=goal_joint_configs,
            goal_found_mask=goal_found_mask,
            target_file_idx=np.array(target_file_idx, dtype=np.int32),
            gt_obj_xy=np.array(GT_OBJ_POS_LIST, dtype=np.float64),
            object_prompts=np.array(object_prompts, dtype=object),
            diag_left_cam_pos=diag_left_cam_pose,
            diag_left_cam_target=diag_left_cam_focus,
            diag_right_cam_pos=diag_right_cam_pose,
            diag_right_cam_target=diag_right_cam_focus,
        )
        print(f"Saved per-object goal configs to {goal_cfg_path} (found {int(goal_found_mask.sum())}/{NUM_OF_OBJECTS})")
        print(
            f"Saved scene images to "
            f"{os.path.join(output_root, f'env_{env_id}_scene_diag_left_views.png')}, "
            f"{os.path.join(output_root, f'env_{env_id}_scene_diag_right_views.png')}, "
            f"{os.path.join(output_root, f'env_{env_id}_scene_center_views.png')}"
        )

        if getattr(args, 'skip_path_planning', False):
            print("Skipping q_start->q_goal path planning (--skip_path_planning).")
        else:
            print("Input object index to grasp")
            target_idx = int(input(f"Choose from 0 to {NUM_OF_OBJECTS - 1}: "))
            saved_q_s = np.array([0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0], dtype=np.float64)
            saved_language = object_prompts[target_idx]

            grasp_file = "./assets/" + "/".join(object_asset_files[target_file_idx[target_idx]].split("/")[:-1]) + "/grasp_dict.npy"
            grasp_data = np.load(grasp_file, allow_pickle=True)

            # generate grasp
            num_grasp = 0
            swept_size = sys.maxsize
            grasp_list = np.arange(len(grasp_data))
            np.random.shuffle(grasp_list)

            swept_volume1 = None
            swept_volume2 = None
            init2grasp_path = None
            grasp2init_path = None
            for grasp_idx in grasp_list:
                target_grasp_pos = grasp_data[grasp_idx]['target_pos']
                target_grasp_quat = grasp_data[grasp_idx]['target_quat']
                target_grasp_pos[:2] = target_grasp_pos[:2] + GT_OBJ_POS_LIST[target_idx][:2]
                init2grasp_angels_temp = rac.grasp_verify(target_grasp_pos, target_grasp_quat)
                grasp2init_angels_temp = rac.grasp_verify(target_grasp_pos + [0,0,0.01], target_grasp_quat)
                if init2grasp_angels_temp is None or grasp2init_angels_temp is None:
                    print("skip imposible grasp")
                    continue

                init2grasp_collision = rac.arm_collision_free(init2grasp_angels_temp, plane_obj, object_collision_models, [])
                grasp2init_collision = rac.arm_collision_free(grasp2init_angels_temp, plane_obj, object_collision_models, [])
                print("check collision init grasp")
                print(init2grasp_collision, grasp2init_collision)
                if not init2grasp_collision or not grasp2init_collision:
                    print("skip collision grasp")
                    continue

                init2grasp_path_temp = RC.get_path2grasp(rac, init2grasp_angels_temp, scene_info, target_mesh=object_mesh[target_idx], time_limit=30, given_static_model = object_collision_models)
                print(init2grasp_path_temp)
                if init2grasp_path_temp is None:
                    print("No path generated\n")
                    continue

                temp_mod_bbox = rac.modify_grasp_bbox(init2grasp_angels_temp, target_mesh=object_mesh[target_idx], visualize=False)
                grasp2init_path_temp = RC.get_path2start(rac, grasp2init_angels_temp, temp_mod_bbox, scene_info, time_limit=30, given_static_model = object_collision_models)
                print(grasp2init_path_temp)
                if grasp2init_path_temp is None:
                    print("No path generated\n")
                    continue
                swept_volume1_temp, swept_verts1_temp = rac.get_swept_volume(init2grasp_path_temp, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                swept_volume2_temp, swept_verts2_temp = rac.get_swept_volume(grasp2init_path_temp, w_target=temp_mod_bbox, frame_rate=60, scene_info=scene_info, animation=False, static_vi=False)
                num_grasp += 1
                is_target_detected = True
                # compare swept volumes
                swept_center_temp, swept_verts_temp = rac.get_swept_center(swept_verts1_temp+swept_verts2_temp, scene_info, 0.6)
                temp_swept_size = get_swept_volume_size(swept_verts_temp)
                if temp_swept_size < swept_size:
                    swept_size = temp_swept_size
                    swept_volume1 = swept_volume1_temp
                    swept_volume2 = swept_volume2_temp
                    init2grasp_path = init2grasp_path_temp
                    grasp2init_path = grasp2init_path_temp
                    swept_center = swept_center_temp
                    swept_verts = swept_verts_temp
                    W_TARGET = temp_mod_bbox
                    saved_q_g = np.array(init2grasp_angels_temp, dtype=np.float64)
                    saved_grasp = np.array(
                        [
                            target_grasp_pos[0],
                            target_grasp_pos[1],
                            target_grasp_pos[2],
                            target_grasp_quat[0],
                            target_grasp_quat[1],
                            target_grasp_quat[2],
                            target_grasp_quat[3],
                        ],
                        dtype=np.float64,
                    )
                if num_grasp == 1:
                    break
            print("\n!!!!!!!!!!!!!!!!!!!", num_grasp ,'grasp generated!!!!!!!!!!!!!!!!!!!!!!!\n')
            robot_path = init2grasp_path

    if robot_path is None or len(robot_path) == 0:
        if getattr(args, 'skip_path_planning', False) and not getattr(args, 'ntfield', False):
            print("Path planning skipped; exiting after saving goal configurations.")
            print('Test Completed Successfully!!')
            if viewer is not None:
                gym.destroy_viewer(viewer)
            gym.destroy_sim(sim)
            sys.exit(0)
        print("Error: No path to animate (grasp planning failed or --ntfield path empty)")
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)
        sys.exit(1)

    path_id_box = [0]
    record_frames = [] if getattr(args, 'record', False) else None
    headless_hold_frames = 120

    def _animate_robot_path_step():
        path_id = path_id_box[0]
        if path_id >= len(robot_path):
            path_id -= 1
        dof_result = robot_path[path_id]

        gym.set_dof_target_position(envs[-1], spj, dof_result[0])
        gym.set_dof_target_position(envs[-1], slj, dof_result[1])
        gym.set_dof_target_position(envs[-1], ej,  dof_result[2])
        gym.set_dof_target_position(envs[-1], wj1, dof_result[3])
        gym.set_dof_target_position(envs[-1], wj2, dof_result[4])
        gym.set_dof_target_position(envs[-1], wj3, dof_result[5])

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.step_graphics(sim)
        if viewer is not None:
            gym.draw_viewer(viewer, sim, True)
            handle_viewer_camera_input(gym, viewer, envs[-1])
            maybe_log_viewer_camera_on_move(gym, viewer, envs[-1], viewer_cam_cache, camera_props)
            gym.sync_frame_time(sim)

        if record_frames is not None:
            gym.render_all_camera_sensors(sim)
            raw = gym.get_camera_image(sim, envs[-1], diag_left_cam_handles[-1], gymapi.IMAGE_COLOR)
            rgba = raw.reshape(camera_props.height, camera_props.width, 4)
            rgb = rgba[..., :3].copy()
            record_frames.append(rgb)

        path_id_box[0] = path_id + 1

    if getattr(args, 'headless', False):
        for _ in range(len(robot_path) + headless_hold_frames):
            _animate_robot_path_step()
    else:
        while not gym.query_viewer_has_closed(viewer):
            _animate_robot_path_step()

    if len(trajectory_joint_configs) > 1:
        saved_speed_qs, saved_speed_qg = estimate_endpoint_joint_speeds(trajectory_joint_configs, sim_params.dt)
    elif robot_path is not None and len(robot_path) > 1:
        saved_speed_qs, saved_speed_qg = estimate_endpoint_joint_speeds(robot_path, sim_params.dt)

    pointcloud_bundle_path = os.path.join(output_root, f'env_{env_id}_pointcloud_capture.npz')
    np.savez(
        pointcloud_bundle_path,
        q_s=saved_q_s,
        q_g=saved_q_g,
        grasp=saved_grasp,
        speed_qs=saved_speed_qs,
        speed_qg=saved_speed_qg,
        language=np.array(saved_language, dtype=object),
        trajectory_joint_configs=np.array(trajectory_joint_configs, dtype=np.float64),
        diag_right_rgb=diag_right_rgb,
        diag_left_rgb=diag_left_rgb,
        point_cloud_center_rgb=point_cloud_center_rgb,
        diag_right_cam_pos=diag_right_cam_pose,
        diag_right_cam_target=diag_right_cam_focus,
        diag_left_cam_pos=diag_left_cam_pose,
        diag_left_cam_target=diag_left_cam_focus,
        point_cloud_center_cam_pos=point_cloud_center_cam_pose,
        point_cloud_center_cam_target=point_cloud_center_cam_focus,
    )
    print(f"Saved point-cloud capture bundle to {pointcloud_bundle_path}")

    # HDF5 export with collect_data-compatible naming style.
    pointcloud_h5_path = os.path.join(output_root, f'env_{env_id}_pointcloud_capture.h5')
    try:
        import h5py

        str_dt = h5py.special_dtype(vlen=str)
        traj_arr = np.array(trajectory_joint_configs, dtype=np.float32)
        goal_joint_configs = np.array(saved_q_g, dtype=np.float32).reshape(1, 6)
        grasp_target_positions = np.array(saved_grasp[:3], dtype=np.float32).reshape(1, 3)
        grasp_target_quaternions = np.array(saved_grasp[3:], dtype=np.float32).reshape(1, 4)
        speed_qs_arr = np.array(saved_speed_qs, dtype=np.float32)
        speed_qg_arr = np.array(saved_speed_qg, dtype=np.float32)
        q_s_arr = np.array(saved_q_s, dtype=np.float32)
        q_g_arr = np.array(saved_q_g, dtype=np.float32)

        with h5py.File(pointcloud_h5_path, "w") as f:
            # Keep collect_data-style core names where applicable.
            f.create_dataset("trajectory_joint_configs", data=traj_arr)
            f.create_dataset("goal_joint_configs", data=goal_joint_configs)
            f.create_dataset("grasp_target_positions", data=grasp_target_positions)
            f.create_dataset("grasp_target_quaternions", data=grasp_target_quaternions)

            # Extra constrained-task metadata requested by user.
            f.create_dataset("q_s", data=q_s_arr)
            f.create_dataset("q_g", data=q_g_arr)
            f.create_dataset("speed_qs", data=speed_qs_arr)
            f.create_dataset("speed_qg", data=speed_qg_arr)
            f.create_dataset("diag_left_rgb", data=np.array(diag_left_rgb, dtype=np.uint8), compression="gzip")
            f.create_dataset("diag_right_rgb", data=np.array(diag_right_rgb, dtype=np.uint8), compression="gzip")
            f.create_dataset("point_cloud_center_rgb", data=np.array(point_cloud_center_rgb, dtype=np.uint8), compression="gzip")
            f.create_dataset("diag_left_cam_pos", data=np.array(diag_left_cam_pose, dtype=np.float32))
            f.create_dataset("diag_left_cam_target", data=np.array(diag_left_cam_focus, dtype=np.float32))
            f.create_dataset("diag_right_cam_pos", data=np.array(diag_right_cam_pose, dtype=np.float32))
            f.create_dataset("diag_right_cam_target", data=np.array(diag_right_cam_focus, dtype=np.float32))
            f.create_dataset("point_cloud_center_cam_pos", data=np.array(point_cloud_center_cam_pose, dtype=np.float32))
            f.create_dataset("point_cloud_center_cam_target", data=np.array(point_cloud_center_cam_focus, dtype=np.float32))
            lang_ds = f.create_dataset("language", (1,), dtype=str_dt)
            lang_ds[:] = np.array([saved_language], dtype=object)

            f.attrs["joint_dim"] = 6
            f.attrs["num_objects"] = 1
            f.attrs["camera_count"] = 3
            f.flush()
        print(f"Saved point-cloud capture HDF5 to {pointcloud_h5_path}")
    except Exception as e:
        print(f"Warning: could not save HDF5 capture file: {e}")

    # save recorded video
    if record_frames and len(record_frames) > 0:
        out_path = getattr(args, 'record_output', 'ntfield_record.mp4')
        out_path = os.path.abspath(out_path)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        saved = False
        try:
            import imageio
            imageio.mimsave(out_path, record_frames, fps=60)
            print(f"Saved video to {out_path}")
            saved = True
        except (ImportError, ValueError) as e:
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
            print("Install imageio[ffmpeg] or opencv-python for video recording: pip install imageio[ffmpeg]")
        if saved:
            sess = getattr(args, '_session_eval_dir', None)
            if sess and getattr(args, 'ntfield', False) and args.h5_path:
                ckpt = os.path.abspath(args.checkpoint) if args.checkpoint else ""
                extra = [
                    f"ntfield_checkpoint: {ckpt}",
                    f"video_ntfield: ntfield.mp4",
                ]
                meta_path = os.path.join(sess, "episode_meta.txt")
                if os.path.isfile(meta_path):
                    h5viz.append_session_meta(sess, extra)
                else:
                    h5viz.write_session_meta(sess, [
                        f"h5_path: {args.h5_path}",
                        f"prompt: {_nt_h5_prompt}",
                        f"object_location (h5): {_nt_h5_ol}",
                        f"object_index (scene): {_nt_h5_obj_idx}",
                    ] + extra)

    print('Test Completed Successfully!!')
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    sys.exit(0)
