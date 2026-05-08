"""
Collect NTField *metric* training data (train/train_arm.py) without OMPL / RRT.

Produces the same three arrays as ntrl-demo/dataprocessing/speed_sampling_arm_normal.py:
  - sampled_points.npy  (N, 12): [x_s, x_g] in normalized joint coords, x = q_rad / SCALE, |x|<=0.5
  - speed.npy           (N, 2):  clearance-based targets (see model_function_metric margin/offset)
  - normal.npy          (N, 12): joint-space direction from finite differences on min link–obstacle clearance

Sampling policy (matches user intent):
  - q_start: collision-free, min clearance to (table/walls/flex) in a *near-obstacle* band
  - q_goal:  collision-free, uniform in [-pi, pi]^6 (then clipped to the normalized box preimage)

Scene geometry must match arm_collision_free: static_env_models + flex_collision_models + ground plane.
"""

from __future__ import annotations

import math
import os
import time
from typing import List, Sequence, Tuple

import fcl
import numpy as np
from scipy.spatial.transform import Rotation as R

# Same as ntrl-demo trajectory_sampler / speed_sampling_arm_normal joint normalization.
SCALE = np.pi / 0.5

# Match models/metric_arm/model_function_metric.py Function.__init__
_LIMIT = 0.5
_MARGIN = _LIMIT / 15.0
_OFFSET = _MARGIN / 10.0

# Match arm_append_list clearance adjustment before the near-obstacle band.
_CLEARANCE_EPS = 0.01


def _as_fcl_collision_object(entry):
    """
    Accept either:
      - python-fcl CollisionObject
      - [CollisionObject, ...] / (CollisionObject, ...)
    and return the CollisionObject.
    """
    if isinstance(entry, (list, tuple)):
        if not entry:
            return None
        candidate = entry[0]
    else:
        candidate = entry
    if candidate is None:
        return None
    # duck typing for python-fcl collision object
    if hasattr(candidate, "getNodeType") or hasattr(candidate, "getObjectType"):
        return candidate
    return None


def _normalize_flex_collision_models(flex_collision_models: Sequence) -> list:
    out = []
    for entry in flex_collision_models:
        obj = _as_fcl_collision_object(entry)
        if obj is not None:
            out.append(obj)
    return out


def _global_coord_converter(coord1, coord2, coord3, offset1, offset2, offset3):
    return (coord1 - offset1, coord3 - offset3, -coord2 + offset2)


def _rotation_concat(quaternion1, quaternion0):
    x0, y0, z0, w0 = quaternion0[0], quaternion0[1], quaternion0[2], quaternion0[3]
    x1, y1, z1, w1 = quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3]
    return [
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
    ]


def _link_collision_objects(rac, q_rad: np.ndarray) -> List[fcl.CollisionObject]:
    pose_array = rac.calculate_transform_from_angles(q_rad.tolist())
    objs: List[fcl.CollisionObject] = []
    for t in range(9):
        rotation = np.array(pose_array[t][1])
        translation = np.array(pose_array[t][0])
        r1 = R.from_quat(rotation)
        tf = fcl.Transform(r1.as_matrix(), translation)
        objs.append(fcl.CollisionObject(rac.fcl_models_[t], tf))
    return objs


def min_link_env_clearance(
    rac,
    q_rad: np.ndarray,
    plane_obj: fcl.CollisionObject,
    static_env_models: Sequence[fcl.CollisionObject],
    flex_collision_models: Sequence,
) -> float:
    """
    Minimum FCL distance from arm links (0..7) to static env and flex objects.

    Note: python-fcl distance(…, Plane) can segfault on some builds; the ground plane is
    still enforced via rac.arm_collision_free(..., plane_obj, ...).
    """
    _ = plane_obj  # kept for API symmetry with arm_collision_free call sites
    links = _link_collision_objects(rac, q_rad)
    dmin = math.inf
    flex_objs = _normalize_flex_collision_models(flex_collision_models)
    for link_idx in range(8):
        link_obj = links[link_idx]
        for env_obj in static_env_models:
            req = fcl.DistanceRequest()
            res = fcl.DistanceResult()
            fcl.distance(link_obj, env_obj, req, res)
            dmin = min(dmin, float(res.min_distance))
        for flex_obj in flex_objs:
            req = fcl.DistanceRequest()
            res = fcl.DistanceResult()
            fcl.distance(link_obj, flex_obj, req, res)
            dmin = min(dmin, float(res.min_distance))
    return float(dmin)


def _speed_from_clearance(clearance_m: float) -> float:
    d_adj = clearance_m - _CLEARANCE_EPS
    return float(np.clip(d_adj, _OFFSET, _MARGIN) / _MARGIN)


def _grad_clearance_wrt_q(
    rac,
    q_rad: np.ndarray,
    plane_obj: fcl.CollisionObject,
    static_env_models: Sequence[fcl.CollisionObject],
    flex_collision_models: Sequence,
    eps: float = 2e-4,
) -> np.ndarray:
    """Central finite-difference gradient of min clearance w.r.t. joint angles (rad)."""
    g = np.zeros(6, dtype=np.float64)
    base = min_link_env_clearance(rac, q_rad, plane_obj, static_env_models, flex_collision_models)
    for i in range(6):
        dq = np.zeros(6, dtype=np.float64)
        dq[i] = eps
        dp = min_link_env_clearance(rac, q_rad + dq, plane_obj, static_env_models, flex_collision_models)
        dm = min_link_env_clearance(rac, q_rad - dq, plane_obj, static_env_models, flex_collision_models)
        g[i] = (dp - dm) / (2.0 * eps)
    # If base clearance is tiny, FD can be noisy; still return normalized direction.
    if not np.all(np.isfinite(g)):
        return np.zeros(6, dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 1e-12:
        return np.zeros(6, dtype=np.float64)
    return (g / n).astype(np.float64)


def _normal_in_normalized_coords(grad_q_unit: np.ndarray) -> np.ndarray:
    """
    training uses X in normalized coords x = q/SCALE.
    For scalar clearance c(q), dc/dx_i = dc/dq_i * SCALE.
    Align with speed_sampling convention: unit 6-vector in normalized space.
    """
    v = (grad_q_unit * SCALE).astype(np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.zeros(6, dtype=np.float64)
    return (v / n).astype(np.float64)


def _sample_point_and_normal_on_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    areas2 = np.linalg.norm(cross, axis=1)
    valid = areas2 > 1e-12
    if not np.any(valid):
        idx = int(rng.integers(0, tri.shape[0]))
    else:
        p = areas2.copy()
        p[~valid] = 0.0
        p /= p.sum()
        idx = int(rng.choice(np.arange(tri.shape[0]), p=p))

    a, b, c = tri[idx]
    u = float(rng.random())
    v = float(rng.random())
    su = math.sqrt(u)
    bary_a = 1.0 - su
    bary_b = su * (1.0 - v)
    bary_c = su * v
    point = bary_a * a + bary_b * b + bary_c * c

    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        n = n / nn
    if rng.random() < 0.5:
        n = -n
    return point.astype(np.float64), n.astype(np.float64)


def _quat_align_x_to_normal(normal: np.ndarray) -> np.ndarray:
    x_axis = normal / (np.linalg.norm(normal) + 1e-12)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x_axis, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis = np.cross(up, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R.from_matrix(rot).as_quat().astype(np.float64)


def _ik_from_target_pose(
    rac,
    target_pos: np.ndarray,
    target_quat_xyzw: np.ndarray,
    rng: np.random.Generator,
    ik_seed_trials: int,
    ik_tool_offset_xyz: Tuple[float, float, float],
) -> np.ndarray | None:
    # Match the conversion used in robot_arm_configuration.grasp_verify().
    r_rot = R.from_quat(target_quat_xyzw)
    cam_offset_vector = np.array(
        [float(ik_tool_offset_xyz[0]), float(ik_tool_offset_xyz[1]), float(ik_tool_offset_xyz[2])],
        dtype=np.float64,
    )
    rot_cam_offset_vector = r_rot.apply(cam_offset_vector)
    converted_coord = _global_coord_converter(
        target_pos[0] - rot_cam_offset_vector[0],
        target_pos[1] - rot_cam_offset_vector[1],
        target_pos[2] - rot_cam_offset_vector[2],
        rac.offset_[0],
        rac.offset_[1],
        rac.offset_[2],
    )
    converted_quat = _rotation_concat(
        [-math.sqrt(2.0) / 2.0, 0.0, 0.0, math.sqrt(2.0) / 2.0],
        target_quat_xyzw,
    )
    seed0 = np.zeros(int(rac.ik_solver_.number_of_joints), dtype=np.float64)
    seeds = [seed0]
    for _ in range(max(ik_seed_trials - 1, 0)):
        seeds.append(rng.uniform(-math.pi, math.pi, size=seed0.shape[0]).astype(np.float64))
    for seed in seeds:
        q = rac.ik_solver_.get_ik(
            seed.tolist(),
            float(converted_coord[0]),
            float(converted_coord[1]),
            float(converted_coord[2]),
            float(converted_quat[0]),
            float(converted_quat[1]),
            float(converted_quat[2]),
            float(converted_quat[3]),
        )
        if q is not None:
            qn = np.asarray(q, dtype=np.float64).reshape(-1)
            if qn.size >= 6 and np.all(np.isfinite(qn[:6])):
                return qn[:6].copy()
    return None


def _sample_q_start_ik_mesh(
    rac,
    plane_obj: fcl.CollisionObject,
    static_env_models: Sequence[fcl.CollisionObject],
    flex_collision_models: Sequence,
    obstacle_meshes: Sequence[Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    ik_pose_trials: int,
    ik_seed_trials: int,
    ik_surface_offset_min: float,
    ik_surface_offset_max: float,
    ik_tool_offset_xyz: Tuple[float, float, float],
) -> Tuple[np.ndarray, float] | None:
    for _ in range(ik_pose_trials):
        mesh_idx = int(rng.integers(0, len(obstacle_meshes)))
        v, f = obstacle_meshes[mesh_idx]
        if v.shape[0] < 3 or f.shape[0] < 1:
            continue
        surf_p, surf_n = _sample_point_and_normal_on_mesh(v, f, rng)

        # Keep target slightly outside obstacle surface; IK solver converts this
        # "tool target" into wrist_3_link configuration internally.
        offset = float(rng.uniform(ik_surface_offset_min, ik_surface_offset_max))
        target_pos = surf_p + offset * surf_n
        target_quat = _quat_align_x_to_normal(-surf_n)
        q_s = _ik_from_target_pose(
            rac, target_pos, target_quat, rng, ik_seed_trials, ik_tool_offset_xyz
        )
        if q_s is None:
            continue
        if not rac.arm_collision_free(
            q_s.tolist(), plane_obj, list(static_env_models), _normalize_flex_collision_models(flex_collision_models)
        ):
            continue
        c_s = min_link_env_clearance(rac, q_s, plane_obj, static_env_models, flex_collision_models)
        d_s_adj = c_s - _CLEARANCE_EPS
        if d_s_adj > 0.0 and d_s_adj < _MARGIN:
            return q_s, float(c_s)
    return None


def collect_metric_dataset(
    rac,
    plane_obj: fcl.CollisionObject,
    static_env_models: Sequence[fcl.CollisionObject],
    flex_collision_models: Sequence,
    num_samples: int,
    output_dir: str,
    seed: int = 0,
    max_tries_factor: int = 2000,
    joint_half_span: float = math.pi,
    sampler_mode: str = "fcl_uniform",
    obstacle_meshes: Sequence[Tuple[np.ndarray, np.ndarray]] | None = None,
    ik_pose_trials: int = 80,
    ik_seed_trials: int = 6,
    ik_surface_offset_min: float = 0.002,
    ik_surface_offset_max: float = 0.03,
    ik_tool_offset_xyz: Tuple[float, float, float] = (0.11, 0.0, 0.08),
    log_every_tries: int = 2000,
    visualize_callback=None,
    visualize_every_accepted: int = 0,
    qstart_only: bool = False,
    save_speed_normal: bool = True,
) -> Tuple[str, ...]:
    """
    Sample (q_s, q_g) pairs and save ntrl-demo-compatible .npy files.

    Returns paths to (sampled_points, speed, normal).
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)

    xs_list: list[np.ndarray] = []
    xg_list: list[np.ndarray] = []
    speed_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []

    tries = 0
    max_tries = max(num_samples * max_tries_factor, num_samples + 1)
    t0 = time.time()

    # Bias start proposals toward a nominal visible configuration (higher hit rate near table).
    nominal = np.array([0.7, -2.0, 2.5, -0.3, 0.7, 0.0], dtype=np.float64)
    sigma_start = 0.95

    sampler_mode = str(sampler_mode).strip().lower()
    if sampler_mode not in {"fcl_uniform", "ik_mesh"}:
        raise ValueError(f"Unknown sampler_mode={sampler_mode}. Use fcl_uniform or ik_mesh.")

    mesh_bank: list[Tuple[np.ndarray, np.ndarray]] = []
    if obstacle_meshes is not None:
        for mesh in obstacle_meshes:
            if mesh is None or len(mesh) < 2:
                continue
            vv = np.asarray(mesh[0], dtype=np.float64)
            ff = np.asarray(mesh[1], dtype=np.int64)
            if vv.ndim == 2 and ff.ndim == 2 and vv.shape[1] == 3 and ff.shape[1] == 3:
                mesh_bank.append((vv, ff))
    if sampler_mode == "ik_mesh" and not mesh_bank:
        raise RuntimeError("sampler_mode=ik_mesh requires obstacle_meshes (non-empty).")

    while len(xs_list) < num_samples and tries < max_tries:
        tries += 1
        if log_every_tries > 0 and tries % log_every_tries == 0:
            accepted = len(xs_list)
            ratio = accepted / max(tries, 1)
            elapsed = max(time.time() - t0, 1e-9)
            acc_per_sec = accepted / elapsed
            remain = max(num_samples - accepted, 0)
            eta_sec = remain / max(acc_per_sec, 1e-9) if accepted > 0 else float("inf")
            eta_msg = f"{eta_sec:.1f}" if np.isfinite(eta_sec) else "inf"
            print(
                "[ntfield_metric_collect_fcl] "
                f"mode={sampler_mode} progress: accepted={accepted}/{num_samples}, "
                f"tries={tries}/{max_tries}, accept_ratio={ratio:.4f}, "
                f"accepted_per_sec={acc_per_sec:.2f}, eta_sec={eta_msg}",
                flush=True,
            )
        if sampler_mode == "ik_mesh":
            out = _sample_q_start_ik_mesh(
                rac=rac,
                plane_obj=plane_obj,
                static_env_models=static_env_models,
                flex_collision_models=flex_collision_models,
                obstacle_meshes=mesh_bank,
                rng=rng,
                ik_pose_trials=ik_pose_trials,
                ik_seed_trials=ik_seed_trials,
                ik_surface_offset_min=ik_surface_offset_min,
                ik_surface_offset_max=ik_surface_offset_max,
                ik_tool_offset_xyz=ik_tool_offset_xyz,
            )
            if out is None:
                continue
            q_s, c_s = out
        else:
            if rng.random() < 0.5:
                q_s = rng.uniform(-joint_half_span, joint_half_span, size=6).astype(np.float64)
            else:
                q_s = nominal + rng.normal(0.0, sigma_start, size=6).astype(np.float64)

            if not rac.arm_collision_free(
                q_s.tolist(), plane_obj, list(static_env_models), _normalize_flex_collision_models(flex_collision_models)
            ):
                continue

            c_s = min_link_env_clearance(rac, q_s, plane_obj, static_env_models, flex_collision_models)
            d_s_adj = c_s - _CLEARANCE_EPS
            if not (d_s_adj > 0.0 and d_s_adj < _MARGIN):
                continue

        x_s = q_s / SCALE
        if np.any(np.abs(x_s) > 0.5):
            continue

        xs_list.append(x_s.astype(np.float64))
        q_goal_for_viz = None

        if not qstart_only:
            q_g = rng.uniform(-joint_half_span, joint_half_span, size=6).astype(np.float64)
            if not rac.arm_collision_free(
                q_g.tolist(), plane_obj, list(static_env_models), _normalize_flex_collision_models(flex_collision_models)
            ):
                xs_list.pop()
                continue
            x_g = q_g / SCALE
            if np.any(np.abs(x_g) > 0.5):
                xs_list.pop()
                continue
            xg_list.append(x_g.astype(np.float64))
            q_goal_for_viz = q_g.copy()

            if save_speed_normal:
                c_g = min_link_env_clearance(rac, q_g, plane_obj, static_env_models, flex_collision_models)
                sp = np.array([_speed_from_clearance(c_s), _speed_from_clearance(c_g)], dtype=np.float64)
                g_s = _grad_clearance_wrt_q(rac, q_s, plane_obj, static_env_models, flex_collision_models)
                g_g = _grad_clearance_wrt_q(rac, q_g, plane_obj, static_env_models, flex_collision_models)
                n_s = _normal_in_normalized_coords(g_s)
                n_g = _normal_in_normalized_coords(g_g)
                n_row = np.concatenate([n_s, n_g], axis=0).astype(np.float64)
                speed_rows.append(sp)
                normal_rows.append(n_row)

        if (
            visualize_callback is not None
            and visualize_every_accepted > 0
            and (len(xs_list) % visualize_every_accepted == 0)
        ):
            try:
                visualize_callback(q_s.copy(), q_goal_for_viz, len(xs_list), tries)
            except Exception as e:
                print(f"[ntfield_metric_collect_fcl] visualize_callback error: {e}", flush=True)

        if log_every_tries > 0 and len(xs_list) == num_samples:
            accepted = len(xs_list)
            ratio = accepted / max(tries, 1)
            elapsed = max(time.time() - t0, 1e-9)
            acc_per_sec = accepted / elapsed
            remain = max(num_samples - accepted, 0)
            eta_sec = remain / max(acc_per_sec, 1e-9)
            print(
                "[ntfield_metric_collect_fcl] "
                f"mode={sampler_mode} progress: accepted={accepted}/{num_samples}, "
                f"tries={tries}/{max_tries}, accept_ratio={ratio:.4f}, "
                f"accepted_per_sec={acc_per_sec:.2f}, eta_sec={eta_sec:.1f}",
                flush=True,
            )

    if log_every_tries > 0 and tries % log_every_tries != 0 and len(xs_list) < num_samples:
        accepted = len(xs_list)
        ratio = accepted / max(tries, 1)
        elapsed = max(time.time() - t0, 1e-9)
        acc_per_sec = accepted / elapsed
        print(
            "[ntfield_metric_collect_fcl] "
            f"mode={sampler_mode} final-progress-before-exit: accepted={accepted}/{num_samples}, "
            f"tries={tries}/{max_tries}, accept_ratio={ratio:.4f}, accepted_per_sec={acc_per_sec:.2f}",
            flush=True,
        )

    if len(xs_list) < num_samples:
        raise RuntimeError(
            f"Collected {len(xs_list)}/{num_samples} metric samples after {tries} tries. "
            "Increase --metric_max_tries_factor, loosen near-obstacle margin, or reduce clutter."
        )

    q_start_norm = np.stack(xs_list, axis=0).astype(np.float32)
    q_start_rad = (q_start_norm.astype(np.float64) * SCALE).astype(np.float32)
    p_qs_norm = os.path.join(output_dir, "q_start_normalized.npy")
    p_qs_rad = os.path.join(output_dir, "q_start_rad.npy")
    np.save(p_qs_norm, q_start_norm)
    np.save(p_qs_rad, q_start_rad)

    if qstart_only:
        print(
            f"[ntfield_metric_collect_fcl] mode={sampler_mode} saved {num_samples} q_start-only samples "
            f"to {output_dir} (q_start_normalized {q_start_norm.shape}, q_start_rad {q_start_rad.shape})."
        )
        return p_qs_norm, p_qs_rad

    sampled_points = np.hstack([q_start_norm, np.stack(xg_list, axis=0)]).astype(np.float32)
    p_pts = os.path.join(output_dir, "sampled_points.npy")
    np.save(p_pts, sampled_points)

    if save_speed_normal:
        speed = np.stack(speed_rows, axis=0).astype(np.float32)
        normal = np.stack(normal_rows, axis=0).astype(np.float32)
        p_sp = os.path.join(output_dir, "speed.npy")
        p_n = os.path.join(output_dir, "normal.npy")
        np.save(p_sp, speed)
        np.save(p_n, normal)
        print(
            f"[ntfield_metric_collect_fcl] mode={sampler_mode} saved {num_samples} samples to {output_dir} "
            f"(sampled_points {sampled_points.shape}, speed {speed.shape}, normal {normal.shape}, "
            f"q_start_normalized {q_start_norm.shape})."
        )
        return p_pts, p_sp, p_n, p_qs_norm, p_qs_rad

    print(
        f"[ntfield_metric_collect_fcl] mode={sampler_mode} saved {num_samples} samples to {output_dir} "
        f"(sampled_points {sampled_points.shape}, q_start_normalized {q_start_norm.shape}, q_start_rad {q_start_rad.shape})."
    )
    return p_pts, p_qs_norm, p_qs_rad
