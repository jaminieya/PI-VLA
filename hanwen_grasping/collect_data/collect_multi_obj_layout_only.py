#
# collect_multi_obj_layout_only.py
# Same tabletop setup as collect_multi_obj_data_for_student.py (3 YCB objects) but does not
# run grasp / motion planning. Saves start_image (RGB from top camera),
# object_locations (NUM_OF_OBJECTS x 3 float32: world-frame centers after settle),
# object_urdf_indices (int32, same row order as object_locations: index into object_urdf_grasp.txt),
# and object_names (UTF-8 human-readable label per row, same convention as collect_multi_obj_data_for_student).
#
# Run (from hanwen_grasping): python collect_data/collect_multi_obj_layout_only.py --headless
# Output: PI-VLA/output/multi_obj_layout/multi_obj_layout_*.h5
# Exits with os._exit() after HDF5 flush to avoid Isaac Gym destroy_sim segfaults on Linux.
#
from datetime import datetime
import math
import sys
import os
import fcl
import numpy as np
from isaacgym import gymapi
from isaacgym import gymutil
import h5py

_script_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.abspath(os.path.join(_script_dir, ".."))

# YCB grasp asset folder or URDF stem -> short phrase (aligned with collect_multi_obj_data_for_student.OBJECT_NAMES)
_OBJECT_DISPLAY_NAMES = {
    "002_master_chef_can": "master chef can",
    "004_sugar_box": "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle": "mustard bottle",
    "036_wood_block": "wood block",
    "011_banana": "banana",
}


def _grasp_asset_path_to_display_name(rel_path: str) -> str:
    """Readable name from a line in object_urdf_grasp.txt (may be urdf/ycb/<id>/file.urdf or flat)."""
    rel = rel_path.replace("\\", "/")
    folder = rel.split("/")[-2] if "/" in rel else ""
    stem = os.path.splitext(os.path.basename(rel))[0]
    for key in (folder, stem):
        if key and key in _OBJECT_DISPLAY_NAMES:
            return _OBJECT_DISPLAY_NAMES[key]
    label = stem or folder
    if len(label) > 4 and label[:3].isdigit() and label[3] == "_":
        label = label[4:]
    return label.replace("_", " ").strip()


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

util_dir = os.path.join(package_root, "util")
grasp_util_dir = os.path.join(package_root, "grasp_util")
sys.path.insert(0, package_root)
sys.path.append(util_dir)
sys.path.append(grasp_util_dir)

from obj_reader import obj_reader

num_of_envs = 1
row_num_of_envs = int(math.sqrt(num_of_envs))

TABLE_DIMS_X = 0.8
TABLE_DIMS_Y = 1.0
TABLE_DIMS_Z = 0.10
DRAWER_HEIGHT = 0.40
max_drawer_height = DRAWER_HEIGHT
min_drawer_height = DRAWER_HEIGHT
table_dims = gymapi.Vec3(TABLE_DIMS_X, TABLE_DIMS_Y, TABLE_DIMS_Z)

max_scaling_factor = 0
NUM_OF_OBJECTS = 3
TARGET_OBJ_INDEX = [1, 3, 5]

if __name__ == "__main__":
    gym = gymapi.acquire_gym()

    args = gymutil.parse_arguments(
        description="Multi-object layout only: RGB + 3 object centers (no grasp goals)",
        custom_parameters=[
            {"name": "--env_id", "type": int, "help": "env_id", "default": 0},
            {
                "name": "--num_episodes",
                "type": int,
                "default": 1,
                "help": "Number of episodes to collect in one invocation",
            },
            {"name": "--headless", "action": "store_true", "help": "Run without creating a viewer"},
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

    object_collision_lib = []

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

    drawer_height = np.random.random() * (max_drawer_height - min_drawer_height) + min_drawer_height

    upper_cover_dims = gymapi.Vec3(table_dims.x, table_dims.y, 0.03)
    upper_cover_asset = gym.create_box(
        sim, upper_cover_dims.x, upper_cover_dims.y, upper_cover_dims.z, asset_options
    )

    os.makedirs(SAVED_RESULT_DIR, exist_ok=True)
    saved_env_name = os.path.join(SAVED_RESULT_DIR, "env_" + str(env_id) + "_scene_info.npy")
    np.save(saved_env_name, np.array([table_dims.x, table_dims.y, table_dims.z, drawer_height]))

    ADD_COVER = False

    asset_options.fix_base_link = False
    object_assets = []
    for ob in object_asset_files:
        object_assets.append(gym.load_asset(sim, asset_root, ob, asset_options))
    asset_options.fix_base_link = True

    ur5e_pose = gymapi.Transform()
    ur5e_pose.p = gymapi.Vec3(0, 0, 0)
    ur5e_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.5 * math.pi)

    table_pose = gymapi.Transform()
    table_pose.p = gymapi.Vec3(table_dims.x * 0.5 + 0.3, 0.0, table_dims.z * 0.5)

    upper_cover_pose = gymapi.Transform()
    upper_cover_pose.p = gymapi.Vec3(table_pose.p.x, 0.0, table_dims.z + drawer_height + 0.015)

    camera_focus = gymapi.Vec3(0, 0, 0)
    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 70.25
    camera_props.width = 1280
    camera_props.height = 720

    table_x_min = table_pose.p.x - table_dims.x * 0.5 + 0.05
    table_x_max = table_pose.p.x + table_dims.x * 0.5 - 0.10
    table_y_min = table_pose.p.y - table_dims.y * 0.5 + 0.10
    table_y_max = table_pose.p.y + table_dims.y * 0.5 - 0.20

    envs = []
    ur5e_handles = []
    object_status_list = []
    object_handles = []
    object_slot_urdf_indices = None
    object_slot_names = None

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
        target_file_idx = np.random.choice(TARGET_OBJ_INDEX, NUM_OF_OBJECTS, replace=False)
        object_slot_urdf_indices = np.asarray(target_file_idx, dtype=np.int32).copy()
        object_slot_names = [
            _grasp_asset_path_to_display_name(object_asset_files[int(object_slot_urdf_indices[k])])
            for k in range(NUM_OF_OBJECTS)
        ]

        object_scaling_factor = np.random.randint(0, max_scaling_factor + 1, size=NUM_OF_OBJECTS) / 10.0 + 1.0

        objs_manager = fcl.DynamicAABBTreeCollisionManager()
        objs_manager.setup()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        GT_TARGET_POS = [
            np.random.uniform(max(table_x_min, 0.20 + table_dims.x / 2), table_x_max),
            np.random.uniform(table_y_min, table_y_max),
            table_dims.z + 0.08,
        ]

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
                t = fcl.Transform(np.array([tx, ty, tz]))

                req = fcl.CollisionRequest()
                rdata = fcl.CollisionData(request=req)
                objs_manager.collide(fcl.CollisionObject(m, t), rdata, fcl.defaultCollisionCallback)

                is_collision = rdata.result.is_collision

                if not is_collision:
                    dist = np.sqrt((tx - GT_TARGET_POS[0]) ** 2 + (ty - GT_TARGET_POS[1]) ** 2)
                    if dist <= 0.2:
                        is_collision = True
                        continue

                    for obj in GT_OBJ_POS_LIST:
                        dist = np.sqrt((tx - obj[0]) ** 2 + (ty - obj[1]) ** 2)
                        if dist <= 0.16:
                            is_collision = True
                            continue

            GT_OBJ_POS_LIST.append([object_pose.p.x, object_pose.p.y])

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
            object_status_list.append([temp_center, temp_bounding_box])
            object_collision_lib.append(m)
            obstacle_objs.append(fcl.CollisionObject(m, t))
            objs_manager.registerObjects(obstacle_objs)
            objs_manager.setup()

        top_cam_handle = gym.create_camera_sensor(envs[-1], camera_props)
        gym.set_camera_location(
            top_cam_handle,
            envs[-1],
            gymapi.Vec3(table_pose.p.x, table_pose.p.y + 0.001, 2),
            gymapi.Vec3(table_pose.p.x - 0.5, table_pose.p.y, table_pose.p.z),
        )

    cam_pos = gymapi.Vec3(2.2, 0, 0.5)
    cam_target = gymapi.Vec3(0, 0, 0.5)
    if viewer is not None:
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
    gym.set_light_parameters(
        sim, 0, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(-1.0, 0.0, 0.0)
    )
    gym.set_light_parameters(
        sim, 1, gymapi.Vec3(0.3, 0.3, 0.3), gymapi.Vec3(1.0, 1.0, 1.0), gymapi.Vec3(1.0, 0.0, 0.0)
    )

    real_position = False
    for _t in range(100):
        if not real_position:
            gym.set_dof_target_position(envs[-1], spj, 0)
            gym.set_dof_target_position(envs[-1], slj, -math.pi / 2)
            gym.set_dof_target_position(envs[-1], ej, 0)
            gym.set_dof_target_position(envs[-1], wj1, -math.pi / 2)
            gym.set_dof_target_position(envs[-1], wj2, 0)
            gym.set_dof_target_position(envs[-1], wj3, 0)
            real_position = True
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)

    original_centers = [status[0].copy() for status in object_status_list]

    num_episodes = max(1, int(getattr(args, "num_episodes", 1)))
    print(f"Layout-only collect: {NUM_OF_OBJECTS} YCB objects; saving start_image + object_locations only.")

    HOME_DOF = [0.7, -2.0, 2.5, -0.3, 0.7, 0.0]
    START_SETTLE_STEPS = 30

    successful_episodes = 0
    objs_manager = fcl.DynamicAABBTreeCollisionManager()

    for episode_idx in range(num_episodes):
        print(f"\n================ Episode {episode_idx + 1}/{num_episodes} ================\n")

        gym.set_dof_target_position(envs[-1], spj, 0)
        gym.set_dof_target_position(envs[-1], slj, -math.pi / 2)
        gym.set_dof_target_position(envs[-1], ej, 0)
        gym.set_dof_target_position(envs[-1], wj1, -math.pi / 2)
        gym.set_dof_target_position(envs[-1], wj2, 0)
        gym.set_dof_target_position(envs[-1], wj3, 0)

        objs_manager.clear()
        obstacle_objs = []
        GT_OBJ_POS_LIST = []
        GT_TARGET_POS = [
            np.random.uniform(max(table_x_min, 0.20 + table_dims.x / 2), table_x_max),
            np.random.uniform(table_y_min, table_y_max),
            table_dims.z + 0.08,
        ]

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
                    dist = np.sqrt((tx - GT_TARGET_POS[0]) ** 2 + (ty - GT_TARGET_POS[1]) ** 2)
                    if dist <= 0.2:
                        is_collision = True
                        continue

                    for obj in GT_OBJ_POS_LIST:
                        dist = np.sqrt((tx - obj[0]) ** 2 + (ty - obj[1]) ** 2)
                        if dist <= 0.16:
                            is_collision = True
                            break

            GT_OBJ_POS_LIST.append([tx, ty])
            obstacle_objs.append(temp_obj)

            handle = object_handles[k]
            states = gym.get_actor_rigid_body_states(envs[-1], handle, gymapi.STATE_POS)
            states["pose"]["p"]["x"] = tx
            states["pose"]["p"]["y"] = ty
            states["pose"]["p"]["z"] = tz
            states["pose"]["r"]["x"] = 0.0
            states["pose"]["r"]["y"] = 0.0
            states["pose"]["r"]["z"] = 0.0
            states["pose"]["r"]["w"] = 1.0

            gym.set_actor_rigid_body_states(envs[-1], handle, states, gymapi.STATE_POS)

        objs_manager.registerObjects(obstacle_objs)
        objs_manager.setup()

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
                    px = states["pose"]["p"]["x"].item()
                    py = states["pose"]["p"]["y"].item()
                    pz = states["pose"]["p"]["z"].item()
                    translation = np.array([px, py, pz])
                    object_status_list[i_obj][0] = original_centers[i_obj] + translation

        object_locations = np.stack(
            [np.array(object_status_list[k][0], dtype=np.float32) for k in range(NUM_OF_OBJECTS)]
        )

        start_dof = HOME_DOF
        for _ in range(START_SETTLE_STEPS):
            gym.set_dof_target_position(envs[-1], spj, start_dof[0])
            gym.set_dof_target_position(envs[-1], slj, start_dof[1])
            gym.set_dof_target_position(envs[-1], ej, start_dof[2])
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

        pi_vla_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        collected_dir = os.path.join(pi_vla_root, "output", "multi_obj_layout")
        os.makedirs(collected_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = os.path.join(collected_dir, f"multi_obj_layout_{timestamp}.h5")
        with h5py.File(out_path, "w") as f:
            f.create_dataset("start_image", data=start_image.astype(np.uint8), compression="gzip")
            f.create_dataset("object_locations", data=object_locations)
            f.create_dataset("object_urdf_indices", data=object_slot_urdf_indices)
            name_dtype = h5py.string_dtype(encoding="utf-8")
            f.create_dataset("object_names", data=np.asarray(object_slot_names, dtype=name_dtype))
            f.attrs["num_objects"] = int(NUM_OF_OBJECTS)
            f.flush()

        successful_episodes += 1
        print(f"Saved 1 sample to {out_path}")

    print(f"Collected {successful_episodes}/{num_episodes} episodes in one simulator setup.")
    print("Test Completed Successfully!!")

    exit_code = 0 if successful_episodes == num_episodes else 1
    os._exit(exit_code)
