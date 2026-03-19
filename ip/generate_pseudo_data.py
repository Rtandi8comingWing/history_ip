"""
Pseudo-demonstration generation script (Appendix D).
Generates synthetic task demonstrations from ShapeNet objects (or primitive fallbacks)
and saves them in the format expected by save_sample() → RunningDataset → train.py.
"""
import os
import sys
import argparse
import glob as glob_mod
import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as Rot, Slerp
from multiprocessing import Pool
from functools import partial

from ip.utils.data_proc import save_sample, sample_to_cond_demo, sample_to_live


# ──────────────────────────────────────────────────────────────────────────────
# 1. Scene construction
# ──────────────────────────────────────────────────────────────────────────────

def load_shapenet_meshes(shapenet_path, n=2):
    """Load n random ShapeNet meshes."""
    obj_files = glob_mod.glob(os.path.join(shapenet_path, '**', '*.obj'), recursive=True)
    if not obj_files:
        obj_files = glob_mod.glob(os.path.join(shapenet_path, '**', '*.off'), recursive=True)
    if len(obj_files) < n:
        raise FileNotFoundError(f"Need at least {n} meshes in {shapenet_path}, found {len(obj_files)}")
    chosen = np.random.choice(obj_files, n, replace=False)
    meshes = []
    for p in chosen:
        try:
            m = trimesh.load(p, force='mesh')
            if isinstance(m, trimesh.Scene):
                m = m.dump(concatenate=True)
            meshes.append(m)
        except Exception:
            meshes.append(trimesh.creation.box(extents=[0.1, 0.1, 0.1]))
    return meshes


def generate_primitive_meshes(n=2):
    """Fallback: generate simple geometric primitives."""
    factories = [
        lambda: trimesh.creation.box(extents=np.random.uniform(0.04, 0.12, 3)),
        lambda: trimesh.creation.cylinder(radius=np.random.uniform(0.02, 0.06),
                                          height=np.random.uniform(0.04, 0.12)),
        lambda: trimesh.creation.icosphere(subdivisions=2,
                                           radius=np.random.uniform(0.03, 0.07)),
    ]
    return [factories[np.random.randint(len(factories))]() for _ in range(n)]


def create_scene(shapenet_path=None, num_objects=2):
    """
    Build a tabletop scene with randomly placed objects.
    Returns list of (mesh, T_w_obj) tuples.
    """
    if shapenet_path and os.path.isdir(shapenet_path):
        meshes = load_shapenet_meshes(shapenet_path, num_objects)
    else:
        meshes = generate_primitive_meshes(num_objects)

    scene_objects = []
    for mesh in meshes:
        # Normalize to unit bounding box then scale to 5-15cm
        ext = mesh.bounding_box.extents
        mesh.apply_scale(1.0 / max(ext))
        target_size = np.random.uniform(0.05, 0.15)
        mesh.apply_scale(target_size)

        # Random position on table (z=0 is table surface)
        x = np.random.uniform(-0.15, 0.15)
        y = np.random.uniform(-0.15, 0.15)
        z = mesh.bounding_box.extents[2] / 2.0  # sit on table

        # Random yaw in [-pi/3, pi/3] (Appendix F)
        yaw = np.random.uniform(-np.pi / 3, np.pi / 3)
        T_w_obj = np.eye(4)
        T_w_obj[:3, :3] = Rot.from_euler('z', yaw).as_matrix()
        T_w_obj[:3, 3] = [x, y, z]

        scene_objects.append((mesh, T_w_obj))
    return scene_objects


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pseudo-task sampling
# ──────────────────────────────────────────────────────────────────────────────

def sample_point_near_object(mesh, T_w_obj, offset=0.03):
    """Sample a point on or near an object surface."""
    pts = mesh.sample(1)
    pt_world = (T_w_obj[:3, :3] @ pts.T).T + T_w_obj[:3, 3]
    pt_world = pt_world.flatten() + np.random.uniform(-offset, offset, 3)
    pt_world[2] = max(pt_world[2], 0.02)  # stay above table
    return pt_world


def sample_pseudo_task(scene_objects):
    """
    Sample 2-6 waypoints for a pseudo-task.
    50% biased (grasp/pick-place patterns), 50% random.
    Returns list of (position, grip_state) tuples.
    """
    num_waypoints = np.random.randint(2, 7)
    waypoints = []

    biased = np.random.random() < 0.5

    if biased and len(scene_objects) >= 1:
        obj_idx = np.random.randint(len(scene_objects))
        mesh, T_w_obj = scene_objects[obj_idx]

        # Approach
        approach_pt = sample_point_near_object(mesh, T_w_obj, offset=0.06)
        approach_pt[2] += 0.05
        waypoints.append((approach_pt, 0))

        # Grasp
        grasp_pt = sample_point_near_object(mesh, T_w_obj, offset=0.01)
        waypoints.append((grasp_pt, 1))

        # Lift
        lift_pt = grasp_pt.copy()
        lift_pt[2] += np.random.uniform(0.05, 0.15)
        waypoints.append((lift_pt, 1))

        # Place
        if len(scene_objects) > 1:
            other_idx = (obj_idx + 1) % len(scene_objects)
            place_pt = sample_point_near_object(scene_objects[other_idx][0],
                                                scene_objects[other_idx][1], offset=0.05)
            place_pt[2] += 0.02
        else:
            place_pt = np.array([np.random.uniform(-0.15, 0.15),
                                 np.random.uniform(-0.15, 0.15),
                                 np.random.uniform(0.02, 0.08)])
        waypoints.append((place_pt, 1))

        # Release
        release_pt = place_pt.copy()
        release_pt[2] += 0.03
        waypoints.append((release_pt, 0))

        waypoints = waypoints[:num_waypoints]
    else:
        grip = 0
        for _ in range(num_waypoints):
            obj_idx = np.random.randint(len(scene_objects))
            mesh, T_w_obj = scene_objects[obj_idx]
            pt = sample_point_near_object(mesh, T_w_obj, offset=0.08)
            if np.random.random() < 0.3:
                grip = 1 - grip
            waypoints.append((pt, grip))

    return waypoints


# ──────────────────────────────────────────────────────────────────────────────
# 3. Trajectory generation
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_poses(T_start, T_end, trans_step=0.01, rot_step_deg=3.0):
    """
    Interpolate between two SE(3) poses with controlled step sizes.
    trans_step: max translation per frame (meters).
    rot_step_deg: max rotation per frame (degrees).
    """
    delta_t = np.linalg.norm(T_end[:3, 3] - T_start[:3, 3])
    R_rel = T_start[:3, :3].T @ T_end[:3, :3]
    delta_r = np.linalg.norm(Rot.from_matrix(R_rel).as_rotvec(degrees=True))

    n_trans = max(1, int(np.ceil(delta_t / trans_step)))
    n_rot = max(1, int(np.ceil(delta_r / rot_step_deg)))
    n_steps = max(n_trans, n_rot)

    if n_steps <= 1:
        return [T_start, T_end]

    key_rots = Rot.from_matrix(np.stack([T_start[:3, :3], T_end[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)

    poses = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        T = np.eye(4)
        T[:3, 3] = T_start[:3, 3] + alpha * (T_end[:3, 3] - T_start[:3, 3])
        T[:3, :3] = slerp(alpha).as_matrix()
        poses.append(T)
    return poses


def sample_ee_orientation():
    """Sample a random but reasonable EE orientation (roughly pointing down)."""
    # Base: z-axis pointing down
    base_rot = Rot.from_euler('x', 180, degrees=True)
    # Add random perturbation
    perturb = Rot.from_euler('xyz',
                             [np.random.uniform(-40, 40),
                              np.random.uniform(-40, 40),
                              np.random.uniform(-180, 180)], degrees=True)
    return (base_rot * perturb).as_matrix()


def generate_trajectory(scene_objects, task_waypoints, trans_step=0.01, rot_step_deg=3.0):
    """
    Generate a full trajectory from task waypoints.
    Returns (traj_poses, traj_grips, obj_transforms_per_frame).
    """
    # Random starting EE pose above the scene
    T_start = np.eye(4)
    T_start[:3, :3] = sample_ee_orientation()
    T_start[:3, 3] = [np.random.uniform(-0.1, 0.1),
                       np.random.uniform(-0.1, 0.1),
                       np.random.uniform(0.2, 0.35)]

    # Build waypoint poses (position from task, orientation smoothly varied)
    wp_poses = [T_start]
    wp_grips = [task_waypoints[0][1] if task_waypoints else 0]

    for pos, grip in task_waypoints:
        T_wp = np.eye(4)
        T_wp[:3, :3] = sample_ee_orientation()
        T_wp[:3, 3] = pos
        wp_poses.append(T_wp)
        wp_grips.append(grip)

    # Interpolate between consecutive waypoints
    traj_poses = []
    traj_grips = []
    for i in range(len(wp_poses) - 1):
        segment = interpolate_poses(wp_poses[i], wp_poses[i + 1], trans_step, rot_step_deg)
        grip_val = wp_grips[i + 1]
        if i == 0:
            traj_poses.extend(segment)
            traj_grips.extend([wp_grips[i]] * (len(segment) - 1) + [grip_val])
        else:
            traj_poses.extend(segment[1:])  # skip duplicate
            traj_grips.extend([grip_val] * len(segment[1:]))

    if len(traj_poses) < 3:
        # Pad with stationary frames
        for _ in range(3 - len(traj_poses)):
            traj_poses.append(traj_poses[-1].copy())
            traj_grips.append(traj_grips[-1])

    # Object attach/detach logic
    attached_obj_idx = None
    T_e_obj = None
    obj_transforms = []

    for i, (T_w_e, grip) in enumerate(zip(traj_poses, traj_grips)):
        frame_objs = [(m, T.copy()) for m, T in scene_objects]

        if grip == 1 and attached_obj_idx is None:
            ee_pos = T_w_e[:3, 3]
            min_dist = float('inf')
            best_idx = None
            for oi, (mesh, T_w_obj) in enumerate(scene_objects):
                dist = np.linalg.norm(T_w_obj[:3, 3] - ee_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_idx = oi
            if min_dist < 0.15:
                attached_obj_idx = best_idx
                T_e_obj = np.linalg.inv(T_w_e) @ scene_objects[attached_obj_idx][1]

        if grip == 0 and attached_obj_idx is not None:
            scene_objects[attached_obj_idx] = (
                scene_objects[attached_obj_idx][0],
                T_w_e @ T_e_obj
            )
            attached_obj_idx = None
            T_e_obj = None

        if attached_obj_idx is not None and T_e_obj is not None:
            frame_objs[attached_obj_idx] = (
                frame_objs[attached_obj_idx][0],
                T_w_e @ T_e_obj
            )

        obj_transforms.append(frame_objs)

    return traj_poses, traj_grips, obj_transforms


# ──────────────────────────────────────────────────────────────────────────────
# 4. Point cloud rendering
# ──────────────────────────────────────────────────────────────────────────────

def setup_cameras(num_cameras=3):
    """Set up depth cameras around the workspace."""
    camera = pyrender.IntrinsicsCamera(fx=600, fy=600, cx=320, cy=240)
    cam_poses = []
    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras + np.random.uniform(-0.2, 0.2)
        radius = np.random.uniform(0.4, 0.6)
        height = np.random.uniform(0.3, 0.5)
        cam_pos = np.array([radius * np.cos(angle), radius * np.sin(angle), height])

        # Look at center of workspace
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        T_cam = np.eye(4)
        T_cam[:3, 0] = right
        T_cam[:3, 1] = up
        T_cam[:3, 2] = -forward  # pyrender convention: -z is forward
        T_cam[:3, 3] = cam_pos
        cam_poses.append(T_cam)

    return camera, cam_poses


def render_object_pcds(frame_objects, camera, cam_poses, img_w=640, img_h=480, num_points=2048):
    """
    Render depth images of objects from multiple cameras, backproject to point clouds.
    Returns concatenated object point cloud (N, 3) in world frame.
    """
    all_points = []

    for cam_pose in cam_poses:
        scene = pyrender.Scene()
        for mesh, T_w_obj in frame_objects:
            py_mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(py_mesh, pose=T_w_obj)

        scene.add(camera, pose=cam_pose)
        renderer = pyrender.OffscreenRenderer(img_w, img_h)
        try:
            _, depth = renderer.render(scene)
        finally:
            renderer.delete()

        # Backproject depth to 3D
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        v, u = np.where(depth > 0)
        if len(v) == 0:
            continue
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)

        # Transform to world frame
        pts_world = (cam_pose[:3, :3] @ pts_cam.T).T + cam_pose[:3, 3]
        all_points.append(pts_world)

    if not all_points:
        # Fallback: sample directly from meshes
        for mesh, T_w_obj in frame_objects:
            pts = mesh.sample(num_points // max(1, len(frame_objects)))
            pts_world = (T_w_obj[:3, :3] @ pts.T).T + T_w_obj[:3, 3]
            all_points.append(pts_world)

    combined = np.concatenate(all_points, axis=0)

    # Subsample to target number of points
    if len(combined) >= num_points:
        idx = np.random.choice(len(combined), num_points, replace=False)
    else:
        idx = np.random.choice(len(combined), num_points, replace=True)
    return combined[idx].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Data augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_trajectory(traj_poses, traj_grips):
    """Apply corrective augmentation and grip noise."""
    poses = [T.copy() for T in traj_poses]
    grips = list(traj_grips)

    # 30% chance: add local perturbation (corrective augmentation)
    if np.random.random() < 0.3:
        num_perturb = np.random.randint(1, max(2, len(poses) // 4))
        for _ in range(num_perturb):
            idx = np.random.randint(0, len(poses))
            poses[idx][:3, 3] += np.random.normal(0, 0.005, 3)
            perturb_rot = Rot.from_rotvec(np.random.normal(0, np.deg2rad(1.5), 3))
            poses[idx][:3, :3] = poses[idx][:3, :3] @ perturb_rot.as_matrix()

    # 10% chance: flip random grip states
    if np.random.random() < 0.1:
        num_flip = np.random.randint(1, max(2, len(grips) // 5))
        for _ in range(num_flip):
            idx = np.random.randint(0, len(grips))
            grips[idx] = 1 - grips[idx]

    return poses, grips


def sample_object_pcds(frame_objects, num_points=2048):
    """
    Fast mode: directly sample points from mesh surfaces (skip rendering).
    ~10x faster than pyrender, slightly less realistic but sufficient for pseudo-data.
    """
    all_points = []
    pts_per_obj = num_points // max(1, len(frame_objects))
    for mesh, T_w_obj in frame_objects:
        pts = mesh.sample(pts_per_obj)
        pts_world = (T_w_obj[:3, :3] @ pts.T).T + T_w_obj[:3, 3]
        # Add small noise to simulate depth sensor
        pts_world += np.random.normal(0, 0.001, pts_world.shape)
        all_points.append(pts_world)

    combined = np.concatenate(all_points, axis=0)
    if len(combined) >= num_points:
        idx = np.random.choice(len(combined), num_points, replace=False)
    else:
        idx = np.random.choice(len(combined), num_points, replace=True)
    return combined[idx].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Single demonstration generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_single_demo(scene_objects, task_waypoints, camera, cam_poses,
                         num_points=2048, fast=False):
    """
    Generate one demonstration: trajectory + rendered point clouds.
    Returns {'pcds': [...], 'T_w_es': [...], 'grips': [...]}.
    """
    # Vary object poses slightly for each demo
    varied_objects = []
    for mesh, T_w_obj in scene_objects:
        T_var = T_w_obj.copy()
        T_var[:3, 3] += np.random.normal(0, 0.01, 3)
        T_var[:3, 3][2] = max(T_var[:3, 3][2], mesh.bounding_box.extents[2] / 2.0)
        yaw_perturb = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        T_var[:3, :3] = T_var[:3, :3] @ Rot.from_euler('z', yaw_perturb).as_matrix()
        varied_objects.append((mesh, T_var))

    # Also vary task waypoint positions slightly
    varied_waypoints = []
    for pos, grip in task_waypoints:
        new_pos = pos.copy() + np.random.normal(0, 0.008, 3)
        new_pos[2] = max(new_pos[2], 0.02)
        varied_waypoints.append((new_pos, grip))

    traj_poses, traj_grips, obj_transforms = generate_trajectory(
        varied_objects, varied_waypoints)

    # Augment
    traj_poses, traj_grips = augment_trajectory(traj_poses, traj_grips)

    # Render point clouds per frame
    pcds = []
    for i in range(len(traj_poses)):
        frame_objs = obj_transforms[min(i, len(obj_transforms) - 1)]
        if fast:
            pcd = sample_object_pcds(frame_objs, num_points=num_points)
        else:
            pcd = render_object_pcds(frame_objs, camera, cam_poses, num_points=num_points)
        pcds.append(pcd)

    return {
        'pcds': pcds,
        'T_w_es': traj_poses,
        'grips': traj_grips,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main generation + save
# ──────────────────────────────────────────────────────────────────────────────

def generate_one_sample(sample_idx, shapenet_path, save_dir, num_demos, num_waypoints_demo,
                        pred_horizon, live_spacing_trans, live_spacing_rot,
                        scene_encoder, offset_base, fast=False):
    """Generate and save one full sample (num_demos + 1 live)."""
    np.random.seed()  # re-seed for multiprocessing

    scene_objects = create_scene(shapenet_path)
    task_waypoints = sample_pseudo_task(scene_objects)
    camera, cam_poses = setup_cameras(num_cameras=3)

    demos_raw = []
    for _ in range(num_demos + 1):
        demo = generate_single_demo(scene_objects, task_waypoints, camera, cam_poses,
                                    fast=fast)
        demos_raw.append(demo)

    # Build full_sample in the format expected by save_sample
    full_sample = {
        'demos': [None] * num_demos,
        'live': None,
    }

    for i in range(num_demos):
        demo_processed = sample_to_cond_demo(demos_raw[i], num_waypoints_demo)
        # Ensure exactly num_waypoints_demo waypoints (extract_waypoints can over/undershoot)
        n = len(demo_processed['obs'])
        if n > num_waypoints_demo:
            # Uniformly subsample to target count
            indices = np.round(np.linspace(0, n - 1, num_waypoints_demo)).astype(int)
            demo_processed['obs'] = [demo_processed['obs'][j] for j in indices]
            demo_processed['grips'] = [demo_processed['grips'][j] for j in indices]
            demo_processed['T_w_es'] = [demo_processed['T_w_es'][j] for j in indices]
        elif n < num_waypoints_demo:
            # Pad by repeating last waypoint
            while len(demo_processed['obs']) < num_waypoints_demo:
                demo_processed['obs'].append(demo_processed['obs'][-1])
                demo_processed['grips'].append(demo_processed['grips'][-1])
                demo_processed['T_w_es'].append(demo_processed['T_w_es'][-1])
        full_sample['demos'][i] = demo_processed

    full_sample['live'] = sample_to_live(demos_raw[-1], pred_horizon, 2048,
                                         live_spacing_trans, live_spacing_rot,
                                         subsample=True)

    # Compute offset: each sample produces len(live['obs']) files
    offset = offset_base
    save_sample(full_sample, save_dir=save_dir, offset=offset, scene_encoder=scene_encoder)

    num_saved = len(full_sample['live']['obs'])
    return num_saved


def generate_and_save(shapenet_path, save_dir, num_samples, num_demos=2,
                      num_waypoints_demo=10, pred_horizon=8,
                      live_spacing_trans=0.01, live_spacing_rot=3,
                      scene_encoder=None, continuous=False, buffer_size=10000,
                      fast=False):
    """
    Main generation loop.
    If continuous=True, runs indefinitely and overwrites a fixed-size buffer of files,
    designed to run in parallel with training (paper: "continuously generated in parallel").
    RunningDataset tolerates missing/stale files by retrying random indices.
    """
    os.makedirs(save_dir, exist_ok=True)

    if continuous:
        print(f"Continuous mode: writing to buffer of {buffer_size} frames in {save_dir}")
        print("Run training in parallel with: RunningDataset(data_path, num_samples=buffer_size)")
        offset = 0
        sample_idx = 0
        while True:
            try:
                num_saved = generate_one_sample(
                    sample_idx, shapenet_path, save_dir, num_demos,
                    num_waypoints_demo, pred_horizon,
                    live_spacing_trans, live_spacing_rot,
                    scene_encoder, offset % buffer_size, fast=fast)
                offset += num_saved
                sample_idx += 1
                if sample_idx % 10 == 0:
                    print(f"[sample {sample_idx}] {offset} total frames generated "
                          f"(buffer wraps at {buffer_size}).")
            except KeyboardInterrupt:
                print(f"\nStopped. {sample_idx} samples, {offset} total frames.")
                break
            except Exception as e:
                print(f"Sample {sample_idx} failed: {e}")
                sample_idx += 1
                continue
    else:
        offset = 0
        for sample_idx in range(num_samples):
            try:
                num_saved = generate_one_sample(
                    sample_idx, shapenet_path, save_dir, num_demos,
                    num_waypoints_demo, pred_horizon,
                    live_spacing_trans, live_spacing_rot,
                    scene_encoder, offset, fast=fast)
                offset += num_saved
                if (sample_idx + 1) % 10 == 0:
                    print(f"[{sample_idx + 1}/{num_samples}] Generated {offset} total frames.")
            except Exception as e:
                print(f"Sample {sample_idx} failed: {e}")
                continue

        print(f"Done. {offset} total frames saved to {save_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-demonstrations (Appendix D)')
    parser.add_argument('--shapenet_path', type=str, default=None,
                        help='Path to ShapeNetCore.v2. If not provided, uses primitive fallback.')
    parser.add_argument('--save_dir', type=str, default='./data/pseudo_demos')
    parser.add_argument('--val_dir', type=str, default=None,
                        help='Path to save validation data. If set, generates --num_val samples first, then trains.')
    parser.add_argument('--num_val', type=int, default=100,
                        help='Number of validation samples to generate (only used with --val_dir).')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_demos', type=int, default=2)
    parser.add_argument('--num_waypoints_demo', type=int, default=10)
    parser.add_argument('--pred_horizon', type=int, default=8)
    parser.add_argument('--continuous', action='store_true',
                        help='Run indefinitely, overwriting a fixed buffer. '
                             'Designed to run in parallel with training.')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='Buffer size for continuous mode (number of .pt files).')
    parser.add_argument('--compute_embeddings', action='store_true',
                        help='Pre-compute scene encoder embeddings. '
                             'Not recommended for continuous mode (requires GPU).')
    parser.add_argument('--encoder_path', type=str, default='./checkpoints/scene_encoder.pt',
                        help='Path to scene_encoder.pt (use ip.extract_scene_encoder to extract from full checkpoint).')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fast', action='store_true',
                        help='Skip pyrender, sample points directly from mesh surfaces (~10x faster).')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    scene_encoder = None
    if args.compute_embeddings:
        import torch
        from ip.models.scene_encoder import SceneEncoder
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(torch.load(args.encoder_path))
        scene_encoder = scene_encoder.to('cuda')
        scene_encoder.eval()
        print("Scene encoder loaded.")

    # Use EGL for headless rendering
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    # Generate validation set first (fixed, not overwritten)
    if args.val_dir:
        print(f"Generating {args.num_val} validation samples → {args.val_dir}")
        generate_and_save(
            shapenet_path=args.shapenet_path,
            save_dir=args.val_dir,
            num_samples=args.num_val,
            num_demos=args.num_demos,
            num_waypoints_demo=args.num_waypoints_demo,
            pred_horizon=args.pred_horizon,
            scene_encoder=scene_encoder,
            continuous=False,
            buffer_size=0,
            fast=args.fast,
        )

    # Generate training data
    generate_and_save(
        shapenet_path=args.shapenet_path,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        num_demos=args.num_demos,
        num_waypoints_demo=args.num_waypoints_demo,
        pred_horizon=args.pred_horizon,
        scene_encoder=scene_encoder,
        continuous=args.continuous,
        buffer_size=args.buffer_size,
        fast=args.fast,
    )


if __name__ == '__main__':
    main()
