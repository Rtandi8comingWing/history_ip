"""
Pseudo-demonstration generation script (Appendix D).
Generates synthetic task demonstrations from ShapeNet objects (or primitive fallbacks)
and saves them in the format expected by save_sample() → RunningDataset → train.py.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import glob as glob_mod
import numpy as np
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as Rot, Slerp

from ip.utils.data_proc import save_sample, sample_to_cond_demo, sample_to_live
from ip.utils.memory_task_generator import MemoryTaskGenerator
from ip.utils.track_builder import (
    build_object_tracks_world,
    project_tracks_to_current_ee,
    compute_track_age_seconds,
    sample_object_surface_points,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Scene construction
# ──────────────────────────────────────────────────────────────────────────────

def normalize(vec, fallback=None):
    vec = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        if fallback is None:
            return None
        fallback = np.asarray(fallback, dtype=np.float64)
        fallback_norm = np.linalg.norm(fallback)
        if fallback_norm < 1e-8:
            return None
        return fallback / fallback_norm
    return vec / norm


def horizontal(vec):
    if vec is None:
        return None
    vec = np.asarray(vec, dtype=np.float64).copy()
    vec[2] = 0.0
    return normalize(vec)


def clamp_above_table(point, min_z=0.02):
    point = np.asarray(point, dtype=np.float64).copy()
    point[2] = max(point[2], min_z)
    return point


def local_to_world(T_w_obj, point_local):
    point_local = np.asarray(point_local, dtype=np.float64)
    return (T_w_obj[:3, :3] @ point_local.T).T + T_w_obj[:3, 3]


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
        ext = mesh.bounding_box.extents
        mesh.apply_scale(1.0 / max(ext))
        target_size = np.random.uniform(0.05, 0.15)
        mesh.apply_scale(target_size)

        x = np.random.uniform(-0.15, 0.15)
        y = np.random.uniform(-0.15, 0.15)
        z = mesh.bounding_box.extents[2] / 2.0

        yaw = np.random.uniform(-np.pi / 3, np.pi / 3)
        T_w_obj = np.eye(4)
        T_w_obj[:3, :3] = Rot.from_euler('z', yaw).as_matrix()
        T_w_obj[:3, 3] = [x, y, z]
        scene_objects.append((mesh, T_w_obj))
    return scene_objects


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pseudo-task sampling
# ──────────────────────────────────────────────────────────────────────────────

def make_waypoint(pos, grip, stage, obj_idx=None, dir_hint=None):
    return {
        'pos': clamp_above_table(pos),
        'grip': int(grip),
        'stage': stage,
        'obj_idx': obj_idx,
        'dir_hint': None if dir_hint is None else np.asarray(dir_hint, dtype=np.float64),
    }


def top_point(mesh, T_w_obj, xy_scale=0.15, z_offset=0.0):
    ext = mesh.bounding_box.extents
    local = np.array([
        np.random.uniform(-xy_scale, xy_scale) * ext[0],
        np.random.uniform(-xy_scale, xy_scale) * ext[1],
        ext[2] / 2.0 + z_offset,
    ])
    return clamp_above_table(local_to_world(T_w_obj, local))


def side_point(mesh, T_w_obj, axis=0, sign=1.0, z_ratio=0.55, clearance=0.005):
    ext = mesh.bounding_box.extents
    other_axis = 1 - axis
    local = np.zeros(3)
    local[axis] = sign * (ext[axis] / 2.0 + clearance)
    local[other_axis] = np.random.uniform(-0.2, 0.2) * ext[other_axis]
    local[2] = (z_ratio - 0.5) * ext[2]
    return clamp_above_table(local_to_world(T_w_obj, local), min_z=0.03)


def random_workspace_point(z_low=0.03, z_high=0.12):
    return np.array([
        np.random.uniform(-0.18, 0.18),
        np.random.uniform(-0.18, 0.18),
        np.random.uniform(z_low, z_high),
    ], dtype=np.float64)


def object_frame_dirs(T_w_obj):
    return T_w_obj[:3, 0].copy(), T_w_obj[:3, 1].copy(), T_w_obj[:3, 2].copy(), T_w_obj[:3, 3].copy()


def select_task_waypoints(waypoints, target_count):
    if len(waypoints) <= target_count:
        return waypoints

    ordered = [0, len(waypoints) - 1]
    for i in range(1, len(waypoints) - 1):
        if waypoints[i]['grip'] != waypoints[i - 1]['grip']:
            ordered.append(i)

    important_stages = ['grasp', 'contact', 'place', 'release', 'lift', 'transfer', 'pull', 'push', 'settle',
                        'retreat', 'approach', 'pregrasp']
    for stage in important_stages:
        for i, waypoint in enumerate(waypoints[1:-1], start=1):
            if waypoint['stage'] == stage:
                ordered.append(i)

    ordered.extend(np.round(np.linspace(0, len(waypoints) - 1, target_count)).astype(int).tolist())
    ordered.extend(range(len(waypoints)))

    selected = []
    for idx in ordered:
        idx = int(idx)
        if idx not in selected:
            selected.append(idx)
        if len(selected) == target_count:
            break

    return [waypoints[i] for i in sorted(selected)]


def build_grasp_task(scene_objects):
    obj_idx = np.random.randint(len(scene_objects))
    mesh, T_w_obj = scene_objects[obj_idx]
    obj_x, obj_y, _, center = object_frame_dirs(T_w_obj)
    align_dir = obj_x if np.random.random() < 0.5 else obj_y

    grasp = top_point(mesh, T_w_obj, xy_scale=0.12, z_offset=0.0)
    approach = grasp + np.array([0.0, 0.0, np.random.uniform(0.06, 0.10)])
    pregrasp = grasp + np.array([0.0, 0.0, np.random.uniform(0.015, 0.03)])
    retreat = grasp + np.array([0.0, 0.0, np.random.uniform(0.08, 0.12)])

    return [
        make_waypoint(approach, 0, 'approach', obj_idx, dir_hint=center - approach),
        make_waypoint(pregrasp, 0, 'pregrasp', obj_idx, dir_hint=center - pregrasp),
        make_waypoint(grasp, 1, 'grasp', obj_idx, dir_hint=align_dir),
        make_waypoint(retreat, 1, 'retreat', obj_idx, dir_hint=align_dir),
    ]


def build_pick_place_task(scene_objects):
    src_idx = np.random.randint(len(scene_objects))
    dst_idx = (src_idx + 1) % len(scene_objects)
    src_mesh, T_src = scene_objects[src_idx]
    dst_mesh, T_dst = scene_objects[dst_idx]
    src_x, _, _, src_center = object_frame_dirs(T_src)
    _, dst_y, _, dst_center = object_frame_dirs(T_dst)

    grasp = top_point(src_mesh, T_src, xy_scale=0.1, z_offset=0.0)
    approach = grasp + np.array([0.0, 0.0, np.random.uniform(0.07, 0.10)])
    lift = grasp + np.array([0.0, 0.0, np.random.uniform(0.08, 0.14)])
    place = top_point(dst_mesh, T_dst, xy_scale=0.15, z_offset=0.01)
    transfer = place + np.array([0.0, 0.0, np.random.uniform(0.06, 0.10)])
    release = place + np.array([0.0, 0.0, np.random.uniform(0.04, 0.07)])

    transfer_dir = transfer - lift
    place_dir = dst_center - place
    return [
        make_waypoint(approach, 0, 'approach', src_idx, dir_hint=src_center - approach),
        make_waypoint(grasp, 1, 'grasp', src_idx, dir_hint=src_x),
        make_waypoint(lift, 1, 'lift', src_idx, dir_hint=transfer_dir),
        make_waypoint(transfer, 1, 'transfer', dst_idx, dir_hint=transfer_dir),
        make_waypoint(place, 1, 'place', dst_idx, dir_hint=place_dir),
        make_waypoint(release, 0, 'release', dst_idx, dir_hint=dst_y),
    ]


def build_opening_task(scene_objects):
    obj_idx = np.random.randint(len(scene_objects))
    mesh, T_w_obj = scene_objects[obj_idx]
    obj_x, obj_y, _, center = object_frame_dirs(T_w_obj)
    axis = np.random.randint(2)
    sign = np.random.choice([-1.0, 1.0])
    radial_dir = (obj_x if axis == 0 else obj_y) * sign
    tangent_dir = (obj_y if axis == 0 else obj_x) * np.random.choice([-1.0, 1.0])

    contact = side_point(mesh, T_w_obj, axis=axis, sign=sign, z_ratio=0.6, clearance=0.004)
    approach = contact - tangent_dir * np.random.uniform(0.04, 0.06) + np.array([0.0, 0.0, 0.05])
    sweep = contact + tangent_dir * np.random.uniform(0.05, 0.08) + radial_dir * 0.01
    pull = sweep + tangent_dir * np.random.uniform(0.03, 0.05) + np.array([0.0, 0.0, 0.02])
    release = pull + np.array([0.0, 0.0, 0.04])

    return [
        make_waypoint(approach, 0, 'approach', obj_idx, dir_hint=center - approach),
        make_waypoint(contact, 1, 'contact', obj_idx, dir_hint=tangent_dir),
        make_waypoint(sweep, 1, 'sweep', obj_idx, dir_hint=tangent_dir),
        make_waypoint(pull, 1, 'pull', obj_idx, dir_hint=tangent_dir),
        make_waypoint(release, 0, 'release', obj_idx, dir_hint=radial_dir),
    ]


def build_closing_task(scene_objects):
    obj_idx = np.random.randint(len(scene_objects))
    mesh, T_w_obj = scene_objects[obj_idx]
    obj_x, obj_y, _, center = object_frame_dirs(T_w_obj)
    axis = np.random.randint(2)
    sign = np.random.choice([-1.0, 1.0])
    radial_dir = (obj_x if axis == 0 else obj_y) * sign
    tangent_dir = (obj_y if axis == 0 else obj_x) * np.random.choice([-1.0, 1.0])

    contact = side_point(mesh, T_w_obj, axis=axis, sign=sign, z_ratio=0.55, clearance=0.006)
    approach = contact - tangent_dir * np.random.uniform(0.05, 0.07) + np.array([0.0, 0.0, 0.04])
    push = contact + tangent_dir * np.random.uniform(0.05, 0.08) - radial_dir * 0.015
    settle = push - tangent_dir * np.random.uniform(0.01, 0.03)
    release = settle + np.array([0.0, 0.0, 0.04])

    return [
        make_waypoint(approach, 0, 'approach', obj_idx, dir_hint=center - approach),
        make_waypoint(contact, 1, 'contact', obj_idx, dir_hint=tangent_dir),
        make_waypoint(push, 1, 'push', obj_idx, dir_hint=tangent_dir),
        make_waypoint(settle, 1, 'settle', obj_idx, dir_hint=-radial_dir),
        make_waypoint(release, 0, 'release', obj_idx, dir_hint=radial_dir),
    ]


def build_random_task(scene_objects, num_waypoints):
    waypoints = []
    grip = 0
    for _ in range(num_waypoints):
        obj_idx = np.random.randint(len(scene_objects))
        mesh, T_w_obj = scene_objects[obj_idx]
        center = T_w_obj[:3, 3]
        if np.random.random() < 0.25:
            grip = 1 - grip
        if np.random.random() < 0.5:
            pt = top_point(mesh, T_w_obj, xy_scale=0.3, z_offset=np.random.uniform(0.0, 0.04))
        else:
            pt = side_point(mesh, T_w_obj, axis=np.random.randint(2), sign=np.random.choice([-1.0, 1.0]),
                            z_ratio=np.random.uniform(0.45, 0.7), clearance=np.random.uniform(0.004, 0.02))
        pt += np.random.normal(0, 0.01, 3)
        pt = clamp_above_table(pt)
        waypoints.append(make_waypoint(pt, grip, 'random', obj_idx, dir_hint=center - pt))
    return waypoints


def sample_pseudo_task(scene_objects):
    """
    Sample 2-6 waypoints for a pseudo-task.
    50% biased (grasp/pick-place/open/close patterns), 50% random.
    Returns a list of waypoint dictionaries.
    """
    num_waypoints = np.random.randint(2, 7)
    biased = np.random.random() < 0.5

    if not biased:
        return build_random_task(scene_objects, num_waypoints)

    family_builders = [
        build_grasp_task,
        build_pick_place_task,
        build_opening_task,
        build_closing_task,
    ]
    waypoints = family_builders[np.random.randint(len(family_builders))](scene_objects)
    return select_task_waypoints(waypoints, num_waypoints)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Trajectory generation
# ──────────────────────────────────────────────────────────────────────────────

def build_downward_orientation(primary_dir=None, prev_R=None, continuity=0.0, tilt_deg=0.0):
    z_axis = np.array([0.0, 0.0, -1.0])
    x_axis = horizontal(primary_dir)

    if prev_R is not None:
        prev_x = horizontal(prev_R[:3, 0])
        if prev_x is not None:
            if x_axis is None:
                x_axis = prev_x
            else:
                x_axis = normalize((1.0 - continuity) * x_axis + continuity * prev_x, fallback=prev_x)

    if x_axis is None:
        x_axis = np.array([1.0, 0.0, 0.0])

    y_axis = normalize(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0]))
    x_axis = normalize(np.cross(y_axis, z_axis), fallback=np.array([1.0, 0.0, 0.0]))
    R = np.column_stack([x_axis, y_axis, z_axis])

    if tilt_deg > 0:
        tilt = Rot.from_euler('xy', np.random.uniform(-tilt_deg, tilt_deg, 2), degrees=True)
        R = R @ tilt.as_matrix()
    return R


def blend_directions(*weighted_dirs):
    accum = np.zeros(3, dtype=np.float64)
    for weight, direction in weighted_dirs:
        direction = horizontal(direction)
        if direction is not None:
            accum += weight * direction
    return normalize(accum)


def waypoint_orientation(waypoint, prev_pose, next_waypoint, scene_objects):
    stage = waypoint['stage']
    obj_dir = None
    if waypoint['obj_idx'] is not None:
        obj_dir = scene_objects[waypoint['obj_idx']][1][:3, 3] - waypoint['pos']
    move_dir = None if next_waypoint is None else next_waypoint['pos'] - waypoint['pos']
    prev_dir = None if prev_pose is None else waypoint['pos'] - prev_pose[:3, 3]

    if stage in {'approach', 'pregrasp', 'grasp', 'contact'}:
        primary_dir = blend_directions(
            (0.55, obj_dir),
            (0.30, waypoint['dir_hint']),
            (0.15, move_dir),
        )
        continuity = 0.35
        tilt_deg = 8.0
    elif stage in {'lift', 'transfer', 'place', 'release', 'retreat'}:
        primary_dir = blend_directions(
            (0.45, waypoint['dir_hint']),
            (0.30, move_dir),
            (0.25, prev_dir),
        )
        continuity = 0.75
        tilt_deg = 5.0
    elif stage in {'sweep', 'pull', 'push', 'settle'}:
        primary_dir = blend_directions(
            (0.50, waypoint['dir_hint']),
            (0.25, move_dir),
            (0.25, obj_dir),
        )
        continuity = 0.65
        tilt_deg = 6.0
    else:
        primary_dir = blend_directions(
            (0.45, waypoint['dir_hint']),
            (0.30, move_dir),
            (0.25, prev_dir),
        )
        continuity = 0.6
        tilt_deg = 8.0

    prev_R = None if prev_pose is None else prev_pose[:3, :3]
    return build_downward_orientation(primary_dir=primary_dir, prev_R=prev_R,
                                      continuity=continuity, tilt_deg=tilt_deg)


def choose_interpolation_mode(mode):
    if mode != 'random':
        return mode
    modes = ['linear', 'cubic', 'spherical']
    probs = np.array([0.4, 0.35, 0.25], dtype=np.float64)
    return np.random.choice(modes, p=probs / probs.sum())


def smoothstep(alpha):
    return alpha * alpha * (3.0 - 2.0 * alpha)


def slerp_unit_vectors(v0, v1, alpha):
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-5:
        vec = (1.0 - alpha) * v0 + alpha * v1
        return normalize(vec, fallback=v0)

    sin_omega = np.sin(omega)
    return (np.sin((1.0 - alpha) * omega) / sin_omega) * v0 + (np.sin(alpha * omega) / sin_omega) * v1


def build_spherical_center(p_start, p_end):
    chord = p_end - p_start
    chord_norm = np.linalg.norm(chord)
    if chord_norm < 1e-6:
        return None

    ref = np.array([0.0, 0.0, 1.0])
    normal = np.cross(chord, ref)
    if np.linalg.norm(normal) < 1e-6:
        ref = np.array([0.0, 1.0, 0.0])
        normal = np.cross(chord, ref)
    normal = normalize(normal)
    if normal is None:
        return None

    midpoint = 0.5 * (p_start + p_end)
    offset = normal * np.random.uniform(0.35, 0.75) * chord_norm
    return midpoint + offset


def interpolate_positions(p_start, p_end, n_steps, mode):
    if n_steps <= 1:
        return [p_start, p_end]

    if mode == 'linear':
        return [
            p_start + (i / n_steps) * (p_end - p_start)
            for i in range(n_steps + 1)
        ]

    if mode == 'cubic':
        return [
            p_start + smoothstep(i / n_steps) * (p_end - p_start)
            for i in range(n_steps + 1)
        ]

    if mode == 'spherical':
        center = build_spherical_center(p_start, p_end)
        if center is None:
            return interpolate_positions(p_start, p_end, n_steps, mode='linear')

        v0 = p_start - center
        v1 = p_end - center
        r0 = np.linalg.norm(v0)
        r1 = np.linalg.norm(v1)
        if r0 < 1e-6 or r1 < 1e-6:
            return interpolate_positions(p_start, p_end, n_steps, mode='linear')

        u0 = v0 / r0
        u1 = v1 / r1
        if abs(np.dot(u0, u1)) > 0.9995:
            return interpolate_positions(p_start, p_end, n_steps, mode='linear')

        positions = []
        for i in range(n_steps + 1):
            alpha = i / n_steps
            direction = slerp_unit_vectors(u0, u1, alpha)
            radius = (1.0 - alpha) * r0 + alpha * r1
            positions.append(center + direction * radius)
        return positions

    raise ValueError(f'Unknown interpolation mode: {mode}')


def estimate_translation_extent(p_start, p_end, mode):
    chord = np.linalg.norm(p_end - p_start)
    if mode == 'linear':
        return chord
    if mode == 'cubic':
        return chord * 1.05
    if mode == 'spherical':
        return chord * 1.35
    raise ValueError(f'Unknown interpolation mode: {mode}')


def interpolate_poses(T_start, T_end, trans_step=0.01, rot_step_deg=3.0, mode='random'):
    """
    Interpolate between two SE(3) poses with controlled step sizes.
    trans_step: max translation per frame (meters).
    rot_step_deg: max rotation per frame (degrees).
    """
    mode = choose_interpolation_mode(mode)
    delta_t = estimate_translation_extent(T_start[:3, 3], T_end[:3, 3], mode)
    R_rel = T_start[:3, :3].T @ T_end[:3, :3]
    delta_r = np.linalg.norm(Rot.from_matrix(R_rel).as_rotvec(degrees=True))

    n_trans = max(1, int(np.ceil(delta_t / trans_step)))
    n_rot = max(1, int(np.ceil(delta_r / rot_step_deg)))
    n_steps = max(n_trans, n_rot)

    if n_steps <= 1:
        return [T_start, T_end]

    key_rots = Rot.from_matrix(np.stack([T_start[:3, :3], T_end[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    positions = interpolate_positions(T_start[:3, 3], T_end[:3, 3], n_steps, mode)

    poses = []
    for i in range(n_steps + 1):
        alpha = i / n_steps
        T = np.eye(4)
        T[:3, 3] = positions[i]
        T[:3, :3] = slerp(alpha).as_matrix()
        poses.append(T)
    return poses


def stage_dwell_count(stage):
    return {
        'grasp': 3,
        'contact': 3,
        'place': 3,
        'release': 2,
        'lift': 1,
        'transfer': 1,
        'retreat': 1,
        'pull': 1,
        'push': 1,
        'settle': 2,
    }.get(stage, 0)


def generate_trajectory(scene_objects, task_waypoints, trans_step=0.01, rot_step_deg=3.0,
                        interpolation_mode='random'):
    """
    Generate a full trajectory from task waypoints.
    Returns (traj_poses, traj_grips, obj_transforms_per_frame).
    """
    first_target = task_waypoints[0]
    T_start = np.eye(4)
    T_start[:3, 3] = first_target['pos'] + np.array([
        np.random.uniform(-0.03, 0.03),
        np.random.uniform(-0.03, 0.03),
        np.random.uniform(0.18, 0.28),
    ])
    T_start[:3, :3] = build_downward_orientation(
        primary_dir=first_target['pos'] - T_start[:3, 3],
        continuity=0.0,
        tilt_deg=10.0,
    )

    wp_poses = [T_start]
    wp_grips = [first_target['grip']]
    prev_pose = T_start
    for i, waypoint in enumerate(task_waypoints):
        next_waypoint = task_waypoints[i + 1] if i + 1 < len(task_waypoints) else None
        T_wp = np.eye(4)
        T_wp[:3, 3] = waypoint['pos']
        T_wp[:3, :3] = waypoint_orientation(waypoint, prev_pose, next_waypoint, scene_objects)
        wp_poses.append(T_wp)
        wp_grips.append(waypoint['grip'])
        prev_pose = T_wp

    traj_poses = [wp_poses[0]]
    traj_grips = [wp_grips[0]]
    for i in range(1, len(wp_poses)):
        segment = interpolate_poses(wp_poses[i - 1], wp_poses[i], trans_step, rot_step_deg,
                                    mode=interpolation_mode)
        traj_poses.extend(segment[1:])
        traj_grips.extend([wp_grips[i]] * len(segment[1:]))

        dwell = stage_dwell_count(task_waypoints[i - 1]['stage'])
        for _ in range(dwell):
            traj_poses.append(wp_poses[i].copy())
            traj_grips.append(wp_grips[i])

    if len(traj_poses) < 3:
        for _ in range(3 - len(traj_poses)):
            traj_poses.append(traj_poses[-1].copy())
            traj_grips.append(traj_grips[-1])

    attached_obj_idx = None
    T_e_obj = None
    obj_transforms = []

    for T_w_e, grip in zip(traj_poses, traj_grips):
        frame_objs = [(m, T.copy()) for m, T in scene_objects]

        if grip == 1 and attached_obj_idx is None:
            ee_pos = T_w_e[:3, 3]
            min_dist = float('inf')
            best_idx = None
            for oi, (_, T_w_obj) in enumerate(scene_objects):
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
                T_w_e @ T_e_obj,
            )
            attached_obj_idx = None
            T_e_obj = None

        if attached_obj_idx is not None and T_e_obj is not None:
            frame_objs[attached_obj_idx] = (
                frame_objs[attached_obj_idx][0],
                T_w_e @ T_e_obj,
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

        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, np.array([0, 0, 1]))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        T_cam = np.eye(4)
        T_cam[:3, 0] = right
        T_cam[:3, 1] = up
        T_cam[:3, 2] = -forward
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

        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        v, u = np.where(depth > 0)
        if len(v) == 0:
            continue
        z = depth[v, u]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)
        pts_world = (cam_pose[:3, :3] @ pts_cam.T).T + cam_pose[:3, 3]
        all_points.append(pts_world)

    if not all_points:
        for mesh, T_w_obj in frame_objects:
            pts = mesh.sample(num_points // max(1, len(frame_objects)))
            pts_world = (T_w_obj[:3, :3] @ pts.T).T + T_w_obj[:3, 3]
            all_points.append(pts_world)

    combined = np.concatenate(all_points, axis=0)
    if len(combined) >= num_points:
        idx = np.random.choice(len(combined), num_points, replace=False)
    else:
        idx = np.random.choice(len(combined), num_points, replace=True)
    return combined[idx].astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Data augmentation
# ──────────────────────────────────────────────────────────────────────────────

def augment_trajectory(traj_poses, traj_grips):
    """Apply corrective augmentation without corrupting grip labels."""
    poses = [T.copy() for T in traj_poses]
    grips = list(traj_grips)

    if np.random.random() < 0.3:
        num_perturb = np.random.randint(1, max(2, len(poses) // 4))
        for _ in range(num_perturb):
            idx = np.random.randint(0, len(poses))
            poses[idx][:3, 3] += np.random.normal(0, 0.005, 3)
            perturb_rot = Rot.from_rotvec(np.random.normal(0, np.deg2rad(1.5), 3))
            poses[idx][:3, :3] = poses[idx][:3, :3] @ perturb_rot.as_matrix()

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
                         num_points=2048, fast=False, interpolation_mode='random'):
    """
    Generate one demonstration: trajectory + rendered point clouds.
    Returns {'pcds': [...], 'T_w_es': [...], 'grips': [...]}.
    """
    varied_objects = []
    for mesh, T_w_obj in scene_objects:
        T_var = T_w_obj.copy()
        T_var[:3, 3] += np.random.normal(0, 0.01, 3)
        T_var[:3, 3][2] = max(T_var[:3, 3][2], mesh.bounding_box.extents[2] / 2.0)
        yaw_perturb = np.random.uniform(-np.deg2rad(10), np.deg2rad(10))
        T_var[:3, :3] = T_var[:3, :3] @ Rot.from_euler('z', yaw_perturb).as_matrix()
        varied_objects.append((mesh, T_var))

    varied_waypoints = []
    for waypoint in task_waypoints:
        new_pos = waypoint['pos'].copy() + np.random.normal(0, 0.008, 3)
        new_pos = clamp_above_table(new_pos)
        varied_waypoints.append({
            'pos': new_pos,
            'grip': waypoint['grip'],
            'stage': waypoint['stage'],
            'obj_idx': waypoint['obj_idx'],
            'dir_hint': waypoint['dir_hint'],
        })

    traj_poses, traj_grips, obj_transforms = generate_trajectory(
        varied_objects, varied_waypoints, interpolation_mode=interpolation_mode)

    traj_poses, traj_grips = augment_trajectory(traj_poses, traj_grips)

    pcds = []
    for i in range(len(traj_poses)):
        frame_objs = obj_transforms[min(i, len(obj_transforms) - 1)]
        if fast:
            pcd = sample_object_pcds(frame_objs, num_points=num_points)
        else:
            pcd = render_object_pcds(frame_objs, camera, cam_poses, num_points=num_points)
        pcds.append(pcd)

    object_poses_seq = []
    object_ids_seq = []
    timestamps = []
    for frame_idx, frame_objs in enumerate(obj_transforms):
        object_poses_seq.append([T_w_obj for _, T_w_obj in frame_objs])
        object_ids_seq.append(list(range(len(frame_objs))))
        timestamps.append(frame_idx / 10.0)

    object_local_points = {}
    for obj_idx, (mesh, _) in enumerate(varied_objects):
        object_local_points[obj_idx] = sample_object_surface_points(mesh, points_per_obj=5)

    return {
        'pcds': pcds,
        'T_w_es': traj_poses,
        'grips': traj_grips,
        'object_poses_seq': object_poses_seq,
        'object_ids_seq': object_ids_seq,
        'timestamps': timestamps,
        'object_local_points': object_local_points,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main generation + save
# ──────────────────────────────────────────────────────────────────────────────

def _compute_offline_tracks(live_raw, live_processed, track_n_max=2, track_history_len=16,
                            track_points_per_obj=5, track_age_norm_max_sec=2.0):
    raw_poses = live_raw['T_w_es']
    raw_obj_poses = live_raw.get('object_poses_seq', [])
    raw_obj_ids = live_raw.get('object_ids_seq', [])
    raw_timestamps = live_raw.get('timestamps', list(range(len(raw_poses))))
    object_local_points = live_raw.get('object_local_points', {})

    current_track_seq = []
    current_track_valid = []
    current_track_age_sec = []

    for live_pose in live_processed['T_w_es']:
        matched_idx = min(len(raw_poses) - 1, len(live_processed['T_w_es']) - 1)
        min_err = float('inf')
        for idx, raw_pose in enumerate(raw_poses):
            err = np.linalg.norm(raw_pose[:3, 3] - live_pose[:3, 3])
            if err < min_err:
                min_err = err
                matched_idx = idx

        start_idx = max(0, matched_idx - track_history_len + 1)
        states = []
        for idx in range(start_idx, matched_idx + 1):
            states.append({
                'object_poses': raw_obj_poses[idx],
                'object_ids': raw_obj_ids[idx],
                'timestamp': raw_timestamps[idx],
            })

        world_tracks = build_object_tracks_world(
            states,
            points_per_obj=track_points_per_obj,
            n_max=track_n_max,
            history_len=track_history_len,
            object_local_points=object_local_points,
        )
        tracks_ee = project_tracks_to_current_ee(
            world_tracks['tracks_world'],
            world_tracks['track_valid'],
            live_pose,
        )
        track_age = compute_track_age_seconds(
            world_tracks['track_timestamps'],
            world_tracks['track_valid'],
            raw_timestamps[matched_idx],
            norm_max_sec=track_age_norm_max_sec,
        )

        current_track_seq.append(tracks_ee)
        current_track_valid.append(world_tracks['track_valid'])
        current_track_age_sec.append(track_age)

    live_processed['current_track_seq'] = current_track_seq
    live_processed['current_track_valid'] = current_track_valid
    live_processed['current_track_age_sec'] = current_track_age_sec
    return live_processed


def _render_memory_demo(memory_task, scene_objects, camera, cam_poses, num_points=2048, fast=False):
    pcds = []
    object_poses_seq = memory_task['object_poses_seq']
    object_ids_seq = memory_task['object_ids_seq']
    timestamps = memory_task['timestamps']
    poses = memory_task['T_w_es']
    grips = memory_task['grips']

    for frame_idx in range(len(poses)):
        frame_objs = []
        for obj_idx, (mesh, _) in enumerate(scene_objects):
            T_w_obj = object_poses_seq[frame_idx][obj_idx]
            frame_objs.append((mesh, T_w_obj))
        if fast:
            pcd = sample_object_pcds(frame_objs, num_points=num_points)
        else:
            pcd = render_object_pcds(frame_objs, camera, cam_poses, num_points=num_points)
        pcds.append(pcd)

    object_local_points = {}
    for obj_idx, (mesh, _) in enumerate(scene_objects):
        object_local_points[obj_idx] = sample_object_surface_points(mesh, points_per_obj=5)

    return {
        'pcds': pcds,
        'T_w_es': poses,
        'grips': grips,
        'object_poses_seq': object_poses_seq,
        'object_ids_seq': object_ids_seq,
        'timestamps': timestamps,
        'object_local_points': object_local_points,
        'meta': memory_task.get('meta'),
    }


def generate_one_sample(sample_idx, shapenet_path, save_dir, num_demos, num_waypoints_demo,
                        pred_horizon, live_spacing_trans, live_spacing_rot,
                        scene_encoder, offset_base, fast=False, interpolation_mode='random',
                        store_tracks=False, task_source='baseline', memory_task_generator=None,
                        track_n_max=2, track_history_len=16, track_points_per_obj=5,
                        track_age_norm_max_sec=2.0):
    """Generate and save one full sample (num_demos + 1 live)."""
    np.random.seed()

    scene_objects = create_scene(shapenet_path)
    task_waypoints = sample_pseudo_task(scene_objects)
    camera, cam_poses = setup_cameras(num_cameras=3)

    demos_raw = []
    if task_source == 'memory' and memory_task_generator is not None:
        memory_task = memory_task_generator.generate_task(scene_objects)
        for _ in range(num_demos + 1):
            demo = _render_memory_demo(memory_task, scene_objects, camera, cam_poses, fast=fast)
            demos_raw.append(demo)
    else:
        for _ in range(num_demos + 1):
            demo = generate_single_demo(scene_objects, task_waypoints, camera, cam_poses,
                                        fast=fast, interpolation_mode=interpolation_mode)
            demos_raw.append(demo)

    full_sample = {
        'demos': [None] * num_demos,
        'live': None,
    }

    for i in range(num_demos):
        demo_processed = sample_to_cond_demo(demos_raw[i], num_waypoints_demo)
        n = len(demo_processed['obs'])
        if n > num_waypoints_demo:
            indices = np.round(np.linspace(0, n - 1, num_waypoints_demo)).astype(int)
            demo_processed['obs'] = [demo_processed['obs'][j] for j in indices]
            demo_processed['grips'] = [demo_processed['grips'][j] for j in indices]
            demo_processed['T_w_es'] = [demo_processed['T_w_es'][j] for j in indices]
        elif n < num_waypoints_demo:
            while len(demo_processed['obs']) < num_waypoints_demo:
                demo_processed['obs'].append(demo_processed['obs'][-1])
                demo_processed['grips'].append(demo_processed['grips'][-1])
                demo_processed['T_w_es'].append(demo_processed['T_w_es'][-1])
        full_sample['demos'][i] = demo_processed

    live_use_subsample = (task_source != 'memory')
    full_sample['live'] = sample_to_live(demos_raw[-1], pred_horizon, 2048,
                                         live_spacing_trans, live_spacing_rot,
                                         subsample=live_use_subsample)
    if store_tracks:
        full_sample['live'] = _compute_offline_tracks(
            demos_raw[-1],
            full_sample['live'],
            track_n_max=track_n_max,
            track_history_len=track_history_len,
            track_points_per_obj=track_points_per_obj,
            track_age_norm_max_sec=track_age_norm_max_sec,
        )

    offset = offset_base
    save_sample(full_sample, save_dir=save_dir, offset=offset, scene_encoder=scene_encoder)

    num_saved = len(full_sample['live']['obs'])
    return num_saved


def generate_and_save(shapenet_path, save_dir, num_samples, num_demos=2,
                      num_waypoints_demo=10, pred_horizon=8,
                      live_spacing_trans=0.01, live_spacing_rot=3,
                      scene_encoder=None, continuous=False, buffer_size=10000,
                      fast=False, interpolation_mode='random', store_tracks=False,
                      task_source='baseline', memory_task_generator=None,
                      track_n_max=2, track_history_len=16,
                      track_points_per_obj=5, track_age_norm_max_sec=2.0):
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
                    scene_encoder, offset % buffer_size, fast=fast,
                    interpolation_mode=interpolation_mode,
                    store_tracks=store_tracks,
                    task_source=task_source,
                    memory_task_generator=memory_task_generator,
                    track_n_max=track_n_max,
                    track_history_len=track_history_len,
                    track_points_per_obj=track_points_per_obj,
                    track_age_norm_max_sec=track_age_norm_max_sec,
                )
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
                    scene_encoder, offset, fast=fast,
                    interpolation_mode=interpolation_mode,
                    store_tracks=store_tracks,
                    task_source=task_source,
                    memory_task_generator=memory_task_generator,
                    track_n_max=track_n_max,
                    track_history_len=track_history_len,
                    track_points_per_obj=track_points_per_obj,
                    track_age_norm_max_sec=track_age_norm_max_sec,
                )
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
                        help='Run indefinitely, overwriting a fixed buffer. Designed to run in parallel with training.')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='Buffer size for continuous mode (number of .pt files).')
    parser.add_argument('--compute_embeddings', action='store_true',
                        help='Pre-compute scene encoder embeddings. Not recommended for continuous mode (requires GPU).')
    parser.add_argument('--encoder_path', type=str, default='./checkpoints/scene_encoder.pt',
                        help='Path to scene_encoder.pt (use ip.extract_scene_encoder to extract from full checkpoint).')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fast', action='store_true',
                        help='Skip pyrender, sample points directly from mesh surfaces (~10x faster).')
    parser.add_argument('--interpolation_mode', type=str, default='random',
                        choices=['random', 'linear', 'cubic', 'spherical'],
                        help='Translation interpolation strategy between waypoints.')
    parser.add_argument('--store_tracks', action='store_true',
                        help='Precompute and store offline track tensors for history-aware training, matching HistRISE train-time convention.')
    parser.add_argument('--track_n_max', type=int, default=2)
    parser.add_argument('--track_history_len', type=int, default=16)
    parser.add_argument('--track_points_per_obj', type=int, default=5)
    parser.add_argument('--track_age_norm_max_sec', type=float, default=2.0)
    parser.add_argument('--task_source', type=str, default='baseline', choices=['baseline', 'memory'],
                        help='Choose baseline pseudo-data or optional memory-task data source.')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    scene_encoder = None
    memory_task_generator = None
    if args.task_source == 'memory':
        memory_task_generator = MemoryTaskGenerator()

    if args.compute_embeddings:
        import torch
        from ip.models.scene_encoder import SceneEncoder
        scene_encoder = SceneEncoder(num_freqs=10, embd_dim=512)
        scene_encoder.load_state_dict(torch.load(args.encoder_path))
        scene_encoder = scene_encoder.to('cuda')
        scene_encoder.eval()
        print("Scene encoder loaded.")

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
            interpolation_mode=args.interpolation_mode,
            store_tracks=args.store_tracks,
            task_source=args.task_source,
            memory_task_generator=memory_task_generator,
            track_n_max=args.track_n_max,
            track_history_len=args.track_history_len,
            track_points_per_obj=args.track_points_per_obj,
            track_age_norm_max_sec=args.track_age_norm_max_sec,
        )

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
        interpolation_mode=args.interpolation_mode,
        store_tracks=args.store_tracks,
        task_source=args.task_source,
        memory_task_generator=memory_task_generator,
        track_n_max=args.track_n_max,
        track_history_len=args.track_history_len,
        track_points_per_obj=args.track_points_per_obj,
        track_age_norm_max_sec=args.track_age_norm_max_sec,
    )


if __name__ == '__main__':
    main()
