import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

from ip.utils.common_utils import transform_pcd


class TrackBuffer:
    def __init__(self,
                 track_n_max: int = 2,
                 track_history_len: int = 16,
                 track_points_per_obj: int = 5,
                 device: str = 'cuda'):
        self.track_n_max = track_n_max
        self.track_history_len = track_history_len
        self.track_points_per_obj = track_points_per_obj
        self.device = device
        self.buffer: List[Dict] = []
        self.next_object_id = 0
        self.active_objects: Dict[int, np.ndarray] = {}

    def reset(self):
        self.buffer = []
        self.next_object_id = 0
        self.active_objects = {}

    def update(self, pcd: np.ndarray, T_w_e: np.ndarray, debug: bool = False):
        pcd_ee = transform_pcd(pcd, np.linalg.inv(T_w_e))
        object_centers, object_points = self._segment_objects(pcd_ee, debug=debug)
        tracked_objects = self._associate_objects(object_centers, object_points, debug=debug)
        self.buffer.append({
            'tracked_objects': tracked_objects,
            'T_w_e': T_w_e.copy(),
            'timestamp': len(self.buffer),
        })
        if len(self.buffer) > self.track_history_len:
            self.buffer.pop(0)

    def _segment_objects(self, pcd: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        if len(pcd) == 0:
            return np.zeros((0, 3)), []

        if len(pcd) > 1000:
            indices = np.random.choice(len(pcd), 1000, replace=False)
            pcd_sample = pcd[indices]
        else:
            pcd_sample = pcd

        n_clusters = min(self.track_n_max, max(1, len(pcd_sample) // 100))
        if n_clusters == 0:
            return np.zeros((0, 3)), []

        center_indices = np.random.choice(len(pcd_sample), n_clusters, replace=False)
        centers = pcd_sample[center_indices].copy()

        for _ in range(3):
            distances = np.linalg.norm(pcd_sample[:, None, :] - centers[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            for i in range(n_clusters):
                mask = labels == i
                if np.any(mask):
                    centers[i] = pcd_sample[mask].mean(axis=0)

        object_points = []
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                cluster_points = pcd_sample[mask]
                replace = len(cluster_points) < self.track_points_per_obj
                indices = np.random.choice(len(cluster_points), self.track_points_per_obj, replace=replace)
                object_points.append(cluster_points[indices])
            else:
                object_points.append(np.zeros((self.track_points_per_obj, 3), dtype=np.float32))

        return centers, object_points

    def _associate_objects(self,
                           object_centers: np.ndarray,
                           object_points: List[np.ndarray],
                           debug: bool = False) -> Dict[int, np.ndarray]:
        tracked_objects: Dict[int, np.ndarray] = {}
        if len(object_centers) == 0:
            return tracked_objects

        if len(self.active_objects) == 0:
            for i, points in enumerate(object_points[:self.track_n_max]):
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.active_objects[obj_id] = object_centers[i]
                tracked_objects[obj_id] = points
            return tracked_objects

        active_ids = list(self.active_objects.keys())
        active_positions = np.array([self.active_objects[oid] for oid in active_ids])
        distances = np.linalg.norm(object_centers[:, None, :] - active_positions[None, :, :], axis=2)

        matched_detections = set()
        matched_objects = set()
        for _ in range(min(len(object_centers), len(active_ids))):
            det_idx, obj_idx = np.unravel_index(distances.argmin(), distances.shape)
            min_dist = distances[det_idx, obj_idx]
            if min_dist < 0.15:
                obj_id = active_ids[obj_idx]
                tracked_objects[obj_id] = object_points[det_idx]
                self.active_objects[obj_id] = object_centers[det_idx]
                matched_detections.add(det_idx)
                matched_objects.add(obj_id)
                distances[det_idx, :] = np.inf
                distances[:, obj_idx] = np.inf
            else:
                break

        for obj_id in list(self.active_objects.keys()):
            if obj_id not in matched_objects:
                del self.active_objects[obj_id]

        for det_idx in range(len(object_centers)):
            if det_idx not in matched_detections and len(tracked_objects) < self.track_n_max:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.active_objects[obj_id] = object_centers[det_idx]
                tracked_objects[obj_id] = object_points[det_idx]

        return tracked_objects

    def get_track_data(self) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.buffer) == 0:
            return None

        history_len = len(self.buffer)
        track_seq = torch.zeros(
            1, self.track_n_max, history_len, self.track_points_per_obj, 3,
            dtype=torch.float32, device=self.device
        )
        track_valid = torch.zeros(1, self.track_n_max, dtype=torch.bool, device=self.device)
        track_lengths = torch.zeros(1, self.track_n_max, dtype=torch.long, device=self.device)
        track_age_sec = torch.zeros(1, self.track_n_max, 1, dtype=torch.float32, device=self.device)

        current_frame = self.buffer[-1]
        current_obj_ids = sorted(list(current_frame['tracked_objects'].keys()))[:self.track_n_max]
        obj_id_to_slot = {}
        for slot, obj_id in enumerate(current_obj_ids):
            obj_id_to_slot[obj_id] = slot
            track_valid[0, slot] = True

        for h, frame in enumerate(self.buffer):
            for obj_id, slot in obj_id_to_slot.items():
                if obj_id in frame['tracked_objects']:
                    points = frame['tracked_objects'][obj_id]
                    track_seq[0, slot, h] = torch.from_numpy(points).float().to(self.device)
                    track_lengths[0, slot] += 1

        max_age_sec = 2.0
        current_time = current_frame['timestamp']
        for obj_id, slot in obj_id_to_slot.items():
            first_seen_time = current_time
            for frame in self.buffer:
                if obj_id in frame['tracked_objects']:
                    first_seen_time = frame['timestamp']
                    break
            age_sec = (current_time - first_seen_time) * 0.1
            track_age_sec[0, slot, 0] = min(age_sec / max_age_sec, 1.0)

        return {
            'track_seq': track_seq,
            'track_valid': track_valid,
            'track_lengths': track_lengths,
            'track_age_sec': track_age_sec,
        }
