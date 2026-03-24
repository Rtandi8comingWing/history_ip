import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import random


@dataclass
class MemoryTaskMeta:
    task_type: str
    decision_points: List[int]
    decision_labels: List[int]
    memory_aspects: List[str]


class MemoryTaskGenerator:
    def __init__(self, control_hz: float = 15.0):
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

    def sample_task_type(self) -> str:
        return random.choice(['counting', 'spatial', 'stage', 'preloaded', 'continuous'])

    def generate_task(self, scene_objects: List, task_type: Optional[str] = None, difficulty: int = 1) -> Dict:
        if task_type is None:
            task_type = self.sample_task_type()

        if task_type == 'counting':
            return self._gen_counting_task(scene_objects, difficulty)
        if task_type == 'spatial':
            return self._gen_spatial_task(scene_objects, difficulty)
        if task_type == 'stage':
            return self._gen_stage_task(scene_objects, difficulty)
        if task_type == 'preloaded':
            return self._gen_preloaded_task(scene_objects, difficulty)
        if task_type == 'continuous':
            return self._gen_continuous_task(scene_objects, difficulty)
        raise ValueError(f'Unknown task type: {task_type}')

    def _base_task(self, scene_objects: List, task_type: str, decision_points: List[int], decision_labels: List[int], memory_aspects: List[str]) -> Dict:
        object_poses_seq = []
        object_ids_seq = []
        timestamps = []
        poses = []
        grips = []

        base_positions = [T_w_obj[:3, 3].copy() for _, T_w_obj in scene_objects]
        current_time = 0.0
        num_steps = 24
        start = np.array([0.0, 0.0, 0.25])
        end = np.array([0.2, 0.0, 0.08])

        for step in range(num_steps):
            alpha = step / max(num_steps - 1, 1)
            T_w_e = np.eye(4)
            T_w_e[:3, 3] = start * (1 - alpha) + end * alpha
            poses.append(T_w_e)
            grips.append(1 if step > num_steps // 2 else 0)
            object_poses = []
            for obj_idx, (_, T_w_obj) in enumerate(scene_objects):
                T_obj = T_w_obj.copy()
                if task_type == 'continuous':
                    jitter = np.array([0.002 * np.sin(step / 3.0 + obj_idx), 0.002 * np.cos(step / 4.0 + obj_idx), 0.0])
                    T_obj[:3, 3] = base_positions[obj_idx] + jitter
                object_poses.append(T_obj)
            object_poses_seq.append(object_poses)
            object_ids_seq.append(list(range(len(scene_objects))))
            timestamps.append(current_time)
            current_time += self.dt

        return {
            'T_w_es': poses,
            'grips': grips,
            'object_poses_seq': object_poses_seq,
            'object_ids_seq': object_ids_seq,
            'timestamps': timestamps,
            'meta': MemoryTaskMeta(
                task_type=task_type,
                decision_points=decision_points,
                decision_labels=decision_labels,
                memory_aspects=memory_aspects,
            ),
        }

    def _gen_counting_task(self, scene_objects: List, difficulty: int) -> Dict:
        return self._base_task(scene_objects, 'counting', [8, 16], [0, 1], ['counting'])

    def _gen_spatial_task(self, scene_objects: List, difficulty: int) -> Dict:
        return self._base_task(scene_objects, 'spatial', [12], [0], ['spatial_memorization'])

    def _gen_stage_task(self, scene_objects: List, difficulty: int) -> Dict:
        return self._base_task(scene_objects, 'stage', [10, 18], [0, 1], ['stage_identification'])

    def _gen_preloaded_task(self, scene_objects: List, difficulty: int) -> Dict:
        return self._base_task(scene_objects, 'preloaded', [14], [0], ['preloaded_memory'])

    def _gen_continuous_task(self, scene_objects: List, difficulty: int) -> Dict:
        return self._base_task(scene_objects, 'continuous', [15], [0], ['continuous_memory'])
