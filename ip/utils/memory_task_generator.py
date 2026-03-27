import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class MemoryTaskMeta:
    task_type: str
    decision_points: List[int]
    decision_labels: List[int]
    memory_aspects: List[str]


class MemoryTaskGenerator:
    """Generate memory-sensitive synthetic tasks for history-aware training.

    Each template is designed so that the current observation near a decision
    point is intentionally similar across labels, while the correct future
    action branch depends on the earlier object-centric history.
    """

    def __init__(self, control_hz: float = 15.0):
        self.control_hz = control_hz
        self.dt = 1.0 / control_hz

    def sample_task_type(self) -> str:
        return random.choice(['counting', 'spatial', 'stage', 'preloaded', 'continuous'])

    def generate_task(self, scene_objects: List, task_type: Optional[str] = None, difficulty: int = 1) -> Dict:
        if len(scene_objects) < 2:
            raise ValueError('MemoryTaskGenerator expects at least two scene objects.')

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

    def _ee_pose(self, position: np.ndarray) -> np.ndarray:
        T_w_e = np.eye(4)
        T_w_e[:3, 3] = np.asarray(position, dtype=np.float64)
        return T_w_e

    def _scene_state(self, base_transforms: List[np.ndarray], object_positions: List[np.ndarray]) -> List[np.ndarray]:
        state = []
        for base_T, pos in zip(base_transforms, object_positions):
            T_obj = base_T.copy()
            T_obj[:3, 3] = np.asarray(pos, dtype=np.float64)
            state.append(T_obj)
        return state

    def _initialize_episode(self, scene_objects: List, ee_start: np.ndarray, object_positions: List[np.ndarray], grip: int = 0) -> Dict:
        base_transforms = [T_w_obj.copy() for _, T_w_obj in scene_objects]
        return {
            'base_transforms': base_transforms,
            'poses': [self._ee_pose(ee_start)],
            'grips': [int(grip)],
            'object_poses_seq': [self._scene_state(base_transforms, object_positions)],
            'object_ids_seq': [list(range(len(scene_objects)))],
            'timestamps': [0.0],
            'current_ee': np.asarray(ee_start, dtype=np.float64),
            'current_objects': [np.asarray(pos, dtype=np.float64).copy() for pos in object_positions],
        }

    def _append_segment(self, episode: Dict, ee_target: np.ndarray, cue_target: np.ndarray, steps: int,
                        grip: int, cue_idx: int = 0) -> None:
        ee_target = np.asarray(ee_target, dtype=np.float64)
        cue_target = np.asarray(cue_target, dtype=np.float64)
        ee_start = episode['current_ee']
        cue_start = episode['current_objects'][cue_idx]
        steps = max(1, int(steps))

        for substep in range(1, steps + 1):
            alpha = substep / steps
            ee_pos = ee_start * (1.0 - alpha) + ee_target * alpha
            cue_pos = cue_start * (1.0 - alpha) + cue_target * alpha
            object_positions = [pos.copy() for pos in episode['current_objects']]
            object_positions[cue_idx] = cue_pos

            episode['poses'].append(self._ee_pose(ee_pos))
            episode['grips'].append(int(grip))
            episode['object_poses_seq'].append(self._scene_state(episode['base_transforms'], object_positions))
            episode['object_ids_seq'].append(list(range(len(object_positions))))
            episode['timestamps'].append((len(episode['poses']) - 1) * self.dt)

        episode['current_ee'] = ee_target
        episode['current_objects'][cue_idx] = cue_target

    def _hold(self, episode: Dict, steps: int, grip: int) -> None:
        self._append_segment(episode, episode['current_ee'], episode['current_objects'][0], steps=steps, grip=grip)

    def _finalize_episode(self, episode: Dict, task_type: str, decision_points: List[int], decision_labels: List[int],
                          memory_aspects: List[str]) -> Dict:
        return {
            'T_w_es': episode['poses'],
            'grips': episode['grips'],
            'object_poses_seq': episode['object_poses_seq'],
            'object_ids_seq': episode['object_ids_seq'],
            'timestamps': episode['timestamps'],
            'meta': MemoryTaskMeta(
                task_type=task_type,
                decision_points=decision_points,
                decision_labels=decision_labels,
                memory_aspects=memory_aspects,
            ),
        }

    def _task_layout(self, scene_objects: List) -> Dict[str, np.ndarray]:
        base_positions = [T_w_obj[:3, 3].copy() for _, T_w_obj in scene_objects]
        cue_home = base_positions[0].copy()
        target_home = base_positions[1].copy()
        workspace_mid = 0.5 * (cue_home + target_home)
        workspace_mid[2] = max(cue_home[2], target_home[2])

        cue_left = cue_home + np.array([-0.10, 0.0, 0.0])
        cue_right = cue_home + np.array([0.10, 0.0, 0.0])
        cue_front = cue_home + np.array([0.0, 0.09, 0.0])
        cue_back = cue_home + np.array([0.0, -0.09, 0.0])
        cue_diag_left = cue_home + np.array([-0.07, 0.07, 0.0])
        cue_diag_right = cue_home + np.array([0.07, 0.07, 0.0])

        observe_pose = workspace_mid + np.array([0.0, -0.14, 0.18])
        decision_pose = target_home + np.array([0.0, -0.11, 0.16])
        left_goal = target_home + np.array([-0.09, 0.0, 0.10])
        right_goal = target_home + np.array([0.09, 0.0, 0.10])
        finish_lift = np.array([0.0, 0.0, 0.05])

        return {
            'base_positions': base_positions,
            'cue_home': cue_home,
            'cue_left': cue_left,
            'cue_right': cue_right,
            'cue_front': cue_front,
            'cue_back': cue_back,
            'cue_diag_left': cue_diag_left,
            'cue_diag_right': cue_diag_right,
            'observe_pose': observe_pose,
            'decision_pose': decision_pose,
            'left_goal': left_goal,
            'right_goal': right_goal,
            'finish_lift': finish_lift,
        }

    def _build_task(self, scene_objects: List, task_type: str, branch_label: int,
                    cue_waypoints: List[np.ndarray], cue_steps: List[int], memory_aspects: List[str],
                    delay_segments: int = 2, oscillate_decision: bool = False) -> Dict:
        layout = self._task_layout(scene_objects)
        episode = self._initialize_episode(scene_objects, layout['observe_pose'], layout['base_positions'], grip=0)

        for cue_target, steps in zip(cue_waypoints, cue_steps):
            ee_target = layout['observe_pose']
            if oscillate_decision:
                ee_target = layout['decision_pose'] if random.random() < 0.5 else layout['observe_pose']
            self._append_segment(episode, ee_target, cue_target, steps=steps, grip=0)

        self._append_segment(episode, layout['decision_pose'], layout['cue_home'], steps=4, grip=0)
        for _ in range(max(0, delay_segments)):
            self._append_segment(episode, layout['decision_pose'], layout['cue_home'], steps=3, grip=0)

        decision_point = len(episode['poses']) - 1
        branch_goal = layout['left_goal'] if branch_label == 0 else layout['right_goal']
        self._append_segment(episode, branch_goal, layout['cue_home'], steps=6, grip=1)
        self._append_segment(episode, branch_goal + layout['finish_lift'], layout['cue_home'], steps=3, grip=1)

        return self._finalize_episode(
            episode,
            task_type=task_type,
            decision_points=[decision_point],
            decision_labels=[branch_label],
            memory_aspects=memory_aspects,
        )

    def _gen_counting_task(self, scene_objects: List, difficulty: int) -> Dict:
        layout = self._task_layout(scene_objects)
        num_visits = random.randint(2, min(5, 3 + max(0, difficulty)))
        cue_waypoints = []
        cue_steps = []
        for idx in range(num_visits):
            cue_waypoints.append(layout['cue_left'] if idx % 2 == 0 else layout['cue_right'])
            cue_steps.append(3)
            cue_waypoints.append(layout['cue_home'])
            cue_steps.append(2)
        branch_label = num_visits % 2
        return self._build_task(
            scene_objects,
            task_type='counting',
            branch_label=branch_label,
            cue_waypoints=cue_waypoints,
            cue_steps=cue_steps,
            memory_aspects=['counting'],
            delay_segments=1,
        )

    def _gen_spatial_task(self, scene_objects: List, difficulty: int) -> Dict:
        layout = self._task_layout(scene_objects)
        branch_label = random.randint(0, 1)
        cue_signal = layout['cue_left'] if branch_label == 0 else layout['cue_right']
        cue_waypoints = [cue_signal, cue_signal, layout['cue_home'], layout['cue_home']]
        cue_steps = [4, 3, 4, 2]
        return self._build_task(
            scene_objects,
            task_type='spatial',
            branch_label=branch_label,
            cue_waypoints=cue_waypoints,
            cue_steps=cue_steps,
            memory_aspects=['spatial_memorization'],
            delay_segments=2 + max(0, difficulty - 1),
        )

    def _gen_stage_task(self, scene_objects: List, difficulty: int) -> Dict:
        layout = self._task_layout(scene_objects)
        branch_label = random.randint(0, 1)
        second_stage = layout['cue_front'] if branch_label == 0 else layout['cue_back']
        cue_waypoints = [
            layout['cue_left'],
            layout['cue_home'],
            second_stage,
            layout['cue_home'],
            layout['cue_diag_left'],
            layout['cue_home'],
        ]
        cue_steps = [3, 2, 3, 2, 3, 2]
        return self._build_task(
            scene_objects,
            task_type='stage',
            branch_label=branch_label,
            cue_waypoints=cue_waypoints,
            cue_steps=cue_steps,
            memory_aspects=['stage_identification'],
            delay_segments=1,
            oscillate_decision=True,
        )

    def _gen_preloaded_task(self, scene_objects: List, difficulty: int) -> Dict:
        layout = self._task_layout(scene_objects)
        branch_label = random.randint(0, 1)
        early_cue = layout['cue_left'] if branch_label == 0 else layout['cue_right']
        cue_waypoints = [early_cue, early_cue, layout['cue_home']]
        cue_steps = [4, 3, 4]
        return self._build_task(
            scene_objects,
            task_type='preloaded',
            branch_label=branch_label,
            cue_waypoints=cue_waypoints,
            cue_steps=cue_steps,
            memory_aspects=['preloaded_memory'],
            delay_segments=4 + max(0, difficulty - 1),
        )

    def _gen_continuous_task(self, scene_objects: List, difficulty: int) -> Dict:
        layout = self._task_layout(scene_objects)
        branch_label = random.randint(0, 1)
        if branch_label == 0:
            cue_waypoints = [
                layout['cue_right'],
                layout['cue_diag_right'],
                layout['cue_front'],
                layout['cue_diag_left'],
                layout['cue_left'],
                layout['cue_home'],
            ]
        else:
            cue_waypoints = [
                layout['cue_left'],
                layout['cue_diag_left'],
                layout['cue_front'],
                layout['cue_diag_right'],
                layout['cue_right'],
                layout['cue_home'],
            ]
        cue_steps = [2 + max(0, difficulty - 1), 2, 2, 2, 2 + max(0, difficulty - 1), 3]
        return self._build_task(
            scene_objects,
            task_type='continuous',
            branch_label=branch_label,
            cue_waypoints=cue_waypoints,
            cue_steps=cue_steps,
            memory_aspects=['continuous_memory'],
            delay_segments=1,
        )
