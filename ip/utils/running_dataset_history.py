from torch.utils.data import Dataset
import torch
import os
import glob
import time
import numpy as np


class RunningDatasetHistory(Dataset):
    def __init__(self, data_path, num_samples=None, rand_g_prob=0.0,
                 track_n_max=2, track_history_len=16,
                 track_points_per_obj=5, track_age_norm_max_sec=2.0,
                 dynamic_num_samples=False, file_pattern='data_*.pt',
                 track_recent_drop_prob=0.0, track_recent_drop_max_steps=0,
                 track_recent_drop_dt_sec=0.1):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.track_n_max = track_n_max
        self.track_history_len = track_history_len
        self.track_points_per_obj = track_points_per_obj
        self.track_age_norm_max_sec = track_age_norm_max_sec
        self.dynamic_num_samples = dynamic_num_samples or num_samples is None
        self.file_pattern = file_pattern
        self.track_recent_drop_prob = track_recent_drop_prob
        self.track_recent_drop_max_steps = track_recent_drop_max_steps
        self.track_recent_drop_dt_sec = track_recent_drop_dt_sec
        self.data_attr = [
            'actions', 'actions_grip',
            'current_track_seq', 'current_track_valid', 'current_track_lengths', 'current_track_age_sec'
        ]

    def _count_samples(self):
        return len(glob.glob(os.path.join(self.data_path, self.file_pattern)))

    def _get_num_samples(self):
        if self.dynamic_num_samples:
            self.num_samples = self._count_samples()
        return 0 if self.num_samples is None else self.num_samples

    def __len__(self):
        return self._get_num_samples()

    def _apply_recent_track_drop(self, data):
        if self.track_recent_drop_prob <= 0 or self.track_recent_drop_max_steps <= 0:
            return data
        if np.random.uniform() >= self.track_recent_drop_prob:
            return data
        if not hasattr(data, 'current_track_seq') or data.current_track_seq is None:
            return data

        seq = data.current_track_seq.clone()
        valid = data.current_track_valid.clone() if hasattr(data, 'current_track_valid') else None
        lengths = data.current_track_lengths.clone() if hasattr(data, 'current_track_lengths') else None
        age = data.current_track_age_sec.clone() if hasattr(data, 'current_track_age_sec') else None

        history_len = seq.shape[-3]
        if history_len <= 1:
            return data

        drop_steps = np.random.randint(1, min(self.track_recent_drop_max_steps, history_len - 1) + 1)

        if lengths is not None:
            flat_lengths = lengths.view(-1)
            flat_valid = valid.view(-1) if valid is not None else torch.ones_like(flat_lengths, dtype=torch.bool)
            flat_seq = seq.view(-1, history_len, seq.shape[-2], seq.shape[-1])
            for idx in range(flat_seq.shape[0]):
                if not flat_valid[idx]:
                    continue
                effective_len = int(flat_lengths[idx].item())
                if effective_len <= 1:
                    continue
                effective_drop = min(drop_steps, effective_len - 1)
                stale_idx = effective_len - 1 - effective_drop
                flat_seq[idx, stale_idx + 1:effective_len] = flat_seq[idx, stale_idx:stale_idx + 1].expand(effective_drop, -1, -1)
                flat_lengths[idx] = stale_idx + 1
            seq = flat_seq.view_as(seq)
            lengths = flat_lengths.view_as(lengths)
            data.current_track_lengths = lengths
        else:
            stale_idx = history_len - 1 - drop_steps
            stale_slice = seq[..., stale_idx:stale_idx + 1, :, :]
            seq[..., stale_idx + 1:, :, :] = stale_slice.expand(*seq[..., stale_idx + 1:, :, :].shape)

        data.current_track_seq = seq

        if age is not None:
            age_delta = drop_steps * self.track_recent_drop_dt_sec / max(self.track_age_norm_max_sec, 1e-6)
            if valid is not None:
                valid_mask = valid.to(dtype=age.dtype).unsqueeze(-1)
                age = torch.clamp(age + age_delta * valid_mask, 0.0, 1.0)
            else:
                age = torch.clamp(age + age_delta, 0.0, 1.0)
            data.current_track_age_sec = age

        return data

    def __getitem__(self, idx):
        while True:
            num_samples = self._get_num_samples()
            if num_samples <= 0:
                time.sleep(0.2)
                continue

            idx = int(idx) % num_samples
            try:
                data = torch.load(os.path.join(self.data_path, f'data_{idx}.pt'), weights_only=False)
                for attr in self.data_attr:
                    assert hasattr(data, attr)

                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                data = self._apply_recent_track_drop(data)
                return data
            except Exception:
                num_samples = self._get_num_samples()
                if num_samples > 0:
                    idx = np.random.randint(0, num_samples)
                else:
                    time.sleep(0.2)
