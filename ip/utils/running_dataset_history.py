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
                 dynamic_num_samples=False, file_pattern='data_*.pt'):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.track_n_max = track_n_max
        self.track_history_len = track_history_len
        self.track_points_per_obj = track_points_per_obj
        self.track_age_norm_max_sec = track_age_norm_max_sec
        self.dynamic_num_samples = dynamic_num_samples or num_samples is None
        self.file_pattern = file_pattern
        self.data_attr = [
            'actions', 'actions_grip',
            'current_track_seq', 'current_track_valid', 'current_track_age_sec'
        ]

    def _count_samples(self):
        return len(glob.glob(os.path.join(self.data_path, self.file_pattern)))

    def _get_num_samples(self):
        if self.dynamic_num_samples:
            self.num_samples = self._count_samples()
        return 0 if self.num_samples is None else self.num_samples

    def __len__(self):
        return self._get_num_samples()

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

                return data
            except Exception:
                num_samples = self._get_num_samples()
                if num_samples > 0:
                    idx = np.random.randint(0, num_samples)
                else:
                    time.sleep(0.2)
