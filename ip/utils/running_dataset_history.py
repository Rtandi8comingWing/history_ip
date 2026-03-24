from torch.utils.data import Dataset
import torch
import os
import numpy as np


class RunningDatasetHistory(Dataset):
    def __init__(self, data_path, num_samples, rand_g_prob=0.0,
                 track_n_max=2, track_history_len=16,
                 track_points_per_obj=5, track_age_norm_max_sec=2.0):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.track_n_max = track_n_max
        self.track_history_len = track_history_len
        self.track_points_per_obj = track_points_per_obj
        self.track_age_norm_max_sec = track_age_norm_max_sec
        self.data_attr = [
            'actions', 'actions_grip',
            'current_track_seq', 'current_track_valid', 'current_track_age_sec'
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        while True:
            try:
                data = torch.load(os.path.join(self.data_path, f'data_{idx}.pt'))
                for attr in self.data_attr:
                    assert hasattr(data, attr)

                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                return data
            except Exception:
                idx = np.random.randint(0, self.num_samples)
