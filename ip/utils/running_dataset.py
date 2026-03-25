from torch.utils.data import Dataset
import torch
import os
import glob
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class RunningDataset(Dataset):
    def __init__(self, data_path, num_samples=None, rec=False, rand_g_prob=0.0,
                 random_rotation=False, dynamic_num_samples=False, file_pattern='data_*.pt'):
        self.data_path = data_path
        self.num_samples = num_samples
        self.rand_g_prob = rand_g_prob
        self.random_rotation = random_rotation
        self.rec = rec
        self.dynamic_num_samples = dynamic_num_samples or num_samples is None
        self.file_pattern = file_pattern
        if rec:
            self.data_attr = [
                'pos',
                'queries',
                'batch_queries',
                'batch_pos',
                'occupancy',
            ]
        else:
            self.data_attr = [
                'actions',
                'actions_grip',
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
                # Make sure that the data has all the required attributes.
                for attr in self.data_attr:
                    assert hasattr(data, attr)

                if np.random.uniform() < self.rand_g_prob:
                    data.current_grip *= -1

                if self.random_rotation and self.rec:
                    R = torch.tensor(Rot.random().as_matrix(), dtype=data.pos.dtype, device=data.pos.device)
                    data.pos = data.pos @ R.T
                    data.queries = data.queries @ R.T
                return data
            except Exception:
                num_samples = self._get_num_samples()
                if num_samples > 0:
                    idx = np.random.randint(0, num_samples)
                else:
                    time.sleep(0.2)
