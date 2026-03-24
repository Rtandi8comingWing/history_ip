import sys
from pathlib import Path
import argparse
import os
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize_track_sample(data, sample_idx):
    seq = data.current_track_seq
    valid = data.current_track_valid
    age = data.current_track_age_sec

    seq_np = seq.cpu().numpy()
    valid_np = valid.cpu().numpy()
    age_np = age.cpu().numpy()

    nonzero = int((seq_np != 0).sum())
    total = int(seq_np.size)
    xyz_min = seq_np.min(axis=(0, 1, 2, 3))
    xyz_max = seq_np.max(axis=(0, 1, 2, 3))

    print(f"sample {sample_idx}:")
    print(f"  current_track_seq shape: {tuple(seq.shape)}")
    print(f"  current_track_valid: {valid_np.astype(int).tolist()}")
    print(f"  current_track_age_sec: {age_np.squeeze(-1).tolist()}")
    print(f"  nonzero elements: {nonzero}/{total}")
    print(f"  xyz range: min={xyz_min.tolist()} max={xyz_max.tolist()}")


def inspect_dataset(data_path, max_samples=5):
    files = sorted([f for f in os.listdir(data_path) if f.startswith('data_') and f.endswith('.pt')])
    if not files:
        raise FileNotFoundError(f'No data_*.pt files found in {data_path}')

    valid_counts = []
    age_values = []
    ranges = []

    for i, fname in enumerate(files[:max_samples]):
        data = torch.load(os.path.join(data_path, fname))
        required = ['current_track_seq', 'current_track_valid', 'current_track_age_sec']
        for attr in required:
            if not hasattr(data, attr):
                raise AttributeError(f'{fname} missing {attr}')

        summarize_track_sample(data, i)
        valid_counts.append(int(data.current_track_valid.sum().item()))
        age_values.extend(data.current_track_age_sec.view(-1).cpu().tolist())
        seq_np = data.current_track_seq.cpu().numpy()
        ranges.append((seq_np.min(), seq_np.max()))

    print('\nsummary:')
    print(f'  inspected samples: {min(max_samples, len(files))}')
    print(f'  mean valid tracks: {float(np.mean(valid_counts)):.3f}')
    print(f'  mean age: {float(np.mean(age_values)):.3f}')
    print(f'  age min/max: {float(np.min(age_values)):.3f}/{float(np.max(age_values)):.3f}')
    print(f'  seq global min/max across samples: {float(min(r[0] for r in ranges)):.4f}/{float(max(r[1] for r in ranges)):.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=5)
    args = parser.parse_args()
    inspect_dataset(args.data_path, args.max_samples)
