import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_sample(data_path, sample_idx):
    return torch.load(os.path.join(data_path, f'data_{sample_idx}.pt'))


def summarize_sample(data, sample_idx):
    seq = data.current_track_seq.squeeze(0).cpu().numpy()  # [N,H,P,3]
    valid = data.current_track_valid.squeeze(0).cpu().numpy()
    age = data.current_track_age_sec.squeeze(0).cpu().numpy()
    print(f'sample {sample_idx}:')
    print(f'  current_track_seq shape: {seq.shape}')
    print(f'  current_track_valid: {valid.astype(int).tolist()}')
    print(f'  current_track_age_sec: {age.squeeze(-1).tolist()}')
    print(f'  xyz min: {seq.min(axis=(0,1,2)).tolist()}')
    print(f'  xyz max: {seq.max(axis=(0,1,2)).tolist()}')


def plot_sample(data, sample_idx, output_dir):
    seq = data.current_track_seq.squeeze(0).cpu().numpy()  # [N,H,P,3]
    valid = data.current_track_valid.squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(12, 5))
    for obj_idx in range(seq.shape[0]):
        ax = fig.add_subplot(1, seq.shape[0], obj_idx + 1, projection='3d')
        if not valid[obj_idx]:
            ax.set_title(f'Object {obj_idx} (invalid)')
            continue
        for point_idx in range(seq.shape[2]):
            point_traj = seq[obj_idx, :, point_idx, :]
            ax.plot(point_traj[:, 0], point_traj[:, 1], point_traj[:, 2], marker='o', linewidth=1)
        ax.set_title(f'Object {obj_idx}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'offline_tracks_sample_{sample_idx}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  saved figure: {out_path}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./track_viz')
    args = parser.parse_args()

    data = load_sample(args.data_path, args.sample_idx)
    summarize_sample(data, args.sample_idx)
    plot_sample(data, args.sample_idx, args.output_dir)


if __name__ == '__main__':
    main()
