import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ip.models.diffusion_history import GraphDiffusionHistory
from ip.utils.rl_bench_utils_history import rollout_model
import argparse
import pickle


def _apply_history_defaults(config):
    config.setdefault('enable_track_nodes', True)
    config.setdefault('track_n_max', 2)
    config.setdefault('track_history_len', 16)
    config.setdefault('track_points_per_obj', 5)
    config.setdefault('track_hidden_dim', 512)
    config.setdefault('track_age_embed_dim', 32)
    config.setdefault('track_age_norm_max_sec', 2.0)
    config['enable_track_nodes'] = True
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_demos', type=int, default=2)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    restrict_rot = bool(parser.parse_args().restrict_rot)
    task_name = parser.parse_args().task_name
    num_demos = parser.parse_args().num_demos
    num_rollouts = parser.parse_args().num_rollouts
    compile_models = bool(parser.parse_args().compile_models)
    ####################################################################################################################
    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))

    config = _apply_history_defaults(config)
    if config.get('pre_trained_encoder', False) and not os.path.exists(config['scene_encoder_path']):
        print(f"Warning: scene_encoder.pt not found at {config['scene_encoder_path']}, disabling pre-trained encoder")
        config['pre_trained_encoder'] = False
    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = 4

    model = GraphDiffusionHistory.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=True,
                                                map_location=config['device']).to(config['device'])

    model.model.reinit_graphs(1, num_demos=num_demos)
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    sr = rollout_model(model, num_demos, task_name, num_rollouts=num_rollouts, execution_horizon=8,
                       num_traj_wp=config['traj_horizon'], restrict_rot=restrict_rot)
    print('Success rate:', sr)
