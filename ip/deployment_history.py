'''
This scripts shows and example of how Instant Policy could be used at deployment.
'''
import sys
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ip.models.diffusion_history import GraphDiffusionHistory
from ip.utils.data_proc import *
from ip.utils.track_buffer import TrackBuffer


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
    ####################################################################################################################
    # Define rollout parameters. 
    num_demos = 2
    num_traj_wp = 10
    num_diffusion_iters = 4
    compile_models = False
    max_execution_steps = 100
    ####################################################################################################################
    # Load and prepare trained model.
    model_path = './checkpoints'
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
    config = _apply_history_defaults(config)
    if config.get('pre_trained_encoder', False) and not os.path.exists(config['scene_encoder_path']):
        print(f"Warning: scene_encoder.pt not found at {config['scene_encoder_path']}, disabling pre-trained encoder")
        config['pre_trained_encoder'] = False
    config['num_layers'] = 2

    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = num_diffusion_iters

    model = GraphDiffusionHistory.load_from_checkpoint(f'{model_path}/model.pt', config=config, strict=False,
                                                map_location=config['device']).to(config['device'])
    model.model.reinit_graphs(1, num_demos=max(num_demos, 1))
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    track_buffer = TrackBuffer(
        track_n_max=config['track_n_max'],
        track_history_len=config['track_history_len'],
        track_points_per_obj=config['track_points_per_obj'],
        device=config['device'],
    )
    track_buffer.reset()
    # Process demonstrations.
    # TODO: Collect or load demonstrations in a form of {'pcds': [], 'T_w_es': [], 'grips': []}
    demos = []    
    full_sample = {
        'demos': [dict()] * num_demos,
        'live': dict(),
    }
    for i, demo in enumerate(demos):
        full_sample['demos'][i] = sample_to_cond_demo(demo, num_traj_wp)
        assert len(full_sample['demos'][i]['obs']) == num_traj_wp
    ####################################################################################################################
    # Rollout the model.
    for k in range(max_execution_steps):
        T_w_e = None  # TODO: end-effector pose in the world frame, [4, 4].
        pcd_w = None  # TODO: segmented point cloud observation in the world frame, [N, 3].
        grip = None  # TODO: whether the gripper is closed or opened, [0, 1]
        full_sample['live']['obs'] = [transform_pcd(subsample_pcd(pcd_w), np.linalg.inv(T_w_e))]
        full_sample['live']['grips'] = [grip]
        full_sample['live']['actions_grip'] = [np.zeros(8)]
        full_sample['live']['T_w_es'] = [T_w_e]
        full_sample['live']['actions'] = [T_w_e.reshape(1, 4, 4).repeat(config['pre_horizon'], axis=0)]
        data = save_sample(full_sample, None)
        track_buffer.update(pcd_w, T_w_e)
        track_data = track_buffer.get_track_data()
        if track_data is not None:
            data.current_track_seq = track_data['track_seq'].cpu()
            data.current_track_valid = track_data['track_valid'].cpu()
            data.current_track_lengths = track_data['track_lengths'].cpu()
            data.current_track_age_sec = track_data['track_age_sec'].cpu()
        else:
            data.current_track_seq = torch.zeros((1, config['track_n_max'], config['track_history_len'], config['track_points_per_obj'], 3), dtype=torch.float32)
            data.current_track_valid = torch.zeros((1, config['track_n_max']), dtype=torch.bool)
            data.current_track_lengths = torch.zeros((1, config['track_n_max']), dtype=torch.long)
            data.current_track_age_sec = torch.ones((1, config['track_n_max'], 1), dtype=torch.float32)
        
        # For efficiency, pre-compute and cache geometry embeddings for the demos. 
        if k == 0:
            demo_scene_node_embds, demo_scene_node_pos = model.model.get_demo_scene_emb(
                data.to(model.config['device']))
        data.live_scene_node_embds, data.live_scene_node_pos =\
            model.model.get_live_scene_emb(data.to(model.config['device']))
        data.demo_scene_node_embds = demo_scene_node_embds.clone()
        data.demo_scene_node_pos = demo_scene_node_pos.clone()
        
        # Inference on the model.
        with torch.no_grad():
            with torch.autocast(dtype=torch.float32, device_type=model.config['device']):
                actions, grips = model.test_step(data.to(model.config['device']), 0)
            actions = actions.squeeze().cpu().numpy()
            grips = grips.squeeze().cpu().numpy()
        
        # TODO: Use whatever controller you have to execute all or part of the predicted actions.
        # TODO: actions: [Pred_horizon, 4, 4] are relative transforms of the end-effector.
        # TODO: To get next pose of the end-effector in the world frame you use T_w_e @ actions[j].
        # TODO: grips: [Pred_horizon, 1] are open and close commands: -1 is close, 1 is open.
