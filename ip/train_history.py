import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ip.models.diffusion_history import *
from ip.configs.base_config_history import config
import pickle
import os
from ip.utils.running_dataset_history import RunningDatasetHistory
from torch_geometric.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse

if __name__ == '__main__':
    ####################################################################################################################
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='test', help='Name of the run.')
    parser.add_argument('--record', type=int, default=0,
                        help='Whether to log the training and save models [0, 1].')
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='Log training on weights and biases [0, 1]. You might need to log in to wandb.')
    parser.add_argument('--save_path', type=str, default='./runs',
                        help='Where the config and models will be saved.')
    parser.add_argument('--fine_tune', type=int, default=0,
                        help='Whether to train from scratch (0), or fine-tune existing model (1).')
    parser.add_argument('--model_path', type=str, default='./checkpoints',
                        help='If fine-tuning, path to where that model is saved.')
    parser.add_argument('--model_name', type=str, default='model.pt',
                        help='If fine-tuning, path to what is the name of the model.')
    parser.add_argument('--compile_models', type=int, default=0,
                        help='If fine-tuning, whether to compile models. When not fine-tuning, it is defined in the config')
    parser.add_argument('--data_path_train', type=str, default='./data/train',
                        help='Path to the training data.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for fine-tuning. When not fine-tuning, it is defined in the config')
    parser.add_argument('--data_path_val', type=str, default='./data/val',
                        help='Path to the validation data.')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (e.g. cpu or cuda).')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Override max training steps for quick smoke tests.')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Dataloader workers. Use 0 for smoke tests.')
    parser.add_argument('--smoke_test', type=int, default=0,
                        help='Enable lightweight config overrides for a quick history-aware smoke test [0,1].')

    record = bool(parser.parse_args().record)
    use_wandb = bool(parser.parse_args().use_wandb)
    fine_tune = bool(parser.parse_args().fine_tune)
    compile_models = bool(parser.parse_args().compile_models)
    run_name = parser.parse_args().run_name
    save_path = parser.parse_args().save_path
    model_path = parser.parse_args().model_path
    model_name = parser.parse_args().model_name
    data_path_train = parser.parse_args().data_path_train
    data_path_val = parser.parse_args().data_path_val
    bs = parser.parse_args().batch_size
    device_override = parser.parse_args().device
    max_steps_override = parser.parse_args().max_steps
    num_workers = parser.parse_args().num_workers
    smoke_test = bool(parser.parse_args().smoke_test)
    ####################################################################################################################
    save_dir = f'{save_path}/{run_name}' if record else None

    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def apply_history_defaults(cfg):
        cfg.setdefault('enable_track_nodes', True)
        cfg.setdefault('track_n_max', 2)
        cfg.setdefault('track_history_len', 16)
        cfg.setdefault('track_points_per_obj', 5)
        cfg.setdefault('track_hidden_dim', 512)
        cfg.setdefault('track_age_embed_dim', 32)
        cfg.setdefault('track_age_norm_max_sec', 2.0)
        if device_override is not None:
            cfg['device'] = device_override
        if max_steps_override is not None:
            cfg['num_iters'] = max_steps_override
        if smoke_test:
            cfg['local_nn_dim'] = min(cfg.get('local_nn_dim', 512), 128)
            cfg['hidden_dim'] = min(cfg.get('hidden_dim', 1024), 256)
            cfg['num_scenes_nodes'] = min(cfg.get('num_scenes_nodes', 16), 8)
            cfg['num_layers'] = min(cfg.get('num_layers', 2), 1)
            cfg['track_hidden_dim'] = min(cfg.get('track_hidden_dim', 512), 128)
            cfg['compile_models'] = False
            cfg['batch_size'] = bs
            cfg['pre_trained_encoder'] = False
            cfg['freeze_encoder'] = False
            if cfg['device'] == 'cuda':
                print('Smoke test mode enabled: using reduced history model size on CUDA')
            else:
                print('Smoke test mode enabled: using reduced history model size on CPU')
        return cfg

    if fine_tune:
        config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
        config['compile_models'] = False
        config['batch_size'] = bs
        config['save_dir'] = save_dir
        config['record'] = record
        # TODO: Here you can change other parameter from the ones used to train initial model.
        config = apply_history_defaults(config)
        if config.get('pre_trained_encoder', False) and not os.path.exists(config['scene_encoder_path']):
            print(f"Warning: scene_encoder.pt not found at {config['scene_encoder_path']}, disabling pre-trained encoder for smoke test")
            config['pre_trained_encoder'] = False
        model = GraphDiffusionHistory.load_from_checkpoint(f'{model_path}/{model_name}', config=config, strict=True,
                                                           map_location=config['device']).to(config['device'])
        if compile_models:
            model.model.compile_models()
    else:
        config['save_dir'] = save_dir
        config['record'] = record
        config = apply_history_defaults(config)
        if config.get('pre_trained_encoder', False) and not os.path.exists(config['scene_encoder_path']):
            print(f"Warning: scene_encoder.pt not found at {config['scene_encoder_path']}, disabling pre-trained encoder for smoke test")
            config['pre_trained_encoder'] = False
        model = GraphDiffusionHistory(config).to(config['device'])
    ####################################################################################################################
    dset_val = RunningDatasetHistory(
        data_path_val,
        len(os.listdir(data_path_val)),
        rand_g_prob=0,
    )
    dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

    dset = RunningDatasetHistory(
        data_path_train,
        len(os.listdir(data_path_train)),
        rand_g_prob=config['randomize_g_prob'],
    )
    dataloader = DataLoader(dset, batch_size=config['batch_size'], drop_last=True, shuffle=True,
                            num_workers=num_workers, pin_memory=(config['device'] == 'cuda'))
    ####################################################################################################################
    logger = None
    if record:
        if use_wandb:
            logger = WandbLogger(project='Instant Policy',
                                 name=f'{run_name}',
                                 save_dir=save_dir,
                                 log_model=False)
        # Dump config to save_dir
        pickle.dump(config, open(f'{save_dir}/config.pkl', 'wb'))
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer_precision = '16-mixed' if config['device'] == 'cuda' and not smoke_test else '32-true'
    trainer = L.Trainer(
        enable_checkpointing=False,  # We save the models manually.
        accelerator=config['device'],
        devices=1,
        max_steps=config['num_iters'],
        enable_progress_bar=True,
        precision=trainer_precision,
        val_check_interval=20000 if not smoke_test else 1,
        num_sanity_val_steps=2 if not smoke_test else 0,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=500 if not smoke_test else 1,
        gradient_clip_val=1,
        gradient_clip_algorithm='norm',
        callbacks=[lr_monitor],
    )

    trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
    )

    # Save last:
    if record:
        model.save_model(f'{save_dir}/last.pt')
