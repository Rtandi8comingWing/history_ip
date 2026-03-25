from ip.models.diffusion import *
from ip.configs.base_config import config
import pickle
import os
import glob
import time
from ip.utils.running_dataset import RunningDataset
from torch_geometric.data import DataLoader
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import argparse


def count_data_files(data_path):
    return len(glob.glob(os.path.join(data_path, 'data_*.pt')))


def wait_for_min_samples(data_path, min_samples, split_name, poll_interval_sec=2.0):
    print(f'Waiting for at least {min_samples} {split_name} samples in {data_path} ...')
    while True:
        num_samples = count_data_files(data_path)
        if num_samples >= min_samples:
            print(f'Found {num_samples} {split_name} samples in {data_path}.')
            return num_samples
        time.sleep(poll_interval_sec)


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
                        help='Batch size override for both scratch training and fine-tuning.')
    parser.add_argument('--data_path_val', type=str, default='./data/val',
                        help='Path to the validation data.')
    parser.add_argument('--wait_for_data', type=int, default=1,
                        help='Wait until enough data files exist before starting training [0,1].')

    args = parser.parse_args()
    record = bool(args.record)
    use_wandb = bool(args.use_wandb)
    fine_tune = bool(args.fine_tune)
    compile_models = bool(args.compile_models)
    run_name = args.run_name
    save_path = args.save_path
    model_path = args.model_path
    model_name = args.model_name
    data_path_train = args.data_path_train
    data_path_val = args.data_path_val
    bs = args.batch_size
    wait_for_data = bool(args.wait_for_data)
    ####################################################################################################################
    save_dir = f'{save_path}/{run_name}' if record else None

    if record and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if fine_tune:
        config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))
        config['compile_models'] = False
        config['save_dir'] = save_dir
        config['record'] = record
        # TODO: Here you can change other parameter from the ones used to train initial model.
        config['batch_size'] = bs
        model = GraphDiffusion.load_from_checkpoint(f'{model_path}/{model_name}', config=config, strict=True,
                                                    map_location=config['device']).to(config['device'])
        if compile_models:
            model.model.compile_models()
    else:
        config['save_dir'] = save_dir
        config['record'] = record
        config['batch_size'] = bs
        model = GraphDiffusion(config).to(config['device'])
    ####################################################################################################################
    train_min_samples = max(1, config['batch_size'])
    val_min_samples = 1
    if wait_for_data:
        wait_for_min_samples(data_path_train, train_min_samples, 'training')
        wait_for_min_samples(data_path_val, val_min_samples, 'validation')

    dset_val = RunningDataset(
        data_path_val,
        num_samples=count_data_files(data_path_val),
        rand_g_prob=0,
        dynamic_num_samples=True,
    )
    dataloader_val = DataLoader(dset_val, batch_size=1, shuffle=False)

    dset = RunningDataset(
        data_path_train,
        num_samples=count_data_files(data_path_train),
        rand_g_prob=config['randomize_g_prob'],
        dynamic_num_samples=True,
    )
    dataloader = DataLoader(dset, batch_size=config['batch_size'], drop_last=True, shuffle=True,
                            num_workers=8, pin_memory=True)
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
    trainer = L.Trainer(
        enable_checkpointing=False,  # We save the models manually.
        accelerator=config['device'],
        devices=1,
        max_steps=config['num_iters'],
        enable_progress_bar=True,
        precision='16-mixed',
        val_check_interval=20000,  # TODO: might want to change that.
        num_sanity_val_steps=2,
        check_val_every_n_epoch=None,
        logger=logger,
        log_every_n_steps=500,  # TODO: might want to change that.
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
