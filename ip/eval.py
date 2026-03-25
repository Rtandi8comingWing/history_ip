from ip.models.diffusion import GraphDiffusion
from ip.utils.rl_bench_utils import rollout_model
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='plate_out')
    parser.add_argument('--num_demos', type=int, default=2)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--restrict_rot', type=int, default=1)
    parser.add_argument('--compile_models', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--model_name', type=str, default='model.pt')
    parser.add_argument('--headless', type=int, default=0,
                        help='Run RLBench/CoppeliaSim without visualization [0,1].')
    args = parser.parse_args()

    restrict_rot = bool(args.restrict_rot)
    task_name = args.task_name
    num_demos = args.num_demos
    num_rollouts = args.num_rollouts
    compile_models = bool(args.compile_models)
    model_path = args.model_path
    model_name = args.model_name
    headless = bool(args.headless)
    ####################################################################################################################
    config = pickle.load(open(f'{model_path}/config.pkl', 'rb'))

    config['compile_models'] = False
    config['batch_size'] = 1
    config['num_demos'] = num_demos
    config['num_diffusion_iters_test'] = 4

    model = GraphDiffusion.load_from_checkpoint(f'{model_path}/{model_name}', config=config, strict=True,
                                                map_location=config['device']).to(config['device'])

    model.model.reinit_graphs(1, num_demos=num_demos)
    model.eval()

    if compile_models:
        model.model.compile_models()
    ####################################################################################################################
    sr = rollout_model(model, num_demos, task_name, num_rollouts=num_rollouts, execution_horizon=8,
                       num_traj_wp=config['traj_horizon'], restrict_rot=restrict_rot, headless=headless)
    print('Success rate:', sr)
