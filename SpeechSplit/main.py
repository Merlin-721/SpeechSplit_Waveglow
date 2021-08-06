import os
import argparse
import torch
from torch.backends import cudnn
from types import SimpleNamespace
import json
from solver import Solver
from data_loader import get_loader, infinite_iter



def str2bool(v):
    return v.lower() in ('true')

def show_hparams(hparams):
    dic = dir(hparams)
    hyps = [f"{name}: {getattr(hparams,name)}" for name in dic if not name.startswith('__')]
    print('\nHyperparameters:\n' + '\n'.join(hyps))


def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(os.path.join(config.save_dir,'G')):
        os.makedirs(os.path.join(config.save_dir,'G'))
    if not os.path.exists(os.path.join(config.save_dir,'P')):
        os.makedirs(os.path.join(config.save_dir,'P'))
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    # Model Hyperparameters
    with open(config.hparams) as f:
        hparams = json.load(f,object_hook=lambda d: SimpleNamespace(**d)) 

    print(config)
    show_hparams(hparams)

    vcc_loader = get_loader(hparams)
    vcc_infinite_loader = infinite_iter(vcc_loader)
    
    # Solver for training
    solver = Solver(vcc_infinite_loader, config, hparams)

    solver.train()
    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # Training configuration.
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--p_lr', type=float, default=0.0001, help='learning rate for P')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--train_G', action='store_true', help='option to train G')
    parser.add_argument('--train_P', action='store_true', help='option to train P')
    parser.add_argument('--hparams', type=str, default='hparams.json')
    parser.add_argument('--val_path', type=str, default='assets/validation.pkl')

    # Miscellaneous.
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--device_id', type=int, default=0)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='run/logs')
    parser.add_argument('--save_dir', type=str, default='run/models')
    parser.add_argument('--sample_dir', type=str, default='run/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=2000)

    config = parser.parse_args()
    main(config)