import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import argparse
import importlib
import numpy as np
import pdb

from utils import get_current_time, set_seed, set_logger
from trainer import Trainer
from tensorboardX import SummaryWriter


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', type=str, default='./config.py', help='the files of config')

    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--print_step', type=bool, default=False, help='print the step information')
    parser.add_argument('--log_dir', type=str, default='./results/logs', help='path that log will be saved')
    parser.add_argument('--checkpoint_dir', type=str, default='./results/checkpoints', help='the path of checkpoint')
    
    parser.add_argument('--gpu_ids', type=list, default='0')
    
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    
    args = parser.parse_args()
    return args

def init_configs():
    args = parse()
    
    # Set train and test datasets and the corresponding data loaders
    spec2 = importlib.util.spec_from_file_location("", args.config_file)
    odm = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(odm)
    config = odm.config

    current_time = get_current_time()
    args.log_dir = os.path.join(args.log_dir, current_time)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, current_time)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    # if not os.path.exists(args.checkpoint_dir):
    #     os.makedirs(args.checkpoint_dir)
        
    args.device = 'cuda:' + args.gpu_ids[0]
    args.gpu_ids = [int(id) for id in args.gpu_ids]
    
    args.writer = SummaryWriter(args.log_dir)
    args.logger = set_logger(args, config)
    
    set_seed(args)

    return args, config

def main():
    args, config = init_configs()
    
    if config['mode'] == 'predict':
        trainer = Trainer(args, config)
        trainer.predict()
    elif config['mode'] == 'test':
        if config['kfold']:
            results_valid = {'loss': [], 'mse': [], 'rmse': [], 'mae': [], 'pearsonr_corr_coefficient': [], 'pearsonr_p': [], 'spearmanr_corr_coefficient': [], 'spearmanr_p': [], 'r2': [], 'kl_divergence': []}
            for fold in config['fold']:
                args.logger.info('')
                args.logger.info("======================================== Starting fold {}: ========================================".format(fold))
                trainer = Trainer(args, config)
                trainer.test()
                
                results_valid['loss'].append(res[2])
                results_valid['mse'].append(res[3]['mse'])
                results_valid['rmse'].append(res[3]['rmse'])
                results_valid['mae'].append(res[3]['mae'])
                results_valid['pearsonr_corr_coefficient'].append(res[3]['pearsonr_corr_coefficient'])
                results_valid['pearsonr_p'].append(res[3]['pearsonr_p'])
                results_valid['spearmanr_corr_coefficient'].append(res[3]['spearmanr_corr_coefficient'])
                results_valid['spearmanr_p'].append(res[3]['spearmanr_p'])
                results_valid['r2'].append(res[3]['r2'])
                results_valid['kl_divergence'].append(res[3]['kl_divergence'])
                
            args.logger.info("Val - Loss {}  MSE {}  R2 {}  RMSE {}  MAE {}  Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {} KL_divergence {}"
                            .format(
                                np.mean(results_valid['loss']),
                                np.mean(results_valid['mse']),
                                np.mean(results_valid['r2']),
                                np.mean(results_valid['rmse']),
                                np.mean(results_valid['mae']),
                                np.mean(results_valid['pearsonr_corr_coefficient']),
                                np.mean(results_valid['pearsonr_p']),
                                np.mean(results_valid['spearmanr_corr_coefficient']),
                                np.mean(results_valid['spearmanr_p']),
                                np.mean(results_valid['kl_divergence']),
                            )
            )
        else:
            trainer = Trainer(args, config)
            res = trainer.test()
            args.logger.info("Val - Loss {}  MSE {}  R2 {}  RMSE {}  MAE {}  Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {} KL_divergence {}"
                            .format(
                                res[0],
                                res[1]['mse'],
                                res[1]['r2'],
                                res[1]['rmse'],
                                res[1]['mae'],
                                res[1]['pearsonr_corr_coefficient'],
                                res[1]['pearsonr_p'],
                                res[1]['spearmanr_corr_coefficient'],
                                res[1]['spearmanr_p'],
                                res[1]['kl_divergence'],
                            )
            )
    else:
        exit("Mode error!")
        
if __name__ == '__main__':
    main()
