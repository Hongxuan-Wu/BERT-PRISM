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

    parser.add_argument('--config_file', type=str, default='1train/expression_intensity_cls/config.py', help='the files of config')

    parser.add_argument('--display_step', type=int, default=100, help='display training information in how many step')
    parser.add_argument('--print_step', type=bool, default=False, help='print the step information')
    parser.add_argument('--log_dir', type=str, default='/results/logs/expression_cls/', help='path that log will be saved')
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
            results_valid = {'loss': [], 'accuracy': [], 'mcc': [], 'f1': [], 'precision': [], 'recall': []}
            for fold in config['fold']:
                args.logger.info('')
                args.logger.info("======================================== Starting fold {}: ========================================".format(fold))
                trainer = Trainer(args, config)
                res = trainer.test(fold)

                results_valid['loss'].append(res[0])
                results_valid['accuracy'].append(res[1]['accuracy'])
                # results_valid['auc'].append(res[1]['auc'])
                results_valid['mcc'].append(res[1]['mcc'])
                results_valid['f1'].append(res[1]['f1'])
                results_valid['precision'].append(res[1]['precision'])
                results_valid['recall'].append(res[1]['recall'])
                
            args.logger.info("Val - Loss {}  Accuracy {}  MCC {}  F1 {}  Precision {}  Recall {}"
                            .format(
                                np.mean(results_valid['loss']),
                                np.mean(results_valid['accuracy']),
                                # np.mean(results_valid['auc']),
                                np.mean(results_valid['mcc']),
                                np.mean(results_valid['f1']),
                                np.mean(results_valid['precision']),
                                np.mean(results_valid['recall']),
                            )
            )
        else:
            trainer = Trainer(args, config)
            res = trainer.test()
            args.logger.info("Val - Loss {}  Accuracy {}  MCC {}  F1 {}  Precision {}  Recall {}"
                            .format(
                                res[0],
                                res[1]['accuracy'],
                                # res[1]['auc'],
                                res[1]['mcc'],
                                res[1]['f1'],
                                res[1]['precision'],
                                res[1]['recall'],
                            )
            )
    else:
        exit("Mode error!")

if __name__ == '__main__':
    main()
