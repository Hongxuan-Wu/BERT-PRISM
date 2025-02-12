from collections import OrderedDict
import os.path as osp

config = OrderedDict()

config['mode'] = 'test'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True
config['save_matrix'] = True

config['root'] = '/data/whx/'
# config['root'] = '/hy-tmp/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # 0-12
config['output_type'] = 'pool'  # pool / cls / mean

# config['pretrained_path'] = config['root'] + "ckpt/DNABERT-2-117M"
config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000/"
# config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000_newtoken/"

config['components'] = True
# config['components'] = False

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'projects/DNABERT_Promotor/1train/components_analysis_regression/data/'
config['root_data'] = config['root'] + 'components/components_prokaryotes/'

# config['catATG'] = True
config['catATG'] = False

################################################################### Training ###############################################################

if config['mode'] == 'test':
    config['valid_batch_size'] = 512  # 2048

    config['checkpoint_dir'] = '/data/whx/ckpt/components/ckpt_kfold/'

    if 'checkpoint-720000_newtoken' in config['pretrained_path']:
        config['checkpoint_dir'] += 'ckpt72new'
    elif 'checkpoint-720000' in config['pretrained_path']:
        config['checkpoint_dir'] += 'ckpt72'
    elif 'DNABERT-2-117M' in config['pretrained_path']:
        config['checkpoint_dir'] += 'dnabert2'
    else:
        pass

    config['dataset'] = 'components'
    # config['dataset'] = 'prokaryotes596'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_LaFleur'
    
    if config['dataset'] == 'components':
        config['datasize'] = 100000000
        config['n_folds'] = 36  # filtered - 36 /  - 
        config['fold'] = [0]
        config['checkpoint_dir'] += '_components_filtered'
    elif config['dataset'] == 'prokaryotes596':
        config['datasize'] = 100000000
        config['n_folds'] = 400  # filtered - 400
        config['fold'] = [0]
        config['checkpoint_dir'] += '_prokaryotes596_filtered'
    
    config['checkpoint_path'] = osp.join(config['checkpoint_dir'], 'last.pth')
    # config['checkpoint_path'] = osp.join(config['checkpoint_dir'], 'best.pth')
    
    config['checkpoint_path'] = osp.join('/data/whx/projects/DNABERT_Promotor_components/1train/components_analysis_regression/results/checkpoints/20250116_205523/', 'last.pth')
