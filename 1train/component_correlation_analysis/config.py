from collections import OrderedDict
import os.path as osp

config = OrderedDict()

config['mode'] = 'test'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True
config['save_matrix'] = False

# config['root'] = '/data/whx/models/BERT-PRISM/'
config['root'] = '/data/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # 0-12
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

config['components'] = True
# config['components'] = False

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'component_correlation_analysis/'
config['root_data'] = config['root_project']

# config['catATG'] = True
config['catATG'] = False

################################################################### Training ###############################################################

if config['mode'] == 'test':
    config['valid_batch_size'] = 256  # 2048

    config['checkpoint_dir'] = config['root'] + 'component_correlation_analysis/BERT-PRISM-1_components_filtered_crossattn/'

    config['dataset'] = 'components'
    
    if config['dataset'] == 'components':
        config['datasize'] = 100000000
        config['n_folds'] = 36  # filtered - 36 /  - 
        config['fold'] = [0]
    elif config['dataset'] == 'prokaryotes596':
        config['datasize'] = 100000000
        config['n_folds'] = 400  # filtered - 400
        config['fold'] = [0]
    
    config['checkpoint_path'] = osp.join(config['checkpoint_dir'], 'last.pth')
