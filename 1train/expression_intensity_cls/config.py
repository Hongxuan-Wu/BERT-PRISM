import os.path as osp
from collections import OrderedDict

config = OrderedDict()

# config['mode'] = 'test'
config['mode'] = 'predict'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True

# 2 / 3 / 4
config['num_class'] = 4

if config['num_class'] == 2:
    # config['strong_weak'] = True
    config['strong_weak'] = False
    
    config['save_roc_auc'] = True
else:
    config['save_roc_auc'] = False

config['root'] = '/data/whx/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # default: 11 (0-12)
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root'] + "ckpt/DNABERT-2-117M"
# config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000/"
# config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000_newtoken/"

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'projects/DNABERT_Promotor/1train/expression_intensity_classification/data/'
config['root_data'] = config['root'] + 'expressions/'

# config['predict_filepath'] = config['root2'] + 'predict/ecos/20240813_035617_ckpt72newtoken_steps1000_ecos_gen10000/gen_seqs.csv'
# config['predict_filepath'] = config['root2'] + 'predict/ecos/20240911_125214_ckpt72newtoken_steps2000_ecos_gen10000/gen_seqs.csv'
# config['predict_filepath'] = config['root2'] + 'predict/ecos/20240912_043045_ckpt72newtoken_steps2000_ecos_gen100000/gen_seqs.csv'

config['catATG'] = True
# config['catATG'] = False

################################################################### Training ###############################################################

############################
# config['kfold'] = True
config['kfold'] = False
############################

if config['kfold']:
    config['n_folds'] = 10  # 5/10
    
    if config['n_folds'] == 5:
        config['fold'] = [0,1,2,3,4]
    elif config['n_folds'] == 10:
        config['fold'] = [0,1,2,3,4,5,6,7,8,9]
    
    # Learning
    config['n_epochs'] = 100
    config['train_batch_size'] = 2048  # 128
    config['valid_batch_size'] = 2048  # 128
    config['lr'] = 3e-5  # 3e-5 / prokaryotes——1e-5
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 100

    config['dataset'] = 'genes_prokaryotes'
    # config['dataset'] = 'genes_escherichia'
    # config['dataset'] = 'genes_streptomyces'
    # config['dataset'] = 'genes_vibrio'
    
    if config['dataset'] == 'genes_escherichia':
        config['datasize'] = 100000000
        config['n_folds'] = 25  # sum - 100 / fromlist - 25 / manual - 50
        config['fold'] = [0]
    elif config['dataset'] == 'genes_streptomyces':
        config['datasize'] = 100000000
        config['n_folds'] = 30
        config['fold'] = [0]
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['dataset'] == 'genes_vibrio':
        config['datasize'] = 100000000
        config['n_folds'] = 16
        config['fold'] = [0]
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['dataset'] == 'genes_prokaryotes':
        config['datasize'] = 100000000
        config['n_folds'] = 400
        config['fold'] = [0]
        config['lr'] = 1e-5

else:
    # Learning
    config['n_epochs'] = 100
    config['train_batch_size'] = 2048  # 128
    config['valid_batch_size'] = 2048  # 128
    config['lr'] = 3e-5  # 3e-5 / prokaryotes —— 1e-5
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 100
    
    # config['train_dataset'] = 'genes_prokaryotes'
    # config['train_dataset'] = 'genes_escherichia'
    # config['train_dataset'] = 'genes_streptomyces'
    config['train_dataset'] = 'genes_vibrio'


    # config['val_dataset'] = 'EColi_GeneExpression'
    # config['val_dataset'] = 'Bsub_GeneExpression'
    # config['val_dataset'] = 'Cglu_GeneExpression'
    # config['val_dataset'] = 'Vibr_GeneExpression'
    config['val_dataset'] = 'WCFS1_GeneExpression'
    
    # config['val_dataset'] = 'mg1655'
    # config['val_dataset'] = 'gnn210095'
    # config['val_dataset'] = 'MG1655-RegluonDB'
    # config['val_dataset'] = 'ecoli_lacUV5'

    if config['train_dataset'] == 'genes_escherichia':
        pass
    elif config['train_dataset'] == 'genes_streptomyces':
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['train_dataset'] == 'genes_vibrio':
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['train_dataset'] == 'genes_prokaryotes':
        config['lr'] = 1e-5

if config['mode'] == 'test':
    # checkpoint_dir = '/data/whx/projects/DNABERT_Promotor/1train/expression_intensity_classification/results/checkpoints/'
    checkpoint_dir = '/data/whx/ckpt/expression_cls/ckptNv1/'
    
    if 'checkpoint-720000_newtoken' in config['pretrained_path']:
        ckpt_name = 'ckpt72new'
    elif 'checkpoint-720000' in config['pretrained_path']:
        ckpt_name = 'ckpt72'
    elif 'DNABERT-2-117M' in config['pretrained_path']:
        ckpt_name = 'dnabert2'
    else:
        pass
    
    if 'genes_' in config['train_dataset']:
        ckpt_name = ckpt_name + '_' + config['train_dataset'][6:]

    if config['num_class'] == 2:
        if config['strong_weak']:
            ckpt_name += '_strongweak'
        else:
            ckpt_name += '_cls2'
    elif config['num_class'] == 3:
        ckpt_name += '_cls3'
    elif config['num_class'] == 4:
        ckpt_name += '_cls4'
    
    config['checkpoint_dir'] = osp.join(checkpoint_dir, ckpt_name)
    

elif config['mode'] == 'predict':
    checkpoint_dir = '/data/whx/ckpt/expression_cls/ckptNv1/'
    
    if 'checkpoint-720000_newtoken' in config['pretrained_path']:
        ckpt_name = 'ckpt72new'
    elif 'checkpoint-720000' in config['pretrained_path']:
        ckpt_name = 'ckpt72'
    elif 'DNABERT-2-117M' in config['pretrained_path']:
        ckpt_name = 'dnabert2'
    else:
        pass
    
    if 'genes_' in config['train_dataset']:
        ckpt_name = ckpt_name + '_' + config['train_dataset'][6:]

    if config['num_class'] == 2:
        if config['strong_weak']:
            ckpt_name += '_strongweak'
        else:
            ckpt_name += '_cls2'
    elif config['num_class'] == 3:
        ckpt_name += '_cls3'
    elif config['num_class'] == 4:
        ckpt_name += '_cls4'
    
    if config['catATG']:
        ckpt_name += '_catATG'
        
    config['checkpoint_dir'] = osp.join(checkpoint_dir, ckpt_name, 'last.pth')
    
    
    gen100000_root = '/data/whx/generation/gen100000/'
    # config['predict_filepath'] = gen100000_root + '20250108_221619_ckpt72_steps2000_escherichia_vibrio_gen100000/gen_seqs.csv'

    # config['predict_filepath'] = gen100000_root + '20250111_083422_ckpt72_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = gen100000_root + '20250122_232752_ckpt72_steps2000_streptomyces_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = gen100000_root + '20250111_083756_ckpt72_steps2000_vibrio_gen100000/gen_seqs.csv'
    
    # config['predict_filepath'] = gen100000_root + '20250113_100814_ckpt72new_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = gen100000_root + '20250122_232818_ckpt72new_steps2000_streptomyces_gen100000/gen_seqs.csv'    
    # config['predict_filepath'] = gen100000_root + '20250111_083726_ckpt72new_steps2000_vibrio_gen100000/gen_seqs.csv'

    # config['predict_filepath'] = gen100000_root + '20250111_083449_dnabert2_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = gen100000_root + '20250111_083648_dnabert2_steps2000_streptomyces_gen100000/gen_seqs.csv'
    config['predict_filepath'] = gen100000_root + '20250111_083816_dnabert2_steps2000_vibrio_gen100000/gen_seqs.csv'
