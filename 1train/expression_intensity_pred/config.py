import os.path as osp
from collections import OrderedDict

config = OrderedDict()

config['mode'] = 'test'
# config['mode'] = 'predict'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_metrics'] = True
config['save_y'] = True
config['save_scatter'] = True

# config['root'] = '/data/whx/models/BERT-PRISM/'
config['root'] = '/data/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # 0-12
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'expression_intensity_pred/'
config['root_data'] = config['root_project']

# config['catATG'] = True
config['catATG'] = False

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
        # config['fold'] = [0]
        config['fold'] = [0,1,2,3,4,5,6,7,8,9]
    
    # Learning
    config['n_epochs'] = 100
    config['train_batch_size'] = 2048  # 128
    config['valid_batch_size'] = 2048  # 128
    config['lr'] = 3e-5  # 3e-5 / 1e-5 / 3e-6 / 1e-6
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 100

    # config['dataset'] = 'genes_prokaryotes'
    # config['dataset'] = 'genes_escherichia'
    # config['dataset'] = 'genes_streptomyces'
    # config['dataset'] = 'genes_vibrio'

    # Paper - 《Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria》
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_LaFleur_train'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_LaFleur'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_Yu'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_36N'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_PL'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_PR'
    # config['dataset'] = '3655_synthetic_promoters'
    config['dataset'] = 'ecoli_lacUV5'
    
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_Urtecho'
    # config['dataset'] = '41467_2022_32829_MOESM5_ESM_Hossain'
    # config['dataset'] = 'ggn210095'  # NDB
    # config['dataset'] = 'tpm_fluorescence'
    
    
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
    elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur_train':
        config['high_quality'] = True
        # config['high_quality'] = False
        config['lr'] = 5e-4
    elif config['dataset'] == 'ecoli_lacUV5':
        config['n_folds'] = 4
        config['fold'] = [3]
        # config['fold'] = [0,1,2,3]
        
        config['n_epochs'] = 300
        bs = 2048
        config['train_batch_size'] = bs
        config['valid_batch_size'] = bs
        config['lr'] = 5e-4
        
else:
    # Learning
    config['n_epochs'] = 100
    config['train_batch_size'] = 2048  # 128
    config['valid_batch_size'] = 2048  # 128
    config['lr'] = 3e-5  # 3e-5 / 1e-5 / 3e-6 / 1e-6
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 100

    # config['train_dataset'] = 'genes_prokaryotes'
    config['train_dataset'] = 'genes_escherichia'
    # config['train_dataset'] = 'genes_streptomyces'
    # config['train_dataset'] = 'genes_vibrio'
    # config['train_dataset'] = '41467_2022_32829_MOESM5_ESM_LaFleur_train'

    if config['train_dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur_train':
            config['high_quality'] = False
            # config['lr'] = 1e-4
            
            # config['high_quality'] = True
            config['lr'] = 5e-4

    config['val_dataset'] = 'EColi_GeneExpression'
    # config['val_dataset'] = 'Bsub_GeneExpression'
    # config['val_dataset'] = 'Cglu_GeneExpression'
    # config['val_dataset'] = 'Vibr_GeneExpression'
    # config['val_dataset'] = 'WCFS1_GeneExpression'
    
    # config['val_dataset'] = '41467_2022_32829_MOESM5_ESM_LaFleur'
    # config['val_dataset'] = '41467_2022_32829_MOESM5_ESM_Yu'
    # config['val_dataset'] = '41467_2022_32829_MOESM5_ESM_Urtecho'
    # config['val_dataset'] = '41467_2022_32829_MOESM5_ESM_Hossain'
    # config['val_dataset'] = 'tpm_fluorescence'


if config['mode'] == 'test':
    config['checkpoint_dir'] = config['root'] + 'expression_intensity_pred/BERT-PRISM-1_escherichia/'

elif config['mode'] == 'predict':
    config['checkpoint_dir'] = config['root'] + 'expression_intensity_pred/BERT-PRISM-1_escherichia_catATG/'
    
    gen100000_root = '/data/whx/generation/gen100000/'
    config['predict_filepath'] = gen100000_root + '20250111_083816_dnabert2_steps2000_vibrio_gen100000/gen_seqs.csv'
