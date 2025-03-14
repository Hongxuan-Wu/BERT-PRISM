from collections import OrderedDict

config = OrderedDict()

config['mode'] = 'test'
# config['mode'] = 'predict'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_roc_auc'] = True
config['save_y'] = True
config['save_scatter'] = True

# config['root'] = '/data/whx/models/BERT-PRISM/'
config['root'] = '/data/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # 0-12
config['output_type'] = 'pool'  # pool / cls / mean

config['pretrained_path'] = config['root']+ "pretrained/BERT-PRISM-1/"

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'authenticity_cls/'

################################################################### Training ###############################################################

############################
# config['kfold'] = True
config['kfold'] = False
############################

if config['kfold']:
    config['n_folds'] = 10  # 5/10
    
    if config['n_folds'] == 5:
        # config['fold'] = [0]
        config['fold'] = [0,1,2,3,4]
    elif config['n_folds'] == 10:
        # config['fold'] = [0]
        config['fold'] = [0,1,2,3,4,5,6,7,8,9]

    # Learning
    config['train_batch_size'] = 128  # 128 / 2048
    config['valid_batch_size'] = 128  # 128 / 2048(ecos_all)
    config['n_epochs'] = 100
    config['lr'] = 3e-5  # 3e-5 / 1e-4(ecos_all)
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    config['lr_decay_epoch'] = config['n_epochs'] // 3
    # config['lr_decay_epoch'] = 100

    # config['dataset'] = 'prokaryotes596'  # 2468649
    # config['dataset'] = 'genes_ecos'
    # config['dataset'] = 'genes_corynebacterium'
    # config['dataset'] = 'genes_synechocystis'
    # config['dataset'] = 'genes_bacillus'
    # config['dataset'] = 'ESM'  # 
    # config['dataset'] = 'mg1655'
    # config['dataset'] = 'MG1655_RegulonDB'
    # config['dataset'] = 'Bacillus'  # 
    # config['dataset'] = 'Burkholderia'  # 
    # config['dataset'] = 'Campylobacter'
    # config['dataset'] = 'Corynebacterium'  # 
    # config['dataset'] = 'Escherichia'  # 
    # config['dataset'] = 'Haloferax'
    # config['dataset'] = 'Klebsiella'  # 
    # config['dataset'] = 'Onion'
    # config['dataset'] = 'Shigella'  # 
    # config['dataset'] = 'Sinorhizobium'  # 
    # config['dataset'] = 'Staphylococcus'
    # config['dataset'] = 'Synechocystis'  # 
    # config['dataset'] = 'Thermococcus'
    # config['dataset'] = 'Xanthomonas'  # 
    
    config['dataset'] = 'escherichia_vibrio'
    
    
    if 'genes_' in config['dataset']:
        config['datasize'] = 58498
        # config['datasize'] = 100000000
        # config['n_folds'] = 100
        # config['fold'] = [0]
    elif config['dataset'] == 'prokaryotes596':
        config['datasize'] = 100000000
        config['n_folds'] = 500
        config['fold'] = [0]

else:
    # Learning
    config['train_batch_size'] = 128  # 128
    config['valid_batch_size'] = 128  # 128
    config['n_epochs'] = 100
    config['lr'] = 3e-5  # 3e-5 / 1e-5 / 3e-6 / 1e-6
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    config['lr_decay_epoch'] = config['n_epochs'] // 3
    # config['lr_decay_epoch'] = 5
    
    # Dataset
    # config['train_dataset'] = 'genes_ecos'  # 58498 - sum: 676318 --- 1e-6
    # config['train_dataset'] = 'genes_corynebacterium'
    # config['train_dataset'] = 'genes_synechocystis'
    # config['train_dataset'] = 'genes_bacillus'
    
    # config['train_dataset'] = 'genes_streptomyces'
    # config['train_dataset'] = 'genes_vibrio'
    
    # config['train_dataset'] = 'ESM'  # 58498
    config['train_dataset'] = 'Escherichia'  # 17232
    # config['train_dataset'] = 'Corynebacterium'  # 7162
    # config['train_dataset'] = 'Synechocystis'  # 1888
    # config['train_dataset'] = 'Bacillus'  # 1526
    
    
    if 'genes_' in config['train_dataset']:
        config['train_datasize'] = 58498
        # config['train_datasize'] = 1000
        # config['train_datasize'] = 2000
        # config['train_datasize'] = 5000
        # config['train_datasize'] = 10000
        # config['train_datasize'] = 20000
        # config['train_datasize'] = 50000
        # config['train_datasize'] = 100000
        # config['train_datasize'] = 200000
        # config['train_datasize'] = 676318

    # config['val_dataset'] = 'mg1655'
    config['val_dataset'] = 'MG1655_RegulonDB'
    # config['val_dataset'] = 'Bacillus'
    # config['val_dataset'] = 'Burkholderia'
    # config['val_dataset'] = 'Campylobacter'
    # config['val_dataset'] = 'Corynebacterium'
    # config['val_dataset'] = 'Escherichia'
    # config['val_dataset'] = 'Haloferax'
    # config['val_dataset'] = 'Klebsiella'
    # config['val_dataset'] = 'Onion'
    # config['val_dataset'] = 'Shigella'
    # config['val_dataset'] = 'Sinorhizobium'
    # config['val_dataset'] = 'Staphylococcus'
    # config['val_dataset'] = 'Synechocystis'
    # config['val_dataset'] = 'Thermococcus'
    # config['val_dataset'] = 'Xanthomonas'


if config['mode'] == 'test':

    config['checkpoint_dir'] = config['root'] + 'authenticity_cls/BERT-PRISM-1_Escherichia/'
    # config['checkpoint_dir'] = config['root'] + 'authenticity_cls/BERT-PRISM-1_ESM/'

elif config['mode'] == 'predict':
    config['valid_batch_size'] = 128
    
    config['checkpoint_dir'] = config['root'] + 'authenticity_cls/BERT-PRISM-1_Escherichia/last.pth'

    root_gen100000 = '/data/whx/generation/gen100000/'

    config['predict_filepath'] = root_gen100000 + '20250111_083816_dnabert2_steps2000_vibrio_gen100000/gen_seqs.csv'
