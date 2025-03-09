from collections import OrderedDict

config = OrderedDict()

# config['mode'] = 'test'
config['mode'] = 'predict'

config['save_last_model'] = True
config['save_best_model'] = False

config['save_roc_auc'] = True
config['save_y'] = True
config['save_scatter'] = True

config['root'] = '/data/whx/'
# config['root'] = '/hy-tmp/'

################################################################### Model ##############################################################

config['n_froze_layers'] = 11  # 0-12
config['output_type'] = 'pool'  # pool / cls / mean

# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000_newtoken/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000/"
config['pretrained_path'] = "/data/whx/ckpt/DNABERT-2-117M/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-1440000_newtoken/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000_newtoken_mlm30/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000_mergetoken/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000_newtoken_onlypromoter/"
# config['pretrained_path'] = "/data/whx/ckpt/checkpoint-720000_newtoken_onlypromoter50tokens/"


#################################################################### Data ##############################################################

# config['root_project'] = config['root'] + 'projects/DNABERT_Promotor/1train/promoter_real_fake_predict/data/'
config['root_project'] = config['root'] + 'projects/DNABERT_Promotor/1train/promoter_real_fake_predict/data/'
config['root_genes_tokens'] = config['root'] + 'realfake/'

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
    
    config['train_dataset'] = 'ESM'  # 58498
    # config['train_dataset'] = 'Escherichia'  # 17232
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
    if "ckpt/checkpoint-720000_newtoken/" in config['pretrained_path']:
        model = 'ckpt72new'
    elif "ckpt/checkpoint-720000/" in config['pretrained_path']:
        model = 'ckpt72'
    elif "ckpt/DNABERT-2-117M/" in config['pretrained_path']:
        model = 'dnabert2'
    elif "ckpt/checkpoint-1440000_newtoken/" in config['pretrained_path']:
        model = 'ckpt144new'
    elif "ckpt/checkpoint-720000_newtoken_mlm30/" in config['pretrained_path']:
        model = 'ckpt72newmlm30'
    elif "ckpt/checkpoint-720000_mergetoken/" in config['pretrained_path']:
        model = 'ckpt72merge'
    elif "ckpt/checkpoint-720000_newtoken_onlypromoter/" in config['pretrained_path']:
        model = 'ckpt72newonlypromoter'
    elif "ckpt/checkpoint-720000_newtoken_onlypromoter50tokens/" in config['pretrained_path']:
        model = 'ckpt72newonlypromoter50tokens'
    
    if config['kfold']:
        checkpoint_root = config['root'] + 'ckpt/realfake/ckpt10fold/'
        config['checkpoint_dir'] = checkpoint_root + model + '_' + config['dataset'] + '_10fold'
    else:
        checkpoint_root = config['root'] + 'ckpt/realfake/ckpt_to1/'
        # config['checkpoint_dir'] = checkpoint_root + model + '_' + config['train_dataset']
        config['checkpoint_dir'] = checkpoint_root + model + '_' + 'ecos'

elif config['mode'] == 'predict':
    config['valid_batch_size'] = 128
    
    # config['checkpoint_dir'] = '/hy-tmp/ckpt/real_fake_ecos_epoch100'
    
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72_ESM/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72_Escherichia/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72_streptomyces/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72_vibrio/last.pth'
    
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72new_ESM/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72new_Escherichia/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72new_streptomyces/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/ckpt72new_vibrio/last.pth'
    
    config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/dnabert2_ESM/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/dnabert2_Escherichia/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/dnabert2_streptomyces/last.pth'
    # config['checkpoint_dir'] = '/data/whx/ckpt/realfake/ckpt_to1/dnabert2_vibrio/last.pth'
    
    

    # config['predict_filepath'] = config['root'] + 'predict/ecos/20240813_035617_ckpt72newtoken_steps1000_ecos_gen10000/gen_seqs.csv'
    # config['predict_filepath'] = config['root'] + 'predict/ecos/20240911_125214_ckpt72newtoken_steps2000_ecos_gen10000/gen_seqs.csv'
    # config['predict_filepath'] = config['root'] + 'predict/ecos/20240912_043045_ckpt72newtoken_steps2000_ecos_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = '/data/whx/projects/DNABERT_Promotor/1train/promoter_generation/results/logs/20250103_165436_ckpt72_steps2000_escherichia_vibrio_gen10000/gen_seqs.csv'
    # config['predict_filepath'] = '/data/whx/projects/DNABERT_Promotor/1train/promoter_generation/results/logs/20250108_221619_ckpt72_steps2000_escherichia_vibrio_gen100000/gen_seqs.csv'

    root_gen100000 = '/data/whx/generation/gen100000/'
    # config['predict_filepath'] = root_gen100000 + '20250111_083422_ckpt72_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = root_gen100000 + '20250122_232752_ckpt72_steps2000_streptomyces_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = root_gen100000 + '20250111_083756_ckpt72_steps2000_vibrio_gen100000/gen_seqs.csv'
    
    # config['predict_filepath'] = root_gen100000 + '20250113_100814_ckpt72new_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = root_gen100000 + '20250122_232818_ckpt72new_steps2000_streptomyces_gen100000/gen_seqs.csv'    
    # config['predict_filepath'] = root_gen100000 + '20250111_083726_ckpt72new_steps2000_vibrio_gen100000/gen_seqs.csv'

    # config['predict_filepath'] = root_gen100000 + '20250111_083449_dnabert2_steps2000_escherichia_gen100000/gen_seqs.csv'
    # config['predict_filepath'] = root_gen100000 + '20250111_083648_dnabert2_steps2000_streptomyces_gen100000/gen_seqs.csv'
    config['predict_filepath'] = root_gen100000 + '20250111_083816_dnabert2_steps2000_vibrio_gen100000/gen_seqs.csv'
