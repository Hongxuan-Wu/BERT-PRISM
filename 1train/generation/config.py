from collections import OrderedDict

config = OrderedDict()

config['token_length'] = 100

config['mode'] = 'generate'

config['save_last_model'] = True
config['save_best_model'] = True

config['save_metrics'] = True

config['root'] = '/data/whx/'
# config['root'] = '/hy-tmp/'

################################################################### Model ##############################################################
# config['pretrained_path'] = config['root'] + "ckpt/DNABERT-2-117M"
# config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000/"
config['pretrained_path'] = config['root'] + "ckpt/checkpoint-720000_newtoken/"


if config['mode'] == 'generate':
    # config['checkpoint_dir'] = config['root'] + 'ckpt/ckpt_ecos_steps2000/'
    # config['checkpoint_dir'] = config['root'] + 'ckpt/generation/ckpt/ckpt72_escherichia_vibrio/'
    # config['checkpoint_dir'] = config['root'] + 'ckpt/generation/ckpt/ckpt72_streptomyces/'
    config['checkpoint_dir'] = config['root'] + 'ckpt/generation/ckpt/ckpt72new_streptomyces/'

# config['n_froze_layers'] = 9  # 0-12

# ------------------------------------------ Diffusion ------------------------------------------
# config['dtype'] = 'fp16'
config['dtype'] = 'bf16'

config['num_steps'] = 2000  # 1000 (500, 1000, 2000, 5000)
config['sample_strategy'] = 'Categorical'
config['word_freq_lambda'] = 0.3
config['timestep'] = 'none'  # 'none', 'token', 'layerwise'
config['hybrid_lambda'] = 1e-2
config['predict_x0'] = True

# config['eval_step_size'] = config['num_steps'] // 50  # eval_num_steps = 50
config['eval_step_size'] = config['num_steps'] // 2  # fastest eval
# config['eval_step_size'] = 1

config['predict_filter_topk'] = 10  # 10
config['predict_filter_topp'] = -1.0  # -1.0
config['predict_num'] = 100000  # 10000, 100000
config['predict_batch'] = 100

#################################################################### Data ##############################################################

config['root_project'] = config['root'] + 'projects/DNABERT_Promotor/1train/promoter_generation/data/'
config['root_data'] = config['root'] + 'expressions/'

config['catATG'] = True
# config['catATG'] = False
# 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'

# config['select_active'] = True
config['select_active'] = False

################################################################## Training ###############################################################

############################
config['kfold'] = True
# config['kfold'] = False
############################

if config['kfold']:
    config['n_folds'] = 10  # 5/10
    
    if config['n_folds'] == 5:
        # config['fold'] = [0]
        config['fold'] = [0,1,2,3,4]
    elif config['n_folds'] == 10:
        config['fold'] = [0]
        # config['fold'] = [0,1,2,3,4,5,6,7,8,9]

    # Learning
    config['n_epochs'] = 200                   # 100 / 20
    config['train_batch_size'] = 128  # 128    256 / 224 / 192 / 160 / 128
    config['valid_batch_size'] = 128  # 128
    config['lr'] = 1e-4  # 1e-4 / 3e-5
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 1000

    # config['dataset'] = 'genes_prokaryotes'
    # config['dataset'] = 'genes_escherichia'
    config['dataset'] = 'genes_streptomyces'
    # config['dataset'] = 'genes_vibrio'
    
    # config['dataset'] = 'escherichia_corynebacterium'
    # config['dataset'] = 'escherichia_vibrio'
    
    # config['dataset'] = 'ESM'
    # config['dataset'] = 'mg1655'
    # config['dataset'] = 'MG1655_RegulonDB'
    # config['dataset'] = 'Bacillus'
    # config['dataset'] = 'Burkholderia'
    # config['dataset'] = 'Campylobacter'
    # config['dataset'] = 'Corynebacterium'
    # config['dataset'] = 'Escherichia'
    # config['dataset'] = 'Haloferax'
    # config['dataset'] = 'Klebsiella'
    # config['dataset'] = 'Onion'
    # config['dataset'] = 'Shigella'
    # config['dataset'] = 'Sinorhizobium'
    # config['dataset'] = 'Staphylococcus'
    # config['dataset'] = 'Synechocystis'
    # config['dataset'] = 'Thermococcus'
    # config['dataset'] = 'Xanthomonas'
    
    if config['dataset'] == 'genes_escherichia':
        config['datasize'] = 100000000
        config['n_folds'] = 25  # sum - 100 / fromlist - 25 / manual - 50
        config['fold'] = [0]
        config['n_epochs'] = 400
        config['lr'] = 7e-5
    elif config['dataset'] == 'genes_streptomyces':
        config['datasize'] = 100000000
        config['n_folds'] = 30
        config['fold'] = [0]
        config['n_epochs'] = 400
        config['lr'] = 7e-5
    elif config['dataset'] == 'genes_vibrio':
        config['datasize'] = 100000000
        config['n_folds'] = 16
        config['fold'] = [0]
        config['n_epochs'] = 400
    elif config['dataset'] == 'genes_prokaryotes':
        config['datasize'] = 100000000
        config['n_folds'] = 400
        config['fold'] = [0]
        # config['lr'] = 1e-5
    elif config['dataset'] == 'escherichia_corynebacterium' or config['dataset'] == 'escherichia_vibrio':
        # config['fold'] = [0]
        # config['n_epochs'] = 400
        config['lr'] = 1e-5
        pass
    
else:
    # Learning
    config['train_batch_size'] = 128  # 128
    config['valid_batch_size'] = 128  # 128
    config['n_epochs'] = 200
    config['lr'] = 1e-4  # 3e-5 / 1e-5 / 3e-6
    config['wd'] = 1e-2  # 1e-2
    config['lr_decay'] = 0.1
    # config['lr_decay_epoch'] = config['n_epochs'] // 3
    config['lr_decay_epoch'] = 100
    
    config['train_dataset'] = 'genes_prokaryotes'
    # config['train_dataset'] = 'genes_escherichia'
    # config['train_dataset'] = 'genes_streptomyces'
    # config['train_dataset'] = 'genes_vibrio'
    
    config['val_dataset'] = 'EColi_GeneExpression'
    # config['val_dataset'] = 'Bsub_GeneExpression'
    # config['val_dataset'] = 'Cglu_GeneExpression'
    # config['val_dataset'] = 'Vibr_GeneExpression'
    # config['val_dataset'] = 'WCFS1_GeneExpression'
    
    if config['train_dataset'] == 'genes_escherichia':
        pass
    elif config['train_dataset'] == 'genes_streptomyces':
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['train_dataset'] == 'genes_vibrio':
        config['lr_decay_epoch'] = config['n_epochs'] // 3
    elif config['train_dataset'] == 'genes_prokaryotes':
        config['lr'] = 1e-5
