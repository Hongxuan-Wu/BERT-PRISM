import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import pdb

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader

def load_dataframe(args, config):
    # read dataframe
    if not config['mode'] == 'predict':
        if config['kfold']:
            if 'genes_' in config['dataset']:
                datapath = os.path.join(config['root_data'], 'data_gene_tpm')
                
                if 'prokaryotes' in config['dataset']:
                    dataname = 'tpm_gene_' + config['dataset'][6:] + '_filtered'
                else:            
                    dataname = 'tpm_gene_' + config['dataset'][6:]
                
                if 'newtoken' in config['pretrained_path']:
                    dataname += '_newtoken'
                elif 'mergetoken' in config['pretrained_path']:
                    dataname += '_mergetoken'
                    
                if config['catATG'] and ('prokaryotes' not in config['dataset']):
                    dataname += '_catATG'
                
                datapath = osp.join(datapath, dataname+'.npz')
                df_raw = np.load(datapath)

                sequences = np.array(df_raw['sequences'])
                masks = np.array(df_raw['masks'])
                intensity = np.array(df_raw['intensity'])
                
                intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'intensity': torch.tensor(intensity, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }

            ########################################################## mutation ##########################################################
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur_train':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='La Fleur et al (Fig 1d, S1)'
                )
                if config['high_quality']:
                    df = df[df['high quality'] == 'Yes'].reset_index(drop=True)
                # pdb.set_trace()
                
                df['Promoter Sequence'] = df['UP'].str.upper() + df['h35'].str.upper() + df['spacs'].str.upper() + df['h10'].str.upper() + df['disc'].str.upper() + df['ITR'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
                
                gene = 'CTGTGCGTTAGGTGATGCCGCGGCTGTACCGCCCATGTGCTCACGAATTCGATCAAATTTCGAGGTTCCATATGGCGAGCTCTGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAAACTAGTAAACGCAGTTACCCCATAGGCT'
                df['genes'] = gene.upper()
                
                intensity = df['intensity'].to_numpy()
                df['intensity'] = intensity
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='La Fleur et al (Fig 3a)'
                )
                df['Promoter Sequence'] = df['Upstream DNA'].str.upper() + df['Promoter Sequence'].str.upper() + df['Downstream DNA'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_Yu':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Yu et al (Fig S3)'
                )
                df['Promoter Sequence'] = df['Promoter Sequence'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_36N':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Lagator 36N'
                )
                df['Promoter'] = df['Promoter'].str.upper()
                df = pd.concat([df['Promoter'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_PL':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Lagator Pl'
                )
                df['Promoter'] = df['Promoter'].str.upper()
                df = pd.concat([df['Promoter'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_PR':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Lagator Pr'
                )
                df['Promoter'] = df['Promoter'].str.upper()
                df = pd.concat([df['Promoter'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '3655_synthetic_promoters':
                df = pd.read_csv(
                    osp.join(config['root_project'], '3665 synthetic promoters.csv'), 
                    delimiter=','
                )
                df.columns=['id', 'promoters', 'intensity']
                
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                df['promoters'] = 'ACATCATAACGGTTCTGGCAAATATTCTGAAATGAGCTG' + df['promoters'].str.upper()
                
                intensity = df['intensity'].to_numpy()
                intensity = np.log1p(intensity)
                intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity
            elif config['dataset'] == 'ecoli_lacUV5':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'mutation/ecoli_lacUV5_250126.xlsx'), 
                    engine='openpyxl',
                    keep_default_na=False
                )
                df.columns = ['promoters', 'intensity']
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                # df = df[df['intensity']>0].reset_index(drop=True)
                
                intensity = df['intensity'].to_numpy()
                intensity = np.log1p(intensity)
                intensity = (intensity - intensity.mean()) / intensity.std()  # 8.798903534559255, 1.064224445763706
                df['intensity'] = intensity
            
            ########################################################## mix ##########################################################
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_Urtecho':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Urtecho et al (Fig 3c, S7b)'
                )
                df['Promoter Sequence'] = df['Promoter Sequence'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_Hossain':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Hossain et al (Fig 3d, S7d)'
                )
                df['Promoter Sequence'] = df['Upstream DNA'].str.upper() + df['Promoter Sequence'].str.upper() + df['Downstream DNA'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
            elif config['dataset'] == 'ggn210095':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'ggn210095-sup-0004-datas3.xlsx'), 
                    engine='openpyxl',
                    sheet_name='NDB',
                    keep_default_na=False
                )
                df.columns = ['promoters', 'intensity']
                df['promoters'] = 'cggaattccctaggggatcc'.upper() + df['promoters'].str.upper() + 'ccaaatacaattggagatgg'.upper()
                
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                # filter
                df = df[df['intensity'] > 0].reset_index(drop=True)
                
                intensity = df['intensity'].to_numpy()
                intensity = np.log1p(intensity)
                intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity
            elif config['dataset'] == 'tpm_fluorescence':
                df = pd.read_csv(
                    osp.join(config['root_project'], 'Merged_TPM_and_Fluorescence/Merged_TPM_and_Fluorescence_unique.txt'), 
                    delimiter=','
                )
                df['promoters'] = df['promoter'].str.upper()
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                intensity = df['fluorescence'].to_numpy()
                intensity = np.log1p(intensity)
                intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity
                
            if '41467_2022_32829_MOESM5_ESM' in config['dataset'] and 'train' not in config['dataset']:
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                intensity = df['intensity'].to_numpy()
                # intensity = -intensity
                # intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity

            # pdb.set_trace()
            
            # KFold
            if 'genes_' in config['dataset']:
                df['fold'] =  torch.tensor([-1 for _ in range(len(df['intensity']))], dtype=torch.int64)
                
                kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'])):
                    df['fold'][val_idx] = fold_idx
            else:
                df["fold"] = -1
                
                kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
                    df.loc[val_idx, "fold"] = fold_idx

                # kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
                # for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df, y=df['quantile_category'])):
                #     df.loc[val_idx, "fold"] = fold_idx
            
            df_val = df[df['fold']==3][['promoters','intensity']]
            df_train = df[df['fold']!=3][['promoters','intensity']]
            path = '/data/whx/projects/DNABERT_Promotor/1train/expression_intensity_prediction/results/logs/20250126_233931/'
            df_val.to_csv(path+'fold3_val.csv', header=True, index=True)
            df_train.to_csv(path+'fold3_train.csv', header=True, index=True)

            pdb.set_trace()
            return df
        else:
            if 'genes_' in config['train_dataset']:
                datapath = os.path.join(config['root_data'], 'data_gene_tpm')
                
                if 'prokaryotes' in config['train_dataset']:
                    dataname = 'tpm_gene_' + config['train_dataset'][6:] + '_filtered'
                else:            
                    dataname = 'tpm_gene_' + config['train_dataset'][6:]
                
                if 'newtoken' in config['pretrained_path']:
                    dataname += '_newtoken'
                elif 'mergetoken' in config['pretrained_path']:
                    dataname += '_mergetoken'
                
                if config['catATG'] and ('prokaryotes' not in config['train_dataset']):
                    dataname += '_catATG'
                
                datapath = osp.join(datapath, dataname+'.npz')
                df_raw = np.load(datapath)
                            
                sequences = np.array(df_raw['sequences'])
                masks = np.array(df_raw['masks'])
                intensity = np.array(df_raw['intensity'])
                
                intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                
                train_dataset = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'intensity': torch.tensor(intensity, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }
            elif config['train_dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur_train':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='La Fleur et al (Fig 1d, S1)'
                )
                if config['high_quality']:
                    df = df[df['high quality'] == 'Yes'].reset_index(drop=True)
                # pdb.set_trace()
                
                df['Promoter Sequence'] = df['UP'].str.upper() + df['h35'].str.upper() + df['spacs'].str.upper() + df['h10'].str.upper() + df['disc'].str.upper() + df['ITR'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)

                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                # gene = 'CTGTGCGTTAGGTGATGCCGCGGCTGTACCGCCCATGTGCTCACGAATTCGATCAAATTTCGAGGTTCCATATGGCGAGCTCTGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAAACTAGTAAACGCAGTTACCCCATAGGCT'
                df['genes'] = gene.upper()
                
                intensity = df['intensity'].to_numpy()
                # intensity = -intensity
                # intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity
                
                train_dataset = df
            
            ########################################################## transcriptome ##########################################################
            if '_GeneExpression' in config['val_dataset']:
                if config['val_dataset'] == 'EColi_GeneExpression':
                    df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/EColi_tpm_gene.txt'))
                elif config['val_dataset'] == 'Bsub_GeneExpression':
                    df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Bsub_tpm_gene.txt'))
                elif config['val_dataset'] == 'Cglu_GeneExpression':
                    df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Cglu_tpm_gene.txt'))
                elif config['val_dataset'] == 'Vibr_GeneExpression':
                    df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/Vibr_tpm_gene.txt'))
                elif config['val_dataset'] == 'WCFS1_GeneExpression':
                    df_tmp = pd.read_csv(os.path.join(config['root_data'], 'data_gene_tpm/transcript/WCFS1_tpm_gene.txt'))
                
                df_tmp.columns=['id', 'promoters', 'genes', 'intensity']
                intensity = np.array(df_tmp['intensity'])
                
                intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                
                df_tmp['intensity'] = intensity
                val_dataset = df_tmp
            elif config['val_dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='La Fleur et al (Fig 3a)'
                )
                df['Promoter Sequence'] = df['Upstream DNA'].str.upper() + df['Promoter Sequence'].str.upper() + df['Downstream DNA'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
                val_dataset = df
            elif config['val_dataset'] == '41467_2022_32829_MOESM5_ESM_Yu':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Yu et al (Fig S3)'
                )
                df['Promoter Sequence'] = df['Promoter Sequence'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
                val_dataset = df
            elif config['val_dataset'] == '41467_2022_32829_MOESM5_ESM_Urtecho':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Urtecho et al (Fig 3c, S7b)'
                )
                df['Promoter Sequence'] = df['Promoter Sequence'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
                val_dataset = df
            elif config['val_dataset'] == '41467_2022_32829_MOESM5_ESM_Hossain':
                df = pd.read_excel(
                    os.path.join(config['root_project'], 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
                    engine='openpyxl',
                    sheet_name='Hossain et al (Fig 3d, S7d)'
                )
                df['Promoter Sequence'] = df['Upstream DNA'].str.upper() + df['Promoter Sequence'].str.upper() + df['Downstream DNA'].str.upper()
                df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
                df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
                val_dataset = df
            elif config['val_dataset'] == 'tpm_fluorescence':
                df = pd.read_csv(
                    osp.join(config['root_project'], 'Merged_TPM_and_Fluorescence/Merged_TPM_and_Fluorescence_unique.txt'), 
                    delimiter=','
                )
                df['promoters'] = df['promoter'].str.upper()
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df['genes'] = gene.upper()
                
                # intensity = df['fluorescence'].to_numpy()
                intensity = df['tpm'].to_numpy()
                intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                df['intensity'] = intensity
                val_dataset = df
                
            if '41467_2022_32829_MOESM5_ESM' in config['val_dataset']:
                gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                # gene = 'CTGTGCGTTAGGTGATGCCGCGGCTGTACCGCCCATGTGCTCACGAATTCGATCAAATTTCGAGGTTCCATATGGCGAGCTCTGAAGACGTTATCAAAGAGTTCATGCGTTTCAAAGTTCGTATGGAAGGTTCCGTTAACGGTCACGAGTTCGAAATCGAAGGTGAAGGTGAAGGTCGTCCGTACGAAGGTACCCAGACCGCTAAACTGAAAGTTACCAAAGGTGGTCCGCTGCCGTTCGCTTGGGACATCCTGTCCCCGCAGTTCCAGTACGGTTCCAAAGCTTACGTTAAACACCCGGCTGACATCCCGGACTACCTGAAACTGTCCTTCCCGGAAGGTTTCAAATGGGAACGTGTTATGAACTTCGAAGACGGTGGTGTTGTTACCGTTACCCAGGACTCCTCCCTGCAAGACGGTGAGTTCATCTACAAAGTTAAACTGCGTGGTACCAACTTCCCGTCCGACGGTCCGGTTATGCAGAAAAAAACCATGGGTTGGGAAGCTTCCACCGAACGTATGTACCCGGAAGACGGTGCTCTGAAAGGTGAAATCAAAATGCGTCTGAAACTGAAAGACGGTGGTCACTACGACGCTGAAGTTAAAACCACCTACATGGCTAAAAAACCGGTTCAGCTGCCGGGTGCTTACAAAACCGACATCAAACTGGACATCACCTCCCACAACGAAGACTACACCATCGTTGAACAGTACGAACGTGCTGAAGGTCGTCACTCCACCGGTGCTTAATAAACTAGTAAACGCAGTTACCCCATAGGCT'
                val_dataset['genes'] = gene.upper()
                
                intensity = val_dataset['intensity'].to_numpy()
                # intensity = -intensity
                # intensity = np.log1p(intensity)
                # intensity = (intensity - intensity.mean()) / intensity.std()
                val_dataset['intensity'] = intensity
            
            # pdb.set_trace()
            return train_dataset, val_dataset
    elif config['mode'] == 'predict':
        df_predict = pd.read_csv(config['predict_filepath'], delimiter=',', header=None)
        return df_predict

def make_loader(args, config, df_train, df_valid, tokenizer):
    if config['kfold']:
        if 'genes_' in config['dataset']:
            train_dataset = NpyDataset(df=df_train)
            valid_dataset = NpyDataset(df=df_valid)
        else:
            train_dataset = PromoterDataset(df=df_train, tokenizer=tokenizer)
            valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    else:
        if 'genes_' in config['train_dataset']:
            train_dataset = NpyDataset(df=df_train)
        else:
            train_dataset = PromoterDataset(df=df_train, tokenizer=tokenizer)
        valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train_batch_size'], 
        shuffle=True, 
        drop_last=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config['valid_batch_size'], 
        shuffle=False, 
        drop_last=False,
        num_workers=args.num_workers, 
        pin_memory=True
    )
    
    data_loader = {'train': train_loader,
                   'val': valid_loader}
    return data_loader

def make_loader_predict(args, config, df_predict, tokenizer):
    predict_dataset = PredictDataset(df=df_predict, tokenizer=tokenizer)
    predict_loader = DataLoader(
            predict_dataset, 
            batch_size=config['valid_batch_size'], 
            shuffle=False, 
            drop_last=False,
            num_workers=args.num_workers, 
            pin_memory=True
    )
    data_loader = {
        'predict': predict_loader
    }
    return data_loader 

class PredictDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PredictDataset, self).__init__()
        self.df = df
        
        # get token id
        file = df.to_numpy()
        sequence = file[:, 0].tolist()
        
        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )  # input_ids, token_type_ids, attention_mask
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx]

class PromoterDataset(Dataset):
    def __init__(self, df, tokenizer):
        super(PromoterDataset, self).__init__()
        self.df = df

        sequence = (df['promoters'] + df['genes']).tolist()
        self.intensity = torch.tensor(df['intensity'].tolist(), dtype=torch.float32).float()
        
        self.seq_output = tokenizer(
            text=sequence, 
            return_tensors="pt", 
            max_length=100, 
            padding=True,
            truncation=True,
        )  # input_ids, token_type_ids, attention_mask
        self.seq_input_ids = self.seq_output['input_ids']
        self.seq_attention_mask = self.seq_output['attention_mask']
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.intensity[idx]

class NpyDataset(Dataset):
    def __init__(self, df):
        super(NpyDataset, self).__init__()
        self.df = df
        
        # get token id
        self.intensity = df['intensity']
        self.seq_input_ids = df['sequences']
        self.masks = df['masks']
        # pdb.set_trace()
        
    def __len__(self):
        return len(self.df['intensity'])

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.masks[idx], self.intensity[idx]
