import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pdb

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import resample
from torch.utils.data import DataLoader

def load_dataframe(args, config):
    # read dataframe
    if config['dataset'] == 'components':
        # filename = 'components_cut'
        filename = 'components_cut_filtered'
        
        if 'newtoken' in config['pretrained_path']:
            filename += '_newtoken'
        elif 'mergetoken' in config['pretrained_path']:
            filename += '_mergetoken'
        else:
            pass
        
        df_raw = np.load(osp.join(config['root_data'], filename+'.npz'))
        # df_raw.files
        
        sequences = df_raw['sequences']
        masks = df_raw['masks']
        masks_blank = df_raw['masks_blank']
        masks_up = df_raw['masks_up']
        masks_core = df_raw['masks_core']
        masks_down = df_raw['masks_down']
        masks_gene = df_raw['masks_gene']
        intensity = df_raw['intensity']
        
        intensity = np.log1p(intensity)
        intensity = (intensity - intensity.mean()) / intensity.std()  # mean - 4.813026468891894, std - 1.6166693708525242
        
        # mean = 4.813026468891894
        # std = 1.6166693708525242
        # y_fix = intensity * std + mean
        # y_fix2 = np.expm1(y_fix)
        # pdb.set_trace()
        
        df = {
            'sequences': torch.tensor(sequences, dtype=torch.int64),
            'intensity': torch.tensor(intensity, dtype=torch.float32),
            'masks': torch.tensor(masks, dtype=torch.int64),
            'masks_blank': torch.tensor(masks_blank, dtype=torch.int64),
            'masks_up' : torch.tensor(masks_up, dtype=torch.int64),
            'masks_core': torch.tensor(masks_core, dtype=torch.int64),
            'masks_down': torch.tensor(masks_down, dtype=torch.int64),
            'masks_gene': torch.tensor(masks_gene, dtype=torch.int64),
        }
    elif config['dataset'] == 'prokaryotes596':
        # df_raw = np.load(osp.join(config['root_data'], 'fromlist/prokaryotes/data/tpm_gene_prokaryotes.npz'))
        df_raw = np.load(osp.join(config['root_data'], 'fromlist/prokaryotes/data/tpm_gene_prokaryotes_filtered.npz'))
        
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
    elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur':
        df = pd.read_excel(
            os.path.join('/hy-tmp/projects/DNABERT_Promotor/1train/expression_intensity_prediction/data/', 'e_coli/41467_2022_32829_MOESM5_ESM.xlsx'), 
            engine='openpyxl',
            sheet_name='La Fleur et al (Fig 3a)'
        )
        df['Promoter Sequence'] = df['Upstream DNA'].str.upper() + df['Promoter Sequence'].str.upper() + df['Downstream DNA'].str.upper()
        df = pd.concat([df['Promoter Sequence'], df['Observed log(TX/Txref)']], axis=1)
        df.rename(columns={'Promoter Sequence': 'promoters', 'Observed log(TX/Txref)': 'intensity'}, inplace=True)
    
    if '41467_2022_32829_MOESM5_ESM' in config['dataset']:
        gene = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
        df['genes'] = gene.upper()

        intensity = df['intensity'].to_numpy()
        intensity = np.log1p(intensity)
        # intensity = (intensity - intensity.mean()) / intensity.std()
        df['intensity'] = intensity

    if config['mode'] == 'test':
        if config['dataset'] == 'components' or config['dataset'] == 'prokaryotes596':
            df['fold'] = torch.tensor([-1 for _ in range(len(df['sequences']))], dtype=torch.int64)
            
            kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'])):
                df['fold'][val_idx] = fold_idx
        else:
            df["fold"] = -1
        # pdb.set_trace()
        return df

def make_loader(args, config, df_train, df_valid, tokenizer):
    if config['mode'] == 'test':
        if config['dataset'] == 'components':
            valid_dataset = NpyComponentDataset(df=df_valid)
        elif config['dataset'] == 'prokaryotes596':
            valid_dataset = NpyDataset(df=df_valid)
        elif config['dataset'] == '41467_2022_32829_MOESM5_ESM_LaFleur':
            dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
            
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config['valid_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=True
            )
        data_loader = {
            'val': valid_loader
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
        
        promoter = df['promoters'].tolist()
        gene = df['genes'].tolist()
        
        sequence = (df['promoters'] + df['genes']).tolist()
        self.intensity = torch.tensor(df['intensity'].tolist())
        
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
        
    def __len__(self):
        return len(self.df['intensity'])

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.masks[idx], self.intensity[idx]

class NpyComponentDataset(Dataset):
    def __init__(self, df):
        super(NpyComponentDataset, self).__init__()
        self.df = df
        
        # get token id
        self.intensity = df['intensity']
        self.seq_input_ids = df['sequences']
        self.masks = df['masks']
        self.masks_blank = df['masks_blank']
        self.masks_up = df['masks_up']
        self.masks_core = df['masks_core']
        self.masks_down = df['masks_down']
        self.masks_gene = df['masks_gene']
        
    def __len__(self):
        return len(self.df['intensity'])

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.masks[idx], self.masks_blank[idx], self.masks_up[idx], self.masks_core[idx], self.masks_down[idx], self.masks_gene[idx], self.intensity[idx]
