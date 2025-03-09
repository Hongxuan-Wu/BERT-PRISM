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

# fromlist30
# {'[0.00, 10.00)': 54208,
# '[10.00, 100.00)': 52026,
# '[100.00, 500.00)': 26059,
# '[500.00, )': 8244}

# manual66
# {'[0.00, 10.00)': 138125,
# '[10.00, 100.00)': 112255,
# '[100.00, 500.00)': 52128,
# '[500.00, )': 20115}

# streptomyces
# {'[0.00, 10.00)': 83068,
#  '[10.00, 100.00)': 59346,
#  '[100.00, 500.00)': 20701,
#  '[500.00, )': 6725}

# vibrio
# {'[0.00, 10.00)': 32118,
#  '[10.00, 100.00)': 31736,
#  '[100.00, 500.00)': 20322,
#  '[500.00, )': 5356}

# genes_prokaryotes
# {'[0.00, 10.00)': 138125,
# '[10.00, 100.00)': 112255,
# '[100.00, 500.00)': 52128,
# '[500.00, )': 20115}

def load_dataframe(args, config):
    # read dataframe
    if config['kfold'] and (config['mode'] != 'test'):
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
            
            # if config['dataset'] == 'genes_escherichia':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_escherichia_newtoken_catATG.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_fromlist_newtoken_catATG.npz')) 
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_manual_newtoken_catATG.npz'))
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_escherichia_newtoken.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_fromlist_newtoken.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_manual_newtoken.npz'))
            # elif config['dataset'] == 'genes_streptomyces':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_streptomyces_newtoken_catATG.npz'))
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_streptomyces_newtoken.npz'))
            # elif config['dataset'] == 'genes_vibrio':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_Vibrio_newtoken_catATG.npz'))
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_Vibrio_newtoken.npz'))
            # elif config['dataset'] == 'genes_prokaryotes':
            #     # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_prokaryotes.npz'))
            #     df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_prokaryotes_filtered.npz'))
            
            sequences = np.array(df_raw['sequences'])
            masks = np.array(df_raw['masks'])
            intensity = np.array(df_raw['intensity'])
            
            df_tmp = pd.DataFrame(intensity, columns=['intensity'])
            df_tmp['intensity_cls'] = -1
            if config['num_class'] == 4:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[(df_tmp['intensity'] >= 100) & (df_tmp['intensity'] < 500), 'intensity_cls'] = 2
                df_tmp.loc[df_tmp['intensity'] >= 500, 'intensity_cls'] = 3
            elif config['num_class'] == 3:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 2
            elif config['num_class'] == 2:
                if not config['strong_weak']:
                    # activity
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 10, 'intensity_cls'] = 1
                else:
                    # strong & weak
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = -1
                    df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 1
                    df_tmp = df_tmp[df_tmp.intensity_cls != -1]
            
            new_df = df_tmp
            
            intensity_cls = np.array(new_df['intensity_cls'])
            index = torch.tensor(new_df.index, dtype=torch.int64)
            if config['num_class'] == 2:
                # df = {
                #     'sequences': torch.tensor(sequences, dtype=torch.int64),
                #     'labels': torch.tensor(intensity_cls, dtype=torch.float32),
                #     'masks': torch.tensor(masks, dtype=torch.int64)
                # }
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64).index_select(0, index),
                    'labels': torch.tensor(intensity_cls, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64).index_select(0, index)
                }
            else:
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'labels': torch.tensor(intensity_cls, dtype=torch.int64),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }

        # KFold
        if 'genes_' in config['dataset']:
            df['fold'] =  torch.tensor([-1 for _ in range(len(df['labels']))], dtype=torch.int64)
            
            kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'], y=df['labels'])):
                df['fold'][val_idx] = fold_idx
        else:
            df["fold"] = -1
            
            # kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            # for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
            #     df.loc[val_idx, "fold"] = fold_idx
            
            kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df, y=df['labels'])):
                df.loc[val_idx, "fold"] = fold_idx

        # pdb.set_trace()
        return df
    elif config['mode'] == 'test':
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
            
            # if config['train_dataset'] == 'genes_escherichia':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_escherichia_newtoken_catATG.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_fromlist_newtoken_catATG.npz')) 
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_manual_newtoken_catATG.npz'))
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_escherichia_newtoken.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_fromlist_newtoken.npz'))
            #         # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/expression_ecos_manual_newtoken.npz'))
            # elif config['train_dataset'] == 'genes_streptomyces':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_streptomyces_newtoken_catATG.npz'))                
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_streptomyces_newtoken.npz'))
            # elif config['train_dataset'] == 'genes_vibrio':
            #     if config['catATG']:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_Vibrio_newtoken_catATG.npz'))                
            #     else:
            #         df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_Vibrio_newtoken.npz'))
            # elif config['train_dataset'] == 'genes_prokaryotes':
            #     # df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_prokaryotes.npz'))
            #     df_raw = np.load(os.path.join(config['root_data'], 'data_gene_tpm/tpm_gene_prokaryotes_filtered.npz'))
            
            sequences = np.array(df_raw['sequences'])
            masks = np.array(df_raw['masks'])
            intensity = np.array(df_raw['intensity'])
            
            df_tmp = pd.DataFrame(intensity, columns=['intensity'])
            df_tmp['intensity_cls'] = -1
            if config['num_class'] == 4:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[(df_tmp['intensity'] >= 100) & (df_tmp['intensity'] < 500), 'intensity_cls'] = 2
                df_tmp.loc[df_tmp['intensity'] >= 500, 'intensity_cls'] = 3
            elif config['num_class'] == 3:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 2
            elif config['num_class'] == 2:
                if not config['strong_weak']:
                    # activity
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 10, 'intensity_cls'] = 1
                else:
                    # strong & weak
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = -1
                    df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 1
                    df_tmp = df_tmp[df_tmp.intensity_cls != -1]
            
            new_df = df_tmp

            intensity_cls = np.array(new_df['intensity_cls'])
            index = torch.tensor(new_df.index, dtype=torch.int64)
            if config['num_class'] == 2:
                # train_dataset = {
                #     'sequences': torch.tensor(sequences, dtype=torch.int64),
                #     'labels': torch.tensor(intensity_cls, dtype=torch.float32),
                #     'masks': torch.tensor(masks, dtype=torch.int64)
                # }
                train_dataset = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64).index_select(0, index),
                    'labels': torch.tensor(intensity_cls, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64).index_select(0, index)
                }
            else:
                train_dataset = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'labels': torch.tensor(intensity_cls, dtype=torch.int64),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }            
        
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
            
            df_tmp['intensity_cls'] = -1
            if config['num_class'] == 4:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[(df_tmp['intensity'] >= 100) & (df_tmp['intensity'] < 500), 'intensity_cls'] = 2
                df_tmp.loc[df_tmp['intensity'] >= 500, 'intensity_cls'] = 3
            elif config['num_class'] == 3:
                df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 1
                df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 2
            elif config['num_class'] == 2:
                if not config['strong_weak']:
                    # activity
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 10, 'intensity_cls'] = 1
                else:
                    # strong & weak
                    df_tmp.loc[df_tmp['intensity'] < 10, 'intensity_cls'] = -1
                    df_tmp.loc[(df_tmp['intensity'] >= 10) & (df_tmp['intensity'] < 100), 'intensity_cls'] = 0
                    df_tmp.loc[df_tmp['intensity'] >= 100, 'intensity_cls'] = 1
                    df_tmp = df_tmp[df_tmp.intensity_cls != -1]
            val_dataset = df_tmp

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
        self.labels = torch.tensor(df['intensity_cls'].tolist(), dtype=torch.float32).float()
        
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
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.labels[idx]

class NpyDataset(Dataset):
    def __init__(self, df):
        super(NpyDataset, self).__init__()
        self.df = df
        
        # get token id
        self.labels = df['labels']
        self.seq_input_ids = df['sequences']
        self.masks = df['masks']
        
    def __len__(self):
        return len(self.df['labels'])

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.masks[idx], self.labels[idx]
