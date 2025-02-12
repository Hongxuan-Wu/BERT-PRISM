import os
import pandas as pd
import numpy as np
import torch
import random
import pdb

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

def load_dataframe(args, config):
    # read dataframe
    if config['mode'] == 'test':
        if config['kfold']:
            if config['dataset'] == 'ESM':
                df = pd.read_csv(os.path.join(config['root_project'], '41592_2018_BFnmeth4633_MOESM3_ESM', 'Nathan_multi_mixed_catATG.csv'), delimiter=',')
            elif config['dataset'] == 'mg1655':
                df = pd.read_csv(os.path.join(config['root_project'], '1-s2.0-S088875431830613X-mmc1', 'iPSW2L_PseKNC_mg1655_catATG.csv'), delimiter=',')
            elif config['dataset'] == 'MG1655_RegulonDB':
                df = pd.read_csv(os.path.join(config['root_project'], 'MG1655-RegulonDB', 'MG1655-RegulonDB_promoters_catATG.csv'), delimiter=',')
            elif config['dataset'] == 'Bacillus':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Bacillus subtilis subsp. subtilis str. 168 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Burkholderia':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Burkholderia cenocepacia J2315 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Campylobacter':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Campylobacter jejuni 81-176 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Corynebacterium':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Corynebacterium glutamicum ATCC 13032 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Escherichia':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Escherichia coli str K-12 substr. MG1655 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Haloferax':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Haloferax volcanii DS2 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Klebsiella':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Klebsiella aerogenes KCTC 2190 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Onion':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Onion yellows phytoplasma OY-M promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Shigella':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Shigella flexneri 5a str. M90T promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Sinorhizobium':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Sinorhizobium meliloti 1021 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Staphylococcus':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Staphylococcus epidermidis ATCC 12228 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Synechocystis':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Synechocystis sp. PCC 6803 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Thermococcus':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Thermococcus kodakarensis KOD1 promoters.csv'), delimiter=',')
            elif config['dataset'] == 'Xanthomonas':
                df = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Xanthomonas campestris pv. campestrie B100 promoters.csv'), delimiter=',')
            elif 'genes_' in config['dataset']:
                prokaryotes_type = config['dataset'][6:]
                
                if 'newtoken' in config['pretrained_path']:
                    df = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed_newtoken.npz'))
                elif 'mergetoken' in config['pretrained_path']:
                    df = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed_mergetoken.npz'))
                else:
                    df = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed.npz'))
                # pdb.set_trace()
                
                length = len(df['sequences'])
                if config['datasize'] < length:
                    # pdb.set_trace()
                    real_ids = random.sample(range(0,length//2), config['datasize']//2)  # ESM - 58498
                    # real_ids = [random.randint(0, length//2-1) for _ in range(58498//2)]
                    fake_ids = [x + length//2 for x in real_ids]
                    ids = real_ids + fake_ids
                    
                    sequences = np.array(df['sequences'])[ids]
                    labels = np.array(df['labels'])[ids]
                    masks = np.array(df['masks'])[ids]
                else:
                    sequences = np.array(df['sequences'])
                    labels = np.array(df['labels'])
                    masks = np.array(df['masks'])
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'labels': torch.tensor(labels, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }
                # pdb.set_trace()
            elif config['dataset'] == 'prokaryotes596':
                df = np.load('/hy-tmp/expressions/fromlist/prokaryotes/tpm_gene_newtoken.npz')
                
                length = len(df['sequences'])
                if config['datasize'] < length:
                    pdb.set_trace()
                    real_ids = random.sample(range(0,length//2), config['datasize']//2)  # ESM - 58498
                    # real_ids = [random.randint(0, length//2-1) for _ in range(58498//2)]
                    fake_ids = [x + length//2 for x in real_ids]
                    ids = real_ids + fake_ids
                    
                    sequences = np.array(df['sequences'])[ids]
                    labels = np.array(df['intensity'])[ids]
                    masks = np.array(df['masks'])[ids]
                else:
                    sequences = np.array(df['sequences'])
                    labels = np.array(df['intensity'])
                    masks = np.array(df['masks'])
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    'labels': torch.tensor(labels, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }
            
            # KFold
            if 'genes_' in config['dataset'] or config['dataset'] == 'prokaryotes596':
                df['fold'] =  torch.tensor([-1 for _ in range(len(labels))], dtype=torch.int64)
                
                kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'], y=df['labels'])):
                    df['fold'][val_idx] = fold_idx
            else:
                df["fold"] = -1
                kfold = StratifiedKFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df, y=df['label'])):
                    df.loc[val_idx, "fold"] = fold_idx
            
            return df
        else:
            if config['train_dataset'] == 'ESM':
                df_train = pd.read_csv(os.path.join(config['root_project'], '41592_2018_BFnmeth4633_MOESM3_ESM', 'Nathan_multi_mixed_catATG.csv'), delimiter=',')
                # df_train = pd.read_csv(os.path.join(config['root_project'], '41592_2018_BFnmeth4633_MOESM3_ESM', 'Nathan_multi_mixed.csv'), delimiter=',')
            elif config['train_dataset'] == 'Bacillus':
                df_train = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Bacillus subtilis subsp. subtilis str. 168 promoters.csv'), delimiter=',')
            elif config['train_dataset'] == 'Corynebacterium':
                df_train = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Corynebacterium glutamicum ATCC 13032 promoters.csv'), delimiter=',')
            elif config['train_dataset'] == 'Escherichia':
                df_train = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Escherichia coli str K-12 substr. MG1655 promoters.csv'), delimiter=',')
            elif config['train_dataset'] == 'Synechocystis':
                df_train = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Synechocystis sp. PCC 6803 promoters.csv'), delimiter=',')
            elif 'genes_' in config['train_dataset']:
                prokaryotes_type = config['train_dataset'][6:]
                
                if 'newtoken' in config['pretrained_path']:
                    df_train = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed_newtoken.npz'))
                elif 'mergetoken' in config['pretrained_path']:
                    df_train = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed_mergetoken.npz'))
                else:
                    df_train = np.load(os.path.join(config['root_genes_tokens'], prokaryotes_type + '/genes_' + prokaryotes_type + '_compressed.npz'))
                # pdb.set_trace()
                
                length = len(df_train['sequences'])
                if config['train_datasize'] < length:
                    real_ids = random.sample(range(0,length//2), config['train_datasize']//2)  # ESM - 58498
                    # real_ids = [random.randint(0, length//2-1) for _ in range(58498//2)]
                    fake_ids = [x + length//2 for x in real_ids]
                    ids = real_ids + fake_ids
                    
                    sequences = np.array(df_train['sequences'])[ids]
                    labels = np.array(df_train['labels'])[ids]
                    masks = np.array(df_train['masks'])[ids]
                else:
                    sequences = np.array(df_train['sequences'])
                    labels = np.array(df_train['labels'])
                    masks = np.array(df_train['masks'])
                    
                df_train = {
                    'sequences': sequences,
                    'labels': labels,
                    'masks': masks
                }
                # pdb.set_trace()
                
            if config['val_dataset'] == 'mg1655':
                df_val = pd.read_csv(os.path.join(config['root_project'], '1-s2.0-S088875431830613X-mmc1', 'iPSW2L_PseKNC_mg1655_catATG.csv'), delimiter=',')
                # val_dataset = pd.read_csv(os.path.join(config['root_project'], '1-s2.0-S088875431830613X-mmc1', 'iPSW2L_PseKNC_mg1655.csv'), delimiter=',')
            elif config['val_dataset'] == 'MG1655_RegulonDB':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'MG1655-RegulonDB', 'MG1655-RegulonDB_promoters_catATG.csv'), delimiter=',')
            elif config['val_dataset'] == 'Bacillus':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Bacillus subtilis subsp. subtilis str. 168 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Burkholderia':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Burkholderia cenocepacia J2315 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Campylobacter':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Campylobacter jejuni 81-176 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Corynebacterium':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Corynebacterium glutamicum ATCC 13032 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Escherichia':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Escherichia coli str K-12 substr. MG1655 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Haloferax':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Haloferax volcanii DS2 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Klebsiella':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Klebsiella aerogenes KCTC 2190 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Onion':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Onion yellows phytoplasma OY-M promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Shigella':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Shigella flexneri 5a str. M90T promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Sinorhizobium':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Sinorhizobium meliloti 1021 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Staphylococcus':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Staphylococcus epidermidis ATCC 12228 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Synechocystis':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Synechocystis sp. PCC 6803 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Thermococcus':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Thermococcus kodakarensis KOD1 promoters.csv'), delimiter=',')
            elif config['val_dataset'] == 'Xanthomonas':
                df_val = pd.read_csv(os.path.join(config['root_project'], 'csvs/catATG', 'Xanthomonas campestris pv. campestrie B100 promoters.csv'), delimiter=',')
            
            return df_train, df_val
    elif config['mode'] == 'predict':
        df_predict = pd.read_csv(config['predict_filepath'], delimiter=',', header=None)
        return df_predict

def make_loader(args, config, df_train, df_valid, tokenizer):
    if config['kfold']:
        if 'genes_' in config['dataset'] or config['dataset'] == 'prokaryotes596':
            train_dataset = NpyDataset(df=df_train)
            valid_dataset = NpyDataset(df=df_valid)
        else:
            train_dataset = PromoterDataset(df=df_train, tokenizer=tokenizer)
            valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    else:
        if 'genes_' in config['train_dataset'] or config['train_dataset'] == 'prokaryotes596':
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
        
        # get token id
        file = df.to_numpy()
        sequence = file[:, 0].tolist()
        self.labels = file[:, 1]
        
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
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32).float()
        
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], label

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
        label = torch.tensor(self.labels[idx], dtype=torch.float32).float()
        
        return self.seq_input_ids[idx], self.masks[idx], label
