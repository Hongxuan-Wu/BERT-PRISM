import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
import random
import pdb

from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, KFold
from torch.utils.data import DataLoader
from collections import Counter

def get_word_freq(tokens, tokenizer):
    counter = Counter(np.array(tokens).reshape([-1]))
    word_freq = torch.zeros((tokenizer.vocab_size,), dtype=torch.int64)
    for i in range(tokenizer.vocab_size):
        word_freq[i] += counter[i]
    word_freq = word_freq_preprocess_fn(word_freq)
    word_freq[tokenizer.pad_token_id] = 0.  # stable training
    word_freq[tokenizer.sep_token_id] = 0.
    word_freq[tokenizer.cls_token_id] = 0.
    word_freq_matrix = torch.stack([process_fn_in_getitem(word_freq.gather(0, d)) for d in tokens], dim=0)
    return word_freq, word_freq_matrix

def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
    return wf

def process_fn_in_getitem(wf):
    return wf - wf.mean()

def load_dataframe(args, config):
    # read dataframe
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
            
            # pdb.set_trace()
            if config['select_active']:
                index = np.where(intensity>=10)[0]
                df = {
                    'sequences': torch.tensor(sequences[index], dtype=torch.int64),
                    'masks': torch.tensor(masks[index], dtype=torch.int64)
                }
            else:
                df = {
                    'sequences': torch.tensor(sequences, dtype=torch.int64),
                    # 'labels': torch.tensor(labels, dtype=torch.float32),
                    'masks': torch.tensor(masks, dtype=torch.int64)
                }
        elif config['dataset'] == 'escherichia_corynebacterium' or config['dataset'] == 'escherichia_vibrio':
            datapath = osp.join(config['root_project'], 'public_promoter', config['dataset']+'.csv')
            df_raw = pd.read_csv(datapath)

            if config['catATG']:
                df_raw['genes'] = 'atggtgagcaagggcgaggagctgttcaccggggtggtgcccatcctggtcgagctggacggcgacgtaaacggccacaagttcagcgtgtccggcgagggcgagggcgatgccacctacggcaagctgaccctgaagttcatctgcaccaccggcaagctgcccgtgccctggcccaccctcgtgaccaccctgacctacggcgtgcagtgcttcagccgctaccccgaccacatgaagcagcacgacttcttcaagtccgccatgcccgaaggctacgtccaggagcgcaccatcttcttcaaggacgacggcaactacaagacccgcgccgaggtgaagttcgagggcgacaccctggtgaaccgcatcgagctgaagggcatcgacttcaaggaggacggcaacatcctggggcacaagctggagtacaactacaacagccacaacgtctatatcatggccgacaagcagaagaacggcatcaaggtgaacttcaagatccgccacaacatcgaggacggcagcgtgcagctcgccgaccactaccagcagaacacccccatcggcgacggccccgtgctgctgcccgacaaccactacctgagcacccagtccgccctgagcaaagaccccaacgagaagcgcgatcacatggtcctgctggagttcgtgaccgccgccgggatcactctcggcatggacgagctgtacaagtaa'
                df_raw['genes'] = df_raw['genes'].str.upper()
            
            df = pd.DataFrame(df_raw['promoters'] + df_raw['genes'])
            df.columns = ['sequences']
        
        # pdb.set_trace()
        
        # KFold
        if 'genes_' in config['dataset']:
            df['fold'] =  torch.tensor([-1 for _ in range(len(df['sequences']))], dtype=torch.int64)
            
            kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df['sequences'])):
                df['fold'][val_idx] = fold_idx
        else:
            df["fold"] = -1
            
            kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=args.seed)
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df)):
                df.loc[val_idx, "fold"] = fold_idx
        
        # pdb.set_trace()
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
            
            train_dataset = {
                'sequences': torch.tensor(sequences, dtype=torch.int64),
                # 'labels': torch.tensor(labels, dtype=torch.float32),
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
            val_dataset = df_tmp
        
        pdb.set_trace()
        return train_dataset, val_dataset

def make_loader(args, config, df_train, df_valid, tokenizer):
    if config['kfold']:
        if 'genes_' in config['dataset']:
            tokens = torch.concat([df_train['sequences'], df_valid['sequences']], dim=0)[:, 1:-1]
            masks = torch.concat([df_train['masks'], df_valid['masks']], dim=0)[:, 1:-1]
            word_freq, word_freq_matrix = get_word_freq(tokens, tokenizer)  # must combine train&valid data to compute word_freq
            
            train_dataset = NpyDataset(
                df=df_train, 
                tokenizer=tokenizer,
                tokens=tokens[:len(df_train['sequences'])],
                masks=masks[:len(df_train['sequences'])],
                word_freq=word_freq,
                word_freq_matrix=word_freq_matrix[:len(df_train['sequences'])],
            )
            valid_dataset = NpyDataset(
                df=df_valid, 
                tokenizer=tokenizer,
                tokens=tokens[-len(df_valid['sequences']):],
                masks=masks[-len(df_valid['sequences']):],
                word_freq=word_freq,
                word_freq_matrix=word_freq_matrix[-len(df_valid['sequences']):],
            )
        else:
            df_combined = pd.concat([df_train, df_valid], ignore_index=True)
            tokenizer_output = tokenizer(
                text=df_combined.to_numpy()[:, 0].tolist(), 
                return_tensors="pt", 
                max_length=config['token_length'], 
                padding=True, 
                truncation=True
            )  # input_ids, token_type_ids, attention_mask
            
            # tokens = tokenizer_output['input_ids']
            # masks = tokenizer_output['attention_mask']
            tokens = tokenizer_output['input_ids'][:, 1:-1]
            masks = tokenizer_output['attention_mask'][:, 1:-1]
            word_freq, word_freq_matrix = get_word_freq(tokens, tokenizer)  # must combine train&valid data to compute word_freq
            
            train_dataset = PromoterDataset(
                df=df_train, 
                tokenizer=tokenizer,
                tokens=tokens[:len(df_train)],
                masks=masks[:len(df_train)],
                word_freq=word_freq,
                word_freq_matrix=word_freq_matrix[:len(df_train)],
            )
            valid_dataset = PromoterDataset(
                df=df_valid, 
                tokenizer=tokenizer,
                tokens=tokens[-len(df_valid):],
                masks=masks[-len(df_valid):],
                word_freq=word_freq,
                word_freq_matrix=word_freq_matrix[-len(df_valid):],
            )
    else:
        pdb.set_trace()
        if config['train_dataset'] == 'genes_ecos':
            train_dataset = NpyDataset(df=df_train, tokenizer=tokenizer)
        else:
            train_dataset = PromoterDataset(df=df_train, tokenizer=tokenizer)
        valid_dataset = PromoterDataset(df=df_valid, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['train_batch_size'], 
                              shuffle=True, 
                              drop_last=False,
                              num_workers=args.num_workers, 
                              pin_memory=True
                              )
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=config['valid_batch_size'], 
                              shuffle=False, 
                              drop_last=False,
                              num_workers=args.num_workers, 
                              pin_memory=True
                              )
    data_loader = {'train': train_loader,
                   'val': valid_loader}
    dataset = {'train': train_dataset,
               'val': valid_dataset}
    return data_loader, dataset

class PromoterDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, masks, word_freq, word_freq_matrix):
        super(PromoterDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.seq_input_ids = tokens
        self.seq_attention_mask = masks
        self.word_freq = word_freq
        self.word_freq_matrix = word_freq_matrix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.word_freq_matrix[idx]

class NpyDataset(Dataset):
    def __init__(self, df, tokenizer, tokens, masks, word_freq, word_freq_matrix):
        super(NpyDataset, self).__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.seq_input_ids = tokens
        self.seq_attention_mask = masks
        self.word_freq = word_freq
        self.word_freq_matrix = word_freq_matrix

    def __len__(self):
        return len(self.seq_input_ids)

    def __getitem__(self, idx):
        return self.seq_input_ids[idx], self.seq_attention_mask[idx], self.word_freq_matrix[idx]
