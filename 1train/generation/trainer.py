import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import transformers
import time
import math
import sklearn
import numpy as np
import pandas as pd
import pdb

import nltk
from nltk.translate.bleu_score import SmoothingFunction as SF
# from rouge import Rouge

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR, LambdaLR
from transformers import AutoTokenizer

from model import PromoterGenerationDiffusion
from dataloader import load_dataframe, make_loader
from utils import AverageMeter, to_gpu


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'train_valid':
            if config['kfold']:
                self.df_all = load_dataframe(args, config)
            else:
                self.df_train, self.df_val = load_dataframe(args, config)
        elif config['mode'] == 'generate':
            self.df_all = load_dataframe(args, config)
        else:
            exit()
            
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = PromoterGenerationDiffusion(config, self.tokenizer, args.device).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        if config['dtype'] == 'fp16':
            self.dtype = torch.float16
        elif config['dtype'] == 'bf16':
            self.dtype = torch.bfloat16
        elif config['dtype'] == 'fp32':
            self.dtype = torch.float32
        
        self.tsne = TSNE(n_components=2, random_state=args.seed)
        
        self.args = args
        self.config = config

    def generate(self, fold=0):
        if self.config['kfold']:
            # split data & dataloader
            if 'genes_' in self.config['dataset']:
                fold_train = torch.where(self.df_all['fold']!=fold)[0] 
                fold_valid = torch.where(self.df_all['fold']==fold)[0]
                
                df_train = {
                    'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_train),
                    'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_train)
                }
                df_valid = {
                    'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_valid),
                    'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_valid)
                }
            else:
                df_train = self.df_all[self.df_all.fold != fold].reset_index(drop=True)
                df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True)
            data_loader, dataset = make_loader(self.args, self.config, df_train, df_valid, self.tokenizer)
        else:
            data_loader, dataset = make_loader(self.args, self.config, self.df_train, self.df_val, self.tokenizer)

        check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last0.pth"), map_location=self.model.device)
        self.model.load_state_dict(check_point)
        
        # generate & save
        df_tokens = pd.DataFrame()
        df_seqs = pd.DataFrame()
        for i in range(self.config['predict_num']//self.config['predict_batch']):
            start_time = time.time()
            res_dict = self.model.generate(dataset['val'].word_freq.to(self.args.device))  # word_freq is same in train_dataset & valid_dataset
            elapsed = time.time() - start_time
            
            sentences = self.tokenizer.batch_decode(res_dict['final_state'])
            seqs = [sentence.replace(' ', '') for sentence in sentences]
            self_bleu = self.self_bleu(sentences)
            diversity = self.calculate_diversity(res_dict['final_state'])
            
            self.logger.info("Generation {}  Time {}s || Self-Bleu {}    Diversity {}".format(
                i,
                round(elapsed, 3), 
                round(self_bleu, 4),
                round(diversity, 4)
            ))
            
            df_tokens_tmp = pd.DataFrame(res_dict['final_state'].cpu())
            df_tokens = pd.concat([df_tokens, df_tokens_tmp])
            df_seqs_tmp = pd.DataFrame(seqs)
            df_seqs = pd.concat([df_seqs, df_seqs_tmp])
            
            # if i == 2:
            #     pdb.set_trace()
        
        df_tokens.to_csv(osp.join(self.args.log_dir, 'gen_tokens.csv'), sep=',', index=False, header=False)
        df_seqs.to_csv(osp.join(self.args.log_dir, 'gen_seqs.csv'), sep=',', index=False, header=False)
        
        return 
    
    def calculate_diversity(self, recovers):
        num_chars = recovers.shape[1]
        diversity = 0.0
        for recover in recovers:
            num_unique_chars = torch.unique(recover).shape[0]
            diversity += num_unique_chars / num_chars
        diversity /= recovers.shape[0]
        return diversity
    
    def bleu(self, referencesList, recoversList):
        referencesList_split = [[reference.split()] for reference in referencesList]
        recoversList_split = [recover.split() for recover in recoversList]
        
        bleu = nltk.translate.bleu_score.corpus_bleu(
            referencesList_split, 
            recoversList_split, 
            smoothing_function=SF().method4
        )
        return bleu

    def self_bleu(self, recoversList):
        """
        This function is a canonical implementation of self-BLEU.
        The deviation from the above one is that the references are ALL THE REST sentences and this one uses CORPUS bleu.
        """
        recoversList = [recover.split() for recover in recoversList]
        
        res = 0.
        for i in range(len(recoversList)):
            res += nltk.translate.bleu_score.corpus_bleu(
                [recoversList[:i] + recoversList[i + 1:]], 
                [recoversList[i]], 
                smoothing_function=SF().method4
            )
        return res / len(recoversList)
