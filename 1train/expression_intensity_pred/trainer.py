import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import scipy.stats
import sklearn.metrics
import torch
import torch.nn as nn
import transformers
import time
import sklearn
import scipy
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR, LambdaLR
from transformers import AutoTokenizer

from model import ExpressionIntesityPrediction
from dataloader import load_dataframe, make_loader, make_loader_predict
from utils import AverageMeter, to_gpu, kl_divergence_score


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'test':
            if config['kfold']:
                self.df_all = load_dataframe(args, config)
            else:
                self.df_train, self.df_val = load_dataframe(args, config)
        elif config['mode'] == 'predict':
            self.df_predict = load_dataframe(args, config)
        else:
            exit()
            
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['pretrained_path'],
            use_fast=True,
            trust_remote_code=True,
        )
        
        # set module
        self.model = ExpressionIntesityPrediction(config).to(args.device)
        self.model = to_gpu(args, self.logger, self.model)
        
        # froze backbone
        for name, param in self.model.named_parameters():
            if "backbone.embeddings" in name:
                param.requires_grad = False
            if 'backbone.encoder.layer' in name:
                if 'module' in name:  # parallel
                    num_layer = int(name.split('.')[4])
                else:
                    num_layer = int(name.split('.')[3])
                if (num_layer+1) <= config['n_froze_layers']:
                    param.requires_grad = False
        
        # set loss
        self.criterion = nn.MSELoss(reduction='mean')
        
        self.tsne = TSNE(n_components=2, random_state=args.seed)
        
        self.args = args
        self.config = config

    def test(self, fold=0):
        if self.config['kfold']:
            # split data & dataloader
            if 'genes_' in self.config['dataset']:
                fold_train = torch.where(self.df_all['fold']!=fold)[0]
                fold_valid = torch.where(self.df_all['fold']==fold)[0]
                
                df_train = {
                    'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_train),
                    'intensity': torch.index_select(self.df_all['intensity'], dim=0, index=fold_train),
                    'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_train)
                }
                df_valid = {
                    'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_valid),
                    'intensity': torch.index_select(self.df_all['intensity'], dim=0, index=fold_valid),
                    'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_valid)
                }
            else:
                df_train = self.df_all[self.df_all.fold != fold].reset_index(drop=True)
                df_valid = self.df_all[self.df_all.fold == fold].reset_index(drop=True)
            data_loader = make_loader(self.args, self.config, df_train, df_valid, self.tokenizer)
            check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last"+str(fold)+".pth"), map_location=self.args.device)
        else:
            data_loader = make_loader(self.args, self.config, self.df_train, self.df_val, self.tokenizer)
            check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last.pth"), map_location=self.args.device)
            # check_point = torch.load(os.path.join(self.config['checkpoint_dir'], "last0.pth"), map_location=self.args.device)
        
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")
        
        losses = AverageMeter()
        total_len = len(data_loader)
        predictions = []
        Y = []
        features = []
        
        for step, (X, mask, y) in enumerate(data_loader['val']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)
            y = y.to(self.args.device)
            batch_size = y.size(0)

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    preds, feature = self.model(X, mask)
                    loss = self.criterion(preds, y.long())
            losses.update(loss.item(), batch_size)
            predictions.append(preds)
            Y.append(y)
            features.append(feature)

            if self.args.print_step:
                if step % self.args.display_step == 0 or step == (total_len - 1):
                    self.logger.info(f"Eval[{step}/{total_len}]  "
                        f"Loss: {losses.val:.5f} ({losses.avg:.5f})  "
                        # f"bce: {weighted_bce_loss:.5f}  "
                        # f"Elapsed: {time_since(start, float(step + 1) / total_len)} "
                        )
        predictions = torch.cat(predictions)
        Y = torch.cat(Y)
        features = torch.cat(features)

        y_true = Y.cpu().numpy()
        y_pred = predictions.detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        
        metrics = self.compute_metrics(y_true, y_pred)
        loss_avg = losses.avg
        
        # pdb.set_trace()
        
        if self.config['save_y']:
            # tpm_true = np.expm1(y_true)
            # tpm_pred = np.expm1(y_pred)
            # df_y = pd.DataFrame([tpm_true, tpm_pred]).T
            
            df_y = pd.DataFrame([y_true, y_pred]).T
            
            if self.config['kfold']:
                df_y.to_csv(osp.join(self.args.log_dir, 'y'+str(fold)+'.csv'), sep=',', index=False, header=['y_true', 'y_pred'])
            else:
                df_y.to_csv(osp.join(self.args.log_dir, 'y.csv'), sep=',', index=False, header=['y_trues', 'y_pred'])
        
        if self.config['save_scatter']:
            features_tsne = self.tsne.fit_transform(features)
            df = pd.DataFrame(features_tsne)
            df['label'] = y_true

            if self.config['kfold']:
                df.to_csv(osp.join(self.args.log_dir, 'scatter'+str(fold)+'.csv'), sep=',', index=False, header=True)
            else:
                df.to_csv(osp.join(self.args.log_dir, 'scatter.csv'), sep=',', index=False, header=True)
            
                colors = ['#179b73' if y==1 else '#d48aaf' for y in y_true]
                plt.scatter(features_tsne[:,0], features_tsne[:,1], s=5, c=colors, alpha=1)
                plt.savefig(osp.join(self.args.log_dir, 'scatter.png'))

        if self.config['save_metrics']:
            df_metrics = pd.DataFrame([metrics]).T
            df_metrics.columns = ['metrics']
            df_metrics.to_csv(osp.join(self.args.log_dir, 'metrics.csv'), sep=',', index=True, header=True)
            
        return [loss_avg, metrics]

        # data_loader = make_loader(self.args, self.config, self.train_dataset, self.val_dataset, self.tokenizer)
        # start_time = time.time()
        # valid_loss_avg, val_metrics = self.valid_step(data_loader['val'], epoch=0)

        # elapsed = time.time() - start_time
        # self.logger.info("Time {}s || Val - Loss {}  MSE {}  R2 {}  RMSE {}  MAE {}  Pearsonr_corr {}  Pearsonr_p {}  Spearmanr_corr {}  Spearmanr_p {}"
        #                 .format(
        #                     round(elapsed,3),
        #                     valid_loss_avg, 
        #                     val_metrics['mse'], 
        #                     val_metrics['r2'], 
        #                     val_metrics['rmse'], 
        #                     val_metrics['mae'], 
        #                     val_metrics['pearsonr_corr_coefficient'], 
        #                     val_metrics['pearsonr_p'],
        #                     val_metrics['spearmanr_corr_coefficient'], 
        #                     val_metrics['spearmanr_p'],
        #                 )
        # )

    def predict(self):
        data_loader = make_loader_predict(self.args, self.config, self.df_predict, self.tokenizer)
        
        check_point = torch.load(self.config['checkpoint_dir'], map_location=self.args.device)
        self.model.load_state_dict(check_point)
        self.logger.info("Loaded model.")

        predictions = []
        
        for step, (X, mask) in enumerate(data_loader['predict']):
            X = X.to(self.args.device)
            mask = mask.to(self.args.device)

            with torch.no_grad():
                # with autocast(device_type='cuda', dtype=torch.float16):
                #     preds = self.model(X, mask)
                preds, feature = self.model(X, mask)
            predictions.append(preds)
        predictions = torch.cat(predictions)
        y_pred = predictions.detach().cpu().numpy()

        df = pd.DataFrame(y_pred)
        df.to_csv(osp.join(self.args.log_dir, 'expression_intensity_prediction.csv'), sep=',', index=False, header=False)
    
        return
    
    def compute_metrics(self, y_true, y_pred):
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)
        mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
        pearsonr_corr_coefficient, pearsonr_p = scipy.stats.pearsonr(y_true, y_pred)
        spearmanr_corr_coefficient, spearmanr_p = scipy.stats.spearmanr(y_true, y_pred)
        kl_divergence = kl_divergence_score(y_true, y_pred)
        
        return {
            'r2': r2, 
            'pearsonr_corr_coefficient': pearsonr_corr_coefficient,
            'pearsonr_p': pearsonr_p,
            'spearmanr_corr_coefficient': spearmanr_corr_coefficient,
            'spearmanr_p': spearmanr_p,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'kl_divergence': kl_divergence,
        }
