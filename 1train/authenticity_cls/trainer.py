import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
import sklearn.metrics
import torch
import torch.nn as nn
import transformers
import time
import sklearn
import numpy as np
import pandas as pd
import pdb

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import StepLR, LambdaLR
from transformers import AutoTokenizer

from model import PromoterRealFakePredict
from dataloader import load_dataframe, make_loader, make_loader_predict
from utils import AverageMeter, to_gpu


class Trainer(object):
    def __init__(self, args, config):
        self.writer = args.writer
        self.logger = args.logger
        
        # load data
        if config['mode'] == 'train_valid' or config['mode'] == 'test':
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
        self.model = PromoterRealFakePredict(config).to(args.device)
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
                    'labels': torch.index_select(self.df_all['labels'], dim=0, index=fold_train),
                    'masks': torch.index_select(self.df_all['masks'], dim=0, index=fold_train)
                }
                df_valid = {
                    'sequences': torch.index_select(self.df_all['sequences'], dim=0, index=fold_valid),
                    'labels': torch.index_select(self.df_all['labels'], dim=0, index=fold_valid),
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
                    loss = self.criterion(preds, y)
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
        
        if self.config['save_roc_auc']:
            df_roc_auc = pd.DataFrame([metrics['fpr'], metrics['tpr']]).T
            if self.config['kfold']:
                df_roc_auc.to_csv(osp.join(self.args.log_dir, 'real_fake_roc_auc'+str(fold)+'.csv'), sep=',', index=False, header=['FPR', 'TPR'])
            else:
                df_roc_auc.to_csv(osp.join(self.args.log_dir, 'real_fake_roc_auc.csv'), sep=',', index=False, header=['FPR', 'TPR'])
        
        if self.config['save_y']:
            df_y = pd.DataFrame([y_true, y_pred]).T
            if self.config['kfold']:
                df_y.to_csv(osp.join(self.args.log_dir, 'real_fake_y'+str(fold)+'.csv'), sep=',', index=False, header=['y_true', 'y_pred'])
            else:
                df_y.to_csv(osp.join(self.args.log_dir, 'real_fake_y.csv'), sep=',', index=False, header=['y_trues', 'y_pred'])
        
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

        return [loss_avg, metrics]
    
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
                    # preds, feature = self.model(X, mask)
                preds, feature = self.model(X, mask)
            predictions.append(preds)
        predictions = torch.cat(predictions)
        y_pred = predictions.detach().cpu().numpy()
        df = pd.DataFrame(y_pred)
        df.to_csv(osp.join(self.args.log_dir, 'real_fake_predict.csv'), sep=',', index=False, header=False)
    
        return

    def compute_metrics(self, y_true, y_pred):
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
        auc = sklearn.metrics.auc(fpr, tpr)
        # auc = sklearn.metrics.roc_auc_score(y_true, y_pred, average="macro")

        accuracy = sklearn.metrics.accuracy_score(y_true, np.round(y_pred))
        precision = sklearn.metrics.precision_score(y_true, np.round(y_pred), average="macro", zero_division=0)
        recall = sklearn.metrics.recall_score(y_true, np.round(y_pred), average="macro", zero_division=0)
        f1 = sklearn.metrics.f1_score(y_true, np.round(y_pred), average="macro", zero_division=0)
        mcc = sklearn.metrics.matthews_corrcoef(y_true, np.round(y_pred))
        
        mse = sklearn.metrics.mean_squared_error(y_true, y_pred)

        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall,
            'f1': f1, 
            'mcc': mcc, 
            'mse': mse
        }
