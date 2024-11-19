import os
from pathlib import Path
import copy
from datetime import datetime
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
import pickle

from tqdm.auto import tqdm

from monai.data import partition_dataset_classes, partition_dataset

import torch
from torch import nn
from torch_geometric.nn import models as tgmodels
from torch_geometric.data.lightning import LightningDataset
import torchvision
import torchmetrics

import pytorch_lightning as L

from hnc_project.pytorch.lightning_GNN import CNN_GNN 
from hnc_project.pytorch.dataset_class import DatasetGeneratorImage

MODELS = ['SimpleGCN',
          'ResNetGCN',
          'ClinicalGatedGCN',
          'GatedGCN',
          'DeepGCN',
          'myGraphUNet']

#models that use edge_attr

class RunModel(object):
    def __init__(self, config):
        self.config = config
        self.clinical_features = pd.read_pickle(os.path.join(self.config['data_path'], self.config['clinical_data']))
        L.seed_everything(self.config['seed'])

        if self.config['log_dir'] is None:
            self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.log_dir = os.path.join('logs', self.config['log_dir'])
        print(f"logs are located at: {self.log_dir}")
        self.metric_dir = os.path.join(self.log_dir, 'metric_dfs')
        Path(self.metric_dir).mkdir(parents=True, exist_ok=True)
    def set_model(self):
        """
        sets and assigns lightning module to self.model
        """
        self.model = CNN_GNN(self.config)



    def set_data(self, resume=None):

        self.data = DatasetGeneratorImage(config=self.config)


    def set_train_test_split_challenge(self):
        if self.config['challenge']:
            train_pats = self.data.y.loc[self.data.y_challenge[self.data.y_challenge['RADCURE-challenge'] == 'training'].index]
            test_pats = self.data.y.loc[self.data.y_challenge[self.data.y_challenge['RADCURE-challenge'] == 'test'].index]
        else:
            train_pats = self.data.y.loc[self.data.y_nocensor[self.data.y_nocensor['RADCURE-challenge'] != 'test'].index]
            
            test_pats = self.data.y.loc[self.data.y_nocensor[self.data.y_nocensor['RADCURE-challenge'] == 'test'].index]
       
        x = [self.data.y.index.get_loc(idx) for idx in train_pats.index]
        y = self.data.y.iloc[x] 
        self.folds = None
        if self.config['preset_folds']:
            self.folds = pd.read_pickle(self.data.data_path.joinpath(self.config['preset_fold_file']))
        else:
            self.folds = partition_dataset_classes(x, y, num_partitions=5, shuffle=True, seed=self.config['seed'])
            with open(self.data.data_path.joinpath(self.config['preset_fold_file']), 'wb') as f:
                pickle.dump(self.folds, f)
                f.close()

        self.train_folds = [[0,1,2,3],
                            [4,0,1,2],
                            [3,4,0,1],
                            [2,3,4,0],
                            [1,2,3,4]]
        self.val_folds = [4, 3, 2, 1, 0]

        self.train_splits = [self.folds[i]+self.folds[j]+self.folds[k]+self.folds[l] for i,j,k,l in self.train_folds]
        self.train_splits_noaug = [self.folds[i]+self.folds[j]+self.folds[k]+self.folds[l] for i,j,k,l in self.train_folds]

        for fold_idx, fold in enumerate(self.train_splits):
            if self.config['challenge']:
                fold_pats = self.data.y_challenge.iloc[fold].index
            else:
                fold_pats = self.data.y_nocensor.iloc[fold].index
            aug_pats = self.data.y[['rotation' in pat for pat in self.data.y.index]].index
            aug_fold_pats = [pat for pat in aug_pats if np.any([fold_pat in pat for fold_pat in fold_pats])]
            self.train_splits[fold_idx].extend(self.data.y.index.get_indexer(aug_fold_pats))
            
        self.val_splits = [self.folds[i] for i in self.val_folds]
        self.test_splits = [self.data.y.index.get_loc(idx) for idx in test_pats.index]            
        self.class_weights = []
        #for split in self.train_splits:
        #    self.class_weights.append([len(self.data.y.values[split][self.data.y.values[split]==0]) / np.sum(self.data.y.values[split])])
        self.class_weights = [1., 1., 1., 1., 1.]
         

    def set_clinical_scaling(self):

        self.clinical_means = []
        self.clinical_stds = []

        for idx, fold in enumerate(self.train_splits_noaug):
            fold_clinical = self.clinical_features.loc[self.data.y.iloc[fold].index]
            self.clinical_means.append(fold_clinical[['age', 'smoke_time', 'prescribed_dose', 'dose_fx']].mean())
            self.clinical_stds.append(fold_clinical[['age', 'smoke_time', 'prescribed_dose', 'dose_fx']].std())


    def set_data_module(self):
        self.data_module_cross_val = [LightningDataset(train_dataset=self.data[fold], val_dataset=self.data[self.val_splits[idx]], test_dataset=self.data[self.test_splits], batch_size=self.config['batch_size'], num_workers=16, pin_memory=True, persistent_workers=False, shuffle=True, drop_last=False) for idx, fold in enumerate(self.train_splits)] 



    def set_callbacks(self, fold_idx):
        self.callbacks = []

        #Checkpoint options
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_auc',
            mode='max',
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, f"top_models_fold_{fold_idx}"),
            filename='model_auc_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            ))     
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, f"top_models_fold_{fold_idx}"),
            filename='model_loss_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            ))     
        self.callbacks.append(L.callbacks.ModelCheckpoint(
            monitor='val_m',
            mode='max',
            save_top_k=self.config['save_top_k'],
            dirpath=os.path.join(self.log_dir, f"top_models_fold_{fold_idx}"),
            filename='model_m_{epoch:02d}_{val_loss:.2f}_{val_auc:.2f}_{val_m:.2f}',
            ))     
        #self.callbacks.append(L.callbacks.EarlyStopping(monitor='val_loss', patience=20, check_on_train_epoch_end=False))

        #self.callbacks.append(L.callbacks.LearningRateFinder(min_lr=1e-8, max_lr=1e-1))


    def run(self, resume=False, resume_idx=None):                                                                                                                                                                       
        if not resume:
            self.trainers = []
            self.test_metrics = {}
            self.val_metrics = {}
            self.test_k_metrics = {}
            self.val_k_metrics = {}

            self.set_callbacks(-1)
            for callback in self.callbacks:
                if 'ModelCheckpoint' in callback.__class__.__name__:
                    self.test_metrics[callback.monitor] = []
                    self.val_metrics[callback.monitor] = []
                    self.test_k_metrics[callback.monitor] = []
                    self.val_k_metrics[callback.monitor] = []

        self.best_checkpoints = {}
        self.best_k_checkpoints = {}
        for idx in range(5):
            L.seed_everything(self.config['seed'])
            self.set_model()
            self.set_callbacks(idx)

            self.trainers.append(L.Trainer(
                max_epochs=self.config['n_epochs'],
                accelerator="auto",
                devices=self.config['gpu_device'] if torch.cuda.is_available() else None,
                logger=[L.loggers.CSVLogger(save_dir=os.path.join(self.log_dir, f"csvlog_fold_{idx}")), L.loggers.TensorBoardLogger(save_dir=os.path.join(self.log_dir, f"tb_fold_{idx}"))],
                callbacks=self.callbacks,
                #check_val_every_n_epoch = 1,
                #auto_lr_find=True
                ))


            self.trainers[idx].fit(self.model, datamodule=self.data_module_cross_val[idx])

            self.best_k_checkpoints[f"fold_{idx}"] = {callback.monitor: list(callback.best_k_models.keys()) for callback in self.trainers[idx].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
            self.best_checkpoints[f"fold_{idx}"] = {callback.monitor: callback.best_model_path for callback in self.trainers[idx].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}

            for monitor, best_model in self.best_checkpoints[f"fold_{idx}"].items():
                self.val_metrics[monitor].append(self.trainers[idx].validate(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])
                self.test_metrics[monitor].append(self.trainers[idx].test(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])

            for monitor, best_model_list in self.best_k_checkpoints[f"fold_{idx}"].items():
                for best_model in best_model_list:
                    self.val_k_metrics[monitor].append(self.trainers[idx].validate(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])
                    self.test_k_metrics[monitor].append(self.trainers[idx].test(self.model, datamodule=self.data_module_cross_val[idx], ckpt_path=best_model)[0])
                    

    def get_metrics_dataframe(self):

        self.val_metrics_df = None
        for idx, (key, values) in enumerate(self.val_metrics.items()):
            if idx == 0:
                self.val_metrics_df = pd.DataFrame(values)
                self.val_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.val_metrics_df = pd.concat([self.val_metrics_df, tmp_df])
        self.val_metrics_df = self.val_metrics_df.set_index(['monitor', self.val_metrics_df.index])

        self.test_metrics_df = None
        for idx, (key, values) in enumerate(self.test_metrics.items()):
            if idx == 0:
                self.test_metrics_df = pd.DataFrame(values)
                self.test_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.test_metrics_df = pd.concat([self.test_metrics_df, tmp_df])
        self.test_metrics_df = self.test_metrics_df.set_index(['monitor', self.test_metrics_df.index])

        self.val_k_metrics_df = None
        for idx, (key, values) in enumerate(self.val_k_metrics.items()):
            if idx == 0:
                self.val_k_metrics_df = pd.DataFrame(values)
                self.val_k_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.val_k_metrics_df = pd.concat([self.val_k_metrics_df, tmp_df])
        self.val_k_metrics_df = self.val_k_metrics_df.set_index(['monitor', self.val_k_metrics_df.index])
        self.test_k_metrics_df = None
        for idx, (key, values) in enumerate(self.test_k_metrics.items()):
            if idx == 0:
                self.test_k_metrics_df = pd.DataFrame(values)
                self.test_k_metrics_df['monitor'] = key
            else:
                tmp_df = pd.DataFrame(values)
                tmp_df['monitor'] = key
                self.test_k_metrics_df = pd.concat([self.test_k_metrics_df, tmp_df])
        self.test_k_metrics_df = self.test_k_metrics_df.set_index(['monitor', self.test_k_metrics_df.index])

        self.val_metrics_df.to_pickle(os.path.join(self.metric_dir, 'val_best_metrics.pkl'))
        self.test_metrics_df.to_pickle(os.path.join(self.metric_dir, 'test_best_metrics.pkl'))

        self.val_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_best_metrics.csv'))
        self.test_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_best_metrics.csv'))

        self.val_k_metrics_df.to_pickle(os.path.join(self.metric_dir, 'val_best_k_metrics.pkl'))
        self.test_k_metrics_df.to_pickle(os.path.join(self.metric_dir, 'test_best_k_metrics.pkl'))


    def get_predictions(self):
        """
        get predictions from list of trainers stored in self.trainers
        requires run() to be executed as a prerequisite
        this will get a set of predictions for each fold
        """

        self.test_predictions_dict = {callback.monitor: [] for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        self.val_predictions_dict = {callback.monitor: [] for callback in self.trainers[0].callbacks if 'ModelCheckpoint' in callback.__class__.__name__}
        self.test_targets = []
        self.val_targets = []

        for idx, trainer in enumerate(self.trainers):
            tmp_test_targets = []
            tmp_val_targets = []
 
            for monitor, best_model in self.best_checkpoints[f"fold_{idx}"].items():
                self.test_predictions_dict[monitor].append(torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].test_dataloader(), ckpt_path=best_model)))
                self.val_predictions_dict[monitor].append(torch.cat(trainer.predict(trainer.model, self.data_module_cross_val[idx].val_dataloader(), ckpt_path=best_model)))

            for batch in self.data_module_cross_val[idx].test_dataloader():
                tmp_test_targets.append(batch.y)
            for batch in self.data_module_cross_val[idx].val_dataloader():
                tmp_val_targets.append(batch.y)

            self.test_targets.append(torch.cat(tmp_test_targets))
            self.val_targets.append(torch.cat(tmp_val_targets))

        self.test_predictions_df = pd.DataFrame(self.test_predictions_dict)
        self.val_predictions_df = pd.DataFrame(self.val_predictions_dict)

        self.test_predictions_df['targets'] = self.test_targets
        self.val_predictions_df['targets'] = self.val_targets

        self.test_predictions_df.to_pickle(os.path.join(self.metric_dir, 'test_predictions.pkl'))
        self.val_predictions_df.to_pickle(os.path.join(self.metric_dir, 'val_predictions.pkl'))


    def get_combined_metrics(self):
        '''
        run after get_predictions() to get metrics that combined all folds
        '''
        test_predictions = {}
        test_targets = []
        val_predictions = {}
        val_targets = []

        for monitor in self.test_predictions_df.columns:
            if 'targets' in monitor:
                continue
            test_predictions[monitor] = []
            val_predictions[monitor] = []
            for idx in range(5):
                if len(self.val_predictions_df.loc[idx, monitor]) != len(self.val_predictions_df.loc[idx, 'targets']):
                    val_predictions[monitor].extend(self.val_predictions_df.loc[idx, monitor][:-1])
                else:
                    val_predictions[monitor].extend(self.val_predictions_df.loc[idx, monitor])

                test_predictions[monitor].extend(self.test_predictions_df.loc[idx, monitor])


        for idx in range(5):
            test_targets.extend(self.test_predictions_df.loc[idx, 'targets'])
            val_targets.extend(self.val_predictions_df.loc[idx, 'targets'])

        for key in test_predictions.keys():
            test_predictions[key] = torch.tensor(test_predictions[key], dtype=torch.float)
            test_targets = torch.tensor(test_targets, dtype = torch.float)
            val_predictions[key] = torch.tensor(val_predictions[key], dtype = torch.float)
            val_targets = torch.tensor(val_targets, dtype=torch.float)
            self.test_metrics_df.loc[(f"{key}_combined", 0), :] = [
                    self.model.test_auc_fn(test_predictions[key], test_targets).numpy(),
                    self.model.test_ap_fn(test_predictions[key], test_targets.to(torch.int64)).numpy(),
                    self.model.m_fn(self.model.test_sen_fn(test_predictions[key], test_targets), self.model.test_spe_fn(test_predictions[key], test_targets)).numpy(),
                    self.model.test_sen_fn(test_predictions[key], test_targets).numpy(),
                    self.model.test_spe_fn(test_predictions[key], test_targets).numpy(),
                    ]
            self.val_metrics_df.loc[(f"{key}_combined", 0), :] = [
                    self.model.loss_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_auc_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_ap_fn(val_predictions[key], val_targets.to(torch.int64)).numpy(),
                    self.model.m_fn(self.model.val_sen_fn(val_predictions[key], val_targets), self.model.val_spe_fn(val_predictions[key], val_targets)).numpy(),
                    self.model.val_sen_fn(val_predictions[key], val_targets).numpy(),
                    self.model.val_spe_fn(val_predictions[key], val_targets).numpy(),
                    ]
        self.test_metrics_df.to_csv(os.path.join(self.metric_dir, 'test_metrics.csv'))
        self.val_metrics_df.to_csv(os.path.join(self.metric_dir, 'val_metrics.csv'))
