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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from monai.data import partition_dataset_classes, partition_dataset

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.loader import DataLoader
from torch_geometric.nn import models as tgmodels
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchmetrics

from hnc_project.pytorch.dataset_class import DatasetGeneratorRadiomics, DatasetGeneratorImage, DatasetGeneratorBoth
from hnc_project.pytorch.simple_gcn import SimpleGCN
from hnc_project.pytorch.gated_gcn import GatedGCN, ClinicalGatedGCN
from hnc_project.pytorch.deep_gcn import DeepGCN, AltDeepGCN
from hnc_project.pytorch.graphu_gcn import myGraphUNet
from hnc_project.pytorch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet200
import hnc_project.pytorch.graphag_resnet as ga
import hnc_project.pytorch.resnet_2d as res2d
from hnc_project.pytorch.cnn import CNN
from hnc_project.pytorch.resnet_spottune import SpotTune
from hnc_project.pytorch.transfer_layer_translation_cfg import layer_loop, layer_loop_downsample

MODELS = ['SimpleGCN',
          'ResNetGCN',
          'ClinicalGatedGCN',
          'GatedGCN',
          'DeepGCN',
          'myGraphUNet']

#models that use edge_attr

class RunModel(object):
    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = config['n_classes']
        self.n_channels = config['n_channels']
        self.seed = config['seed']
        self.class_weights = None
        self.data_type = config['data_type']
        self.model_name = config['model_name']
        self.extractor_name = config['extractor_name']
        self.with_edge_attr = config['with_edge_attr']
        self.n_clinical = config['n_clinical']
        self.edge_dim = config['edge_dim']
        self.scaling_type = config['scaling_type']
        self.cross_val = config['cross_val']
        self.nested_cross_val = config['nested_cross_val']
        self.augment = config['augment']
        self.external_data = None

        if f"{self.seed}" in self.config['clinical_mean'].keys():
            self.clinical_mean = torch.tensor(self.config['clinical_mean'][f"{self.seed}"], dtype=torch.float).to(self.device)
            self.clinical_std = torch.tensor(self.config['clinical_std'][f"{self.seed}"], dtype=torch.float).to(self.device)
        else:
            self.clinical_mean = None
            self.clinical_std = None

        if self.config['radiomics_mean'] is not None:
            self.radiomics_mean = torch.load(self.config['radiomics_mean']).to(self.device)
            self.radiomics_std = torch.load(self.config['radiomics_std']).to(self.device)
        else:
            self.radiomics_mean = None
            self.radiomics_std = None
 
        if self.n_classes == 1:
            self.loss_fn = nn.BCEWithLogitsLoss().to(self.device)
            #self.loss_fn = torchvision.ops.sigmoid_focal_loss().to(self.device)
            self.acc_fn = torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(self.device)
            self.auc_fn = torchmetrics.classification.BinaryAUROC().to(self.device)
            self.ap_fn = torchmetrics.classification.BinaryAveragePrecision().to(self.device)
            self.spe_fn = torchmetrics.classification.BinarySpecificity().to(self.device)
            self.sen_fn = torchmetrics.classification.BinaryRecall().to(self.device)
            self.confusion = torchmetrics.classification.BinaryConfusionMatrix().to(self.device)
        else:
            self.loss_fn = nn.CrossEntropyLoss().to(self.device)
            self.acc_fn = torchmetrics.classification.Accuracy(task='multiclass', num_classes=self.n_classes).to(self.device)
            self.auc_fn = torchmetrics.classification.AUROC(task='multiclass', num_classes=self.n_classes).to(self.device)
            self.spe_fn = torchmetrics.classification.Specificity(task='multiclass', num_classes=self.n_classes).to(self.device)
            self.sen_fn = torchmetrics.classification.Recall(task='multiclass', num_classes=self.n_classes).to(self.device)
            self.confusion = torchmetrics.classification.ConfusionMatrix(task='multiclass', num_classes=self.n_classes).to(self.device)

        if self.config['log_dir'] is None:
            self.log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
        else:
            self.log_dir = os.path.join('logs', self.config['log_dir'])
        print(f"logs are located at: {self.log_dir}")
        self.writer = SummaryWriter(self.log_dir)
        print('remember to set the data')
        self.epoch = 0
        self.best_auc =0.
        self.best_ap =0.
        self.best_M =0.
        self.best_loss = 5.0
        self.best_sum = 0.
        self.best_model = None

        self.feature_extractor = None


    def reset_metrics(self):
        self.best_auc =0.
        self.best_ap =0.
        self.best_M =0.
        self.best_loss = 5.0
        self.best_sum = 0.
        self.best_model = None
    

    def set_model(self):
        """
        sets and assigns GNN model to self.model
        make sure to assign dataset before calling set_model
        """
        if self.feature_extractor is None and 'image' in self.config['data_type'] and 'EXT' in self.model_name:
            raise ValueError(f"Need to set feature extractor before initializing model")

        if self.feature_extractor is not None:
            if 'SpotTune' in self.extractor_name:
                in_channels = self.feature_extractor.resnet.classify.out_features
            else:
                in_channels = self.feature_extractor.classify.out_features
        else:
            in_channels = 64

        if 'both' in self.data_type:
            in_channels += len(self.data[0].radiomics[0])

        if self.model_name == 'SimpleGCN':
            self.model = SimpleGCN(self.data.num_node_features, self.config['n_hidden_channels'], self.n_classes, self.config['dropout']).to(self.device)
            print(f"{self.model_name} set")
        elif self.model_name == 'ClinicalGatedGCN':
            if 'image' in self.config['data_type']:
                self.model = ClinicalGatedGCN(in_channels, self.config['n_hidden_channels'], self.n_classes, self.n_clinical, self.edge_dim, self.config['dropout']).to(self.device) 
            elif self.config['data_type'] == 'both':
                self.model = ClinicalGatedGCN(in_channels, self.config['n_hidden_channels'], self.n_classes, self.n_clinical, self.edge_dim, self.config['dropout']).to(self.device) 
            else:
                self.model = ClinicalGatedGCN(self.data.num_node_features, self.config['n_hidden_channels'], self.n_classes, self.n_clinical, self.edge_dim, self.config['dropout']).to(self.device) 
            print(f"{self.model_name} set")

        elif self.model_name == 'GatedGCN':
            if 'image' in self.config['data_type']:
                self.model = GatedGCN(in_channels, self.config['n_hidden_channels'], self.n_classes, self.edge_dim, self.config['dropout']).to(self.device)
            else:
                self.model = GatedGCN(self.data._data.num_node_features, self.config['n_hidden_channels'], self.n_classes, self.edge_dim, self.config['dropout']).to(self.device) 
            print(f"{self.model_name} set")

        elif self.model_name == 'EXTGatedResNetGCN':
            self.model = GatedGCN(in_channels, self.config['n_hidden_channels'], self.n_classes, self.edge_dim, self.config['dropout']).to(self.device)

        elif self.model_name == 'myGraphUNet':
            self.model = myGraphUNet(in_channels, self.config['n_hidden_channels'], self.n_classes, self.n_clinical, self.config['dropout']).to(self.device)
            print(f"{self.model_name} does not use edge_attr, setting with_edge_attr to False")
            self.with_edge_attr = False
            self.config['with_edge_attr'] = False 
            print(f"{self.model_name} set")

        elif self.model_name == 'DeepGCN':
            self.model = DeepGCN(in_channels, self.config['n_hidden_channels'], self.config['num_deep_layers'], self.n_classes, self.n_clinical, self.edge_dim, self.config['dropout']).to(self.device) 
            print(f"{self.model_name} set")

        elif self.model_name == 'AltDeepGCN':
            in_channels = len(self.data[0].radiomics[0]) + in_channels
            self.model = AltDeepGCN(in_channels, self.config['n_hidden_channels'], self.config['num_deep_layers'], self.n_classes, self.n_clinical, self.edge_dim, self.config['dropout']).to(self.device) 
            print(f"{self.model_name} set")
        elif self.model_name == 'CNN':
            self.model = CNN(self.config['n_channels'], self.config['dropout']).to(self.device) 
            print(f"{self.model_name} set")
        elif self.model_name == 'ResNet':
            #self.model = resnet50(num_classes=self.n_classes, in_channels=self.n_channels, dropout=self.config['dropout'], n_clinical=self.n_clinical).to(self.device)
            self.model = resnet101(num_classes=self.n_classes, in_channels=self.n_channels, dropout=self.config['dropout'], n_clinical=self.n_clinical).to(self.device)
        elif self.model_name == 'GraphAgResNet':
            self.model = ga.resnet101(num_classes=self.n_classes, in_channels=self.n_channels, dropout=self.config['dropout'], n_clinical=self.n_clinical).to(self.device)
            
          
        else:
            self.model = getattr(tgmodels, self.model_name)(in_channels=self.data.num_node_features, 
                                                            hidden_channels=self.config['n_hidden_channels'], 
                                                            depth=3, 
                                                            #num_layers=3, 
                                                            out_channels=self.n_classes, 
                                                            #dropout=self.config['dropout']
                                                           ).to(self.device)
            print(f"{self.model_name} set")
             
        # set the optimizer here, since it needs the model parameters in its initialization
        if self.feature_extractor is not None:
            self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.feature_extractor.parameters())},
                                               {'params': filter(lambda p: p.requires_grad, self.model.parameters())}], lr=self.config['learning_rate'], weight_decay=self.config['l2_reg'])
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config['learning_rate'], weight_decay=self.config['l2_reg'])

        self.lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=self.config['lr_factor'], patience=self.config['lr_patience'], mode=self.config['lr_mode'], verbose=True)
        #self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['lr_steps'], gamma=self.config['lr_factor'], verbose=True)



    def set_feature_extractor(self, transfer=None):
        if self.extractor_name == 'ResNet':
            if '2d' in self.config['data_type']:
                self.feature_extractor = res2d.resnet101(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
            else:
                #self.feature_extractor = resnet18(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
                #self.feature_extractor = resnet34(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
                self.feature_extractor = resnet50(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
                #self.feature_extractor = resnet101(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
                #self.feature_extractor = resnet152(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
                #self.feature_extractor = resnet200(num_classes=self.config['n_extracted_features'], in_channels=self.n_channels, dropout=self.config['ext_dropout']).to(self.device)
            #self.feature_extractor.classify = nn.Identity() 
            if transfer == 'MedicalNet':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                initial_state = torch.load('./models/resnet_101.pth', map_location=self.device)['state_dict']
                fixed_state = {}
                for k, v in initial_state.items():
                    if 'layer' in k:
                        mod_name = k.replace('module', 'blocks')
                    else:
                        mod_name = k.replace('module.', '')
                    for name, new in layer_loop.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    for name, new in layer_loop_downsample.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    fixed_state[mod_name] = v

                if self.n_channels > 1:
                    fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels  

                self.feature_extractor.load_state_dict(fixed_state, strict=False)
                for name, p in self.feature_extractor.named_parameters():
                    if 'classify' in name: continue
                    #if 'blocks.3' in name: continue
                    p.requires_grad = False

            if transfer == 'ImageNet':
                initial_state = torchvision.models.resnet50(weights='DEFAULT').state_dict()

                fixed_state = OrderedDict()
                for k, v in initial_state.items():
                    mod_name = k
                    parallel_mod_name = k
                    for name, new in layer_loop.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, f"blocks.{new}")
                    for name, new in layer_loop_downsample.items():
                        if name in mod_name:
                            mod_name = mod_name.replace(name, new)
                    fixed_state[mod_name] = v

                for name, w in fixed_state.items():
                    if 'blocks' in name and any(i in name for i in ('downsample', 'conv1', 'conv3')):
                        fixed_state[name] = w.unsqueeze(-1)
                    elif 'conv1' in name:
                        #fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,7,1,1)
                        fixed_state[name] = w.unsqueeze(-1).repeat(1,1,1,1,7)
                    elif 'conv2' in name:
                        #fixed_state[name] = w.unsqueeze(-1).transpose(-1, 2).repeat(1,1,3,1,1)
                        fixed_state[name] = w.unsqueeze(-1).repeat(1,1,1,1,3)
                    else:
                        fixed_state[name] = w

                fixed_state['conv1.weight'] = fixed_state['conv1.weight'].mean(axis=1).unsqueeze(1).repeat(1,self.n_channels,1,1,1)/self.n_channels

                self.feature_extractor.load_state_dict(fixed_state, strict=False)
                for name, p in self.feature_extractor.named_parameters():
                    p.requires_grad=False
            print(f"feature extractor with transfer learning: {self.config['transfer']} set")

        if self.extractor_name == 'SpotTune':
            self.feature_extractor = SpotTune(num_classes=64, in_channels=self.n_channels, dropout=self.config['dropout']).to(self.device)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            initial_state = torch.load('./models/resnet_50.pth', map_location=self.device)['state_dict']
            fixed_state = {}
            for k, v in initial_state.items():
                if 'layer' in k:
                    mod_name = k.replace('module', 'blocks')
                else:
                    mod_name = k.replace('module.', '')
                for name, new in layer_loop.items():
                    if name in mod_name:
                        mod_name = mod_name.replace(name, new)
                for name, new in layer_loop_downsample.items():
                    if name in mod_name:
                        mod_name = mod_name.replace(name, new)
                fixed_state[mod_name] = v

                fixed_state_v2 = {}
                for k, v in fixed_state.items():
                    fixed_state_v2[k] = v
                    fixed_state_v2[k.replace('blocks', 'parallel_blocks')] = v

            if self.n_channels > 1:
                fixed_state['conv1.weight'] = fixed_state['conv1.weight'].repeat(1,self.n_channels,1,1,1)/self.n_channels  

            self.feature_extractor.load_state_dict(fixed_state, strict=False)
            self.freeze_layers(ignore=['classify', 'parallel_blocks', 'agent']) 
            print(f"feature extractor SpotTune set")





    def freeze_layers(self, ignore=['conv_seg']):
        '''
        Freeze all layers except for those in the ignore list. Function can be tuned later when attempting to implement the adaptive transfer learning
        '''
        #make sure modules in the ignore list are still trainable
        if ignore[0] == 'all':
            self.unfreeze_layers()
        else:
            for name, p in self.feature_extractor.named_parameters():
                if any([i in name for i in ignore]):
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    def unfreeze_layers(self, ignore=[]):
        '''
        unfreeze all layers except for those in the ignore list
        '''
        for name, p in self.model.named_parameters():
            if any([i in name for i in ignore]):
                p.requires_grad = False
            else:
                p.requires_grad = True

    


    def set_data(self, patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2', radiomics_dir='../../data/HNSCC/radiomics', edge_file='../../data/HNSCC/edge_staging/edges_122823.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', clinical_data='../../data/HNSCC/clinical_features.pkl', version='v1', pre_transform=None):

        if self.config['dataset_name'] == 'HNSCC':
            patch_dir = '../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2'
            radiomics_dir = '../../data/HNSCC/radiomics'
            edge_file = '../../data/HNSCC/edge_staging/edges_122823.pkl'
            locations_file = '../../data/HNSCC/edge_staging/centered_locations_010424.pkl'
            clinical_data = f"../../data/HNSCC/{self.config['clinical_data']}"

        elif self.config['dataset_name'] == 'UTSW_HNC':
            patch_dir = '../../data/UTSW_HNC/Nii_222_50_50_60_Crop'
            radiomics_dir = None
            edge_file = '../../data/UTSW_HNC/edge_staging/edges_utsw_040224.pkl'
            locations_file = '../../data/UTSW_HNC/edge_staging/centered_locations_utsw_040324.pkl'
            clinical_data = f"../../data/UTSW_HNC/{self.config['clinical_data']}"
        elif self.config['dataset_name'] == 'RADCURE':
            patch_dir = '../../data/RADCURE/Nii_222_50_50_60_Crop'
            radiomics_dir = None
            edge_file = '../../data/RADCURE/edge_staging/edges_radcure_053024.pkl'
            locations_file = '../../data/RADCURE/edge_staging/centered_locations_radcure_060424.pkl'
            clinical_data = f"../../data/RADCURE/{self.config['clinical_data']}"
        elif self.config['dataset_name'] == 'Combined':
            patch_dir = '../../data/Combined/Nii_222_50_50_60_Crop'
            radiomics_dir = None
            edge_file = '../../data/Combined/edge_staging/edges_combined.pkl'
            locations_file = '../../data/Combined/edge_staging/centered_locations_combined.pkl'
            clinical_data = f"../../data/Combined/{self.config['clinical_data']}"

        if self.data_type == 'radiomics':
            self.data = DatasetGeneratorRadiomics(patch_dir, radiomics_dir, edge_file, locations_file, clinical_data, version, pre_transform=None, config=self.config)
        elif 'image' in self.data_type:
            self.data = DatasetGeneratorImage(self.config['dataset_name'], patch_dir, edge_file, locations_file, clinical_data, version, pre_transform=self.scaling_type, config=self.config)
        elif self.data_type == 'both':
            self.data = DatasetGeneratorBoth(patch_dir, radiomics_dir, edge_file, locations_file, clinical_data, version, pre_transform=self.scaling_type, config=self.config)

    def set_external_test(self, patch_dir='../../data/UTSW/Nii_222_50_50_60_Crop', radiomics_dir='../../data/HNSCC/radiomics', edge_file='../../data/UTSW_HNC/edge_staging/edges_utsw_040224.pkl', locations_file='../../data/UTSW_HNC/edge_staging/centered_locations_utsw_040324.pkl', clinical_data='../../data/UTSW_HNC/clinical_features_v2.pkl', version='v1', pre_transform=None):

        if self.config['external_dataset_name'] == 'HNSCC':
            patch_dir = '../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2'
            radiomics_dir = '../../data/HNSCC/radiomics'
            edge_file = '../../data/HNSCC/edge_staging/edges_122823.pkl'
            locations_file = '../../data/HNSCC/edge_staging/centered_locations_010424.pkl'
            clinical_data = '../../data/HNSCC/clinical_features_sorted_v2.pkl'

        elif self.config['external_dataset_name'] == 'UTSW_HNC':
            patch_dir = '../../data/UTSW_HNC/Nii_222_50_50_60_Crop'
            radiomics_dir = None
            edge_file = '../../data/UTSW_HNC/edge_staging/edges_utsw_040224.pkl'
            locations_file = '../../data/UTSW_HNC/edge_staging/centered_locations_utsw_040324.pkl'
            clinical_data = '../../data/UTSW_HNC/clinical_features_sorted_v5.pkl'

        if 'image' in self.data_type:
            self.external_data = DatasetGeneratorImage(self.config['external_dataset_name'], patch_dir, edge_file, locations_file, clinical_data, version, pre_transform=self.scaling_type, config=self.config)
        else:
            print(f"input data is set as {self.data_type}, and needs to be image")
        self.external_dataloader = DataLoader(self.external_data, batch_size=self.batch_size, shuffle=True, pin_memory=True)


    def set_scaled_data(self, patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2', radiomics_dir='../../data/HNSCC/radiomics', edge_file='../../data/HNSCC/edge_staging/edges_122823.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', version='v1'):
        if self.data_type == 'radiomics':
            self.get_std_norm()
            self.data = DatasetGeneratorRadiomics(patch_dir, radiomics_dir, edge_file, locations_file, version, pre_transform=self.transform, config=self.config)
        


    def set_train_test_split(self):
        if self.cross_val:
            augments = np.where(['rotation' in pat for pat in self.data.patients])[0]
            for aug in self.config['augments']:
                if 'rotation' in aug: continue
                augments = np.append(augments, np.where([aug in pat for pat in self.data.patients]))
            if len(augments) > 0:
                x = np.array(range(self.data.len()))
                x = np.delete(x, augments)

                y_aug = self.data.y.iloc[augments]
                y = self.data.y.drop(y_aug.index)
            else:
                x = np.array(range(self.data.len()))
                y_aug = None
                y = self.data.y
            self.folds = partition_dataset_classes(x, y, num_partitions=5, shuffle=True, seed=self.config['seed'])

            self.train_folds = [[0,1,2],
                                [4,0,1],
                                [3,4,0],
                                [2,3,4],
                                [1,2,3]]
            self.val_folds = [3, 2, 1, 0, 4]
            self.test_folds = [4, 3, 2, 1, 0]

            self.nested_train_folds = [[[0,1,2],[1,2,3],[2,3,0],[3,0,1]],
                                       [[4,0,1],[0,1,2],[1,2,4],[2,4,0]],
                                       [[3,4,0],[4,0,1],[0,1,3],[1,3,4]],
                                       [[2,3,4],[3,4,0],[4,0,2],[0,2,3]],
                                       [[1,2,3],[2,3,4],[3,4,1],[4,1,2]]]
            self.nested_val_folds = [[3,0,1,2],
                                     [2,4,0,1],
                                     [1,3,4,0],
                                     [0,2,3,4],
                                     [4,1,2,3]]

            self.train_splits = [self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in self.train_folds]

            self.aug_splits = []
            if len(augments) > 0:
                aug_pats = y_aug.index
                for split in self.train_splits:
                    aug_split = []
                    split_pats = y.index[split]
                    for pat in aug_pats:
                        if pat.split('_')[0] in split_pats:
                            aug_split.append(pat)
                    self.aug_splits.append(aug_split)
                for idx in range(len(self.train_splits)):
                    self.train_splits[idx].extend([self.data.y.index.get_loc(pat) for pat in self.aug_splits[idx]])
            self.val_splits = [self.folds[i] for i in self.val_folds]
            self.test_splits = [self.folds[i] for i in self.test_folds]
            self.class_weights = []
            for split in self.train_splits:
                self.class_weights.append([len(self.data.y.values[split][self.data.y.values[split]==0]) / np.sum(self.data.y.values[split])])

            self.fold_df = pd.DataFrame(self.data.y).copy(deep=True)
            self.train_fold_df = pd.DataFrame(self.data.y_source).copy(deep=True)
            for idx in range(5):
                self.fold_df[f"train_fold_{idx}"] = [True if pat in self.data.y.iloc[self.train_splits[idx]].index else False for pat in self.fold_df.index] 
                self.train_fold_df[f"fold_{idx}"] = [True if pat in self.data.y_source.iloc[self.folds[idx]].index else False for pat in self.train_fold_df.index]              

            self.nested_train_splits = [[self.folds[i]+self.folds[j]+self.folds[k] for i,j,k in nest] for nest in self.nested_train_folds]
            self.nested_val_splits = [[self.folds[i] for i in nest] for nest in self.nested_val_folds] 
            self.nested_aug_splits = []
            if len(augments) > 0:
                aug_pats = y_aug.index
                for split in self.nested_train_splits:
                    aug_split = []
                    for nest in split:
                        aug_nest = []
                        nest_pats = y.index[nest]
                        for pat in aug_pats:
                            if pat.split('_')[0] in nest_pats:
                                aug_nest.append(pat)
                        aug_split.append(aug_nest)
                    self.nested_aug_splits.append(aug_split)

                for idx in range(len(self.nested_train_splits)):
                    for jdx in range(len(self.nested_train_splits[idx])):
                        self.nested_train_splits[idx][jdx].extend([self.data.y.index.get_loc(pat) for pat in self.nested_aug_splits[idx][jdx]])   


            
        else:
            if self.data_type == 'outside':
                idx_train, self.idx_test, y_train, y_test = train_test_split(list(range(len(self.data._data.y))), self.data._data.y, test_size=0.2, random_state=42, stratify=self.data._data.y) 
                self.idx_train, self.idx_val, self.y_train, self.y_val = train_test_split(idx_train, y_train, test_size=0.4, random_state=42, stratify=y_train) 
                self.class_weights = [len(self.y_train[self.y_train==0]) / self.y_train.sum()]
            else:
                idx_train, self.idx_test, y_train, y_test = train_test_split(list(range(self.data.len())), self.data.y.values, test_size=0.2, random_state=42, stratify=self.data.y.values) 
                self.idx_train, self.idx_val, self.y_train, self.y_val = train_test_split(idx_train, y_train, test_size=0.4, random_state=42, stratify=y_train) 
                self.class_weights = [len(self.y_train[self.y_train==0]) / np.sum(self.y_train)]



    def get_std_norm(self):

        if self.config['use_clinical'] and (self.clinical_mean is None or len(self.clinical_mean) < 1):
            scale_features = None
            self.clinical_mean = []
            self.clinical_std = []
            for idx in tqdm(self.train_splits):
                for i, feat in tqdm(list(enumerate(self.data[idx]))):
                    if i == 0:
                        scale_features = feat.clinical[0][[0, 1]].unsqueeze(0)
                        continue
                    scale_features = torch.cat((scale_features, feat.clinical[0][[0, 1]].unsqueeze(0)), 0)
                
                self.clinical_mean.append(scale_features.mean(0))
                self.clinical_std.append(scale_features.std(0))
            print(f"Mean with splitting seed {self.seed}")
            print(self.clinical_mean)
            print(f"STD with splitting seed {self.seed}")
            print(self.clinical_std)
                

        if self.data_type == 'both' and self.radiomics_mean is None:
            self.radiomics_mean = []
            self.radiomics_std = []
            node_features = None
            for idx in self.train_splits:
                for i, feat in enumerate(self.data[idx]):
                    if i == 0:
                        node_features = feat.radiomics
                        continue
                    node_features = torch.cat((node_features, feat.radiomics), 0)

                self.radiomics_mean.append(node_features.mean(0))
                self.radiomics_std.append(node_features.std(0))

      

            
            


    def set_train_loader(self):
        if self.nested_cross_val:
            self.train_nested_dataloaders = [[DataLoader(self.data[nest], batch_size=self.batch_size, shuffle=True, pin_memory=True) for nest in fold] for fold in self.nested_train_splits]
        elif self.cross_val:
            self.train_cross_dataloaders = [DataLoader(self.data[fold], batch_size=self.batch_size, shuffle=True, pin_memory=True) for fold in self.train_splits]
        else:
            self.train_dataloader = DataLoader(self.data[self.idx_train], batch_size=self.batch_size, shuffle=True, pin_memory=True)


    def set_val_loader(self):
        if self.nested_cross_val:
            self.val_nested_dataloaders = [[DataLoader(self.data[nest], batch_size=self.batch_size, shuffle=False, pin_memory=True) for nest in fold] for fold in self.nested_val_splits]
        elif self.cross_val:
            self.val_cross_dataloaders = [DataLoader(self.data[fold], batch_size=self.batch_size, shuffle=False, pin_memory=True) for fold in self.val_splits]
        else:
            self.val_dataloader = DataLoader(self.data[self.idx_val], batch_size=self.batch_size, shuffle=False, pin_memory=True)


    def set_test_loader(self):
        if self.cross_val:
            self.test_cross_dataloaders = [DataLoader(self.data[fold], batch_size=self.batch_size, shuffle=False, pin_memory=True) for fold in self.test_splits]
        else:
            self.test_dataloader = DataLoader(self.data[self.idx_test], batch_size=self.batch_size, shuffle=False, pin_memory=True)


    def set_loss_fn(self, cross_idx=None):
        if cross_idx is not None:
            if self.n_classes == 1:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weights[cross_idx])).to(self.device)
            else:
                self.loss_fn = nn.CrossEntropyLoss().to(pos_weight=torch.tensor(self.class_weights[cross_idx])).to(self.device)
        else:
            if self.n_classes == 1:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weights)).to(self.device)
            else:
                self.loss_fn = nn.CrossEntropyLoss().to(pos_weight=torch.tensor(self.class_weights)).to(self.device)
        
 
    def train(self, data_to_use='train', cross_idx=None, nest_idx=None):
        if data_to_use == 'val':
            if cross_idx is not None:
                if nest_idx is not None:
                    dataloader = self.val_nested_dataloaders[cross_idx][nest_idx]
                else:
                    dataloader = self.val_cross_dataloaders[cross_idx]
            else:
                dataloader = self.val_dataloader
        elif data_to_use == 'test':
            if cross_idx is not None:
                dataloader = self.test_cross_dataloaders[cross_idx]
            else:
                dataloader = self.test_dataloader
        elif data_to_use == 'train':
            if cross_idx is not None:
                if nest_idx is not None:
                    dataloader = self.train_nested_dataloaders[cross_idx][nest_idx]
                else:
                    dataloader = self.train_cross_dataloaders[cross_idx]
            else:
                dataloader = self.train_dataloader

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if self.feature_extractor is not None:
            self.feature_extractor.train()
        self.model.train()
        total_loss = 0.
        
        for batch_idx, batch in enumerate(dataloader):
            #batch = batch.to(self.device)
            batch.x = batch.x.to(self.device, dtype=torch.float)
            batch.y = batch.y.to(self.device, dtype=torch.float)
            batch.edge_index = batch.edge_index.to(self.device, dtype=torch.int64)
            
            #edge_attr = None
            if batch.edge_attr is not None:
                batch.edge_attr = batch.edge_attr.to(self.device, dtype=torch.float)
            batch.batch = batch.batch.to(self.device)

            if self.feature_extractor is not None:
                x = self.feature_extractor(batch.x)
            else:
                x = batch.x
            #Compute prediction error
            if self.data_type in ['both', 'radiomics']:
                batch.radiomics = batch.radiomics.to(self.device, dtype=torch.float)
                batch.radiomics = (batch.radiomics - self.radiomics_mean[cross_idx]) / self.radiomics_mean[cross_idx]

            if self.with_edge_attr:

                if self.config['use_clinical']:
                    batch.clinical = batch.clinical.to(self.device, dtype=torch.float)
                    batch.clinical[:, 0] = (batch.clinical[:,0] - self.clinical_mean[cross_idx][0]) / self.clinical_std[cross_idx][0]
                    batch.clinical[:, 1] = (batch.clinical[:,1] - self.clinical_mean[cross_idx][1]) / self.clinical_std[cross_idx][1]
                    if self.data_type == 'both':
                        pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, clinical=batch.clinical, radiomics=batch.radiomics)
                    else:
                        pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, clinical=batch.clinical)
                else:
                    pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
            else:
                if self.config['use_clinical']:
                    batch.clinical = batch.clinical.to(self.device, dtype=torch.float)
                    batch.clinical[:, 0] = (batch.clinical[:,0] - self.clinical_mean[cross_idx][0]) / self.clinical_std[cross_idx][0]
                    batch.clinical[:, 1] = (batch.clinical[:,1] - self.clinical_mean[cross_idx][1]) / self.clinical_std[cross_idx][1]
                    if self.data_type == 'both':
                        pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch, clinical=batch.clinical, radiomics=batch.radiomics)
                    else:
                        #pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch, clinical=batch.clinical)
                        pred = self.model(x=x, batch=batch.batch, clinical=batch.clinical)
                else:
                    pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch)
            loss = self.loss_fn(pred, batch.y)
            total_loss += loss 
           
            if self.n_classes > 1:
                acc = self.acc_fn(pred, torch.argmax(batch.y.squeeze(), dim=1))
                auc = self.auc_fn(pred, torch.argmax(batch.y.squeeze(), dim=1))
                ap = self.ap_fn(pred, torch.argmax(batch.y.squeeze(), dim=1))
                sen = self.sen_fn(pred, torch.argmax(batch.y.squeeze(), dim=1))
                spe = self.spe_fn(pred, torch.argmax(batch.y.squeeze(), dim=1))
            else:
                acc = self.acc_fn(pred, batch.y)
                auc = self.auc_fn(pred, batch.y)
                ap = self.ap_fn(pred, batch.y.to(torch.long))
                sen = self.sen_fn(pred, batch.y)
                spe = self.spe_fn(pred, batch.y)
        
        #Backpropagate
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            print(f"[{batch_idx+1:02g}/{num_batches}][{'='*int((100*((batch_idx+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch_idx+1))/num_batches))//5)}] "
                  f"loss: {loss:>0.4f}, "
                  f"AP: {self.ap_fn.compute():>0.4f}, "
                  f"AUC: {self.auc_fn.compute():>0.4f}", end='\r')

        iacc = self.acc_fn.compute()
        iauc = self.auc_fn.compute()
        iap = self.ap_fn.compute()
        total_loss /= num_batches

        ispe = self.spe_fn.compute()
        isen = self.sen_fn.compute()
        metric_M = 0.6*isen + 0.4*ispe

        print(f"\nAvg Train Loss: {total_loss:>0.4f}; "
              f"Total Train AP: {iap:>0.4f}; "
              f"Total Train AUC: {iauc:>0.4f}")   
        if cross_idx:
            displace = cross_idx * self.n_epochs
        else:
            displace = 0 
        self.writer.add_scalars('Loss', {f"{data_to_use}_loss": total_loss}, self.epoch+displace)
        self.writer.add_scalars('ACC', {f"{data_to_use}_acc": iacc}, self.epoch+displace)
        self.writer.add_scalars('AUC', {f"{data_to_use}_auc": iauc}, self.epoch+displace)
        self.writer.add_scalars('AP', {f"{data_to_use}_ap": iap}, self.epoch+displace)

        self.acc_fn.reset()
        self.auc_fn.reset()
        self.ap_fn.reset()
        self.sen_fn.reset()
        self.spe_fn.reset()

        return total_loss.cpu().detach().numpy(), iacc.cpu().detach().numpy(), iauc.cpu().detach().numpy()



    def test(self, data_to_use='test', cross_idx=None, nest_idx=None):
        if data_to_use == 'val':
            if cross_idx is not None:
                if nest_idx is not None:
                    dataloader = self.val_nested_dataloaders[cross_idx][nest_idx]
                else:
                    dataloader = self.val_cross_dataloaders[cross_idx]
            else:
                dataloader = self.val_dataloader
        elif data_to_use == 'test':
            if cross_idx is not None:
                dataloader = self.test_cross_dataloaders[cross_idx]
            else:
                dataloader = self.test_dataloader
        elif data_to_use == 'train':
            if cross_idx is not None:
                if nest_idx is not None:
                    dataloader = self.train_nested_dataloaders[cross_idx][nest_idx]
                else:
                    dataloader = self.train_cross_dataloaders[cross_idx]
            else:
                dataloader = self.train_dataloader
        elif data_to_use == 'external':
            dataloader = self.external_dataloader

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        if self.feature_extractor is not None:
            self.feature_extractor.eval()
        self.model.eval()
        test_loss = 0
        total_pred = []
        total_target = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch.x = batch.x.to(self.device, dtype=torch.float)
                batch.y = batch.y.to(self.device, dtype=torch.float)
                batch.edge_index = batch.edge_index.to(self.device, dtype=torch.int64)
                if batch.edge_attr is not None:
                    batch.edge_attr = batch.edge_attr.to(self.device, dtype=torch.float)
                batch.batch = batch.batch.to(self.device)

                if self.feature_extractor is not None:
                    x = self.feature_extractor(batch.x)
                else:
                    x = batch.x

                if self.data_type in ['both', 'radiomics']:
                    batch.radiomics = batch.radiomics.to(self.device, dtype=torch.float)
                    batch.radiomics = (batch.radiomics - self.radiomics_mean[cross_idx]) / self.radiomics_mean[cross_idx]
                if self.with_edge_attr:
                    if self.config['use_clinical']:
                        batch.clinical = batch.clinical.to(self.device, dtype=torch.float)
                        batch.clinical[:, 0] = (batch.clinical[:,0] - self.clinical_mean[cross_idx][0]) / self.clinical_std[cross_idx][0]
                        batch.clinical[:, 1] = (batch.clinical[:,1] - self.clinical_mean[cross_idx][1]) / self.clinical_std[cross_idx][1]
                        if self.data_type == 'both':
                            pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, clinical=batch.clinical, radiomics=batch.radiomics)
                        else:
                            pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch, clinical=batch.clinical)
                    else:
                        pred = self.model(x=x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                else:
                    if self.config['use_clinical']:
                        batch.clinical = batch.clinical.to(self.device, dtype=torch.float)
                        batch.clinical[:, 0] = (batch.clinical[:,0] - self.clinical_mean[cross_idx][0]) / self.clinical_std[cross_idx][0]
                        batch.clinical[:, 1] = (batch.clinical[:,1] - self.clinical_mean[cross_idx][1]) / self.clinical_std[cross_idx][1]
                        if self.data_type == 'both':
                            pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch, clinical=batch.clinical, radiomics=batch.radiomics)
                        else:
                            #pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch, clinical=batch.clinical)
                            pred = self.model(x=x, batch=batch.batch, clinical=batch.clinical)
                    else:
                        pred = self.model(x=x, edge_index=batch.edge_index, batch=batch.batch)

                test_loss += self.loss_fn(pred, batch.y)
                total_pred.extend(pred)
                total_target.extend(batch.y)
                if self.n_classes > 1:
                    acc = self.acc_fn(pred, torch.argmax(y.squeeze(), dim=1))
                    auc = self.auc_fn(pred, torch.argmax(y.squeeze(), dim=1))
                    ap = self.ap_fn(pred, torch.argmax(y.squeeze(), dim=1))
                    sen = self.sen_fn(pred, torch.argmax(y.squeeze(), dim=1))
                    spe = self.spe_fn(pred, torch.argmax(y.squeeze(), dim=1))
                else:
                    acc = self.acc_fn(pred, batch.y)
                    auc = self.auc_fn(pred, batch.y)
                    ap = self.ap_fn(pred, batch.y.to(torch.long))
                    sen = self.sen_fn(pred, batch.y)
                    spe = self.spe_fn(pred, batch.y)
                
                print(f"[{batch_idx+1}/{num_batches}][{'='*int((100*((batch_idx+1)/num_batches))//5) + '.'*int((100*((num_batches-(batch_idx+1))/num_batches))//5)}]", end='\r')
                #print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')
                      
        iacc = self.acc_fn.compute()
        iauc = self.auc_fn.compute()
        iap = self.ap_fn.compute()
        ispe = self.spe_fn.compute()
        isen = self.sen_fn.compute()

        iacc_err = torch.sqrt((1/size)*iacc*(1-iacc))
        iauc_err = torch.sqrt((1/size)*iauc*(1-iauc))
        iap_err = torch.sqrt((1/size)*iap*(1-iap))

        metric_M = 0.6*isen + 0.4*ispe

        test_loss /= num_batches
        print(f"\nEpoch {data_to_use.capitalize()}   Loss: {test_loss:>0.3f}; "
              f"      {data_to_use.capitalize()} AP: {iap:>0.3f} \u00B1 {iap_err:>0.2}; "
              f"        {data_to_use.capitalize()} AUC: {iauc:>0.3f} \u00B1 {iauc_err:>0.2};"
              f"        {data_to_use.capitalize()} SEN: {isen:>0.3f}; "
              f"        {data_to_use.capitalize()} SPE: {ispe:>0.3f} ")
 
        if cross_idx:
            displace = cross_idx * self.n_epochs
        else:
            displace = 0 
        self.writer.add_scalars('Loss', {f"{data_to_use}_loss": test_loss}, self.epoch+displace)
        self.writer.add_scalars('ACC', {f"{data_to_use}_acc": iacc}, self.epoch+displace)
        self.writer.add_scalars('AUC', {f"{data_to_use}_auc": iauc}, self.epoch+displace)
        self.writer.add_scalars('AP', {f"{data_to_use}_ap": iap}, self.epoch+displace)

        if data_to_use == 'val':
            #if iap > self.best_ap and self.epoch > 40:
            if (iauc >= self.best_auc or iap >= self.best_ap) and self.epoch > 25:
            #if iap >= self.best_ap and self.epoch > 25:
                print(f"#################new best model saved###############")
                if iauc >= self.best_auc:
                    self.best_auc = iauc
                if iap >= self.best_ap:
                    self.best_ap = iap
                #self.best_loss = test_loss
                out_path = os.path.join(self.log_dir, f"best_model_{self.epoch}_{test_loss:0.2f}_{metric_M:>0.2f}_{iauc:>0.2f}.pth")
                if self.feature_extractor is None:
                    self.best_model = {
                        'model_state_dict': copy.deepcopy(self.model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                        'config' : self.config,
                        'epoch': self.epoch,
                        'loss': test_loss,}
                else:
                    self.best_model = {
                        'model_state_dict': copy.deepcopy(self.model.state_dict()),
                        'extractor_state_dict': copy.deepcopy(self.feature_extractor.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                        'config' : self.config,
                        'epoch': self.epoch,
                        'loss': test_loss,}
        total_pred = torch.tensor(total_pred)
        total_target = torch.tensor(total_target)
        self.acc_fn.reset()
        self.auc_fn.reset()
        self.ap_fn.reset()
        self.sen_fn.reset()
        self.spe_fn.reset()
        self.policy = None
        return [test_loss.cpu().detach().numpy(), iap.cpu().detach().numpy(), iauc.cpu().detach().numpy(), isen.cpu().detach().numpy(), ispe.cpu().detach().numpy()], [torch.flatten(total_pred).cpu().detach().numpy(), torch.flatten(total_target).cpu().detach().numpy()]

   

    def run_crossval(self):
        if not self.cross_val:
            raise ValueError('cross_val needs to be set to True to run')
        self.fold_metrics = []
        self.fold_probs = {}
        self.fold_targets = {}
        self.fold_probs = {}
        self.fold_targets = {}
        keys = ['train', 'val', 'test']
        for k in keys:
            self.fold_probs[k] = []
            self.fold_targets[k] = []

        for fold_idx in range(5):
            if self.feature_extractor is not None and self.data_type!='radiomics':
                self.set_feature_extractor(transfer=self.config['transfer'])
            self.reset_metrics()
            self.set_model()
            self.set_loss_fn(cross_idx=fold_idx)
            print(f"Fold: {fold_idx}")
            results = self.run(cross_idx=fold_idx)
            self.fold_metrics.append(results[0])
            self.fold_probs['train'].append(results[1][0][0])
            self.fold_targets['train'].append(results[1][0][1])
            self.fold_probs['val'].extend(results[1][1][0])
            self.fold_targets['val'].extend(results[1][1][1])
            self.fold_probs['test'].extend(results[1][2][0])
            self.fold_targets['test'].extend(results[1][2][1])

    
        metric_results = np.array(self.fold_metrics)

        for k in self.fold_probs.keys():
            if 'train' in k: continue
            self.fold_probs[k] = torch.tensor(np.array(self.fold_probs[k]))
            self.fold_targets[k] = torch.tensor(np.array(self.fold_targets[k]), dtype=torch.long)

        overall_test_metrics = [
                                self.acc_fn(self.fold_probs['test'], self.fold_targets['test']),
                                self.ap_fn(self.fold_probs['test'], self.fold_targets['test']),
                                self.auc_fn(self.fold_probs['test'], self.fold_targets['test']),
                                self.sen_fn(self.fold_probs['test'], self.fold_targets['test']),
                                self.spe_fn(self.fold_probs['test'], self.fold_targets['test']),
                               ]
        overall_val_metrics = [
                                self.acc_fn(self.fold_probs['val'], self.fold_targets['val']),
                                self.ap_fn(self.fold_probs['val'], self.fold_targets['val']),
                                self.auc_fn(self.fold_probs['val'], self.fold_targets['val']),
                                self.sen_fn(self.fold_probs['val'], self.fold_targets['val']),
                                self.spe_fn(self.fold_probs['val'], self.fold_targets['val']),
                               ]
        for i, r in enumerate(overall_test_metrics):
            overall_test_metrics[i] = r.cpu().detach().numpy()
        for i, r in enumerate(overall_val_metrics):
            overall_val_metrics[i] = r.cpu().detach().numpy()

        with open(f"{self.log_dir}/overall_results.csv", 'w') as out_file:
            out_file.write('loss,AP,AUC,SEN,SPE\n')
            out_file.writelines([f"Mean: {r}\n" for r in metric_results.mean(axis=0)])
            out_file.writelines([f"Overall Val: {met}\n" for met in overall_val_metrics])
            out_file.writelines([f"Overall Test: {met}\n" for met in overall_test_metrics])
            #out_file.write(f"Mean Val: {np.mean(overall_val_metrics, axis=0)}\n")
            #out_file.write(f"Mean Test: {np.mean(overall_test_metrics, axis=0)}\n")
        
        return metric_results, metric_results.mean(axis=0), overall_test_metrics, overall_val_metrics
            


    def run_nested_crossval(self):
        if not self.cross_val and not self.nested_cross_val:
            raise ValueError('cross_val and nested_cross_val need to be set to True to run')
        self.fold_metrics = []
        self.fold_probs = {}
        self.fold_targets = {}
        self.fold_probs = {}
        self.fold_targets = {}
        keys = ['train', 'val', 'test']
        for k in keys:
            self.fold_probs[k] = []
            self.fold_targets[k] = []

        for fold_idx in range(5):
            self.fold_metrics.append([])
            for k in keys:
                self.fold_probs[k].append([])
                self.fold_targets[k].append([])
            for nest_idx in range(4):
                if self.feature_extractor is not None and self.data_type!='radiomics':
                    self.set_feature_extractor(transfer=self.config['transfer'])
                self.reset_metrics()
                self.set_model()
                self.set_loss_fn(cross_idx=fold_idx)
                print(f"Fold: {fold_idx}, Nest: {nest_idx}")
                results = self.run(cross_idx=fold_idx, nest_idx=nest_idx)
                self.fold_metrics[fold_idx].append(results[0])
                self.fold_probs['train'][fold_idx].append(results[1][0][0])
                self.fold_targets['train'][fold_idx].append(results[1][0][1])
                self.fold_probs['val'][fold_idx].extend(results[1][1][0])
                self.fold_targets['val'][fold_idx].extend(results[1][1][1])
                self.fold_probs['test'][fold_idx].append(results[1][2][0])
                self.fold_targets['test'][fold_idx].append(results[1][2][1])

   
        
        metric_results = np.array(self.fold_metrics)

        overall_test_metrics = []
        overall_val_metrics = []
        final_test_probs = []
        final_test_targets = []
        final_val_probs = []
        final_val_targets = []
        for fold_idx in range(5):
            self.fold_probs['val'][fold_idx] = torch.tensor(self.fold_probs['val'][fold_idx]).to(self.device)
            self.fold_targets['val'][fold_idx] = torch.tensor([int(x) for x in self.fold_targets['val'][fold_idx]], dtype=torch.long).to(self.device)
            self.fold_probs['test'][fold_idx] = torch.tensor(self.fold_probs['test'][fold_idx]).to(self.device).mean(axis=0)
            self.fold_targets['test'][fold_idx] = torch.tensor(self.fold_targets['test'][fold_idx][0], dtype=torch.long).to(self.device)
            overall_test_metrics.append([
                                    self.acc_fn(self.fold_probs['test'][fold_idx], self.fold_targets['test'][fold_idx]),
                                    self.ap_fn(self.fold_probs['test'][fold_idx], self.fold_targets['test'][fold_idx]),
                                    self.auc_fn(self.fold_probs['test'][fold_idx], self.fold_targets['test'][fold_idx]),
                                    self.sen_fn(self.fold_probs['test'][fold_idx], self.fold_targets['test'][fold_idx]),
                                    self.spe_fn(self.fold_probs['test'][fold_idx], self.fold_targets['test'][fold_idx]),
                                   ])
            overall_val_metrics.append([
                                    self.acc_fn(self.fold_probs['val'][fold_idx], self.fold_targets['val'][fold_idx]),
                                    self.ap_fn(self.fold_probs['val'][fold_idx], self.fold_targets['val'][fold_idx]),
                                    self.auc_fn(self.fold_probs['val'][fold_idx], self.fold_targets['val'][fold_idx]),
                                    self.sen_fn(self.fold_probs['val'][fold_idx], self.fold_targets['val'][fold_idx]),
                                    self.spe_fn(self.fold_probs['val'][fold_idx], self.fold_targets['val'][fold_idx]),
                                   ])
            #final_val_probs.extend(self.fold_probs['val']['fold_idx'])
            #final_val_targets.extend(self.fold_targets['val']['fold_idx'])
            final_test_probs.extend(self.fold_probs['test'][fold_idx])
            final_test_targets.extend(self.fold_targets['test'][fold_idx])

        #final_val_probs = torch.tensor(final_val_probs).to(self.device)
        #final_val_targets = torch.tensor(final_val_targets).to(self.device)
        self.final_test_probs = torch.tensor(final_test_probs).to(self.device)
        self.final_test_targets = torch.tensor(final_test_targets).to(self.device)
        final_test_metrics = [
                                self.acc_fn(self.final_test_probs, self.final_test_targets),
                                self.ap_fn(self.final_test_probs, self.final_test_targets),
                                self.auc_fn(self.final_test_probs, self.final_test_targets),
                                self.sen_fn(self.final_test_probs, self.final_test_targets),
                                self.spe_fn(self.final_test_probs, self.final_test_targets),
                               ]
        for fold_idx in range(5):
            for i, r in enumerate(overall_test_metrics[fold_idx]):
                overall_test_metrics[fold_idx][i] = r.cpu().detach().numpy()
            for i, r in enumerate(overall_val_metrics[fold_idx]):
                overall_val_metrics[fold_idx][i] = r.cpu().detach().numpy()

        
        with open(f"{self.log_dir}/overall_results.csv", 'w') as out_file:
            out_file.write('loss,AP,AUC,SEN,SPE\n')
            out_file.writelines([f"Mean: {r}\n" for r in metric_results.mean(axis=1)])
            out_file.writelines([f"Fold Val {idx}: {met}\n" for idx, met in enumerate(overall_val_metrics)])
            out_file.writelines([f"Fold Test {idx}: {met}\n" for idx, met in enumerate(overall_test_metrics)])
            out_file.write(f"Overall Mean Val: {np.mean(overall_val_metrics, axis=0)}\n")
            out_file.write(f"Fold Test: {final_test_metrics}\n")
            out_file.write(f"Config: {self.config}\n")
      
        self.fold_df.to_csv(f"{self.log_dir}/folds.csv")
        self.train_fold_df.to_csv(f"{self.log_dir}/train_folds.csv")

        with open(f"{self.log_dir}/model_probabilities.pkl", 'wb') as f:
            pickle.dump(self.fold_probs, f)
            pickle.dump(self.fold_targets, f)
            f.close()

        return metric_results, overall_test_metrics, overall_val_metrics


    def run(self, cross_idx=None, nest_idx=None):
        out_csv = []
        out_csv.append(f"epoch,train_loss,train_acc,train_auc,val_loss,val_acc,val_auc\n")
        self.epoch = 0
        for t in range(self.n_epochs):
            print(f"Epoch {t+1}/{self.n_epochs}")
            self.epoch = t + 1
            train_results = self.train('train', cross_idx=cross_idx, nest_idx=nest_idx)
            #train_test_results = self.test('train', cross_idx=cross_idx, nest_idx=nest_idx)
            val_results = self.test('val', cross_idx=cross_idx, nest_idx=nest_idx)
            if self.config['lr_sched']:
                print('sched step')
                #### 0 - loss; 1 - ap; 2 - auc; 3 - sen; 4 - spe
                self.lr_sched.step(val_results[0][2])   
                # for running on unix/linux
                #self.lr_sched.get_last_lr()   
                #self.lr_sched.step()   
        
            out_csv.append(f"{self.epoch},{train_results[0]},{train_results[1]},{train_results[2]},{val_results[0][0]},{val_results[0][1]},{val_results[0][2]}\n")
            print(f"-----------------------------------------------------------")
        self.epoch += 1

        if self.best_model is not None:
             if self.feature_extractor is not None:
                 self.feature_extractor.load_state_dict(self.best_model['extractor_state_dict'])
             self.model.load_state_dict(self.best_model['model_state_dict'])
             self.optimizer.load_state_dict(self.best_model['optimizer_state_dict'])
        print("Train Total")
        final_train = self.test('train', cross_idx=cross_idx, nest_idx=nest_idx)
        print("Val Total")
        final_val = self.test('val', cross_idx=cross_idx, nest_idx=nest_idx)
        print("Test")
        final_test = self.test('test', cross_idx=cross_idx)
        out_csv.append(f"       loss, AP, AUC, SEN, SPE\n")
        out_csv.append(f"Train: {final_train[0]}\n")
        out_csv.append(f"Val: {final_val[0]}\n")
        out_csv.append(f"Test: {final_test[0]}\n")

        model_name = "best_model.pth"
        if self.nested_cross_val:
            model_name = f"best_model_{cross_idx}_{nest_idx}.pth"
        elif self.cross_val:
            model_name = f"best_model_{cross_idx}.pth"
        if self.best_model is None:
            if self.feature_extractor is not None:
                torch.save({'model_state_dict': copy.deepcopy(self.model.state_dict()),
                            'extractor_state_dict': copy.deepcopy(self.feature_extractor.state_dict()),
                            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                            'config'    : self.config, 
                            'model_name': self.model.__class__.__name__}, os.path.join(self.log_dir, model_name))
            else:
                torch.save({'model_state_dict': copy.deepcopy(self.model.state_dict()),
                            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                            'config'    : self.config, 
                            'model_name': self.model.__class__.__name__}, os.path.join(self.log_dir, model_name))
        else:
            torch.save(self.best_model, os.path.join(self.log_dir, model_name))

        print("Done")
        out_csv.append(f"{self.config}\n")
        if self.cross_val:
            if self.nested_cross_val:
                with open(f"{self.log_dir}/results_{cross_idx}_{nest_idx}.csv", 'w') as out_file:
                    out_file.writelines(out_csv)
            else:
                with open(f"{self.log_dir}/results_{cross_idx}.csv", 'w') as out_file:
                    out_file.writelines(out_csv)
        else:
            with open(f"{self.log_dir}/results.csv", 'w') as out_file:
                out_file.writelines(out_csv)
        return [final_train[0], final_val[0], final_test[0]], [final_train[1], final_val[1], final_test[1]]



    def predict(self, data_to_use='test'):
        if data_to_use == 'test':
            dataloader = self.test_dataloader
        elif data_to_use == 'train':
            dataloader = self.train_dataloader
        elif data_to_use == 'val':
            dataloader = self.val_dataloader

        self.model.eval()
        predict_fn = nn.Sigmoid()
        num_batches = len(dataloader)
        results = []
        true_val = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                if self.use_clinical:
                    X, y = (X[0].to(self.device, dtype=torch.float), X[1].to(self.device, dtype=torch.float)), y.to(self.device, dtype=torch.long)
                else:
                    X, y = X.to(self.device, dtype=torch.float), y.to(self.device, dtype=torch.long)

                if self.spottune:
                    probs = self.agent(X)
                    action = self.gumbel_softmax(probs.view(probs.size(0), -1, 2), self.gumbel_temperature)
                    self.policy = action[:,:,1]

                pred = self.model(X, self.policy)
                pred = predict_fn(pred)
                pred_np = pred.cpu().detach().numpy()
                for p in pred_np:
                    results.append(p)
                for t in y:
                    true_val.append(t.cpu().detach().numpy())
                print(f"[{batch+1}/{num_batches}][{'='*(batch+1) + '.'*(num_batches-(batch+1))}]", end='\r')

        return results, true_val

    def test_average(self, data_to_use='test'):
        torch.manual_seed(self.seed)
        tests = []
        random_gen = np.random.default_rng(self.seed)
        
        for i in range(100):
            print(i)
            #i_rand = random_gen.integers(1, high=1000)
            #print(i_rand)

            torch.manual_seed(i)
            tests.append(self.test(data_to_use))
            print(tests[i])

        tests = np.array(tests)

        acc_err = np.sqrt(np.sum(tests[:,2]**2))/len(tests)
        auc_err = np.sqrt(np.sum(tests[:,2]**2))/len(tests)
        
        acc_std = np.std(tests[:,1])
        auc_std = np.std(tests[:,3])

        acc_tot_err = np.sqrt(acc_err**2 + acc_std**2)
        auc_tot_err = np.sqrt(acc_err**2 + acc_std**2)

        print(f"Average loss: {np.mean(tests[:,0]):>0.3f} {np.std(tests[:,0]):>0.3f}")
        print(f"Average ACC : {np.mean(tests[:,1]):>0.3f} {acc_tot_err:>0.3f}")
        print(f"Average AUC : {np.mean(tests[:,3]):>0.3f} {auc_tot_err:>0.3f}")
        
        return tests


    def sample_gumbel(self, shape, eps=1e-20):
        if self.seed_switch == 'low':
            torch.manual_seed(self.seed)
        U = torch.cuda.FloatTensor(shape).uniform_()
        #U = torch.FloatTensor(shape).uniform_()
        return -Variable(torch.log(-torch.log(U + eps) + eps))


    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)


    def gumbel_softmax(self, logits, temperature = 5):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
