#!/usr/bin/env python
from pathlib import Path
import random, datetime
from math import floor
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from scipy.ndimage import center_of_mass, rotate

import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.util import random_noise

from hnc_foundation_dm_prediction import data_prep as dp
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
import torch_geometric.transforms as T

VALID_DATASETS = [
'HNSCC',
'UTSW_HNC',
'RADCURE',
'Combined',
]

class DatasetGeneratorImage(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, config, pre_transform=None, transform=None):
        self.config = config 
        self.dataset_name=self.config['dataset_name']
        if self.dataset_name == 'RADCURE':
            self.data_path = Path('../../data/RADCURE')
            self.patient_skip = pd.read_pickle(self.data_path.joinpath(self.config['patch_dir']).joinpath(self.config['no_GTVp_pickle']).as_posix())
        else:
            raise ValueError(f"need to set a dataset to use, valid ones include: {VALID_DATASETS}")

        self.data_path = Path(f"../../data/{self.dataset_name}")
        self.patch_path = self.data_path.joinpath(self.config['patch_dir'])
        self.edge_dict = pd.read_pickle(self.data_path.joinpath(f"edge_staging/{self.config['edge_file']}").as_posix())
        self.locations = pd.read_pickle(self.data_path.joinpath(f"edge_staging/{self.config['locations_file']}").as_posix())
        self.pdir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{self.config['edge_file'].replace('.pkl', '')}_{self.config['data_version']}")
        self.patients = [pat.as_posix().split('/')[-1] for pat in sorted(self.patch_path.glob('*/')) if '.pkl' not in str(pat)]
        #self.patients = [pat.as_posix().split('/')[-1] for pat in self.patch_path.glob('*/') if '.pkl' not in str(pat)]
        self.patients = [pat for pat in self.patients if pat not in self.patient_skip]
        self.years = 2

        self.rng_noise = np.random.default_rng(42)
        self.rng_rotate = np.random.default_rng(42)
        self.rng_rotate_axis = np.random.default_rng(42)
           
        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(self.data_path.joinpath(self.config['clinical_data']).as_posix())
        else:
            self.clinical_features = None

        labels = dp.retrieve_patients(self.data_path.as_posix(), dataset=self.dataset_name)
        self.y_source = labels.loc[self.patients]

        if self.config['remove_censored']:
            self.y_processing = self.y_source[(self.y_processing['last_fu_days'] >= datetime.timedelta(days = 2*365)) | (self.y_processing['survival_dm_yrs'] > 0)]

        if self.config['challenge'] == True:
            self.y_processing = self.y_processing[(self.y_processing['RADCURE-challenge'] == 'training') | (self.y_processing['RADCURE-challenge'] == 'test')]

        self.patients = list(self.y_processing.index)

        #self.y = self.y_processing['survival_dm_yrs'].notna() & (self.y_processing['survival_dm_yrs'] < self.years) & (self.y_processing['survival_dm_yrs'] > 0)
        self.dm = (self.y_processing['survival_dm_yrs'].notna() & (self.y_processing['survival_dm_yrs'] < self.years) & (self.y_processing['survival_dm_yrs'] > 0)).rename('dm_2year')
        self.lm = (self.y_processing['survival_lm_yrs'].notna() & (self.y_processing['survival_lm_yrs'] < self.years) & (self.y_processing['survival_lm_yrs'] > 0)).rename('lm_2year')
        self.rm = (self.y_processing['survival_rm_yrs'].notna() & (self.y_processing['survival_rm_yrs'] < self.years) & (self.y_processing['survival_rm_yrs'] > 0)).rename('rm_2year')
        self.death = (self.y_processing['survival_death_yrs'].notna() & (self.y_processing['survival_death_yrs'] < self.years) & (self.y_processing['survival_death_yrs'] > 0) & (self.y_processing['Cause of Death'].str.contains('Index'))).rename('death_2year')
        self.any_rec = (self.dm | self.lm | self.rm | self.death).rename('os_2year')

        self.y = pd.concat([self.dm, self.lm, self.rm, self.death, self.any_rec], axis=1)

        if self.config['augment']:
            aug_pos_pats = self.y[(self.y['dm_2year']==1)]
            aug_pats = self.y
               
            if 'rotation' in self.config['augments']:
                for rot in range(self.config['n_rotations']):
                    aug_rot_pats = aug_pats.copy(deep=True)
                    aug_rot_pats.index = aug_pats.index + f"_rotation_{rot+1}"
                    self.patients.extend(aug_rot_pats.index)
                    
                    self.y = pd.concat([self.y, aug_rot_pats])
                if self.config['balance_classes']:
                    ratio_classes = int(floor(len(self.y[self.y==0]) / len(self.y[self.y==1])))
                    for rot in range(ratio_classes):
                        aug_rot_pats = aug_pos_pats.copy(deep=True)
                        aug_rot_pats.index = aug_pos_pats.index + f"_rotation_pos_{rot+1}"
                        self.patients.extend(aug_rot_pats.index)
                        self.y = pd.concat([self.y, aug_rot_pats])
                    


        super(DatasetGeneratorImage, self).__init__(pre_transform=pre_transform)

    @property
    def raw_paths(self):
        return [f"{self.raw_dir}/{pat}" for pat in self.patients]

    @property
    def raw_dir(self):
        return str(self.patch_path)

    @property
    def processed_dir(self):
        return str(self.pdir)

    @property
    def processed_file_names(self):
        return [f"graph_{idx}_{pat}.pt" for idx, pat in enumerate(self.patients)]


    def download(self):
        pass


    def process(self):
        print("processed graph files not present, starting graph production")
        norm_filter = sitk.NormalizeImageFilter()
        idx = 0
        for full_pat in tqdm(self.patients):
            pat = full_pat.split('_')[0]
            if pat in self.patient_skip: continue
            graph_nx = self.edge_dict[pat]
            graph_array = []
            patches = list(sorted(self.patch_path.joinpath(pat).glob('image*.nii.gz')))

            if 'rotation' in full_pat:
                angle = self.rng_rotate.integers(-30, high=30)
                rotate_axes = self.rng_rotate_axis.integers(0, high=3)
            patch_list = []
            for i, patch in enumerate(patches):
                patch_name = '_'.join(patch.as_posix().split('/')[-1].split('_')[1:]).replace('.nii.gz','')
                patch_list.append(patch_name)

                patch_image = sitk.ReadImage(patch.as_posix())
                patch_struct = sitk.ReadImage(str(patch).replace('image', 'Struct'))

                #Image currently given as 2-channels as image and mask
                patch_array = sitk.GetArrayFromImage(patch_image)
                struct_array = sitk.GetArrayFromImage(patch_struct)

                if self.config['scaling_type'] is not None:
                    if self.config['scaling_type'] == 'MinMax':
                        #patch_std = (patch_array - (-500.) / 1000.
                        #patch_scaled = patch_std * (1 - (-1)) + (-1)
                        patch_scaled = patch_array / 500.
                        patch_scaled = np.where(struct_array, patch_scaled, 0)

                    elif self.config['scaling_type'] == 'ZScore':
                        patch_scaled = (patch_array - patch_array.mean()) / patch_array.std()
                        patch_scaled = np.where(struct_array, patch_scaled, 0)

                    else:
                        raise Exception(f"scaling is set to {self.config['scaling_type']}, but it is not implemented")
                else:
                    patch_scaled = patch_array

                if 'rotation' in full_pat:
                    patch_scaled = self.apply_rotation(patch_scaled, angle, rotate_axes)
                    struct_array = self.apply_rotation(struct_array, angle, rotate_axes)

                if 'noise' in full_pat:
                    patch_scaled = self.apply_noise(patch_scaled)

                if 'flip' in full_pat:
                    patch_scaled = self.apply_flip(patch_scaled)
                    struct_array = self.apply_flip(struct_array)

                
                graph_nx.nodes[patch_name]['pos'] = torch.tensor(np.array(self.locations[pat][patch_name]), dtype=torch.float)
                graph_nx.nodes[patch_name]['x'] = torch.tensor(np.expand_dims(patch_scaled, 0), dtype=torch.float)
                #graph_nx.nodes[patch_name]['y1'] = torch.tensor(np.expand_dims(struct_array, 0), dtype=torch.float)

            if self.config['use_clinical']:
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None

            graph_nx['y'] = torch.tensor([int(self.y[pat])])
            graph_nx['clinical'] = clinical
            graph_nx['patients'] = full_pat

            graph_pyg = from_networkx(graph_nx)

            if self.config['with_edge_attr'] and len(graph_nx.nodes) != 1:
                dist_transform = T.Cartesian()
                graph_pyg = dist_transform(graph_pyg)
            elif len(graph_nx.nodes) == 1:
                graph_pyg.edge_attr = graph_nx['pos'] - graph_nx['pos']
            
            torch.save(graph_pyg, f"{self.processed_dir}/graph_{idx}_{full_pat}.pt")
            #torch.save(data, f"{self.processed_dir}/graph_{idx}.pt")
            idx += 1
        

    def len(self):
        return len(self.patients)


    def get(self, idx):
        pat = self.patients[idx]
        data = torch.load(f"{self.processed_dir}/graph_{idx}_{pat}.pt")
        #data = torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        return data


    def apply_noise(self, arr):
        return random_noise(arr, mode='gaussian', seed=self.rng_noise)


    def apply_rotation(self, arr, angle, rotate_axes):
        axis_tuples = [(0,1), (0,2), (1,2)]
        arr = rotate(arr, angle, axes=axis_tuples[rotate_axes], reshape=False)
        return arr



    def apply_flip(self, arr):
        arr = np.flip(arr, axis=(0,1,2)).copy()
        return arr



