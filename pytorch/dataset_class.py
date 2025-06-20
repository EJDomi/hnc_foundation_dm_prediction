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
from torch.utils.data import Sampler
from torchmtlr.utils import make_time_bins, encode_survival
from torch_geometric.data import Dataset, Data
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
        if self.dataset_name == 'HNSCC':
            self.data_path = Path('../../data/HNSCC')
            self.patient_skip = [ 
                'HNSCC-01-0271',
                'HNSCC-01-0379',
                ]
        elif self.dataset_name == 'UTSW_HNC':
            self.data_path = Path('../../data/UTSW_HNC')
            self.patient_skip = [
                '91352703',
                '91486155',
                '92910065',
                '91333995',]
        elif self.dataset_name == 'RADCURE':
            self.data_path = Path('../../data/RADCURE')
            self.patient_skip = pd.read_pickle(self.data_path.joinpath(self.config['patch_dir']).joinpath(self.config['no_GTVp_pickle']).as_posix())
        elif self.dataset_name == 'Combined':
            self.data_path = Path('../../data/Combined')
            self.patient_skip = [
                'HNSCC-01-0271',
                'HNSCC-01-0379',
                '91352703',
                '91486155',
                '92910065',
                '91333995',]
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
        if self.config['cut_on_ngtvs']:
            self.patients = [pat for pat in self.patients if len(list(self.patch_path.joinpath(pat).glob('*.gz'))) / 2 <= 10]

        self.years = 2

        self.rng_noise = np.random.default_rng(42)
        self.rng_rotate = np.random.default_rng(42)
        self.rng_rotate_axis = np.random.default_rng(42)
           
        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(self.data_path.joinpath(self.config['clinical_data']).as_posix())
        else:
            self.clinical_features = None

        if self.dataset_name == 'Combined': 
            labels_hnscc = dp.retrieve_patients(self.data_path.as_posix(), dataset='HNSCC')
            labels_utsw = dp.retrieve_patients(self.data_path.as_posix(), dataset='UTSW_HNC')
            y_hnscc = labels_hnscc.loc[[pat for pat in self.patients if 'HNSCC' in pat]]
            y_hnscc = y_hnscc['has_dm'] & (y_hnscc['survival_dm_yrs'] < self.years)
            
            y_utsw = labels_utsw.loc[[pat for pat in self.patients if 'HNSCC' not in pat]]
            y_utsw = y_utsw.notna() & (y_utsw < self.years) & (y_utsw > 0)
            self.y = pd.concat([y_hnscc, y_utsw])
        else:
            labels = dp.retrieve_patients(self.data_path.as_posix(), dataset=self.dataset_name)
        if self.dataset_name == 'HNSCC':
            self.y_source = labels.loc[self.patients]
            self.y = self.y_source['has_dm'] & (self.y_source['survival_dm_yrs'] < self.years)
            #self.y = self.y_source['has_lr'] & (self.y_source['survival_lr'] < self.years)
            #self.y = self.y_source['has_dm']
        elif self.dataset_name in ['UTSW_HNC', 'RADCURE']:
            self.y_source = labels.loc[self.patients]

            if self.config['challenge'] == True:
                self.y_challenge = self.y_source[(self.y_source['RADCURE-challenge'] == 'training') | (self.y_source['RADCURE-challenge'] == 'test')]
                if self.config['remove_censored']:
                    self.y_challenge = self.y_challenge[(self.y_challenge['last_fu_yrs'] >= self.years) | (self.y_challenge['survival_dm_yrs'] > 0)]
                self.patients = list(self.y_challenge.index)
                self.y = self.y_challenge['survival_dm_yrs'].notna() & (self.y_challenge['survival_dm_yrs'] < self.years) & (self.y_challenge['survival_dm_yrs'] > 0)

            else:
                if self.config['remove_censored']:
                    self.y_nocensor = self.y_source[(self.y_source['last_fu_yrs'] >= 2) | (self.y_source['survival_dm_yrs'] > 0)]
                    self.patients = list(self.y_nocensor.index)
                    self.y = self.y_nocensor['survival_dm_yrs'].notna() & (self.y_nocensor['survival_dm_yrs'] < self.years) & (self.y_nocensor['survival_dm_yrs'] > 0)
                else:
                    self.patients = list(self.y_source.index)
                    self.y = self.y_source['survival_dm_yrs'].notna() & (self.y_source['survival_dm_yrs'] < self.years) & (self.y_source['survival_dm_yrs'] > 0)

                self.events = self.y_source.copy(deep=True)
                self.events['lm_event'] = (self.events['survival_lm_yrs'].notna()) & (self.events['survival_lm_yrs'] > 0)
                self.events['lm_time'] = self.events.survival_lm_yrs.fillna(self.events.last_fu_yrs)
                self.events['rm_event'] = (self.events['survival_rm_yrs'].notna()) & (self.events['survival_rm_yrs'] > 0)
                self.events['rm_time'] = self.events.survival_rm_yrs.fillna(self.events.last_fu_yrs) 
                self.events['dm_event'] = (self.events['survival_dm_yrs'].notna()) & (self.events['survival_dm_yrs'] > 0)
                self.events['dm_time'] = self.events.survival_dm_yrs.fillna(self.events.last_fu_yrs)
                self.events['death_event'] = (self.events['survival_death_yrs'].notna()) & (self.events['survival_death_yrs'] > 0) * (self.events['Cause of Death'].str.contains('Index'))
                self.events['death_time'] = self.events.survival_death_yrs.fillna(self.events.last_fu_yrs)
                self.events.drop(columns=['Cause of Death',
                                      'survival_lm_yrs',
                                      'survival_rm_yrs',
                                      'survival_dm_yrs',
                                      'survival_death_yrs',
                                      'last_fu_yrs',
                                      'RADCURE-challenge',
                                      ])
                self.time_bins = {}
                self.survival = {}
                self.bin_names = [f'{self.config["survival_types"][0]}_bin_{idx}' for idx in range(self.config['time_bins']+1)] 
                for event in ['lm', 'rm', 'dm', 'death']:
                    self.time_bins[f'{event}'] = make_time_bins(times=self.events[f'{event}_time'], num_bins=self.config['time_bins'], event=self.events[f'{event}_event'])
                    self.survival[f'{event}_y'] = encode_survival(self.events[f'{event}_time'].values, self.events[f'{event}_event'].values, self.time_bins[f'{event}'])
                    self.events[[f'{event}_bin_{idx}' for idx in range(self.config['time_bins']+1)]] = encode_survival(self.events[f'{event}_time'].values, self.events[f'{event}_event'].values, self.time_bins[f'{event}']).numpy()
                    


           

        if self.config['augment']:
            if self.config['challenge']:
                aug_pos_pats = self.y[(self.y==1) & (self.y_challenge['RADCURE-challenge'] == 'training')]
                aug_pats = (self.y) & (self.y_challenge['RADCURE-challenge'] == 'training')
               
            else:
                aug_pos_pats = self.y[self.y==1]
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
                if self.config['true_balance_classes']:
                    ratio_classes = int(floor(len(self.y[self.y==0]) / len(self.y[self.y==1])))
                    for rot in range(ratio_classes-1):
                        aug_rot_pats = aug_pos_pats.copy(deep=True)
                        aug_rot_pats.index = aug_pos_pats.index + f"_rotation_pos_{rot+1}"
                        self.patients.extend(aug_rot_pats.index)
                        self.y = pd.concat([self.y, aug_rot_pats])

        if self.config['store_radiomics']:
            self.radiomics_data = None
            rad_file_list = sorted(list(self.data_path.joinpath('radiomics').glob('radiomics_*.csv')),
                                   key=lambda f: int(str(f).split('/')[-1].split('_')[-1].strip('.csv')))
            print("extracting radiomics from csv files")
            for idx, rad_file in tqdm(list(enumerate(rad_file_list))):
                rad_tmp = pd.read_csv(rad_file)
                rad_tmp[['patient_id', 'struct']] = rad_tmp['Unnamed: 0'].str.split('__', expand=True)
                rad_tmp.drop(columns=['Unnamed: 0', 'Image', 'Mask'], inplace=True)
                rad_tmp.set_index(['patient_id', 'struct'], inplace=True)
                rad_tmp.drop(columns=[col for col in rad_tmp.columns if 'diagnostics' in col], inplace=True)
            
                if idx == 0:
                    self.radiomics_data = rad_tmp
                else:
                    self.radiomics_data = pd.concat([self.radiomics_data, rad_tmp])
            if self.config['scale_radiomics']:
                rad_mean = pd.read_pickle(self.config['radiomics_mean'])
                rad_std = pd.read_pickle(self.config['radiomics_std'])

                self.radiomics_data = (self.radiomics_data - rad_mean) / rad_std
            print("done extracting radiomics")
        else:
            self.radiomics_data = None

        if self.config['store_embeddings']:
            self.embedding_data = pd.read_pickle(self.data_path.joinpath(self.config['embedding_file']).as_posix())
        else:
            self.embedding_data = None


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
            #print(f"    {full_pat}, {idx}")
            graph_array = []
            #all_struct_array = []
            edge_idx_map = {}
            #patches = list(self.patch_path.joinpath(pat).glob('image*.nii.gz'))
            #patches = list(reversed(sorted(self.patch_path.joinpath(pat).glob('image*.nii.gz'))))
            patches = list(sorted(self.patch_path.joinpath(pat).glob('image*.nii.gz')))

            if pat == 'RADCURE-2638' and np.any(['image_GTVn_[a].nii.gz' in str(p) for p in patches]):
                for p in patches:
                    if 'image_GTVn_[a].nii.gz' in p:
                        patches.remove(p)
            if pat == 'RADCURE-2638' and np.any(['image_GTVn_a.nii.gz' in str(p) for p in patches]):
                for p in patches:
                    if 'image_GTVn_a.nii.gz' in p.as_posix():
                        patches.remove(p)
            if pat == 'RADCURE-3697' and np.any(['image_GTVn_[A].nii.gz' in str(p) for p in patches]):
                for p in patches: 
                    if 'image_GTVn_[A].nii.gz' in p:
                        patches.remove(p)
            if pat == 'RADCURE-3697' and np.any(['image_GTVn_A.nii.gz' in str(p) for p in patches]):
                for p in patches: 
                    if 'image_GTVn_A.nii.gz' in p.as_posix():
                        patches.remove(p)

            # reorder patches glob so that GTVp will always be first entry (if it exists) (and so will always have an index of 0 in the graph)
            #if np.any(['GTVp' in str(l) for l in patches]):
            #    patches_reorder = patches[-1:]
            #    patches_reorder.extend(patches[:-1])
            #    patches = patches_reorder
            #if not np.any(['GTVp' in str(patch) for patch in patches]):
            #    print(f"skipping {pat}")
            #    continue
            #if 'rotation' in full_pat:
            #    angle = self.rng_rotate.integers(-180, high=180)
            if 'rotation' in full_pat:
                angle = self.rng_rotate.integers(-30, high=30)
                #sign = self.rng_rotate.uniform()
                #if sign < 0.5:
                #    angle = -angle
                rotate_axes = self.rng_rotate_axis.integers(0, high=3)
            patch_list = []
            for i, patch in enumerate(patches):
                patch_name = '_'.join(patch.as_posix().split('/')[-1].split('_')[1:]).replace('.nii.gz','')
                if self.config['use_GTVp_only']:
                    if 'GTVp' not in patch_name:
                        continue
                    if 'GTVp2' in patch_name:
                        continue
                    if 'GTVp3' in patch_name:
                        continue
                    #if i > 0: continue
                patch_list.append(patch_name)

                edge_idx_map[patch_name] = i
                patch_image = sitk.ReadImage(patch.as_posix())
                patch_struct = sitk.ReadImage(str(patch).replace('image', 'Struct'))

                #Image normalization done in SimpleITK
                #patch_image_norm = norm_filter.Execute(patch_image)

                
                #Image currently given as 2-channels as image and mask
                patch_array = sitk.GetArrayFromImage(patch_image)
                struct_array = sitk.GetArrayFromImage(patch_struct)

                #patch_array = np.where(struct_array, patch_array, 0)

                if '2d' in self.config['data_type']:
                    com = center_of_mass(struct_array)
                    patch_array = patch_array[int(com[0])]
                    struct_array = struct_array[int(com[0])]

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

                #node_image = np.stack((patch_scaled, struct_array))
                #node_image = np.moveaxis(node_image, [0, 1, 2, 3], [-1, -4, -3, -2]) 
                #print(f"        {patch_name}")
                #print(f"        {np.shape(node_image)}")
                #print(f"        {np.shape(patch_scaled)}")
                #graph_array.append(node_image)
                graph_array.append(np.expand_dims(patch_scaled, 0))
                #all_struct_array.append(np.expand_dims(struct_array, 0))

            graph_array = np.array(graph_array)
            #all_struct_array = np.array(all_struct_array)

            graph_array = torch.tensor(graph_array, dtype=torch.float)
            #all_struct_array = torch.tensor(all_struct_array, dtype=torch.float)

            if self.config['store_radiomics']:
                radiomics = torch.tensor([self.radiomics_data.loc[(pat, gtv)].values for gtv in patch_list], dtype=torch.float)
            else:
                radiomics = None

            if self.config['store_embeddings']:
                embeddings = torch.tensor(np.array([self.embedding_data.loc[(pat, gtv), 'embedding'] for gtv in patch_list]), dtype=torch.float)
            else:
                embeddings = None
            #graph_array = torch.permute(graph_array, (3, 0, 1, 2))
            node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patch_list]))
            if self.config['use_clinical']:
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None

            #if len(self.edge_dict[pat]) == 0:
            if len(patch_list) == 1:
                if self.config['with_edge_attr']:
                    data = Data(x=graph_array, 
                                edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), 
                                edge_attr=torch.tensor([[0.]]), 
                                pos=node_pos, 
                                y=torch.tensor([int(self.y[pat])], dtype=torch.float), 
                                clinical=clinical, 
                                patient=full_pat, 
                                radiomics=radiomics, 
                                embeddings=embeddings,
                                survival=torch.tensor([self.events.loc[pat, self.bin_names]]), 
                                event=torch.tensor([self.events.loc[pat, [f'{self.config["survival_types"][0]}_event', f'{self.config["survival_types"][0]}_time']]]))
                else:
                    data = Data(x=graph_array, 
                                edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), 
                                pos=node_pos, 
                                y=torch.tensor([int(self.y[pat])], dtype=torch.float), 
                                clinical=clinical, 
                                patient=full_pat, 
                                radiomics=radiomics, 
                                embeddings=embeddings, 
                                survival=torch.tensor([self.events.loc[pat, self.bin_names]]), 
                                event=torch.tensor([self.events.loc[pat, [f'{self.config["survival_types"][0]}_event', f'{self.config["survival_types"][0]}_time']]]))
            else:
                if self.config['reverse_edges']:
                    edges = torch.tensor([[edge_idx_map[gtv2], edge_idx_map[gtv]] for gtv, gtv2 in self.edge_dict[pat] if gtv in patch_list and gtv2 in patch_list], dtype=torch.int64)
                elif self.config['complete_graph']:
                    edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv in patch_list for gtv2 in patch_list if gtv != gtv2], dtype=torch.int64)
                elif self.config['undirected_graph']:
                    edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv, gtv2 in self.edge_dict[pat] if gtv in patch_list and gtv2 in patch_list], dtype=torch.int64)
                    edges2 = torch.tensor([[edge_idx_map[gtv2], edge_idx_map[gtv]] for gtv, gtv2 in self.edge_dict[pat] if gtv in patch_list and gtv2 in patch_list], dtype=torch.int64)
                    edges = torch.cat((edges, edges2), 0)
                elif self.config['star_graph']:
                    edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map['GTVp']] for gtv in patch_list], dtype=torch.int64)
                else:
                    edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv, gtv2 in self.edge_dict[pat] if gtv in patch_list and gtv2 in patch_list], dtype=torch.int64)
                #full_edges = []
                #for gtv in patch_list:
                #    for gtv2 in patch_list:
                #        if gtv == gtv2: continue
                #        full_edges.append([edge_idx_map[gtv], edge_idx_map[gtv2]])
                #full_edges_ten = torch.tensor(full_edges, dtype=torch.int64)
                #edges_op = torch.tensor([[edge_idx_map[gtv2], edge_idx_map[gtv]] for gtv, gtv2 in self.edge_dict[pat]], dtype=torch.int64)
                #edges = torch.cat((edges, edges_op), 0)
                data = Data(x=graph_array, 
                            edge_index=edges.t().contiguous(), 
                            pos=node_pos, 
                            y=torch.tensor([int(self.y[pat])], dtype=torch.float), 
                            clinical=clinical, 
                            patient=full_pat,
                            radiomics=radiomics, 
                            embeddings=embeddings, 
                            survival=torch.tensor([self.events.loc[pat, self.bin_names]]), 
                            event=torch.tensor([self.events.loc[pat, [f'{self.config["survival_types"][0]}_event', f'{self.config["survival_types"][0]}_time']]]))

                #data = Data(x=graph_array, edge_index=full_edges_ten.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=pat)


            if self.config['with_edge_attr'] and len(patch_list) != 1:
                #sph_transform = T.Spherical()
                #norm_transform = T.Cartesian()
                #dist_transform = T.Distance()
                #data = sph_transform(data) 
                data = T.Distance()(data) 
                #data = norm_transform(data) 
            #if self.config['undirected_graph']:
            #    data = T.ToUndirected()(data)

            torch.save(data, f"{self.processed_dir}/graph_{idx}_{full_pat}.pt")
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


class PositiveSampler(Sampler):
    def __init__(self, dataset, config):
        self.config = config
        self.dataset = dataset
        self.num_pats = len(dataset)
        self.positive_indices = [i for i, data in enumerate(dataset) if dataset.events.iloc[i][f"{self.config['survival_types'][0]}_event"]]
        self.negative_indices = [i for i, data in enumerate(dataset) if not dataset.events.iloc[i][f"{self.config['survival_types'][0]}_event"]]
        self.drop_last = False

    def __iter__(self):
        indices = []
        negative_fraction = (self.config['batch_size']*2) // 3
        positive_fraction = self.config['batch_size'] - negative_fraction
        for _ in range(self.num_pats // self.config['batch_size']):
            batch = np.random.choice(self.negative_indices, negative_fraction, replace=False).tolist()
            batch.extend(np.random.choice(self.positive_indices, positive_fraction, replace=False))
            np.random.shuffle(batch)
            yield batch
