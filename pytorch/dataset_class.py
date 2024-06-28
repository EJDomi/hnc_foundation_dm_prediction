#!/usr/bin/env python
from pathlib import Path
import random
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from scipy.ndimage import center_of_mass

import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from skimage.transform import rotate
from skimage.util import random_noise

from hnc_project import data_prep as dp
import torch
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
    def __init__(self, dataset_name='HNSCC', patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2',  edge_file='../../data/HNSCC/edge_staging/edges_112723.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', clinical_data=None, version='1', pre_transform=None, config=None):
        self.config = config 
        self.dataset_name=dataset_name
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
            self.patient_skip = [
'RADCURE-0321',
'RADCURE-0341',
'RADCURE-0346',
'RADCURE-0401',
'RADCURE-0425',
'RADCURE-0448',
'RADCURE-0457',
'RADCURE-0460',
'RADCURE-0492',
'RADCURE-0551',
'RADCURE-0575',
'RADCURE-0872',
'RADCURE-0896',
'RADCURE-0974',
'RADCURE-0988',
'RADCURE-1088',
'RADCURE-1127',
'RADCURE-1248',
'RADCURE-1251',
'RADCURE-1267',
'RADCURE-1329',
'RADCURE-1384',
'RADCURE-1423',
'RADCURE-1595',
'RADCURE-1623',
'RADCURE-1634',
'RADCURE-1659',
'RADCURE-1686',
'RADCURE-1773',
'RADCURE-1834',
'RADCURE-1921',
'RADCURE-1980',
'RADCURE-2007',
'RADCURE-2045',
'RADCURE-2064',
'RADCURE-2072',
'RADCURE-2147',
'RADCURE-2166',
'RADCURE-2245',
'RADCURE-2254',
'RADCURE-2274',
'RADCURE-2278',
'RADCURE-2304',
'RADCURE-2327',
'RADCURE-2510',
'RADCURE-2558',
'RADCURE-2565',
'RADCURE-2619',
'RADCURE-2705',
'RADCURE-2775',
'RADCURE-2787',
'RADCURE-2821',
'RADCURE-2957',
'RADCURE-2983',
'RADCURE-3009',
'RADCURE-3057',
'RADCURE-3074',
'RADCURE-3076',
'RADCURE-3088',
'RADCURE-3112',
'RADCURE-3130',
'RADCURE-3141',
'RADCURE-3306',
'RADCURE-3388',
'RADCURE-3394',
'RADCURE-3413',
'RADCURE-3438',
'RADCURE-3571',
'RADCURE-3594',
'RADCURE-3611',
'RADCURE-3618',
'RADCURE-3659',
'RADCURE-3682',
'RADCURE-3693',
'RADCURE-3714',
'RADCURE-3720',
'RADCURE-3742',
'RADCURE-3753',
'RADCURE-3786',
'RADCURE-3795',
'RADCURE-3808',
'RADCURE-3851',
'RADCURE-3853',
'RADCURE-3862',
'RADCURE-3898',
'RADCURE-3900',
'RADCURE-3907',
'RADCURE-4011',
'RADCURE-4075',
                ]
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
        self.patch_path = Path(patch_dir)
        self.data_path = Path(f"../../data/{self.dataset_name}")
        self.edge_dict = pd.read_pickle(edge_file)
        self.locations = pd.read_pickle(locations_file)
        self.pdir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{edge_file.split('/')[-1].replace('.pkl', '')}_{version}")
        self.patients = [pat.as_posix().split('/')[-1] for pat in sorted(self.patch_path.glob('*/')) if '.pkl' not in str(pat)]
        #self.patients = [pat.as_posix().split('/')[-1] for pat in self.patch_path.glob('*/') if '.pkl' not in str(pat)]
        self.patients = [pat for pat in self.patients if pat not in self.patient_skip]
        self.years = 2

        self.rng_noise = np.random.default_rng(42)
        self.rng_rotate = np.random.default_rng(42)
           
        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(clinical_data)
        else:
            self.clinical_features = None

        if self.dataset_name == 'Combined': 
            labels_hnscc = dp.retrieve_patients(self.data_path, dataset='HNSCC')
            labels_utsw = dp.retrieve_patients(self.data_path, dataset='UTSW_HNC')
            y_hnscc = labels_hnscc.loc[[pat for pat in self.patients if 'HNSCC' in pat]]
            y_hnscc = y_hnscc['has_dm'] & (y_hnscc['survival_dm'] < self.years)
            
            y_utsw = labels_utsw.loc[[pat for pat in self.patients if 'HNSCC' not in pat]]
            y_utsw = y_utsw.notna() & (y_utsw < self.years) & (y_utsw > 0)
            self.y = pd.concat([y_hnscc, y_utsw])
        else:
            labels = dp.retrieve_patients(self.data_path, dataset=self.dataset_name)
        if self.dataset_name == 'HNSCC':
            self.y_source = labels.loc[self.patients]
            self.y = self.y_source['has_dm'] & (self.y_source['survival_dm'] < self.years)
            #self.y = self.y_source['has_lr'] & (self.y_source['survival_lr'] < self.years)
            #self.y = self.y_source['has_dm']
        elif self.dataset_name in ['UTSW_HNC', 'RADCURE']:
            self.y_source = labels.loc[self.patients]

            if self.config['challenge'] == True:
                self.y_challenge = self.y_source[(self.y_source['RADCURE-challenge'] == 'training') | (self.y_source['RADCURE-challenge'] == 'test')]
                self.patients = list(self.y_challenge.index)
                self.y = self.y_challenge['survival_dm'].notna() & (self.y_challenge['survival_dm'] < self.years) & (self.y_challenge['survival_dm'] > 0)

            else:
                self.y = self.y_source['survival_dm'].notna() & (self.y_source['survival_dm'] < self.years) & (self.y_source['survival_dm'] > 0)
            
           

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
                    aug_rot_pats.index = aug_pats.index + f"_rotation_{rot}"
                    self.patients.extend(aug_rot_pats.index)
                    
                    self.y = pd.concat([self.y, aug_rot_pats])
                if self.config['positive_increase'] > 0:
                    for rot in range(self.config['positive_increase']):
                        aug_rot_pats = aug_pos_pats.copy(deep=True)
                        aug_rot_pats.index = aug_pos_pats.index + f"_rotation_pos_{rot}"
                        self.patients.extend(aug_rot_pats.index)
                        self.y = pd.concat([self.y, aug_rot_pats])
                    
                #aug_rot_pats = aug_pats.copy(deep=True)
                #aug_rot_pats.index = aug_pats.index + f"_rotation_0"
                #self.patients.extend(aug_rot_pats.index)
                
                #self.y = pd.concat([self.y, aug_rot_pats])
                

            if 'noise' in self.config['augments']:
                aug_noise_pats = aug_pats.copy(deep=True)
                aug_noise_pats.index = aug_pats.index + f"_noise"
                self.patients.extend(aug_noise_pats.index)
                self.y = pd.concat([self.y, aug_noise_pats])

            if 'flip' in self.config['augments']:
                aug_flip_pats = aug_pats.copy(deep=True)
                aug_flip_pats.index = aug_pats.index + f"_flip"
                self.patients.extend(aug_flip_pats.index)
                self.y = pd.concat([self.y, aug_flip_pats])
          
             


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
            edge_idx_map = {}
            #patches = list(self.patch_path.joinpath(pat).glob('image*.nii.gz'))
            #patches = list(reversed(sorted(self.patch_path.joinpath(pat).glob('image*.nii.gz'))))
            patches = list(sorted(self.patch_path.joinpath(pat).glob('image*.nii.gz')))

            # reorder patches glob so that GTVp will always be first entry (if it exists) (and so will always have an index of 0 in the graph)
            #if np.any(['GTVp' in str(l) for l in patches]):
            #    patches_reorder = patches[-1:]
            #    patches_reorder.extend(patches[:-1])
            #    patches = patches_reorder
            #if not np.any(['GTVp' in str(patch) for patch in patches]):
            #    print(f"skipping {pat}")
            #    continue
            if 'rotation' in full_pat:
                angle = self.rng_rotate.integers(-30, high=30)
            patch_list = []
            for i, patch in enumerate(patches):
                patch_name = '_'.join(patch.as_posix().split('/')[-1].split('_')[1:]).replace('.nii.gz','')
                if 'GTVp' not in patch_name:
                    continue
                if 'GTVp2' in patch_name:
                    continue
                #if i > 0: continue
                patch_list.append(patch_name)

                edge_idx_map[patch_name] = i
                patch_image = sitk.ReadImage(patch)
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

                if self.pre_transform is not None:
                    if self.pre_transform == 'MinMax':
                        patch_std = (patch_array - 500.) / 1000.
                        patch_scaled = patch_std * (1 - (-1)) + (-1)
                        patch_scaled = np.where(struct_array, patch_scaled, 0)

                    elif self.pre_transform == 'ZScore':
                        patch_scaled = (patch_array - patch_array.mean()) / patch_array.std()
                        patch_scaled = np.where(struct_array, patch_scaled, 0)

                    else:
                        raise Exception(f"pre_transform is set to {self.pre_transform}, but it is not implemented")
                else:
                    patch_scaled = patch_array

                if 'rotation' in full_pat:
                    patch_scaled = self.apply_rotation(patch_scaled, angle)
                    struct_array = self.apply_rotation(struct_array, angle)

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

            graph_array = np.array(graph_array)

            graph_array = torch.tensor(graph_array, dtype=torch.float)

            
            #graph_array = torch.permute(graph_array, (3, 0, 1, 2))
            node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patch_list]))
            if self.config['use_clinical']: 
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None
            #if len(self.edge_dict[pat]) == 0:
            if len(patch_list) == 1:
                if self.config['with_edge_attr']:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), edge_attr=torch.tensor([[0.]]), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=full_pat)
                else:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=full_pat)
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
                data = Data(x=graph_array, edge_index=edges.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=full_pat)
                #data = Data(x=graph_array, edge_index=full_edges_ten.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=pat)


            if self.config['with_edge_attr'] and len(patch_list) != 1:
                sph_transform = T.Spherical()
                norm_transform = T.Cartesian()
                dist_transform = T.Distance()
                #data = sph_transform(data) 
                data = dist_transform(data) 
                #data = norm_transform(data) 
            

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


    def apply_rotation(self, arr, angle):
        arr = rotate(arr, angle, preserve_range=True)
        return arr



    def apply_flip(self, arr):
        arr = np.flip(arr, axis=(0,1,2)).copy()
        return arr



class DatasetGeneratorBoth(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2', radiomics_dir='../../data/HNSCC/radiomics',  edge_file='../../data/HNSCC/edge_staging/edges_112723.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', clinical_data=None, version='1', pre_transform=None, config=None):
        self.config = config 
        self.patch_path = Path(patch_dir)
        self.data_path = Path('../../data/HNSCC')
        self.radiomics_path = Path(radiomics_dir)
        self.edge_dict = pd.read_pickle(edge_file)
        self.locations = pd.read_pickle(locations_file)
        self.pdir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{edge_file.split('/')[-1].replace('.pkl', '')}_{version}")
        self.patients = [pat.as_posix().split('/')[-1] for pat in self.patch_path.glob('*/')]
        self.years = 2

        self.rng_noise = np.random.default_rng(42)
        self.rng_rotate = np.random.default_rng(42)
           
        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(clinical_data)
        else:
            self.clinical_features = None
 
        labels = dp.retrieve_patients(self.data_path)
        y = labels.loc[self.patients]
        self.y = y['has_dm'] & (y['survival_dm'] < self.years)

        if self.config['augment']:
            aug_pos_pats = self.y[self.y==1]
            aug_pats = self.y
            if 'rotation' in self.config['augments']:
                for rot in range(self.config['n_rotations']):
                    aug_rot_pats = aug_pats.copy(deep=True)
                    aug_rot_pats.index = aug_pats.index + f"_rotation_{rot}"
                    self.patients.extend(aug_rot_pats.index)
                    
                    self.y = pd.concat([self.y, aug_rot_pats])
                if self.config['positive_increase'] > 0:
                    for rot in range(self.config['positive_increase']):
                        aug_rot_pats = aug_pos_pats.copy(deep=True)
                        aug_rot_pats.index = aug_pos_pats.index + f"_rotation_pos_{rot}"
                        self.patients.extend(aug_rot_pats.index)
                        self.y = pd.concat([self.y, aug_rot_pats])

                #aug_rot_pats = aug_pats.copy(deep=True)
                #aug_rot_pats.index = aug_pats.index + f"_rotation_0"
                #self.patients.extend(aug_rot_pats.index)
                
                #self.y = pd.concat([self.y, aug_rot_pats])
                

            if 'noise' in self.config['augments']:
                aug_noise_pats = aug_pats.copy(deep=True)
                aug_noise_pats.index = aug_pats.index + f"_noise"
                self.patients.extend(aug_noise_pats.index)
                self.y = pd.concat([self.y, aug_noise_pats])

            if 'flip' in self.config['augments']:
                aug_flip_pats = aug_pats.copy(deep=True)
                aug_flip_pats.index = aug_pats.index + f"_flip"
                self.patients.extend(aug_flip_pats.index)
                self.y = pd.concat([self.y, aug_flip_pats])
          
             


        super(DatasetGeneratorBoth, self).__init__(pre_transform=pre_transform)

    @property
    def raw_paths(self):
        return [self.raw_dir.joinpath(pat) for pat in self.patients]

    @property
    def raw_dir(self):
        return self.patch_path

    @property
    def processed_dir(self):
        return self.pdir 

    @property
    def processed_file_names(self):
        return [f"graph_{idx}.pt" for idx, pat in enumerate(self.patients)]


    def download(self):
        pass

    # Function for processing images as node features of graph (doesn't work properly ---- yet ...
    # for this function to be usable again, would need to add ResNet or similar to process the node feature image to etract features for use as node features
    def process(self):
        print("processed graph files not present, starting graph production")
        norm_filter = sitk.NormalizeImageFilter()
        for idx, full_pat in enumerate(self.patients):
            pat = full_pat.split('_')[0]
            print(f"    {full_pat}, {idx}")
            graph_array = []
            rad_array = []
            edge_idx_map = {}
            patches = list(self.patch_path.joinpath(pat).glob('image*.nii.gz'))
            radiomics = pd.read_pickle(self.radiomics_path.joinpath(f"features_{pat}.pkl"))

            # reorder patches glob so that GTVp will always be first entry (if it exists) (and so will always have an index of 0 in the graph)
            if np.any(['GTVp' in str(l) for l in patches]):
                patches_reorder = patches[-1:]
                patches_reorder.extend(patches[:-1])
                patches = patches_reorder
    
            if 'rotation' in full_pat:
                angle = self.rng_rotate.integers(-30, high=30)

            patch_list = []
            for i, patch in enumerate(patches):
                patch_name = patch.as_posix().split('/')[-1].split('_')[-1].replace('.nii.gz','')
                patch_list.append(patch_name)

                edge_idx_map[patch_name] = i
                patch_image = sitk.ReadImage(patch)
                patch_struct = sitk.ReadImage(str(patch).replace('image', 'Struct'))

                rad_array.append(np.array(list(radiomics[patch_name].values())))
                
                #Image normalization done in SimpleITK
                #patch_image_norm = norm_filter.Execute(patch_image)

                

                #Image currently given as 2-channels as image and mask
                patch_array = sitk.GetArrayFromImage(patch_image)
                struct_array = sitk.GetArrayFromImage(patch_struct)

                if self.pre_transform is not None:
                    if self.pre_transform == 'MinMax':
                        patch_std = (patch_array - 500) / (500 - (-500))
                        patch_scaled = patch_std * (1 - (-1)) + (-1)


                if 'rotation' in full_pat:
                    patch_scaled = self.apply_rotation(patch_scaled, angle)
                    struct_array = self.apply_rotation(struct_array, angle)

                if 'noise' in full_pat:
                    patch_scaled = self.apply_noise(patch_scaled)

                if 'flip' in full_pat:
                    patch_scaled = self.apply_flip(patch_scaled)
                    struct_array = self.apply_flip(struct_array)


                node_image = np.stack((patch_scaled, struct_array))
                #node_image = np.moveaxis(node_image, [0, 1, 2, 3], [-1, -4, -3, -2]) 
                print(f"        {patch_name}")
                print(f"        {np.shape(node_image)}")
                graph_array.append(node_image)

            graph_array = np.array(graph_array)
            rad_array = np.array(rad_array)

            graph_array = torch.tensor(graph_array, dtype=torch.float)
            rad_array = torch.tensor(rad_array, dtype=torch.float)

            #graph_array = torch.permute(graph_array, (3, 0, 1, 2))
            node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patch_list]))
            if self.config['use_clinical']: 
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None
            if len(self.edge_dict[pat]) == 0:
                if self.config['with_edge_attr']:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), edge_attr=torch.tensor([[0.]]), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, radiomics=rad_array)
                else:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, radiomics=rad_array)
            else:
                edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv, gtv2 in self.edge_dict[pat]], dtype=torch.int64)
                #edges_op = torch.tensor([[edge_idx_map[gtv2], edge_idx_map[gtv]] for gtv, gtv2 in self.edge_dict[pat]], dtype=torch.int64)
                #edges = torch.cat((edges, edges_op), 0)
                data = Data(x=graph_array, edge_index=edges.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, radiomics=rad_array)


            if self.config['with_edge_attr'] and len(self.edge_dict[pat]) != 0:
                sph_transform = T.Spherical()
                norm_transform = T.Cartesian()
                dist_transform = T.Distance()
                #data = sph_transform(data) 
                data = dist_transform(data) 
                #data = norm_transform(data) 
            

            torch.save(data, self.processed_dir.joinpath(f"graph_{idx}.pt"))
        

    def len(self):
        return len(self.patients)


    def get(self, idx):
        data = torch.load(self.processed_dir.joinpath(f"graph_{idx}.pt"))
        return data


    def apply_noise(self, arr):
        return random_noise(arr, mode='gaussian', seed=self.rng_noise)


    def apply_rotation(self, arr, angle):
        arr = rotate(arr, angle, preserve_range=True)
        return arr


    def apply_flip(self, arr):
        arr = np.flip(arr, axis=(0,1,2)).copy()
        return arr


class DatasetGeneratorRadiomics(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2', radiomics_dir='../../data/HNSCC/radiomics',  edge_file='../../data/HNSCC/edge_staging/edges_112723.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', clinical_data=None, version='1', pre_transform=None, config=None):
       
        self.config = config 
        self.patch_path = Path(patch_dir)
        self.radiomics_path = Path(radiomics_dir)
        self.data_path = Path('../../data/HNSCC')
        self.edge_dict = pd.read_pickle(edge_file)
        self.locations = pd.read_pickle(locations_file)
        self.pdir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{edge_file.split('/')[-1].replace('.pkl', '')}_{version}")
        self.patients = [pat.as_posix().split('/')[-1] for pat in self.patch_path.glob('*/')]
        self.years = 2

        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(clinical_data)
        else:
            self.clinical_features = None

        labels = dp.retrieve_patients(self.data_path)
        y = labels.loc[self.patients]
        self.y = y['has_dm'] & (y['survival_dm'] < self.years)
        #self.y = y['has_dm']

        super(DatasetGeneratorRadiomics, self).__init__(pre_transform=pre_transform)

    @property
    def raw_paths(self):
        return [self.raw_dir.joinpath(pat) for pat in self.patients]

    @property
    def raw_dir(self):
        return self.patch_path

    @property
    def processed_dir(self):
        return self.pdir 

    @property
    def processed_file_names(self):
        return [f"graph_{idx}.pt" for idx, pat in enumerate(self.patients)]


    def download(self):
        pass

    # function for using radiomics as node features of the graph.
    def process(self):
        print("processed graph files not present, starting graph production")
        for idx, pat in enumerate(self.patients):
            print(f"    {pat}, {idx}")
            graph_array = []
            edge_idx_map = {} 
 
            patches = pd.read_pickle(self.radiomics_path.joinpath(f"features_{pat}.pkl"))

            for i, patch in enumerate(patches.keys()):
                edge_idx_map[patch] = i

                node_features = np.array(list(patches[patch].values()))
                print(f"        {patch}")
                graph_array.append(node_features)

            graph_array = np.array(graph_array)
            if self.pre_transform is not None:
                graph_array = self.pre_transform.transform(graph_array)

            if self.config['use_clinical']: 
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None
            x = torch.tensor(graph_array, dtype=torch.float)
            if len(self.edge_dict[pat]) == 0:
                node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patches.keys()]))
                if self.config['with_edge_attr']:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), edge_attr=torch.tensor([[0.]]), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical)
                else:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical)
            else:
                edges = torch.tensor(np.array([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv, gtv2 in self.edge_dict[pat]]), dtype=torch.int64)
                node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patches.keys()]))
                data = Data(x=x, edge_index=edges.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical)

            if self.config['with_edge_attr'] and len(self.edge_dict[pat]) != 0:
                sph_transform = T.Spherical()
                norm_transform = T.Cartesian()
                dist_transform = T.Distance()
                #data = sph_transform(data) 
                #data = dist_transform(data) 
                data = norm_transform(data) 
            

            torch.save(data, self.processed_dir.joinpath(f"graph_{idx}.pt"))


    def len(self):
        return len(self.patients)


    def get(self, idx):
        data = torch.load(self.processed_dir.joinpath(f"graph_{idx}.pt"))
        return data
