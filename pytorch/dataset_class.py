#!/usr/bin/env python
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rotate
from skimage.util import random_noise

from hnc_project import data_prep as dp
import torch
from torch_geometric.data import Dataset, Data
import torch_geometric.transforms as T

import elasticdeform


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



class DatasetGeneratorImage(Dataset):
    """
    generate images for pytorch dataset
    """
    def __init__(self, patch_dir='../../data/HNSCC/HNSCC_Nii_222_50_50_60_Crop_v2',  edge_file='../../data/HNSCC/edge_staging/edges_112723.pkl', locations_file='../../data/HNSCC/edge_staging/centered_locations_010424.pkl', clinical_data=None, version='1', pre_transform=None, config=None):
        self.config = config 
        self.patch_path = Path(patch_dir)
        self.data_path = Path('../../data/HNSCC')
        self.edge_dict = pd.read_pickle(edge_file)
        self.locations = pd.read_pickle(locations_file)
        self.pdir = self.data_path.joinpath(f"graph_staging/{self.patch_path.name}_{edge_file.split('/')[-1].replace('.pkl', '')}_{version}")
        self.patients = [pat.as_posix().split('/')[-1] for pat in self.patch_path.glob('*/') if '.pkl' not in str(pat)]
        self.years = 2

        self.rng_noise = np.random.default_rng(42)
        self.rng_rotate = np.random.default_rng(42)
           
        if self.config['use_clinical']:
            self.clinical_features = pd.read_pickle(clinical_data)
        else:
            self.clinical_features = None
 
        labels = dp.retrieve_patients(self.data_path)
        self.y_source = labels.loc[self.patients]
        self.y = self.y_source['has_dm'] & (self.y_source['survival_dm'] < self.years)

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

            if 'deform' in self.config['augments']:
                for dfm in range(self.config['n_deforms']):
                    aug_deform_pats = aug_pats.copy(deep=True)
                    aug_deform_pats.index = aug_pats.index + f"_deform_{dfm}"
                    self.patients.extend(aug_deform_pats.index)
                    self.y = pd.concat([self.y, aug_deform_pats])
                if self.config['positive_increase'] > 0:
                    for dfm in range(self.config['positive_increase']):
                        aug_deform_pats = aug_pos_pats.copy(deep=True)
                        aug_deform_pats.index = aug_pos_pats.index + f"_deformation_pos_{dfm}"
                        self.patients.extend(aug_deform_pats.index)
                        self.y = pd.concat([self.y, aug_deform_pats])
                 
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
            edge_idx_map = {}
            patches = list(self.patch_path.joinpath(pat).glob('image*.nii.gz'))

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

                #Image normalization done in SimpleITK
                #patch_image_norm = norm_filter.Execute(patch_image)

                

                #Image currently given as 2-channels as image and mask
                patch_array = sitk.GetArrayFromImage(patch_image)
                struct_array = sitk.GetArrayFromImage(patch_struct)

                if self.pre_transform is not None:
                    if self.pre_transform == 'MinMax':
                        patch_std = (patch_array - patch_array.min()) / (patch_array.max() - patch_array.min())
                        patch_scaled = patch_std * (1 - (-1)) + (-1)


                if 'rotation' in full_pat:
                    patch_scaled = self.apply_rotation(patch_scaled, angle)
                    struct_array = self.apply_rotation(struct_array, angle)

                if 'noise' in full_pat:
                    patch_scaled = self.apply_noise(patch_scaled)

                if 'flip' in full_pat:
                    patch_scaled = self.apply_flip(patch_scaled)
                    struct_array = self.apply_flip(struct_array)

                if 'deform' in full_pat:
                    [patch_scaled, struct_array] = self.apply_deformation(patch_scaled, struct_array)


                node_image = np.stack((patch_scaled, struct_array))
                #node_image = np.moveaxis(node_image, [0, 1, 2, 3], [-1, -4, -3, -2]) 
                print(f"        {patch_name}")
                print(f"        {np.shape(node_image)}")
                graph_array.append(node_image)

            graph_array = np.array(graph_array)

            graph_array = torch.tensor(graph_array, dtype=torch.float)

            #graph_array = torch.permute(graph_array, (3, 0, 1, 2))
            node_pos = torch.from_numpy(np.array([self.locations[pat][gtv] for gtv in patch_list]))
            if self.config['use_clinical']: 
                clinical = torch.tensor(pd.to_numeric(self.clinical_features.loc[pat]).values, dtype=torch.float).unsqueeze(0)
            else:
                clinical = None
            if len(self.edge_dict[pat]) == 0:
                if self.config['with_edge_attr']:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), edge_attr=torch.tensor([[0.]]), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=pat)
                else:
                    data = Data(x=graph_array, edge_index=torch.tensor([[0,0]], dtype=torch.int64).t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=pat)
            else:
                edges = torch.tensor([[edge_idx_map[gtv], edge_idx_map[gtv2]] for gtv, gtv2 in self.edge_dict[pat]], dtype=torch.int64)
                #edges_op = torch.tensor([[edge_idx_map[gtv2], edge_idx_map[gtv]] for gtv, gtv2 in self.edge_dict[pat]], dtype=torch.int64)
                #edges = torch.cat((edges, edges_op), 0)
                data = Data(x=graph_array, edge_index=edges.t().contiguous(), pos=node_pos, y=torch.tensor([int(self.y[pat])], dtype=torch.float), clinical=clinical, patient=pat)


            if self.config['with_edge_attr'] and len(self.edge_dict[pat]) != 0:
                sph_transform = T.Spherical()
                norm_transform = T.Cartesian()
                dist_transform = T.Distance()
                #data = sph_transform(data) 
                data = dist_transform(data) 
                #data = norm_transform(data) 
            

            torch.save(data, f"{self.processed_dir}/graph_{idx}.pt")
        

    def len(self):
        return len(self.patients)


    def get(self, idx):
        data = torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        return data


    def apply_noise(self, arr):
        return random_noise(arr, mode='gaussian', seed=self.rng_noise)


    def apply_rotation(self, arr, angle):
        arr = rotate(arr, angle, preserve_range=True)
        return arr


    def apply_deformation(self, arr, struct):
        arr = elasticdeform.deform_random_grid([arr, struct], sigma=5, order=3, axis=[(0,1,2), (0,1,2)])
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

            if 'deform' in self.config['augments']:
                for dfm in range(self.config['n_deforms']):
                    aug_deform_pats = aug_pats.copy(deep=True)
                    aug_deform_pats.index = aug_pats.index + f"_deform_{dfm}"
                    self.patients.extend(aug_deform_pats.index)
                    self.y = pd.concat([self.y, aug_deform_pats])
                if self.config['positive_increase'] > 0:
                    for dfm in range(self.config['positive_increase']):
                        aug_deform_pats = aug_pos_pats.copy(deep=True)
                        aug_deform_pats.index = aug_pos_pats.index + f"_deformation_pos_{dfm}"
                        self.patients.extend(aug_deform_pats.index)
                        self.y = pd.concat([self.y, aug_deform_pats])
                 
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
                        patch_std = (patch_array - patch_array.min()) / (patch_array.max() - patch_array.min())
                        patch_scaled = patch_std * (1 - (-1)) + (-1)


                if 'rotation' in full_pat:
                    patch_scaled = self.apply_rotation(patch_scaled, angle)
                    struct_array = self.apply_rotation(struct_array, angle)

                if 'noise' in full_pat:
                    patch_scaled = self.apply_noise(patch_scaled)

                if 'flip' in full_pat:
                    patch_scaled = self.apply_flip(patch_scaled)
                    struct_array = self.apply_flip(struct_array)

                if 'deform' in full_pat:
                    [patch_scaled, struct_array] = self.apply_deformation(patch_scaled, struct_array)


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


    def apply_deformation(self, arr, struct):
        arr = elasticdeform.deform_random_grid([arr, struct], sigma=5, order=3, axis=[(0,1,2), (0,1,2)])
        return arr


    def apply_flip(self, arr):
        arr = np.flip(arr, axis=(0,1,2)).copy()
        return arr
